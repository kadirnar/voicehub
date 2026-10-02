from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from voicehub.hub import resolve_pretrained_file
from voicehub.hub_transport import (
    _REDIRECT_HANDLER,
    HubDownloadError,
    _SafeRedirectHandler,
    download_hugging_face_file,
    download_hugging_face_snapshot,
    get_cached_hugging_face_commit,
)


class _CaseInsensitiveHeaders(dict):

    def get(self, key, default=None):
        lowered = key.lower()
        for candidate, value in self.items():
            if candidate.lower() == lowered:
                return value
        return default


class _Response(io.BytesIO):

    def __init__(self, content: bytes, headers=None, status: int = 200):
        super().__init__(content)
        self.headers = _CaseInsensitiveHeaders(headers or {})
        self.status = status

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class HubTransportTests(unittest.TestCase):

    @staticmethod
    def _json_response(value, *, headers=None):
        content = json.dumps(value).encode("utf-8")
        merged_headers = {"Content-Length": str(len(content))}
        merged_headers.update(headers or {})
        return _Response(content, merged_headers)

    def test_resolve_pretrained_file_preserves_local_path_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "assets" / "config.json"
            nested.parent.mkdir()
            nested.write_text("{}", encoding="utf-8")

            self.assertEqual(
                resolve_pretrained_file(
                    root,
                    "config.json",
                    subfolder="assets",
                ),
                nested,
            )
            self.assertEqual(
                resolve_pretrained_file(nested, "config.json"),
                nested,
            )
            with self.assertRaisesRegex(FileNotFoundError, "Could not find"):
                resolve_pretrained_file(root, "missing.json")
            with self.assertRaisesRegex(FileNotFoundError, "is a file"):
                resolve_pretrained_file(nested, "other.json")

    def test_download_encodes_url_and_never_persists_token(self):
        content = b"native checkpoint"
        sha256 = hashlib.sha256(content).hexdigest()
        response = _Response(
            content,
            {
                "Content-Length": str(len(content)),
                "ETag": f'"{sha256}"',
                "X-Repo-Commit": "a" * 40,
            },
        )
        captured = {}

        def open_request(request, *, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return response

        with tempfile.TemporaryDirectory() as directory, patch(
                "voicehub.hub_transport._URL_OPENER.open",
                side_effect=open_request,
        ):
            resolved = download_hugging_face_file(
                "owner/model",
                "config file.json",
                subfolder="nested assets",
                cache_dir=directory,
                revision="refs/pr/1",
                token="hf_secret",
            )

            self.assertEqual(resolved.read_bytes(), content)
            request = captured["request"]
            self.assertEqual(
                request.full_url,
                "https://huggingface.co/owner/model/resolve/"
                "refs%2Fpr%2F1/nested%20assets/config%20file.json",
            )
            self.assertEqual(
                request.get_header("Authorization"),
                "Bearer hf_secret",
            )
            self.assertGreater(captured["timeout"], 0)

            serialized_cache = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore") for path in Path(directory).rglob("*")
                if path.is_file())
            self.assertNotIn("hf_secret", serialized_cache)
            self.assertEqual(
                get_cached_hugging_face_commit(
                    "owner/model",
                    "config file.json",
                    subfolder="nested assets",
                    cache_dir=directory,
                    revision="refs/pr/1",
                ),
                "a" * 40,
            )

    def test_local_files_only_reuses_voicehub_cache_without_http(self):
        content = b"cached"
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                    "voicehub.hub_transport._URL_OPENER.open",
                    return_value=_Response(
                        content,
                        {
                            "Content-Length": str(len(content)),
                            "X-Repo-Commit": "b" * 40,
                        },
                    ),
            ):
                downloaded = download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                )

            with patch(
                    "voicehub.hub_transport._URL_OPENER.open",
                    side_effect=AssertionError("offline resolution attempted HTTP"),
            ):
                cached = download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                    local_files_only=True,
                )
            self.assertEqual(cached, downloaded)
            self.assertEqual(cached.read_bytes(), content)

    def test_standard_hugging_face_cache_is_read_in_offline_mode(self):
        commit = "c" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "models--owner--model"
            (repository / "refs").mkdir(parents=True)
            (repository / "refs" / "main").write_text(commit, encoding="utf-8")
            cached = repository / "snapshots" / commit / "nested" / "model.bin"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"legacy")

            resolved = download_hugging_face_file(
                "owner/model",
                "model.bin",
                subfolder="nested",
                cache_dir=root,
                local_files_only=True,
            )

        self.assertEqual(resolved, cached)

    def test_hugging_face_cache_with_shared_blob_symlinks_is_read(self):
        # huggingface_hub 1.x links snapshot files to blobs shared by all
        # repositories (``<cache>/blobs/<xx>/<sha256>``), outside the
        # repository folder but still inside the cache root.
        commit = "c" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "blobs" / "ab" / ("ab" + "0" * 62)
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"shared")
            snapshot = root / "models--owner--model" / "snapshots" / commit
            snapshot.mkdir(parents=True)
            cached = snapshot / "model.pth"
            cached.symlink_to(Path("..", "..", "..", "blobs", "ab", blob.name))

            resolved = download_hugging_face_file(
                "owner/model",
                "model.pth",
                cache_dir=root,
                revision=commit,
                local_files_only=True,
            )
            self.assertEqual(resolved, cached)
            self.assertEqual(resolved.read_bytes(), b"shared")

    def test_hugging_face_cache_symlink_outside_cache_root_is_rejected(self):
        commit = "c" * 40
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            target = Path(outside) / "model.pth"
            target.write_bytes(b"outside")
            snapshot = root / "models--owner--model" / "snapshots" / commit
            snapshot.mkdir(parents=True)
            (snapshot / "model.pth").symlink_to(target)

            with self.assertRaisesRegex(ValueError, "escapes its root"):
                download_hugging_face_file(
                    "owner/model",
                    "model.pth",
                    cache_dir=root,
                    revision=commit,
                    local_files_only=True,
                )

    def test_cached_etag_uses_conditional_request_and_handles_not_modified(self):
        content = b"unchanged"
        etag = '"cache-etag"'
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                    "voicehub.hub_transport._URL_OPENER.open",
                    return_value=_Response(
                        content,
                        {
                            "Content-Length": str(len(content)),
                            "ETag": etag,
                            "X-Repo-Commit": "d" * 40,
                        },
                    ),
            ):
                first = download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                )

            captured = {}

            def not_modified(request, *, timeout):
                del timeout
                captured["request"] = request
                raise HTTPError(
                    request.full_url,
                    304,
                    "Not Modified",
                    _CaseInsensitiveHeaders(),
                    None,
                )

            with patch(
                    "voicehub.hub_transport._URL_OPENER.open",
                    side_effect=not_modified,
            ):
                second = download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                )

        self.assertEqual(first, second)
        request_headers = {name.lower(): value for name, value in captured["request"].header_items()}
        self.assertEqual(request_headers["if-none-match"], etag)

    def test_network_failure_falls_back_to_complete_cached_file(self):
        content = b"available offline after timeout"
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                    "voicehub.hub_transport._URL_OPENER.open",
                    return_value=_Response(
                        content,
                        {
                            "Content-Length": str(len(content)),
                            "ETag": '"known"',
                        },
                    ),
            ):
                cached = download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                )
            with patch(
                    "voicehub.hub_transport._URL_OPENER.open",
                    side_effect=URLError("network unavailable"),
            ):
                resolved = download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                )

        self.assertEqual(resolved, cached)

    def test_hub_redirect_metadata_governs_immutable_lfs_download(self):
        content = b"large-file-content"
        sha256 = hashlib.sha256(content).hexdigest()

        def redirected_download(request, *, timeout):
            del timeout
            redirected = _REDIRECT_HANDLER.redirect_request(
                request,
                Mock(),
                302,
                "Found",
                {
                    "X-Linked-ETag": f'"{sha256}"',
                    "X-Linked-Size": str(len(content)),
                    "X-Repo-Commit": "f" * 40,
                },
                "https://cdn-lfs.huggingface.co/model.bin",
            )
            self.assertIsNotNone(redirected)
            return _Response(
                content,
                {
                    "Content-Length": str(len(content)),
                    # Object-storage ETags are not guaranteed to hash the file.
                    "ETag": '"' + ("0" * 64) + '"',
                },
            )

        with tempfile.TemporaryDirectory() as directory, patch(
                "voicehub.hub_transport._URL_OPENER.open",
                side_effect=redirected_download,
        ):
            resolved = download_hugging_face_file(
                "owner/model",
                "model.bin",
                cache_dir=directory,
            )
            metadata_paths = tuple(Path(directory).rglob("*.json"))
            self.assertEqual(len(metadata_paths), 1)
            metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(resolved.read_bytes(), content)

        self.assertEqual(metadata["commit"], "f" * 40)
        self.assertEqual(metadata["etag"], f'"{sha256}"')

    def test_integrity_failures_do_not_publish_partial_files(self):
        content = b"corrupt"
        with tempfile.TemporaryDirectory() as directory, patch(
                "voicehub.hub_transport._URL_OPENER.open",
                return_value=_Response(
                    content,
                    {
                        "Content-Length": str(len(content)),
                        "X-Checksum-Sha256": "0" * 64,
                        "X-Repo-Commit": "e" * 40,
                    },
                ),
        ):
            with self.assertRaisesRegex(HubDownloadError, "integrity"):
                download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                )

            self.assertFalse(tuple(Path(directory).rglob("*.incomplete")))
            self.assertFalse(tuple(Path(directory).rglob("*.json")))

    def test_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "voicehub.hub_transport._URL_OPENER.open",
                return_value=_Response(b"short", {"Content-Length": "99"}),
        ):
            with self.assertRaisesRegex(HubDownloadError, "unexpected size"):
                download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                )

    def test_remote_paths_and_revisions_reject_traversal(self):
        cases = (
            {
                "filename": "../secret"
            },
            {
                "filename": "model.bin",
                "subfolder": "../private"
            },
            {
                "filename": "model.bin",
                "revision": "../main"
            },
            {
                "filename": "model.bin",
                "revision": ""
            },
            {
                "filename": r"..\secret"
            },
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    download_hugging_face_file(
                        "owner/model",
                        cache_dir=directory,
                        **kwargs,
                    )

    def test_offline_cache_miss_has_actionable_location(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                    FileNotFoundError,
                    r"owner/model@main/config\.json",
            ):
                download_hugging_face_file(
                    "owner/model",
                    "config.json",
                    cache_dir=directory,
                    local_files_only=True,
                )

    def test_token_true_reads_runtime_environment_only(self):
        response = _Response(b"x", {"Content-Length": "1"})
        captured = {}

        def open_request(request, *, timeout):
            del timeout
            captured["authorization"] = request.get_header("Authorization")
            return response

        with tempfile.TemporaryDirectory() as directory:
            runtime_environment = {"HF_TOKEN": "runtime-only"}
            with patch.dict("os.environ", runtime_environment, clear=True):
                with patch(
                        "voicehub.hub_transport._URL_OPENER.open",
                        side_effect=open_request,
                ):
                    download_hugging_face_file(
                        "owner/model",
                        "model.bin",
                        cache_dir=directory,
                        token=True,
                    )
        self.assertEqual(captured["authorization"], "Bearer runtime-only")

    def test_cross_host_redirect_strips_authorization(self):
        handler = _SafeRedirectHandler()
        request = Request(
            "https://huggingface.co/owner/model/resolve/main/model.bin",
            headers={"Authorization": "Bearer secret"},
        )

        redirected = handler.redirect_request(
            request,
            Mock(),
            302,
            "Found",
            {},
            "https://cdn-lfs.huggingface.co/model.bin",
        )

        self.assertIsNotNone(redirected)
        self.assertIsNone(redirected.get_header("Authorization"))
        with self.assertRaisesRegex(HTTPError, "insecure"):
            handler.redirect_request(
                request,
                Mock(),
                302,
                "Found",
                {},
                "http://huggingface.co/model.bin",
            )

    def test_snapshot_download_resolves_commit_and_publishes_complete_manifest(self):
        commit = "1" * 40
        files = {
            "config.json": b"{}",
            "weights/model.bin": b"native weights",
        }
        responses = [
            self._json_response({"sha": commit}),
            self._json_response([{
                "path": path,
                "size": len(content),
                "type": "file",
            } for path, content in files.items()]),
            *[
                _Response(
                    content,
                    {
                        "Content-Length": str(len(content)),
                        "X-Repo-Commit": commit,
                    },
                ) for content in files.values()
            ],
        ]
        requests = []

        def open_request(request, *, timeout):
            self.assertGreater(timeout, 0)
            requests.append(request)
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as directory, patch(
                "voicehub.hub_transport._URL_OPENER.open",
                side_effect=open_request,
        ):
            snapshot = download_hugging_face_snapshot(
                "owner/model",
                cache_dir=directory,
                revision="main",
                token="hf_runtime_only",
            )
            self.assertEqual(
                (snapshot / "config.json").read_bytes(),
                files["config.json"],
            )
            self.assertEqual(
                (snapshot / "weights" / "model.bin").read_bytes(),
                files["weights/model.bin"],
            )
            manifests = tuple(Path(directory).glob("voicehub/repos/*/snapshot_refs/*.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["commit"], commit)
            self.assertEqual(
                [entry["path"] for entry in manifest["files"]],
                list(files),
            )

            with patch(
                    "voicehub.hub_transport._URL_OPENER.open",
                    side_effect=AssertionError("offline snapshot attempted HTTP"),
            ):
                offline = download_hugging_face_snapshot(
                    "owner/model",
                    cache_dir=directory,
                    revision="main",
                    local_files_only=True,
                )

            serialized_cache = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore") for path in Path(directory).rglob("*")
                if path.is_file())
            self.assertNotIn("hf_runtime_only", serialized_cache)

        self.assertEqual(offline, snapshot)
        self.assertIn("/api/models/owner/model/revision/main", requests[0].full_url)
        self.assertIn(
            f"/api/models/owner/model/tree/{commit}",
            requests[1].full_url,
        )
        for request in requests:
            self.assertEqual(
                request.get_header("Authorization"),
                "Bearer hf_runtime_only",
            )

    def test_snapshot_tree_pagination_and_patterns_are_bounded_and_deterministic(self):
        commit = "2" * 40
        api_path = f"/api/models/owner/model/tree/{commit}"
        next_link = (f"<https://huggingface.co{api_path}?cursor=next>; rel=\"next\"")
        selected = b"configuration"
        responses = [
            self._json_response({"sha": commit}),
            self._json_response(
                [{
                    "path": "README.md",
                    "size": 6,
                    "type": "file",
                }],
                headers={"Link": next_link},
            ),
            self._json_response([
                {
                    "path": "config.json",
                    "size": len(selected),
                    "type": "file",
                },
                {
                    "path": "private/config.json",
                    "size": 1,
                    "type": "file",
                },
            ]),
            _Response(
                selected,
                {
                    "Content-Length": str(len(selected)),
                    "X-Repo-Commit": commit,
                },
            ),
        ]
        requests = []

        def open_request(request, *, timeout):
            del timeout
            requests.append(request)
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as directory, patch(
                "voicehub.hub_transport._URL_OPENER.open",
                side_effect=open_request,
        ):
            snapshot = download_hugging_face_snapshot(
                "owner/model",
                cache_dir=directory,
                allow_patterns="*config.json",
                ignore_patterns="private/*",
            )
            resolved_content = (snapshot / "config.json").read_bytes()
            readme_exists = (snapshot / "README.md").exists()
            private_config_exists = (snapshot / "private" / "config.json").exists()

        self.assertEqual(resolved_content, selected)
        self.assertFalse(readme_exists)
        self.assertFalse(private_config_exists)
        self.assertEqual(len(requests), 4)
        self.assertEqual(requests[2].full_url, next_link.split("<", 1)[1].split(">", 1)[0])

    def test_snapshot_rejects_cross_origin_pagination(self):
        commit = "3" * 40
        responses = [
            self._json_response({"sha": commit}),
            self._json_response(
                [],
                headers={
                    "Link": (
                        "<https://attacker.invalid/api/models/owner/model/"
                        f"tree/{commit}?cursor=secret>; rel=\"next\"")
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
                "voicehub.hub_transport._URL_OPENER.open",
                side_effect=responses,
        ):
            with self.assertRaisesRegex(HubDownloadError, "untrusted"):
                download_hugging_face_snapshot(
                    "owner/model",
                    cache_dir=directory,
                    token="never-leak",
                )

    def test_hub_api_rejects_ambiguous_json_before_using_a_commit(self):
        documents = {
            "duplicate": (
                '{"sha":"discarded-secret-value","sha":"' + ("8" * 40) + '"}',
                "Duplicate JSON object key 'sha'",
            ),
            "constant": (
                '{"sha":"' + ("8" * 40) + '","metadata":{"score":NaN}}',
                "non-finite.*NaN",
            ),
            "overflow": (
                '{"sha":"' + ("8" * 40) + '","metadata":{"score":1e400}}',
                r"\$\.metadata\.score.*non-finite",
            ),
        }
        for name, (document, message) in documents.items():
            requests = []

            def open_request(request, *, timeout):
                del timeout
                requests.append(request)
                if len(requests) > 1:
                    raise AssertionError("Hub tree lookup must not run")
                content = document.encode("utf-8")
                return _Response(content, {"Content-Length": str(len(content))})

            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory, patch(
                    "voicehub.hub_transport._URL_OPENER.open",
                    side_effect=open_request,
            ):
                with self.assertRaisesRegex(HubDownloadError, message) as raised:
                    download_hugging_face_snapshot(
                        "owner/model",
                        cache_dir=directory,
                        token="runtime-only-token",
                    )

                self.assertEqual(len(requests), 1)
                self.assertNotIn("discarded-secret-value", str(raised.exception))
                self.assertNotIn("runtime-only-token", str(raised.exception))

    def test_ambiguous_file_cache_metadata_is_an_offline_cache_miss(self):
        content = b"cached"
        commit = "9" * 40
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                    "voicehub.hub_transport._URL_OPENER.open",
                    return_value=_Response(
                        content,
                        {
                            "Content-Length": str(len(content)),
                            "X-Repo-Commit": commit,
                        },
                    ),
            ):
                download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                )

            metadata_paths = tuple(Path(directory).glob("voicehub/repos/*/refs/**/*.json"))
            self.assertEqual(len(metadata_paths), 1)
            metadata_path = metadata_paths[0]
            encoded = metadata_path.read_text(encoding="utf-8")
            metadata_path.write_text(
                '{"repo_id":"discarded-secret-value",' + encoded[1:],
                encoding="utf-8",
            )

            self.assertIsNone(
                get_cached_hugging_face_commit(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                ))
            with self.assertRaisesRegex(FileNotFoundError, "not available"):
                download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                    local_files_only=True,
                )

    def test_ambiguous_snapshot_cache_manifest_is_an_offline_cache_miss(self):
        commit = "a" * 40
        content = b"{}"
        responses = [
            self._json_response({"sha": commit}),
            self._json_response([{
                "path": "config.json",
                "size": len(content),
                "type": "file",
            }]),
            _Response(
                content,
                {
                    "Content-Length": str(len(content)),
                    "X-Repo-Commit": commit,
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                    "voicehub.hub_transport._URL_OPENER.open",
                    side_effect=responses,
            ):
                download_hugging_face_snapshot(
                    "owner/model",
                    cache_dir=directory,
                )

            manifest_paths = tuple(Path(directory).glob("voicehub/repos/*/snapshot_refs/*.json"))
            self.assertEqual(len(manifest_paths), 1)
            manifest_path = manifest_paths[0]
            encoded = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text(
                '{"repo_id":"discarded-secret-value",' + encoded[1:],
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "not available"):
                download_hugging_face_snapshot(
                    "owner/model",
                    cache_dir=directory,
                    local_files_only=True,
                )

    def test_snapshot_failure_never_publishes_a_complete_manifest(self):
        commit = "4" * 40
        first = b"first"
        responses = [
            self._json_response({"sha": commit}),
            self._json_response([
                {
                    "path": "a.bin",
                    "size": len(first),
                    "type": "file",
                },
                {
                    "path": "b.bin",
                    "size": 6,
                    "type": "file",
                },
            ]),
            _Response(
                first,
                {
                    "Content-Length": str(len(first)),
                    "X-Repo-Commit": commit,
                },
            ),
            URLError("interrupted"),
        ]

        def open_request(*_args, **_kwargs):
            response = responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response

        with tempfile.TemporaryDirectory() as directory, patch(
                "voicehub.hub_transport._URL_OPENER.open",
                side_effect=open_request,
        ):
            with self.assertRaisesRegex(HubDownloadError, "Could not reach"):
                download_hugging_face_snapshot(
                    "owner/model",
                    cache_dir=directory,
                )
            self.assertFalse(tuple(Path(directory).glob("voicehub/repos/*/snapshot_refs/*.json")))

    def test_snapshot_offline_reads_an_existing_legacy_repository(self):
        commit = "5" * 40
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "models--owner--model"
            (repository / "refs").mkdir(parents=True)
            (repository / "refs" / "main").write_text(
                commit,
                encoding="utf-8",
            )
            snapshot = repository / "snapshots" / commit
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")

            resolved = download_hugging_face_snapshot(
                "owner/model",
                cache_dir=directory,
                local_files_only=True,
            )

        self.assertEqual(resolved, snapshot.resolve())

    def test_immutable_file_download_rejects_a_different_server_commit(self):
        requested = "6" * 40
        with tempfile.TemporaryDirectory() as directory, patch(
                "voicehub.hub_transport._URL_OPENER.open",
                return_value=_Response(
                    b"x",
                    {
                        "Content-Length": "1",
                        "X-Repo-Commit": "7" * 40,
                    },
                ),
        ):
            with self.assertRaisesRegex(HubDownloadError, "different commit"):
                download_hugging_face_file(
                    "owner/model",
                    "model.bin",
                    cache_dir=directory,
                    revision=requested,
                )


if __name__ == "__main__":
    unittest.main()
