"""Legacy checkpoint conversions must never write into the Hub cache."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from voicehub.architectures.f5tts.artifacts import resolve_f5tts_artifacts
from voicehub.architectures.f5tts.metadata import (
    VOCOS_CHECKPOINT_REVISION,
    VOCOS_LEGACY_CHECKPOINT,
    VOCOS_REPOSITORY,
)
from voicehub.architectures.kokoro.checkpoint import import_legacy_kokoro_voice
from voicehub.architectures.voxcpm2.checkpoint import export_voxcpm_checkpoint
from voicehub.architectures.voxcpm2.codec import VoxCPMAudioVAE
from voicehub.architectures.voxcpm2.modeling import VoxCPM2Model
from voicehub.architectures.voxcpm2.runtime import load_voxcpm2_runtime
from voicehub.models.kokoro.pipeline import KPipeline
from voicehub.path_utils import derived_artifact_path, voicehub_cache_root

from .test_native_voxcpm2 import _tiny_config, _tiny_tokenizer

_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_KOKORO_CONFIG = {
    "vocab": {
        " ": 1,
        "h": 2,
        "o": 3,
    },
    "n_token": 8,
    "hidden_dim": 512,
    "n_layer": 1,
    "max_dur": 4,
    "style_dim": 8,
    "n_mels": 8,
    "dim_in": 8,
    "max_conv_dim": 16,
    "plbert": {
        "hidden_size": 16,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 16,
        "num_hidden_layers": 1,
        "embedding_size": 8,
    },
    "istftnet": {
        "upsample_kernel_sizes": [20, 12],
        "upsample_rates": [10, 6],
        "gen_istft_hop_size": 5,
        "gen_istft_n_fft": 20,
        "resblock_dilation_sizes": [[1, 3, 5]],
        "resblock_kernel_sizes": [3],
        "upsample_initial_channel": 512,
    },
}


class _Pickle:
    """A payload written with ``torch.save`` (a legacy checkpoint)."""

    def __init__(self, value: object) -> None:
        self.value = value


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
    }


class DerivedArtifactCacheTest(unittest.TestCase):

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.hub = self.root / "hub"
        self.voicehub_cache = self.root / "voicehub-cache"
        environment = patch.dict(os.environ, {"VOICEHUB_CACHE": str(self.voicehub_cache)})
        environment.start()
        self.addCleanup(environment.stop)

    def _snapshot(self, repo_id: str, files: dict[str, object], *, revision: str = _COMMIT) -> Path:
        """Write a read-only Hub cache snapshot with regular files.

        This is the layout of VoiceHub's own download cache (and of Hub
        caches without symlink support).
        """
        repository = self.hub / f"models--{repo_id.replace('/', '--')}"
        snapshot = repository / "snapshots" / revision
        for name, value in files.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, _Pickle):
                torch.save(value.value, path)
            elif isinstance(value, Path):
                path.write_bytes(value.read_bytes())
            elif isinstance(value, str):
                path.write_text(value, encoding="utf-8")
            else:
                path.write_text(json.dumps(value), encoding="utf-8")
        self._make_read_only(repository)
        return snapshot

    def _make_read_only(self, repository: Path) -> None:
        paths = [repository, *repository.rglob("*")]
        for path in paths:
            path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

        def restore():
            for path in paths:
                if path.exists():
                    path.chmod(path.stat().st_mode | stat.S_IWUSR)

        self.addCleanup(restore)

    def _assert_unchanged(self, before: dict[str, tuple[int, int]]) -> None:
        after = _tree(self.hub)
        self.assertEqual(sorted(set(after) - set(before)), [], "files were written into the Hub cache")
        self.assertEqual(after, before)

    def _assert_derived(self, path: Path) -> None:
        self.assertTrue(path.is_file())
        self.assertTrue(path.resolve().is_relative_to(self.voicehub_cache.resolve()))

    def test_cache_root_unifies_environment_variables(self):
        with patch.dict(os.environ, {"VOICEHUB_CACHE": "/a", "VOICEHUB_CACHE_DIR": "/b"}):
            self.assertEqual(voicehub_cache_root(), Path("/a"))
            self.assertEqual(voicehub_cache_root("/c"), Path("/c"))
        with patch.dict(os.environ, {"VOICEHUB_CACHE_DIR": "/b", "XDG_CACHE_HOME": "/x"}):
            os.environ.pop("VOICEHUB_CACHE")
            self.assertEqual(voicehub_cache_root(), Path("/b"))
            os.environ.pop("VOICEHUB_CACHE_DIR")
            self.assertEqual(voicehub_cache_root(), Path("/x/voicehub"))

    def test_derived_path_is_keyed_by_source_identity(self):
        source = self.root / "weights.bin"
        source.write_bytes(b"one")
        first = derived_artifact_path(source, "weights.safetensors", namespace="demo")
        self.assertTrue(first.is_relative_to(self.voicehub_cache / "derived" / "demo"))
        self.assertEqual(first.name, "weights.safetensors")
        self.assertEqual(first, derived_artifact_path(source, "weights.safetensors", namespace="demo"))
        source.write_bytes(b"three")
        self.assertNotEqual(first, derived_artifact_path(source, "weights.safetensors", namespace="demo"))
        with self.assertRaises(ValueError):
            derived_artifact_path(source, "../escape", namespace="demo")
        with self.assertRaises(ValueError):
            derived_artifact_path(source, "x.safetensors", namespace="../demo")

    def test_kokoro_voice_conversion_leaves_hub_snapshot_untouched(self):
        voice = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 1, 256)
        self._snapshot(
            "acme/kokoro",
            {
                "config.json": _KOKORO_CONFIG,
                "kokoro.pth": _Pickle({}),
                "voices/af_test.pt": _Pickle(voice),
            },
        )
        before = _tree(self.hub)
        pipeline = KPipeline(
            "a",
            repo_id="acme/kokoro",
            model=False,
            checkpoint_filename="kokoro.pth",
            revision=_COMMIT,
            cache_dir=str(self.hub),
            local_files_only=True,
            allow_legacy_checkpoint_conversion=True,
        )

        loaded = pipeline.load_single_voice("af_test")

        self.assertTrue(torch.equal(loaded, voice))
        self._assert_unchanged(before)
        converted = list(self.voicehub_cache.rglob("af_test.voicehub.safetensors"))
        self.assertEqual(len(converted), 1)
        self._assert_derived(converted[0])

    def test_kokoro_reads_a_previously_converted_sibling(self):
        voice = torch.ones(3, 1, 256)
        sibling = self.root / "af_test.voicehub.safetensors"
        import_legacy_kokoro_voice(self._legacy_voice(voice * 2), output_path=sibling)
        self._snapshot(
            "acme/kokoro",
            {
                "config.json": _KOKORO_CONFIG,
                "kokoro.pth": _Pickle({}),
                "voices/af_test.pt": _Pickle(voice),
                "voices/af_test.voicehub.safetensors": sibling,
            },
        )
        pipeline = KPipeline(
            "a",
            repo_id="acme/kokoro",
            model=False,
            checkpoint_filename="kokoro.pth",
            revision=_COMMIT,
            cache_dir=str(self.hub),
            local_files_only=True,
        )

        self.assertTrue(torch.equal(pipeline.load_single_voice("af_test"), voice * 2))
        self.assertFalse(self.voicehub_cache.exists())

    def _legacy_voice(self, voice: torch.Tensor) -> Path:
        path = self.root / "legacy-voice.pt"
        torch.save(voice, path)
        return path

    def test_voxcpm_audiovae_conversion_leaves_hub_snapshot_untouched(self):
        config = _tiny_config()
        model_path = export_voxcpm_checkpoint(VoxCPM2Model(config), self.root / "model.safetensors")
        codec = VoxCPMAudioVAE(config.audio_vae_config)
        self._snapshot(
            "acme/voxcpm",
            {
                "model.safetensors": model_path,
                "config.json": config.to_dict(),
                "tokenizer.json": {},
                "audiovae.pth": _Pickle({"state_dict": codec.state_dict()}),
            },
        )
        before = _tree(self.hub)

        # The tokenizer is irrelevant here; only the AudioVAE is converted.
        tokenizer = _tiny_tokenizer(self.root / "tokenizer.json")
        with patch(
                "voicehub.architectures.voxcpm2.runtime.VoxCPM2Tokenizer.from_file",
                return_value=tokenizer,
        ):
            runtime = load_voxcpm2_runtime(
                "acme/voxcpm",
                revision=_COMMIT,
                trust_legacy_codec=True,
                cache_dir=str(self.hub),
                local_files_only=True,
            )

        self._assert_unchanged(before)
        converted = list(self.voicehub_cache.rglob("audiovae.safetensors"))
        self.assertEqual(len(converted), 1)
        self._assert_derived(converted[0])
        for name, tensor in codec.state_dict().items():
            self.assertTrue(torch.equal(runtime.codec.state_dict()[name], tensor))

    def test_f5tts_conversions_leave_hub_snapshots_untouched(self):
        vocoder_state = {"head.out.weight": torch.randn(4, 2)}
        model_state = {"transformer.weight": torch.randn(3)}
        self._snapshot(
            VOCOS_REPOSITORY,
            {VOCOS_LEGACY_CHECKPOINT: _Pickle(vocoder_state)},
            revision=VOCOS_CHECKPOINT_REVISION,
        )
        model_snapshot = self._snapshot(
            "acme/f5tts",
            {
                "model.pt": _Pickle({"ema_model": model_state}),
                "vocab.txt": "a\n",
            },
        )
        before = _tree(self.hub)

        artifacts = resolve_f5tts_artifacts(
            model_snapshot,
            model_name="custom",
            cache_dir=str(self.hub),
            local_files_only=True,
        )

        self._assert_unchanged(before)
        self._assert_derived(artifacts.checkpoint)
        self._assert_derived(artifacts.vocoder)
        self.assertEqual(artifacts.checkpoint.name, "model.safetensors")
        self.assertEqual(artifacts.vocoder.name, "pytorch_model.safetensors")


if __name__ == "__main__":
    unittest.main()
