import ast
import base64
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Event, Thread
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import torch

from voicehub import AutoConfig, AutoModelForTextToSpeech, PreTrainedTTSModel, TTSOutput, VoiceHubConfig
from voicehub.errors import LLMBackendCompatibilityError, LLMBackendRequestError
from voicehub.generation import GenerationConfig
from voicehub.llm_serving import (
    LLMBackend,
    LLMBackendConfig,
    LLMBackendSupport,
    LLMBackendTransport,
    LLMServingClient,
    RemoteCausalLMProxy,
    TokenGenerationRequest,
    TokenGenerationResult,
    get_llm_backend_support,
    list_llm_backend_support,
    register_llm_backend_support,
    unregister_llm_backend_support,
)
from voicehub.llm_serving.http import HTTPBackendClient, HTTPBackendResponse, join_endpoint


class _FakeHTTPResponse:

    def __init__(self, body, *, headers=None, status=200):
        self.body = body
        self.headers = {} if headers is None else headers
        self.status = status
        self.read_limit = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def read(self, limit=-1):
        self.read_limit = limit
        return self.body


@contextmanager
def _running_http_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _wave_bytes(samples, *, sample_rate=16000, channels=1):
    payload = io.BytesIO()
    with wave.open(payload, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return payload.getvalue()


class _RemoteSpeechConfig(VoiceHubConfig):
    model_type = "qwen3tts"


class _RemoteSpeechModel(PreTrainedTTSModel):
    config_class = _RemoteSpeechConfig

    def __init__(self, config=None, **kwargs):
        self.load_count = 0
        self.validation_count = 0
        super().__init__(
            self._coerce_config(config),
            **kwargs,
        )

    def _load_pretrained_model(self):
        self.load_count += 1
        self.model = object()

    def _validate_generation_inputs(self, model_inputs):
        del model_inputs
        self.validation_count += 1
        raise AssertionError("Native validation must not run for speech transport.")

    def _generate(self, text, **kwargs):
        del text, kwargs
        raise AssertionError("Native generation must not run for speech transport.")


class _RecordingSpeechClient:

    def __init__(self):
        self.calls = []

    def synthesize(self, model_type, inputs, *, default_sample_rate):
        self.calls.append((model_type, dict(inputs), default_sample_rate))
        return TTSOutput(
            audio=[0.0, 0.25],
            sample_rate=default_sample_rate,
            metadata={"backend": "recording"},
        )


class LLMBackendConfigurationTests(unittest.TestCase):

    def test_external_configuration_normalizes_aliases_and_endpoint(self):
        config = LLMBackendConfig(
            backend="vllm-omni",
            endpoint="https://engine.example/v1/",
            transport="audio",
            model="  org/model  ",
        )

        self.assertIs(config.backend, LLMBackend.VLLM)
        self.assertIs(config.transport, LLMBackendTransport.SPEECH)
        self.assertEqual(config.endpoint, "https://engine.example/v1")
        self.assertEqual(config.model, "org/model")

    def test_runtime_secrets_are_redacted_from_diagnostics(self):
        config = LLMBackendConfig(
            backend="sglang",
            endpoint="http://localhost:30000",
            api_key="top-secret-api-key",
            headers={"X-Private": "top-secret-header"},
            extra_body={"adapter": "top-secret-adapter"},
        )

        diagnostic = config.to_dict()
        rendered = repr(config)
        self.assertEqual(diagnostic["api_key"], "<redacted>")
        self.assertEqual(diagnostic["headers"], {"X-Private": "<redacted>"})
        self.assertEqual(diagnostic["extra_body_keys"], ("adapter", ))
        for secret in (
                "top-secret-api-key",
                "top-secret-header",
                "top-secret-adapter",
        ):
            self.assertNotIn(secret, rendered)
            self.assertNotIn(secret, repr(diagnostic))

    def test_extra_body_is_deeply_immutable_and_request_copies_are_isolated(self):
        source = {
            "extra_params": {
                "stages": [{
                    "name": "semantic",
                    "options": {
                        "temperature": 0.7
                    },
                }],
            },
        }
        config = LLMBackendConfig(
            backend="vllm",
            endpoint="http://localhost:8000",
            extra_body=source,
        )

        source["extra_params"]["stages"][0]["options"]["temperature"] = 9.0
        source["extra_params"]["stages"].append({"name": "acoustic"})
        first = config.request_extra_body()
        self.assertEqual(
            first,
            {
                "extra_params": {
                    "stages": [{
                        "name": "semantic",
                        "options": {
                            "temperature": 0.7
                        },
                    }],
                },
            },
        )
        with self.assertRaises(TypeError):
            config.extra_body["extra_params"]["new_field"] = True
        with self.assertRaises(TypeError):
            config.extra_body["extra_params"]["stages"][0]["name"] = "changed"

        first["extra_params"]["stages"][0]["options"]["temperature"] = 0.1
        first["extra_params"]["stages"].append({"name": "vocoder"})
        second = config.request_extra_body()
        self.assertEqual(
            second["extra_params"]["stages"],
            [{
                "name": "semantic",
                "options": {
                    "temperature": 0.7
                },
            }],
        )

    def test_configuration_rejects_unsafe_connection_settings(self):
        invalid_values = (
            {
                "backend": "vllm"
            },
            {
                "backend": "vllm",
                "endpoint": "ftp://engine.example",
            },
            {
                "backend": "vllm",
                "endpoint": "https://user:password@engine.example",
            },
            {
                "backend": "vllm",
                "endpoint": "https://engine.example?token=secret",
            },
            {
                "backend": "vllm",
                "endpoint": "https://engine.example",
                "headers": {
                    "Host": "other.example"
                },
            },
            {
                "backend": "vllm",
                "endpoint": "https://engine.example",
                "headers": {
                    "X-Test": "one\nInjected: two"
                },
            },
            {
                "backend": "vllm",
                "endpoint": "https://engine.example",
                "api_key": "secret",
                "headers": {
                    "Authorization": "Bearer another"
                },
            },
            {
                "backend": "vllm",
                "endpoint": "https://engine.example",
                "extra_body": {
                    "input": "owned-by-request"
                },
            },
            {
                "backend": "vllm",
                "endpoint": "https://engine.example",
                "extra_body": {
                    "temperature_override": float("nan")
                },
            },
        )

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    LLMBackendConfig(**values)

    def test_native_configuration_rejects_external_settings(self):
        with self.assertRaisesRegex(ValueError, "native backend"):
            LLMBackendConfig(
                backend="native",
                endpoint="http://localhost:8000",
            )

    def test_from_value_rejects_unknown_options_and_backend_disagreement(self):
        with self.assertRaises(TypeError):
            LLMBackendConfig.from_value(
                {
                    "endpoint": "http://localhost:8000",
                    "unknown_option": True,
                },
                backend="vllm",
            )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            LLMBackendConfig.from_value(
                {
                    "backend": "sglang",
                    "endpoint": "http://localhost:8000",
                },
                backend="vllm",
            )

    def test_token_request_validates_prompt_and_generation_length(self):
        for prompt in ([], [-1], [True], "1,2"):
            with self.subTest(prompt=prompt):
                with self.assertRaises((TypeError, ValueError)):
                    TokenGenerationRequest(
                        prompt_token_ids=prompt,
                        max_new_tokens=1,
                    )
        with self.assertRaises(ValueError):
            TokenGenerationRequest(
                prompt_token_ids=[1],
                max_new_tokens=0,
            )

    def test_token_request_seed_is_a_nonnegative_signed_int64(self):
        signed_int64_max = 2**63 - 1
        for seed in (0, signed_int64_max):
            with self.subTest(seed=seed):
                request = TokenGenerationRequest(
                    prompt_token_ids=[1],
                    max_new_tokens=1,
                    seed=seed,
                )
                self.assertEqual(request.seed, seed)

        for seed in (-1, signed_int64_max + 1):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "seed"):
                    TokenGenerationRequest(
                        prompt_token_ids=[1],
                        max_new_tokens=1,
                        seed=seed,
                    )

    def test_token_ids_are_nonnegative_signed_int64_values(self):
        signed_int64_max = 2**63 - 1
        request = TokenGenerationRequest(
            prompt_token_ids=[0, signed_int64_max],
            stop_token_ids=[signed_int64_max],
            max_new_tokens=1,
        )
        result = TokenGenerationResult(token_ids=[0, signed_int64_max])

        self.assertEqual(request.prompt_token_ids, (0, signed_int64_max))
        self.assertEqual(result.token_ids, (0, signed_int64_max))
        for field, value in (
            ("prompt_token_ids", [signed_int64_max + 1]),
            ("stop_token_ids", [signed_int64_max + 1]),
        ):
            request_values = {
                "prompt_token_ids": [1],
                "max_new_tokens": 1,
                field: value,
            }
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError,
                    "signed 64-bit",
            ):
                TokenGenerationRequest(**request_values)
        with self.assertRaisesRegex(ValueError, "signed 64-bit"):
            TokenGenerationResult(token_ids=[signed_int64_max + 1])


class LLMBackendSupportTests(unittest.TestCase):

    def test_support_matrix_resolves_verified_transport(self):
        orpheus, orpheus_transport = get_llm_backend_support(
            "orpheustts",
            "vllm",
        )
        qwen, qwen_transport = get_llm_backend_support(
            "qwen3-tts",
            "sglang-omni",
        )

        self.assertEqual(orpheus.model_type, "orpheustts")
        self.assertIs(orpheus_transport, LLMBackendTransport.TOKENS)
        self.assertEqual(qwen.engine, "SGLang-Omni")
        self.assertIs(qwen_transport, LLMBackendTransport.SPEECH)
        self.assertEqual(qwen.task_type_without_reference, "CustomVoice")
        self.assertEqual(qwen.task_type_with_reference, "Base")
        self.assertEqual(
            get_llm_backend_support("fishtts", "sglang")[0].reference_format,
            "references",
        )
        self.assertEqual(
            get_llm_backend_support("mosstts", "vllm")[0].speech_string_options,
            ("ambient_sound", ),
        )
        sglang_models = {item.model_type for item in list_llm_backend_support(backend="sglang")}
        self.assertIn("fishtts", sglang_models)
        self.assertNotIn("higgstts", sglang_models)

    def test_support_matrix_fails_closed_with_architecture_reason(self):
        with self.assertRaisesRegex(
                LLMBackendCompatibilityError,
                "64-token repetition window",
        ):
            get_llm_backend_support("outetts", "vllm")
        with self.assertRaisesRegex(
                LLMBackendCompatibilityError,
                "through speech, not tokens",
        ):
            get_llm_backend_support(
                "qwen3tts",
                "vllm",
                transport="tokens",
            )
        with self.assertRaisesRegex(
                LLMBackendCompatibilityError,
                "does not use an external",
        ):
            get_llm_backend_support("qwen3tts", "native")
        with self.assertRaisesRegex(
                LLMBackendCompatibilityError,
                "No verified engine adapter exists for this architecture",
        ):
            get_llm_backend_support("future-unregistered-tts", "vllm")

    def test_specialized_failure_reasons_are_owned_by_architectures(self):
        from voicehub.registry import list_model_specs

        blocked = {}
        for spec in list_model_specs():
            architecture = spec.native_architecture
            if architecture is None:
                continue
            reason = architecture.metadata.get("external_llm_backend_blocker")
            if reason is not None:
                blocked[spec.model_type] = reason

        self.assertEqual(
            set(blocked),
            {
                "bark",
                "chatterbox",
                "conversationtts",
                "csm",
                "dia",
                "gptsovits",
                "neutts",
                "outetts",
                "parlertts",
                "vibevoice",
                "vui",
                "xtts",
                "zonos",
                "zonos2",
            },
        )
        registered_backends = {}
        for support in list_llm_backend_support():
            registered_backends.setdefault(support.model_type, set()).add(support.backend)
        for model_type, reason in blocked.items():
            with self.subTest(model_type=model_type):
                self.assertIsInstance(reason, str)
                self.assertEqual(reason, reason.strip())
                backend = next(
                    candidate for candidate in (
                        LLMBackend.VLLM,
                        LLMBackend.SGLANG,
                    ) if candidate not in registered_backends.get(model_type, set()))
                with self.assertRaises(LLMBackendCompatibilityError) as raised:
                    get_llm_backend_support(model_type, backend)
                self.assertIn(reason, str(raised.exception))

    def test_support_metadata_is_validated_and_json_serializable(self):
        support = LLMBackendSupport(
            model_type=" future-remote-tts ",
            backend="vllm",
            transports=("speech", ),
            default_transport="speech",
            engine=" Future engine ",
            checkpoint_family=" Future checkpoint ",
            task_type_without_reference="FutureVoice",
            task_type_with_reference="FutureClone",
            task_type_aliases=(("speak", "FutureGenerate"), ),
            reference_format="references",
            speech_string_options=("emotion_prompt", ),
        )

        self.assertEqual(support.model_type, "future-remote-tts")
        serialized = json.loads(json.dumps(support.to_dict()))
        self.assertEqual(
            serialized,
            {
                "model_type": "future-remote-tts",
                "backend": "vllm",
                "transports": ["speech"],
                "default_transport": "speech",
                "engine": "Future engine",
                "checkpoint_family": "Future checkpoint",
                "notes": "",
                "task_type_without_reference": "FutureVoice",
                "task_type_with_reference": "FutureClone",
                "task_type_aliases": {
                    "speak": "FutureGenerate"
                },
                "reference_format": "references",
                "speech_string_options": ["emotion_prompt"],
                "speech_input_options": list(support.speech_input_options),
                "speech_default_options": list(support.speech_default_options),
                "speech_native_only_options": list(support.speech_native_only_options),
            },
        )
        self.assertIn("emotion_prompt", support.speech_input_options)
        self.assertIn("emotion_prompt", support.speech_default_options)
        self.assertNotIn("emotion_prompt", support.speech_native_only_options)
        with self.assertRaisesRegex(ValueError, "concrete transports"):
            LLMBackendSupport(
                model_type="bad-auto-transport",
                backend="vllm",
                transports=("auto", ),
                default_transport="auto",
                engine="engine",
                checkpoint_family="checkpoint",
            )
        with self.assertRaisesRegex(ValueError, "request-owned"):
            LLMBackendSupport(
                model_type="bad-owned-field",
                backend="vllm",
                transports=("speech", ),
                default_transport="speech",
                engine="engine",
                checkpoint_family="checkpoint",
                speech_string_options=("input", ),
            )
        with self.assertRaisesRegex(ValueError, "request-owned"):
            LLMBackendSupport(
                model_type="bad-typed-field",
                backend="vllm",
                transports=("speech", ),
                default_transport="speech",
                engine="engine",
                checkpoint_family="checkpoint",
                speech_string_options=("temperature", ),
            )

    def test_extension_support_drives_payload_without_shared_model_branch(self):
        support = LLMBackendSupport(
            model_type="future-remote-tts",
            backend="vllm",
            transports=("speech", ),
            default_transport="speech",
            engine="Future engine",
            checkpoint_family="Future checkpoint",
            task_type_without_reference="FutureVoice",
            task_type_with_reference="FutureClone",
            task_type_aliases=(("speak", "FutureGenerate"), ),
            reference_format="references",
            speech_string_options=("emotion_prompt", ),
        )
        register_llm_backend_support(support)
        try:
            resolved, transport = get_llm_backend_support(
                "future-remote-tts",
                "vllm",
            )
            self.assertIs(resolved, support)
            self.assertIs(transport, LLMBackendTransport.SPEECH)
            client = LLMServingClient(
                LLMBackendConfig(
                    backend="vllm",
                    endpoint="http://localhost:8091",
                    transport="speech",
                ))
            payload = client._speech_payload(
                "future-remote-tts",
                {
                    "text": "Hello",
                    "reference_audio": "https://example.com/reference.wav",
                    "reference_text": "Reference",
                    "mode": "speak",
                    "emotion_prompt": "A quiet room",
                },
            )
            self.assertEqual(payload["task_type"], "FutureGenerate")
            self.assertEqual(payload["emotion_prompt"], "A quiet room")
            self.assertEqual(
                payload["references"],
                [{
                    "audio_path": "https://example.com/reference.wav",
                    "text": "Reference",
                }],
            )
        finally:
            self.assertIs(
                unregister_llm_backend_support("future-remote-tts", "vllm"),
                support,
            )
        self.assertIsNone(unregister_llm_backend_support(
            "future-remote-tts",
            "vllm",
            missing_ok=True,
        ))
        with self.assertRaisesRegex(ValueError, "cannot be unregistered"):
            unregister_llm_backend_support("qwen3tts", "vllm")

    def test_support_listing_remains_backend_and_framework_lazy(self):
        script = (
            "import sys;"
            "from voicehub.llm_serving import list_llm_backend_support;"
            "print(len(list_llm_backend_support()),"
            "'torch' in sys.modules,"
            "'voicehub.llm_serving.backends' in sys.modules)")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "14 False False")

    def test_architecture_failure_lookup_remains_framework_lazy(self):
        script = (
            "import sys\nfrom voicehub.errors import LLMBackendCompatibilityError\nfrom voicehub.llm_serving import get_llm_backend_support\ntry:\n    get_llm_backend_support('outetts', 'vllm')\nexcept LLMBackendCompatibilityError as error:\n    print('64-token repetition window' in str(error), 'torch' in sys.modules, 'voicehub.models.outetts.modeling' in sys.modules)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "True False False")

    def test_shared_support_has_no_provider_keyed_failure_map(self):
        from voicehub.registry import list_model_specs

        registered = {spec.model_type for spec in list_model_specs()}
        path = (Path(__file__).parents[1] / "voicehub" / "llm_serving" / "support.py")
        provider_maps = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                key.value
                for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            matched = sorted(keys & registered)
            if matched:
                provider_maps.append((node.lineno, matched))

        self.assertEqual(provider_maps, [])

    def test_shared_runtime_has_no_registered_model_comparisons(self):
        from voicehub.registry import list_model_specs

        registered = {spec.model_type for spec in list_model_specs()}
        comparisons = []
        package = Path(__file__).parents[1] / "voicehub"
        for path in package.rglob("*.py"):
            relative = path.relative_to(package)
            if relative.parts[0] in {"models", 'architectures', "components"}:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                for operand in (node.left, *node.comparators):
                    if (isinstance(operand, ast.Constant) and operand.value in registered):
                        comparisons.append((str(relative), node.lineno, operand.value), )

        self.assertEqual(comparisons, [])


class HTTPBackendTests(unittest.TestCase):

    def test_endpoint_joining_handles_bases_and_complete_routes(self):
        self.assertEqual(
            join_endpoint("http://localhost:8000", "/v1/completions"),
            "http://localhost:8000/v1/completions",
        )
        self.assertEqual(
            join_endpoint("http://localhost:8000/v1", "/v1/completions"),
            "http://localhost:8000/v1/completions",
        )
        self.assertEqual(
            join_endpoint(
                "http://localhost:8000/v1/completions",
                "/v1/completions",
            ),
            "http://localhost:8000/v1/completions",
        )

    def test_request_errors_do_not_leak_credentials(self):
        secret = "super-secret-token"
        config = LLMBackendConfig(
            backend="vllm",
            endpoint="http://localhost:8000",
            api_key=secret,
        )
        client = HTTPBackendClient(config)
        captured = {}

        def unavailable(request, **kwargs):
            captured["authorization"] = request.get_header("Authorization")
            captured["timeout"] = kwargs["timeout"]
            raise URLError(f"connection failed while using {secret}")

        with patch(
                "voicehub.llm_serving.http.urlopen",
                side_effect=unavailable,
        ), self.assertRaises(LLMBackendRequestError) as raised:
            client.post_json_document("/generate", {"input_ids": [1]})

        self.assertEqual(captured["authorization"], f"Bearer {secret}")
        self.assertEqual(captured["timeout"], 300.0)
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(secret, repr(raised.exception))

    def test_http_error_response_stream_is_closed(self):
        client = HTTPBackendClient(LLMBackendConfig(
            backend="vllm",
            endpoint="http://localhost:8000",
        ))
        response_stream = io.BytesIO(b'{"error":"rejected"}')
        backend_error = HTTPError(
            "http://localhost:8000/v1/completions",
            422,
            "Unprocessable Entity",
            {},
            response_stream,
        )

        with patch(
                "voicehub.llm_serving.http.urlopen",
                side_effect=backend_error,
        ), self.assertRaisesRegex(LLMBackendRequestError, "HTTP 422"):
            client.post_json_document("/v1/completions", {"prompt": [1]})

        self.assertTrue(response_stream.closed)

    def test_response_size_is_bounded_before_and_during_read(self):
        config = LLMBackendConfig(
            backend="sglang",
            endpoint="http://localhost:30000",
            max_response_bytes=4,
        )
        client = HTTPBackendClient(config)
        declared = _FakeHTTPResponse(
            b"small",
            headers={"Content-Length": "5"},
        )
        with patch(
                "voicehub.llm_serving.http.urlopen",
                return_value=declared,
        ), self.assertRaisesRegex(LLMBackendRequestError, "exceeds 4 bytes"):
            client.post_json_document("/generate", {})

        undeclared = _FakeHTTPResponse(b"large")
        with patch(
                "voicehub.llm_serving.http.urlopen",
                return_value=undeclared,
        ), self.assertRaisesRegex(LLMBackendRequestError, "exceeds 4 bytes"):
            client.post_json_document("/generate", {})
        self.assertEqual(undeclared.read_limit, 5)

    def test_json_response_rejects_ambiguous_keys_and_non_finite_numbers(self):
        client = HTTPBackendClient(LLMBackendConfig(
            backend="sglang",
            endpoint="http://localhost:30000",
        ))
        ambiguities = (
            (
                "duplicate",
                b'{"token_ids":["discarded-secret"],"token_ids":[7]}',
                r"Duplicate JSON object key 'token_ids'",
            ),
            (
                "constant",
                b'{"meta":{"score":NaN}}',
                r"non-finite JSON constant 'NaN'",
            ),
            (
                "overflow",
                b'{"meta":{"score":1e400}}',
                r"\$\.meta\.score.*non-finite",
            ),
        )

        for name, body, diagnostic in ambiguities:
            with self.subTest(name=name):
                with patch(
                        "voicehub.llm_serving.http.urlopen",
                        return_value=_FakeHTTPResponse(body),
                ), self.assertRaisesRegex(
                        LLMBackendRequestError,
                        diagnostic,
                ) as raised:
                    client.post_json_document("/generate", {"input_ids": [1]})

                rendered = str(raised.exception)
                self.assertIn("sglang", rendered)
                self.assertIn("/generate", rendered)
                self.assertNotIn("discarded-secret", rendered)

    def test_redirects_fail_closed_without_forwarding_runtime_secrets(self):
        target_requests = []

        class RedirectTargetHandler(BaseHTTPRequestHandler):

            def _capture(self):
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length:
                    self.rfile.read(content_length)
                target_requests.append({
                    "method": self.command,
                    "headers": dict(self.headers.items()),
                })
                payload = b'{"unexpected":"redirect followed"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _capture
            do_POST = _capture

            def log_message(self, format_string, *args):
                del format_string, args

        api_secret = "authorization-must-not-leak"
        header_secret = "custom-header-must-not-leak"
        with _running_http_server(RedirectTargetHandler) as target:
            target_url = (f"http://127.0.0.1:{target.server_address[1]}/captured")

            class RedirectSourceHandler(BaseHTTPRequestHandler):

                def do_POST(self):
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if content_length:
                        self.rfile.read(content_length)
                    status = int(self.path.rsplit("/", 1)[-1])
                    self.send_response(status)
                    self.send_header("Location", target_url)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

                def log_message(self, format_string, *args):
                    del format_string, args

            with _running_http_server(RedirectSourceHandler) as source:
                client = HTTPBackendClient(
                    LLMBackendConfig(
                        backend="vllm",
                        endpoint=(f"http://127.0.0.1:{source.server_address[1]}"),
                        api_key=api_secret,
                        headers={"X-VoiceHub-Secret": header_secret},
                        timeout=2,
                    ))
                for status in (301, 302, 303, 307, 308):
                    with self.subTest(status=status):
                        target_requests.clear()
                        raised = None
                        try:
                            client.post_json_document(
                                f"/redirect/{status}",
                                {"prompt": [1]},
                            )
                        except LLMBackendRequestError as error:
                            raised = error

                        received_values = {
                            value
                            for request in target_requests
                            for value in request["headers"].values()
                        }
                        self.assertNotIn(api_secret, received_values)
                        self.assertNotIn(header_secret, received_values)
                        self.assertEqual(
                            target_requests,
                            [],
                            "The HTTP client followed a redirect to another "
                            "origin.",
                        )
                        self.assertIsNotNone(
                            raised,
                            "Redirect responses must fail closed.",
                        )


class TokenBackendTests(unittest.TestCase):

    def test_vllm_token_payload_and_response_parsing(self):
        response = _FakeHTTPResponse(
            json.dumps({
                "choices": [{
                    "token_ids": [40, 41],
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                },
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        captured = {}

        def respond(request, **kwargs):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = kwargs["timeout"]
            return response

        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8000/v1",
                model="org/orpheus",
                api_key="runtime-key",
                extra_body={"truncate_prompt_tokens": 2048},
            ))
        request = TokenGenerationRequest(
            prompt_token_ids=[10, 20, 30],
            max_new_tokens=12,
            temperature=0.7,
            top_p=0.9,
            top_k=25,
            min_p=0.05,
            repetition_penalty=1.1,
            stop_token_ids=[99, 100],
            seed=7,
        )

        with patch(
                "voicehub.llm_serving.http.urlopen",
                side_effect=respond,
        ):
            result = client.generate_tokens(request)

        self.assertEqual(captured["url"], "http://localhost:8000/v1/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer runtime-key")
        self.assertEqual(captured["timeout"], 300.0)
        payload = captured["payload"]
        self.assertEqual(payload["model"], "org/orpheus")
        self.assertEqual(payload["prompt"], [10, 20, 30])
        self.assertEqual(payload["max_tokens"], 12)
        self.assertEqual(payload["stop_token_ids"], [99, 100])
        self.assertEqual(payload["truncate_prompt_tokens"], 2048)
        self.assertTrue(payload["return_token_ids"])
        self.assertFalse(payload["skip_special_tokens"])
        self.assertEqual(result.token_ids, (40, 41))
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, 3)
        self.assertEqual(result.completion_tokens, 2)

    def test_sglang_token_payload_and_response_parsing(self):
        response = _FakeHTTPResponse(
            json.dumps({
                "meta_info": {
                    "output_token_ids": [70, 71, 72],
                    "finish_reason": {
                        "type": "length"
                    },
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                },
            }).encode("utf-8"), )
        captured = {}

        def respond(request, **kwargs):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = kwargs["timeout"]
            return response

        client = LLMServingClient(
            LLMBackendConfig(
                backend="sglang",
                endpoint="http://localhost:30000",
                model="org/llasa",
            ))
        request = TokenGenerationRequest(
            prompt_token_ids=[11, 12],
            max_new_tokens=20,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            min_p=0.02,
            repetition_penalty=1.2,
            stop_token_ids=[2],
            seed=123,
        )

        with patch(
                "voicehub.llm_serving.http.urlopen",
                side_effect=respond,
        ):
            result = client.generate_tokens(request)

        self.assertEqual(captured["url"], "http://localhost:30000/generate")
        payload = captured["payload"]
        self.assertEqual(payload["input_ids"], [11, 12])
        self.assertNotIn("model", payload)
        self.assertFalse(payload["stream"])
        self.assertEqual(
            payload["sampling_params"],
            {
                "max_new_tokens": 20,
                "temperature": 0.8,
                "repetition_penalty": 1.2,
                "top_p": 0.95,
                "top_k": 50,
                "min_p": 0.02,
                "stop_token_ids": [2],
                "sampling_seed": 123,
            },
        )
        self.assertEqual(result.token_ids, (70, 71, 72))
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.prompt_tokens, 2)
        self.assertEqual(result.completion_tokens, 3)

    def test_sglang_maps_disabled_top_k_and_rejects_large_penalty(self):
        client = LLMServingClient(
            LLMBackendConfig(
                backend="sglang",
                endpoint="http://localhost:30000",
                model="org/llasa",
            ))
        response = {
            "output_ids": [7],
            "meta_info": {
                "finish_reason": {
                    "type": "stop"
                }
            },
        }
        with patch.object(
                client.http,
                "post_json_document",
                return_value=response,
        ) as post:
            client.generate_tokens(
                TokenGenerationRequest(
                    prompt_token_ids=[1, 2],
                    max_new_tokens=4,
                    top_k=0,
                ))

        route, payload = post.call_args.args
        self.assertEqual(route, "/generate")
        self.assertEqual(payload["sampling_params"]["top_k"], -1)
        self.assertNotIn("model", payload)

        with patch.object(client.http, "post_json_document") as post:
            with self.assertRaisesRegex(
                    LLMBackendCompatibilityError,
                    "repetition_penalty.*2",
            ):
                client.generate_tokens(
                    TokenGenerationRequest(
                        prompt_token_ids=[1, 2],
                        max_new_tokens=4,
                        repetition_penalty=2.01,
                    ))
        post.assert_not_called()

    def test_token_protocol_errors_are_actionable(self):
        no_token_ids = _FakeHTTPResponse(
            json.dumps({
                "choices": [{
                    "text": "decoded text only"
                }]
            }).encode("utf-8"))
        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8000",
                model="org/orpheus",
            ))
        with patch(
                "voicehub.llm_serving.http.urlopen",
                return_value=no_token_ids,
        ), self.assertRaisesRegex(
                LLMBackendRequestError,
                "did not return token IDs",
        ):
            client.generate_tokens(TokenGenerationRequest(
                prompt_token_ids=[1],
                max_new_tokens=1,
            ))

        malformed = _FakeHTTPResponse(b"not-json")
        with patch(
                "voicehub.llm_serving.http.urlopen",
                return_value=malformed,
        ), self.assertRaisesRegex(
                LLMBackendRequestError,
                "malformed JSON",
        ):
            client.generate_tokens(TokenGenerationRequest(
                prompt_token_ids=[1],
                max_new_tokens=1,
            ))

    def test_vllm_token_generation_requires_server_model(self):
        client = LLMServingClient(LLMBackendConfig(
            backend="vllm",
            endpoint="http://localhost:8000",
        ))
        with self.assertRaisesRegex(
                LLMBackendRequestError,
                "requires `LLMBackendConfig.model`",
        ):
            client.generate_tokens(TokenGenerationRequest(
                prompt_token_ids=[1],
                max_new_tokens=1,
            ))


class RemoteCausalLMProxyTests(unittest.TestCase):

    def test_proxy_preserves_prompt_and_appends_remote_suffix(self):

        class RecordingClient:

            def __init__(self):
                self.request = None

            def generate_tokens(self, request):
                self.request = request
                return TokenGenerationResult(
                    token_ids=(20, 21),
                    finish_reason="stop",
                )

        client = RecordingClient()
        proxy = RemoteCausalLMProxy(client, model_type="orpheustts")
        input_ids = torch.tensor([[0, 10, 11]], dtype=torch.long)
        attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
        generation_config = GenerationConfig(
            max_new_tokens=8,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            min_p=0.1,
            repetition_penalty=1.05,
            eos_token_id=(98, 99),
            seed=4,
        )

        output = proxy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

        self.assertIs(proxy.eval(), proxy)
        self.assertEqual(client.request.prompt_token_ids, (10, 11))
        self.assertEqual(client.request.stop_token_ids, (98, 99))
        self.assertEqual(client.request.max_new_tokens, 8)
        torch.testing.assert_close(
            output.sequences,
            torch.tensor([[0, 10, 11, 20, 21]]),
        )
        torch.testing.assert_close(
            output.generated_lengths,
            torch.tensor([2]),
        )
        torch.testing.assert_close(
            output.finished,
            torch.tensor([True]),
        )

    def test_proxy_rejects_batching_and_unknown_generate_options(self):

        class UnusedClient:

            def generate_tokens(self, request):
                del request
                raise AssertionError("The invalid request must not be sent.")

        proxy = RemoteCausalLMProxy(UnusedClient(), model_type="llasa")
        config = GenerationConfig(max_new_tokens=1)
        with self.assertRaisesRegex(
                LLMBackendCompatibilityError,
                "batch size 1",
        ):
            proxy.generate(
                input_ids=torch.tensor([[1], [2]]),
                generation_config=config,
            )
        with self.assertRaisesRegex(
                LLMBackendCompatibilityError,
                "unsupported option.*use_model_defaults",
        ):
            proxy.generate(
                input_ids=torch.tensor([[1]]),
                generation_config=config,
                use_model_defaults=True,
            )

    def test_proxy_cannot_export_remote_weights(self):
        proxy = RemoteCausalLMProxy(object(), model_type="orpheustts")
        with self.assertRaisesRegex(RuntimeError, "owns the language-model weights"):
            proxy.save_pretrained("unused")


class SpeechBackendTests(unittest.TestCase):

    def test_direct_speech_client_rejects_unknown_input_from_capability(self):
        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8091",
                transport="speech",
            ))

        with self.assertRaisesRegex(
                ValueError,
                "Unsupported external speech option.*temperatur",
        ):
            client._speech_payload(
                "qwen3tts",
                {
                    "text": "Hello",
                    "temperatur": 0.7,
                },
            )

    def test_vllm_speech_nests_sampling_fields_and_rejects_penalty(self):
        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8091",
                transport="speech",
                model="Qwen/Qwen3-TTS",
            ))

        payload = client._speech_payload(
            "qwen3tts",
            {
                "text": "Hello",
                "temperature": 0.65,
                "top_p": 0.9,
                "top_k": 40,
            },
        )

        self.assertEqual(
            payload["extra_params"],
            {
                "temperature": 0.65,
                "top_p": 0.9,
                "top_k": 40,
            },
        )
        self.assertEqual(
            set(payload),
            {
                "extra_params",
                "input",
                "model",
                "response_format",
                "stream",
                "task_type",
            },
        )
        for field in ("temperature", "top_p", "top_k"):
            self.assertNotIn(field, payload)

        with self.assertRaisesRegex(
                LLMBackendCompatibilityError,
                "repetition_penalty",
        ):
            client._speech_payload(
                "qwen3tts",
                {
                    "text": "Hello",
                    "repetition_penalty": 1.1,
                },
            )

    def test_sglang_speech_uses_its_direct_sampling_schema(self):
        client = LLMServingClient(
            LLMBackendConfig(
                backend="sglang",
                endpoint="http://localhost:8000",
                transport="speech",
                model="OpenMOSS-Team/MOSS-TTS-v1.5",
            ))

        payload = client._speech_payload(
            "mosstts",
            {
                "text": "Hello",
                "duration_tokens": 200,
                "repetition_penalty": 1.1,
                "seed": 42,
                "temperature": 0.7,
                "token_count": 200,
                "top_k": 0,
                "top_p": 0.9,
            },
        )

        self.assertEqual(payload["top_k"], -1)
        self.assertNotIn("extra_params", payload)
        self.assertEqual(
            set(payload),
            {
                "duration_tokens",
                "input",
                "model",
                "repetition_penalty",
                "response_format",
                "seed",
                "stream",
                "temperature",
                "token_count",
                "top_k",
                "top_p",
            },
        )
        for option in (
            {"stage_params": {}},
            {"repetition_penalty": 2.01},
        ):
            with self.subTest(option=option), self.assertRaises(LLMBackendCompatibilityError):
                client._speech_payload(
                    "mosstts",
                    {
                        "text": "Hello",
                        **option,
                    },
                )

    def test_speech_payload_validates_speed_and_boolean_modes(self):
        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8091",
                transport="speech",
                model="Qwen/Qwen3-TTS",
            ))

        payload = client._speech_payload(
            "qwen3tts",
            {
                "text": "Hello",
                "speed": 0.25,
                "non_streaming_mode": False,
                "x_vector_only_mode": True,
            },
        )
        self.assertEqual(payload["speed"], 0.25)
        self.assertIs(payload["non_streaming_mode"], False)
        self.assertIs(payload["x_vector_only_mode"], True)

        for option, message in (
            ({"speed": 0.24}, "speed"),
            ({"speed": 4.01}, "speed"),
            ({"speed": float("nan")}, "speed"),
            ({"non_streaming_mode": "false"}, "non_streaming_mode"),
            ({"x_vector_only_mode": 1}, "x_vector_only_mode"),
        ):
            with self.subTest(option=option), self.assertRaisesRegex(
                (TypeError, ValueError),
                    message,
            ):
                client._speech_payload(
                    "qwen3tts",
                    {
                        "text": "Hello",
                        **option,
                    },
                )

    def test_vllm_moss_maps_supported_ambient_sound(self):
        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8091",
                transport="speech",
                model="OpenMOSS-Team/MOSS-SoundEffect",
            ))

        payload = client._speech_payload(
            "mosstts",
            {
                "text": "Distant waves",
                "ambient_sound": "Ocean surf on rocks",
            },
        )

        self.assertEqual(payload["ambient_sound"], "Ocean surf on rocks")

    def test_wav_response_is_decoded_with_server_sample_rate(self):
        wav = _wave_bytes(
            [0, 16384, -16384],
            sample_rate=22050,
        )
        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8091",
                transport="speech",
                model="Qwen/Qwen3-TTS",
            ))
        response = HTTPBackendResponse(
            body=wav,
            headers={"content-type": "audio/wav"},
            status=200,
        )

        with patch.object(
                client.http,
                "post_json",
                return_value=response,
        ) as post:
            output = client.synthesize(
                "qwen3tts",
                {
                    "text": "Hello",
                    "speaker": "Ryan",
                    "language": "English",
                },
                default_sample_rate=24000,
            )

        self.assertEqual(output.sample_rate, 22050)
        torch.testing.assert_close(
            output.audio,
            torch.tensor([0.0, 0.5, -0.5]),
        )
        self.assertEqual(output.metadata["backend"], "vllm")
        route, payload = post.call_args.args
        self.assertEqual(route, "/v1/audio/speech")
        self.assertEqual(payload["input"], "Hello")
        self.assertEqual(payload["voice"], "Ryan")
        self.assertEqual(payload["language"], "English")
        self.assertEqual(payload["task_type"], "CustomVoice")

    def test_raw_stereo_pcm_is_downmixed_and_uses_response_metadata(self):
        raw = struct.pack(
            "<hhhh",
            32767,
            -32768,
            16384,
            16384,
        )
        client = LLMServingClient(
            LLMBackendConfig(
                backend="sglang",
                endpoint="http://localhost:30000",
                transport="speech",
                model="Qwen/Qwen3-TTS",
            ))
        response = HTTPBackendResponse(
            body=raw,
            headers={
                "content-type": "audio/pcm",
                "x-sample-rate": "16000",
                "x-channels": "2",
                "x-bit-depth": "16",
            },
            status=200,
        )

        with patch.object(
                client.http,
                "post_json",
                return_value=response,
        ):
            output = client.synthesize(
                "qwen3tts",
                {"text": "Hello"},
                default_sample_rate=24000,
            )

        self.assertEqual(output.sample_rate, 16000)
        torch.testing.assert_close(
            output.audio,
            torch.tensor([-1.0 / 65536.0, 0.5]),
        )

    def test_local_reference_audio_is_sent_as_data_url(self):
        wav = _wave_bytes([0, 1000], sample_rate=16000)
        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8091",
                transport="speech",
                model="Qwen/Qwen3-TTS",
            ))
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "reference.wav"
            audio_path.write_bytes(wav)
            payload = client._speech_payload(
                "qwen3tts",
                {
                    "text": "Clone this voice",
                    "speaker_audio_path": audio_path,
                    "reference_text": "Reference transcript",
                    "mode": "voice_clone",
                },
            )

        prefix, encoded = payload["ref_audio"].split(",", 1)
        self.assertEqual(prefix, "data:audio/x-wav;base64")
        self.assertEqual(base64.b64decode(encoded), wav)
        self.assertEqual(payload["ref_text"], "Reference transcript")
        self.assertEqual(payload["task_type"], "Base")

    def test_inline_reference_audio_is_validated_and_bounded(self):
        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8091",
                transport="speech",
                model="Qwen/Qwen3-TTS",
            ))
        wav = _wave_bytes([0, 1000], sample_rate=16000)
        encoded = base64.b64encode(wav).decode("ascii")
        client._REFERENCE_LIMIT = len(wav)

        payload = client._speech_payload(
            "qwen3tts",
            {
                "text": "Clone this voice",
                "ref_audio": f"data:audio/wav;base64,{encoded}",
            },
        )
        self.assertEqual(payload["ref_audio"], f"data:audio/wav;base64,{encoded}")

        for value, message in (
            ("data:audio/wav,not-base64", "base64-encoded"),
            ("data:audio/wav;base64,not!base64", "invalid base64"),
            ("data:audio/wav;base64,", "cannot be empty"),
            (
                "data:audio/wav;base64," + base64.b64encode(wav + b"x").decode("ascii"),
                "remote-reference limit",
            ),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                client._speech_payload(
                    "qwen3tts",
                    {
                        "text": "Clone this voice",
                        "ref_audio": value,
                    },
                )

    def test_sglang_fish_reference_uses_references_envelope(self):
        wav = _wave_bytes([0, 1000], sample_rate=16000)
        client = LLMServingClient(
            LLMBackendConfig(
                backend="sglang",
                endpoint="http://localhost:30000",
                transport="speech",
                model="fishaudio/s2-pro",
            ))
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "reference.wav"
            audio_path.write_bytes(wav)
            payload = client._speech_payload(
                "fishtts",
                {
                    "text": "Speak",
                    "reference_audio": audio_path,
                    "ref_text": "Reference",
                },
            )

        self.assertNotIn("ref_audio", payload)
        self.assertEqual(len(payload["references"]), 1)
        reference = payload["references"][0]
        self.assertTrue(reference["audio_path"].startswith("data:audio/"))
        self.assertEqual(reference["text"], "Reference")

    def test_speech_payload_rejects_native_only_and_ambiguous_inputs(self):
        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8091",
                transport="speech",
                model="Qwen/Qwen3-TTS",
            ))
        with self.assertRaisesRegex(
                LLMBackendCompatibilityError,
                "native-only option.*guidance_scale",
        ):
            client._speech_payload(
                "qwen3tts",
                {
                    "text": "Hello",
                    "guidance_scale": 3.0,
                },
            )
        for option in (
            {"duration": 2.0},
            {"max_len": 200},
        ):
            with self.subTest(option=option), self.assertRaisesRegex(
                    LLMBackendCompatibilityError,
                    "native-only option",
            ):
                client._speech_payload(
                    "omnivoice",
                    {
                        "text": "Hello",
                        **option,
                    },
                )
        with self.assertRaisesRegex(ValueError, "only one reference-audio"):
            client._speech_payload(
                "qwen3tts",
                {
                    "text": "Hello",
                    "ref_audio": "https://example.com/one.wav",
                    "reference_audio": "https://example.com/two.wav",
                },
            )
        with self.assertRaisesRegex(
                LLMBackendCompatibilityError,
                "speaker-embedding",
        ):
            client._speech_payload(
                "qwen3tts",
                {
                    "text": "Hello",
                    "speaker_embedding": torch.zeros(4),
                },
            )

    def test_speech_response_errors_are_actionable(self):
        client = LLMServingClient(
            LLMBackendConfig(
                backend="vllm",
                endpoint="http://localhost:8091",
                transport="speech",
                model="Qwen/Qwen3-TTS",
            ))
        responses = (
            (
                HTTPBackendResponse(
                    body=b'{"error":"failed"}',
                    headers={"content-type": "application/json"},
                    status=200,
                ),
                "returned JSON instead of audio",
            ),
            (
                HTTPBackendResponse(
                    body=b"encoded-mp3",
                    headers={"content-type": "audio/mpeg"},
                    status=200,
                ),
                "requested PCM WAVE",
            ),
            (
                HTTPBackendResponse(
                    body=b"\x00",
                    headers={"content-type": "audio/pcm"},
                    status=200,
                ),
                "not aligned",
            ),
        )
        for response, message in responses:
            with self.subTest(message=message), patch.object(
                    client.http,
                    "post_json",
                    return_value=response,
            ), self.assertRaisesRegex(LLMBackendRequestError, message):
                client.synthesize(
                    "qwen3tts",
                    {"text": "Hello"},
                    default_sample_rate=24000,
                )


class PreTrainedTTSLLMServingTests(unittest.TestCase):

    def test_extension_option_drives_wrapper_validation_and_defaults(self):

        class ExtensionConfig(VoiceHubConfig):
            model_type = "future-wrapper-tts"

        class ExtensionModel(_RemoteSpeechModel):
            config_class = ExtensionConfig

        support = LLMBackendSupport(
            model_type="future-wrapper-tts",
            backend="vllm",
            transports=("speech", ),
            default_transport="speech",
            engine="Future engine",
            checkpoint_family="Future checkpoint",
            speech_string_options=("emotion_prompt", ),
        )
        register_llm_backend_support(support)
        try:
            model = ExtensionModel(
                ExtensionConfig(
                    name_or_path="future/checkpoint",
                    generation_config={"emotion_prompt": "Calm"},
                ))
            model.set_llm_backend(
                "vllm",
                endpoint="http://localhost:8091",
                transport="speech",
            )
            recording = _RecordingSpeechClient()
            model._llm_backend_client = recording

            model.generate("Default emotion")
            model.forward("Explicit emotion", emotion_prompt="Bright")
            with self.assertRaisesRegex(
                    ValueError,
                    "Unsupported external speech option.*emotion_promt",
            ):
                model.forward("Misspelled", emotion_promt="Quiet")

            self.assertEqual(
                [inputs[1]["emotion_prompt"] for inputs in recording.calls],
                ["Calm", "Bright"],
            )
        finally:
            self.assertIs(
                unregister_llm_backend_support("future-wrapper-tts", "vllm"),
                support,
            )

    def test_remote_speech_forward_bypasses_native_load_and_validation(self):
        model = _RemoteSpeechModel(_RemoteSpeechConfig(
            sample_rate=22050,
            name_or_path="remote/qwen",
        ))
        model.set_llm_backend(
            "vllm",
            endpoint="http://localhost:8091",
            transport="speech",
        )
        recording = _RecordingSpeechClient()
        model._llm_backend_client = recording

        output = model.forward(
            "Hello",
            voice="Ryan",
        )

        self.assertEqual(model.load_count, 0)
        self.assertEqual(model.validation_count, 0)
        self.assertFalse(model.is_loaded)
        self.assertEqual(output.sample_rate, 22050)
        self.assertEqual(
            recording.calls,
            [(
                "qwen3tts",
                {
                    "text": "Hello",
                    "voice": "Ryan",
                },
                22050,
            )],
        )

    def test_remote_speech_requests_can_overlap_for_server_batching(self):
        rendezvous = Barrier(2)

        class ConcurrentSpeechClient:

            def synthesize(self, model_type, inputs, *, default_sample_rate):
                del model_type, inputs
                rendezvous.wait(timeout=3)
                return TTSOutput(
                    audio=[0.0],
                    sample_rate=default_sample_rate,
                )

        model = _RemoteSpeechModel(_RemoteSpeechConfig(
            sample_rate=24000,
            name_or_path="remote/qwen",
        ))
        model.set_llm_backend(
            "vllm",
            endpoint="http://localhost:8091",
            transport="speech",
        )
        model._llm_backend_client = ConcurrentSpeechClient()
        outputs = []
        errors = []

        def generate(text):
            try:
                outputs.append(model.forward(text))
            except BaseException as error:
                errors.append(error)

        threads = [Thread(target=generate, args=(text, ), daemon=True) for text in ("first", "second")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(model._active_llm_requests, 0)

    def test_backend_cannot_be_cleared_during_remote_synthesis(self):
        entered = Event()
        release = Event()

        class BlockingSpeechClient:

            def synthesize(self, model_type, inputs, *, default_sample_rate):
                del model_type, inputs
                entered.set()
                if not release.wait(timeout=3):
                    raise TimeoutError("Test did not release remote synthesis.")
                return TTSOutput(
                    audio=[0.0],
                    sample_rate=default_sample_rate,
                )

        model = _RemoteSpeechModel(_RemoteSpeechConfig(
            sample_rate=24000,
            name_or_path="remote/qwen",
        ))
        model.set_llm_backend(
            "vllm",
            endpoint="http://localhost:8091",
            transport="speech",
        )
        model._llm_backend_client = BlockingSpeechClient()
        errors = []

        def generate():
            try:
                model.forward("Hello")
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=generate, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            with self.assertRaisesRegex(RuntimeError, "requests are active"):
                model.clear_llm_backend()
        finally:
            release.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(model._active_llm_requests, 0)

    def test_backend_is_reserved_before_request_preprocessing(self):
        entered = Event()
        release = Event()

        class BlockingProcessor:

            def __call__(self, text, **kwargs):
                entered.set()
                if not release.wait(timeout=3):
                    raise TimeoutError("Test did not release request preprocessing.")
                return {
                    "text": text,
                    **kwargs,
                }

        model = _RemoteSpeechModel(_RemoteSpeechConfig(
            sample_rate=24000,
            name_or_path="remote/qwen",
        ))
        model.set_llm_backend(
            "vllm",
            endpoint="http://localhost:8091",
            transport="speech",
        )
        recording = _RecordingSpeechClient()
        model._llm_backend_client = recording
        model.processor = BlockingProcessor()
        errors = []

        def generate():
            try:
                model.generate("Hello", voice="Ryan")
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=generate, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            self.assertEqual(model._active_llm_requests, 1)
            with self.assertRaisesRegex(RuntimeError, "requests are active"):
                model.clear_llm_backend()
            with self.assertRaisesRegex(RuntimeError, "requests are active"):
                model.set_llm_backend(
                    "sglang",
                    endpoint="http://localhost:30000",
                    transport="speech",
                )
        finally:
            release.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(model.llm_backend.value, "vllm")
        self.assertEqual(model._active_llm_requests, 0)
        self.assertEqual(len(recording.calls), 1)

    def test_native_request_blocks_external_backend_selection_during_preprocessing(self):
        entered = Event()
        release = Event()

        class NativeRaceModel(_RemoteSpeechModel):

            def _validate_generation_inputs(self, model_inputs):
                del model_inputs
                self.validation_count += 1

            def _generate(self, text, **kwargs):
                del text, kwargs
                if self.model is None:
                    raise AssertionError("Native generation ran without a loaded model.")
                return TTSOutput(
                    audio=[0.0],
                    sample_rate=self.sample_rate,
                )

        class BlockingProcessor:

            def __call__(self, text, **kwargs):
                entered.set()
                if not release.wait(timeout=3):
                    raise TimeoutError("Test did not release request preprocessing.")
                return {
                    "text": text,
                    **kwargs,
                }

        model = NativeRaceModel(_RemoteSpeechConfig(
            sample_rate=24000,
            name_or_path="native/qwen",
        ))
        model.processor = BlockingProcessor()
        errors = []

        def generate():
            try:
                model.forward("Hello")
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=generate, daemon=True)
        thread.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            self.assertEqual(model._active_generation_requests, 1)
            with self.assertRaisesRegex(RuntimeError, "requests are active"):
                model.set_llm_backend(
                    "vllm",
                    endpoint="http://localhost:8091",
                    transport="speech",
                )
        finally:
            release.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(model.llm_backend.value, "native")
        self.assertEqual(model.load_count, 1)
        self.assertEqual(model.validation_count, 1)
        self.assertEqual(model._active_generation_requests, 0)
        self.assertEqual(model._active_llm_requests, 0)

    def test_external_backend_is_rejected_for_training(self):
        model = _RemoteSpeechModel(_RemoteSpeechConfig(name_or_path="remote/qwen"))
        model.set_llm_backend(
            "sglang",
            endpoint="http://localhost:30000",
        )

        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            model.load_for_training()
        self.assertEqual(model.load_count, 0)

    def test_remote_forward_rejects_unknown_option_before_network_or_load(self):
        model = _RemoteSpeechModel(_RemoteSpeechConfig(name_or_path="remote/qwen"))
        model.set_llm_backend(
            "vllm",
            endpoint="http://localhost:8091",
        )
        recording = _RecordingSpeechClient()
        model._llm_backend_client = recording

        with self.assertRaisesRegex(
                ValueError,
                "Unsupported external speech option.*temperatur",
        ):
            model.forward("Hello", temperatur=0.7)

        self.assertEqual(model.load_count, 0)
        self.assertEqual(recording.calls, [])

    def test_auto_from_config_plumbs_runtime_only_backend_configuration(self):
        config = AutoConfig.for_model(
            "qwen3tts",
            name_or_path="remote/qwen-checkpoint",
        )
        model = AutoModelForTextToSpeech.from_config(
            config,
            llm_backend="sglang",
            llm_backend_config={
                "endpoint": "http://localhost:30000/v1",
                "api_key": "runtime-only",
            },
        )

        self.assertIs(model.llm_backend, LLMBackend.SGLANG)
        self.assertIs(
            model.llm_backend_transport,
            LLMBackendTransport.SPEECH,
        )
        self.assertEqual(
            model.llm_backend_config.model,
            "remote/qwen-checkpoint",
        )
        self.assertFalse(model.is_loaded)
        serialized = config.to_dict()
        self.assertNotIn("llm_backend", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("runtime-only", repr(serialized))


if __name__ == "__main__":
    unittest.main()
