import json
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

from voicehub.runtime import (
    ARCHITECTURE_REGISTRY,
    ArchitectureCapabilities,
    ArchitectureCompatibilityError,
    ArchitectureRegistrationError,
    ArchitectureRegistry,
    ArchitectureSpec,
    ComponentResolutionError,
    LazyComponentReference,
    RuntimeBundle,
    RuntimeRequirements,
    UnknownArchitectureError,
    ensure_compatible,
    inspect_compatibility,
    register_builtin_architectures,
)
from voicehub.tasks import SpeechTask


class _FakeModel:
    pass


def _capabilities(**overrides):
    values = {
        "tasks": ("asr", ),
        "devices": ("cpu", "cuda"),
        "dtypes": ("float32", "bfloat16"),
        "checkpoint_formats": ("safetensors", "pt"),
        "training": True,
        "streaming": True,
        "batched_inference": True,
        "distributed_training": True,
        "export_formats": ("onnx", ),
        "optimization_passes": ("compile", ),
        "features": ("timestamps", ),
    }
    values.update(overrides)
    return ArchitectureCapabilities(**values)


def _spec(architecture_id="test-whisper", **overrides):
    values = {
        "architecture_id": architecture_id,
        "version": "1.0",
        "model_builder": f"{__name__}:_FakeModel",
        "capabilities": _capabilities(),
    }
    values.update(overrides)
    return ArchitectureSpec(**values)


class LazyComponentReferenceTests(unittest.TestCase):

    def test_reference_does_not_import_until_resolution(self):
        module_name = "tests._voicehub_lazy_architecture"
        sys.modules.pop(module_name, None)
        reference = LazyComponentReference(
            module_name,
            "namespace.Model",
        )

        self.assertNotIn(module_name, sys.modules)

        module = ModuleType(module_name)
        module.namespace = SimpleNamespace(Model=_FakeModel)
        sys.modules[module_name] = module
        try:
            self.assertIs(reference.resolve(), _FakeModel)
            self.assertIs(reference.instantiate().__class__, _FakeModel)
        finally:
            sys.modules.pop(module_name, None)

    def test_resolution_errors_include_the_import_target(self):
        reference = LazyComponentReference.from_path("tests._voicehub_missing_architecture:Model")

        with self.assertRaisesRegex(
                ComponentResolutionError,
                "_voicehub_missing_architecture",
        ):
            reference.resolve()

    def test_non_callable_components_cannot_be_instantiated(self):
        reference = LazyComponentReference.from_path(f"{__name__}:SpeechTask")

        self.assertIs(reference.resolve(), SpeechTask)
        with self.assertRaisesRegex(ComponentResolutionError, "not callable"):
            LazyComponentReference.from_path(f"{__name__}:__doc__").instantiate()


class ArchitectureSpecificationTests(unittest.TestCase):

    def test_spec_normalizes_and_freezes_declarative_metadata(self):
        metadata = {"source": {"files": ["config.json"]}}
        spec = _spec(
            architecture_id=" Test_Whisper ",
            processor=f"{__name__}:SimpleNamespace",
            components={"feature_extractor": f"{__name__}:SimpleNamespace"},
            metadata=metadata,
        )
        metadata["source"]["files"].append("weights.pt")

        self.assertEqual(spec.architecture_id, "test-whisper")
        self.assertEqual(spec.qualified_id, "test-whisper@1.0")
        self.assertEqual(
            spec.get_component_reference("feature-extractor").path,
            f"{__name__}:SimpleNamespace",
        )
        self.assertIs(spec.resolve_component("builder"), _FakeModel)
        self.assertEqual(
            spec.metadata["source"]["files"],
            ("config.json", ),
        )
        with self.assertRaises(TypeError):
            spec.components["other"] = LazyComponentReference(
                __name__,
                "_FakeModel",
            )

    def test_capabilities_are_open_ended_and_normalized(self):
        capabilities = _capabilities(
            dtypes=("fp32", "bf16"),
            checkpoint_formats=(".safetensors", "GGUF"),
        )

        self.assertEqual(
            capabilities.tasks,
            (SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
        )
        self.assertTrue(capabilities.supports_device("cuda:1"))
        self.assertTrue(capabilities.supports_dtype("bfloat16"))
        self.assertTrue(capabilities.supports_checkpoint_format("safetensors"))
        self.assertTrue(capabilities.supports_optimization("compile"))

    def test_distributed_training_requires_a_training_path(self):
        with self.assertRaisesRegex(ValueError, "requires training"):
            ArchitectureCapabilities(
                tasks=("tts", ),
                training=False,
                distributed_training=True,
            )


class ArchitectureRegistryTests(unittest.TestCase):

    def setUp(self):
        self.registry = ArchitectureRegistry()

    def test_registration_aliases_filters_and_live_views(self):
        view = self.registry.specs
        whisper = _spec()
        synthesis = _spec(
            "native-vits",
            capabilities=_capabilities(
                tasks=("tts", ),
                training=False,
                streaming=False,
                distributed_training=False,
            ),
        )

        self.registry.register(whisper, aliases=("Whisper_Native", ))
        self.registry.register(synthesis)

        self.assertIs(view["test-whisper"], whisper)
        self.assertIs(self.registry.get("whisper-native"), whisper)
        self.assertEqual(
            [spec.architecture_id for spec in self.registry.list(task="tts")],
            ["native-vits"],
        )
        self.assertEqual(
            [spec.architecture_id for spec in self.registry.list(training=True)],
            ["test-whisper"],
        )
        with self.assertRaises(TypeError):
            view["other"] = whisper

    def test_registration_is_explicit_and_replacement_preserves_order(self):
        first = _spec("first", version="1")
        second = _spec("second")
        replacement = _spec("first", version="2")
        self.registry.register(first)
        self.registry.register(second)

        with self.assertRaisesRegex(
                ArchitectureRegistrationError,
                "already registered",
        ):
            self.registry.register(replacement)
        self.registry.register(replacement, exist_ok=True)

        self.assertIs(self.registry.get("first"), replacement)
        self.assertEqual(tuple(self.registry), ("first", "second"))

    def test_unregister_removes_targeted_aliases(self):
        spec = _spec()
        self.registry.register(spec, aliases=("alias-one", "alias-two"))

        self.assertIs(self.registry.unregister("alias-one"), spec)
        self.assertEqual(dict(self.registry.aliases), {})
        with self.assertRaises(UnknownArchitectureError):
            self.registry.get(spec.architecture_id)

    def test_registry_mutation_is_thread_safe(self):
        architecture_ids = tuple(f"parallel-{index}" for index in range(32))

        with ThreadPoolExecutor(max_workers=8) as executor:
            tuple(
                executor.map(
                    lambda architecture_id: self.registry.register(_spec(architecture_id)),
                    architecture_ids,
                ))

        self.assertEqual(len(self.registry), len(architecture_ids))
        self.assertEqual(set(self.registry), set(architecture_ids))

    def test_builtin_catalog_is_idempotent_and_model_graphs_remain_lazy(self):
        specs = register_builtin_architectures()

        self.assertEqual(
            tuple(spec.architecture_id for spec in specs),
            (
                "native-asr-dispatch",
                "whisper",
                "wav2vec2",
                "hubert",
                "wavlm",
                "moonshine",
                "qwen3-asr",
                "granite-speech",
                "parakeet-tdt",
                "nemotron-3.5-rnnt",
                "cohere-asr",
                "seamless-m4t-v2-s2t",
                "vibevoice-asr",
                "lasr-ctc",
                "sensevoice-small",
                "nemo-quartznet-ctc",
                "wenet-gigaspeech-u2pp",
                "espnet-librispeech-transformer-e18",
                "vits",
                "vui",
                "chatterbox",
                "csm",
                "conversationtts",
                "llasa",
                "neutts",
                "outetts",
                "speecht5",
                "kokoro",
                "f5tts",
                "gptsovits",
                "moss-tts",
                "melotts",
                "openvoice-v2-converter",
                "parlertts",
                "styletts2",
                "qwen3-tts",
                "vibevoice-tts",
                "voxcpm2",
                "omnivoice",
                "higgs-audio-v2",
                "irodoritts-rf-dit",
                "cosyvoice-native",
                "xtts2",
                "fish-s2",
                "zonos",
                "zonos2",
                "supertonic",
                "bark",
                "inflecttts",
                "echo-dit",
                "native-vad-dispatch",
                "silero-vad",
                "fsmn-vad",
                "speechbrain-crdnn-asr",
                "speechbrain-crdnn-vad",
                "ten-vad",
                "webrtc-vad",
                "marblenet-vad",
                "pyannet",
                "energy-vad",
                "causal-lm",
                "dac",
                "encodec",
                "dia",
            ),
        )
        self.assertIs(
            ARCHITECTURE_REGISTRY.get("native-whisper"),
            ARCHITECTURE_REGISTRY.get("whisper"),
        )
        self.assertIs(
            ARCHITECTURE_REGISTRY.get("native-hubert"),
            ARCHITECTURE_REGISTRY.get("hubert"),
        )
        self.assertIs(
            ARCHITECTURE_REGISTRY.get("sesame-csm"),
            ARCHITECTURE_REGISTRY.get("csm"),
        )
        self.assertNotIn("csm", ARCHITECTURE_REGISTRY.aliases)
        csm = ARCHITECTURE_REGISTRY.get("csm")
        self.assertTrue(csm.capabilities.training)
        self.assertFalse(csm.capabilities.batched_inference)
        self.assertIn(
            "two-level-codebook-cross-entropy",
            csm.capabilities.features,
        )
        fish = ARCHITECTURE_REGISTRY.get("fishtts")
        self.assertIs(fish, ARCHITECTURE_REGISTRY.get("fish-s2"))
        self.assertTrue(fish.capabilities.training)
        self.assertEqual(
            fish.capabilities.checkpoint_formats,
            ("safetensors", ),
        )
        self.assertFalse(fish.metadata["commercial_use"])

        from voicehub.registry import get_model_spec, list_model_specs

        model_specs = list_model_specs(task=None)
        self.assertEqual(len(model_specs), 70)
        self.assertTrue(all(spec.is_voicehub_native for spec in model_specs))
        moss_model = get_model_spec("mosstts")
        self.assertEqual(moss_model.architecture, "moss-tts")
        self.assertNotIn("streaming", moss_model.capabilities)
        self.assertIn("buffered-generation", moss_model.capabilities)
        self.assertIn("raw-audio-fine-tuning", moss_model.capabilities)
        self.assertIn("preencoded-rvq-fine-tuning", moss_model.capabilities)
        moss = ARCHITECTURE_REGISTRY.get("moss-tts")
        self.assertIs(moss, ARCHITECTURE_REGISTRY.get("mosstts"))
        self.assertIs(moss, ARCHITECTURE_REGISTRY.get("moss_tts"))
        self.assertTrue(moss.capabilities.training)
        self.assertFalse(moss.capabilities.streaming)
        self.assertEqual(
            moss.capabilities.checkpoint_formats,
            ("safetensors", ),
        )
        self.assertIn("native-codec-v1-v2", moss.capabilities.features)

        code = """
import json
import sys
import voicehub.runtime as architectures
print(json.dumps({
    "ids": list(architectures.ARCHITECTURES),
    "torch": "torch" in sys.modules,
    "numpy": "numpy" in sys.modules,
    "modeling": sorted(
        name for name in sys.modules
        if name.startswith("voicehub.models.")
        and name.endswith(".modeling")
    ),
    "kokoro_runtime": sorted(
        name for name in sys.modules
        if name in {
            "voicehub.models.kokoro.native.albert",
            "voicehub.models.kokoro.native.checkpoint",
        }
        or name == "voicehub.models.kokoro.modeling"
    ),
    "csm_runtime": sorted(
        name for name in sys.modules
        if name in {
            "voicehub.models.csm.native.checkpoint",
            "voicehub.models.csm.native.mimi",
            "voicehub.models.csm.native.modeling",
            "voicehub.models.csm.native.processing",
            "voicehub.models.csm.native.runtime",
        }
        or name == "voicehub.models.csm.modeling"
    ),
    "outetts_runtime": sorted(
        name for name in sys.modules
        if name.startswith("voicehub.models.outetts.native.")
        and not name.endswith((".metadata", ".registration"))
        or name == "voicehub.models.outetts.modeling"
    ),
    "fishtts_runtime": sorted(
        name for name in sys.modules
        if name.startswith("voicehub.models.fishtts.native.")
        and not name.endswith((".metadata", ".registration"))
        or name == "voicehub.models.fishtts.modeling"
    ),
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(
            payload["ids"],
            [
                "native-asr-dispatch",
                "whisper",
                "wav2vec2",
                "hubert",
                "wavlm",
                "moonshine",
                "qwen3-asr",
                "granite-speech",
                "parakeet-tdt",
                "nemotron-3.5-rnnt",
                "cohere-asr",
                "seamless-m4t-v2-s2t",
                "vibevoice-asr",
                "lasr-ctc",
                "sensevoice-small",
                "nemo-quartznet-ctc",
                "wenet-gigaspeech-u2pp",
                "espnet-librispeech-transformer-e18",
                "vits",
                "vui",
                "chatterbox",
                "csm",
                "conversationtts",
                "llasa",
                "neutts",
                "outetts",
                "speecht5",
                "kokoro",
                "f5tts",
                "gptsovits",
                "moss-tts",
                "melotts",
                "openvoice-v2-converter",
                "parlertts",
                "styletts2",
                "qwen3-tts",
                "vibevoice-tts",
                "voxcpm2",
                "omnivoice",
                "higgs-audio-v2",
                "irodoritts-rf-dit",
                "cosyvoice-native",
                "xtts2",
                "fish-s2",
                "zonos",
                "zonos2",
                "supertonic",
                "bark",
                "inflecttts",
                "echo-dit",
                "native-vad-dispatch",
                "silero-vad",
                "fsmn-vad",
                "speechbrain-crdnn-asr",
                "speechbrain-crdnn-vad",
                "ten-vad",
                "webrtc-vad",
                "marblenet-vad",
                "pyannet",
                "energy-vad",
                "causal-lm",
                "dac",
                "encodec",
                "dia",
            ],
        )
        self.assertFalse(payload["torch"])
        self.assertFalse(payload["numpy"])
        self.assertEqual(payload["modeling"], [])
        self.assertEqual(payload["kokoro_runtime"], [])
        self.assertEqual(payload["csm_runtime"], [])
        self.assertEqual(payload["outetts_runtime"], [])
        self.assertEqual(payload["fishtts_runtime"], [])

    def test_every_native_model_spec_resolves_a_task_compatible_architecture(self):
        from voicehub.registry import list_model_specs

        for model_spec in list_model_specs(task=None):
            if not model_spec.is_voicehub_native:
                continue
            with self.subTest(model_type=model_spec.model_type):
                architecture = model_spec.native_architecture
                self.assertIn(
                    model_spec.task,
                    architecture.capabilities.tasks,
                )


class ArchitectureRuntimeTests(unittest.TestCase):

    def test_compatibility_reports_every_incompatible_capability(self):
        spec = _spec(
            capabilities=_capabilities(
                devices=("cpu", ),
                dtypes=("float32", ),
                training=False,
                streaming=False,
                batched_inference=False,
                distributed_training=False,
                export_formats=(),
                optimization_passes=(),
                features=(),
            ))
        requirements = RuntimeRequirements(
            task="asr",
            device="cuda:0",
            dtype="bf16",
            training=True,
            streaming=True,
            batched=True,
            export_format="onnx",
            optimization_passes=("compile", ),
            required_features=("timestamps", ),
        )

        issues = inspect_compatibility(spec, requirements)

        self.assertEqual(
            {issue.capability
             for issue in issues},
            {
                "device",
                "dtype",
                "training",
                "streaming",
                "batched-inference",
                "export-format",
                "optimization-pass",
                "feature",
            },
        )
        with self.assertRaises(ArchitectureCompatibilityError) as context:
            ensure_compatible(spec, requirements)
        self.assertEqual(context.exception.issues, issues)
        self.assertIn("device 'cuda:0'", str(context.exception))

    def test_runtime_bundle_exposes_resolved_components_read_only(self):
        spec = _spec()
        requirements = RuntimeRequirements(
            task="asr",
            device="cuda:0",
            dtype="bf16",
            checkpoint_format="safetensors",
            training=True,
            distributed=True,
            streaming=True,
            batched=True,
            export_format="onnx",
            optimization_passes=("compile", ),
            required_features=("timestamps", ),
        )
        model = _FakeModel()
        bundle = RuntimeBundle(
            spec=spec,
            model=model,
            processor=object(),
            components={"feature_extractor": object()},
            requirements=requirements,
        )

        self.assertIs(bundle.get_component("model"), model)
        self.assertIs(
            bundle.get_component("feature-extractor"),
            bundle.components["feature-extractor"],
        )
        self.assertIsNone(bundle.get_component("decoder", None))
        self.assertTrue(bundle.training_ready)
        self.assertTrue(bundle.streaming_ready)
        with self.assertRaises(TypeError):
            bundle.components["other"] = object()

    def test_runtime_bundle_validates_requirements_before_use(self):
        with self.assertRaises(ArchitectureCompatibilityError):
            RuntimeBundle(
                spec=_spec(),
                model=_FakeModel(),
                requirements=RuntimeRequirements(
                    task="vad",
                    device="cpu",
                ),
            )


if __name__ == "__main__":
    unittest.main()
