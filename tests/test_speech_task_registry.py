import ast
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from voicehub.auto import (
    AutoConfig,
    AutoModel,
    AutoModelForSpeechRecognition,
    AutoModelForTextToSpeech,
    AutoModelForVoiceActivityDetection,
    AutoProcessor,
)
from voicehub.components import MODEL_COMPONENTS, components_for_model
from voicehub.configuration import VoiceHubConfig
from voicehub.processing.processor import VoiceHubProcessor
from voicehub.registry import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    ModelRegistry,
    ModelSpec,
    get_default_model_spec,
    get_model_spec,
    list_model_specs,
    register_model_alias,
    register_model_spec,
    unregister_model_alias,
    unregister_model_spec,
)
from voicehub.tasks import SpeechTask


class _FakeSpeechModel:

    def __init__(self, config, **kwargs):
        self.config = config
        self.init_kwargs = kwargs
        self.loaded = False
        self.inference_strategy = None

    def set_inference_strategy(self, strategy):
        self.inference_strategy = strategy

    def load(self):
        self.loaded = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        config,
        inference_strategy=None,
        **kwargs,
    ):
        model = cls(config, **kwargs)
        model.pretrained_model_name_or_path = pretrained_model_name_or_path
        model.inference_strategy = inference_strategy
        return model


class _FakeASRModel(_FakeSpeechModel):
    pass


class _FakeVADModel(_FakeSpeechModel):
    pass


class _FakeDefaultTTSModel:

    def __init__(self, config=None, *, model_path=None, device="auto", **kwargs):
        self.config = config
        self.model_path = model_path
        self.device = device
        self.init_kwargs = kwargs
        self.loaded = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        config,
        inference_strategy=None,
        **kwargs,
    ):
        model = cls(
            config,
            model_path=pretrained_model_name_or_path,
            **kwargs,
        )
        model.inference_strategy = inference_strategy
        return model


class _ExtensionASRConfig(VoiceHubConfig):
    model_type = "test-auto-register-asr"


class _ExtensionProcessor(VoiceHubProcessor):
    pass


class _ExtensionASRModelWithProcessor(_FakeASRModel):
    processor_class = _ExtensionProcessor


class SpeechTaskRegistryTests(unittest.TestCase):
    ASR_MODEL_TYPE = "test-speech-registry-asr"
    ASR_ALIAS = "test-speech-registry-stt"
    VAD_MODEL_TYPE = "test-speech-registry-vad"
    VAD_ALIAS = "test-speech-registry-activity"
    FAKE_MODULE = "tests._voicehub_fake_speech_backends"

    def tearDown(self):
        for model_type in (self.ASR_MODEL_TYPE, self.VAD_MODEL_TYPE):
            unregister_model_spec(model_type, missing_ok=True)

    def _spec(
        self,
        model_type,
        class_name,
        task,
        *,
        module=None,
    ):
        return ModelSpec(
            model_type=model_type,
            module=module or self.FAKE_MODULE,
            class_name=class_name,
            default_model_path=f"acme/{model_type}",
            install_extra=model_type,
            task=task,
            architecture="test-speech-family",
        )

    def _register_task_backends(self):
        register_model_spec(
            self._spec(
                self.ASR_MODEL_TYPE,
                "_FakeASRModel",
                SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
            ),
            aliases=(self.ASR_ALIAS, ),
        )
        register_model_spec(
            self._spec(
                self.VAD_MODEL_TYPE,
                "_FakeVADModel",
                "vad",
            ),
            aliases=(self.VAD_ALIAS, ),
        )

    def test_model_spec_normalizes_task_metadata(self):
        tts_spec = self._spec(
            self.ASR_MODEL_TYPE,
            "_FakeASRModel",
            SpeechTask.TEXT_TO_SPEECH,
        )
        asr_spec = self._spec(
            self.ASR_MODEL_TYPE,
            "_FakeASRModel",
            "stt",
        )

        self.assertIs(tts_spec.task, SpeechTask.TEXT_TO_SPEECH)
        self.assertIs(
            asr_spec.task,
            SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        )
        self.assertEqual(
            asr_spec.capabilities,
            (SpeechTask.AUTOMATIC_SPEECH_RECOGNITION.value, ),
        )
        self.assertEqual(asr_spec.architecture, "test-speech-family")
        self.assertEqual(asr_spec.components, ())
        self.assertFalse(asr_spec.default_for_task)
        self.assertEqual(
            (
                tts_spec.processor_module,
                tts_spec.processor_class,
            ),
            (
                "voicehub.processing.processor",
                "VoiceHubProcessor",
            ),
        )
        self.assertEqual(
            (
                asr_spec.processor_module,
                asr_spec.processor_class,
            ),
            (
                "voicehub.processing.processor",
                "AudioProcessor",
            ),
        )
        with self.assertRaisesRegex(ValueError, "declared together"):
            replace(asr_spec, processor_class=None)
        self.assertTrue(asr_spec.supports_task("asr"))
        self.assertFalse(asr_spec.supports_task("vad"))

    def test_all_registered_processors_load_without_heavy_backends(self):
        code = """
import json
import sys

from voicehub import AutoProcessor, VoiceHubConfig, list_model_specs

processor_classes = set()
input_contracts = {}
for spec in list_model_specs(task=None):
    config = VoiceHubConfig()
    config.model_type = spec.model_type
    constructed = AutoProcessor.from_config(config)
    loaded = AutoProcessor.from_pretrained(
        "",
        config=config,
        local_files_only=True,
    )
    if spec.task.value == "text-to-speech":
        values = loaded("processor contract")
    else:
        values = loaded([0.0, 0.0], sampling_rate=16_000)
    processor_classes.add(loaded.__class__.__name__)
    input_contracts[spec.model_type] = sorted(values)
    if constructed.__class__ is not loaded.__class__:
        raise AssertionError(
            f"Processor class changed while loading {spec.model_type!r}."
        )

heavy_modules = sorted(
    name for name in (
        "torch",
        "transformers",
        "faster_whisper",
        "nemo",
        "speechbrain",
        "funasr",
        "espnet2",
        "wenet",
    )
    if name in sys.modules
)
print(json.dumps({
    "count": len(input_contracts),
    "heavy_modules": heavy_modules,
    "input_contracts": input_contracts,
    "processor_classes": sorted(processor_classes),
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["count"], 70)
        self.assertEqual(result["heavy_modules"], [])
        self.assertEqual(
            result["processor_classes"],
            ["AudioProcessor", "VoiceHubProcessor"],
        )
        for spec in list_model_specs(task=None):
            with self.subTest(model_type=spec.model_type):
                expected_keys = (["text"]
                                 if spec.task is SpeechTask.TEXT_TO_SPEECH else ["audio", "sampling_rate"])
                self.assertEqual(
                    result["input_contracts"][spec.model_type],
                    expected_keys,
                )

    def test_all_registered_configs_load_without_heavy_backends(self):
        code = """
import json
import sys

from voicehub import AutoConfig, list_model_specs

config_classes = {}
for spec in list_model_specs(task=None):
    config = AutoConfig.for_model(spec.model_type)
    if config.model_type != spec.model_type:
        raise AssertionError(
            f"Config type {config.model_type!r} does not match {spec.model_type!r}."
        )
    if config.architectures != [spec.class_name]:
        raise AssertionError(
            f"Config architectures do not match {spec.model_type!r}."
        )
    json.dumps(config.to_dict(), sort_keys=True)
    config_classes[spec.model_type] = config.__class__.__name__

heavy_modules = sorted(
    name for name in (
        "torch",
        "transformers",
        "faster_whisper",
        "nemo",
        "speechbrain",
        "funasr",
        "espnet2",
        "wenet",
    )
    if name in sys.modules
)
print(json.dumps({
    "config_classes": config_classes,
    "count": len(config_classes),
    "heavy_modules": heavy_modules,
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual(result["count"], 70)
        self.assertEqual(result["heavy_modules"], [])
        self.assertEqual(
            set(result["config_classes"]),
            {spec.model_type
             for spec in list_model_specs(task=None)},
        )

    def test_registry_enforces_one_declarative_default_per_task(self):
        first = replace(
            self._spec(
                self.ASR_MODEL_TYPE,
                "_FakeASRModel",
                "asr",
            ),
            default_for_task=True,
        )
        second = replace(
            self._spec(
                "test-second-default-asr",
                "_FakeASRModel",
                "asr",
            ),
            default_for_task=True,
        )
        registry = ModelRegistry((first, ))

        self.assertIs(registry.get_default("speech-to-text"), first)
        self.assertIsNone(registry.get_default("vad"))
        with self.assertRaisesRegex(
                ValueError,
                "already declares default model",
        ):
            registry.register(second)
        with self.assertRaisesRegex(TypeError, "default_for_task"):
            replace(first, default_for_task="yes")

    def test_model_spec_normalizes_component_declarations(self):
        spec = ModelSpec(
            model_type=self.ASR_MODEL_TYPE,
            module=self.FAKE_MODULE,
            class_name="_FakeASRModel",
            default_model_path="acme/asr",
            task="asr",
            components=(" DAC ", "vocos"),
        )

        self.assertEqual(spec.components, ("dac", "vocos"))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            ModelSpec(
                model_type=self.ASR_MODEL_TYPE,
                module=self.FAKE_MODULE,
                class_name="_FakeASRModel",
                default_model_path="acme/asr",
                task="asr",
                components=("dac", " DAC "),
            )

    def test_registration_aliases_and_task_filters_are_live(self):
        self._register_task_backends()

        self.assertIs(
            get_model_spec(self.ASR_ALIAS),
            MODEL_REGISTRY[self.ASR_MODEL_TYPE],
        )
        self.assertEqual(
            MODEL_ALIASES[self.VAD_ALIAS],
            self.VAD_MODEL_TYPE,
        )
        self.assertEqual(
            get_model_spec(self.ASR_MODEL_TYPE).class_name,
            "_FakeASRModel",
        )
        self.assertIn(
            self.ASR_MODEL_TYPE,
            {spec.model_type
             for spec in list_model_specs(task="speech-to-text")},
        )
        self.assertNotIn(
            self.VAD_MODEL_TYPE,
            {spec.model_type
             for spec in list_model_specs(task="speech-to-text")},
        )

    def test_isolated_model_registry_keeps_live_read_only_views(self):
        spec = ModelSpec.from_classes(
            model_type=self.ASR_MODEL_TYPE,
            model_class=_FakeASRModel,
            config_class=_ExtensionASRConfig,
            task="asr",
        )
        registry = ModelRegistry()
        specs = registry.specs
        aliases = registry.aliases

        registry.register(spec, aliases=(self.ASR_ALIAS, ))

        self.assertIs(specs[self.ASR_MODEL_TYPE], spec)
        self.assertEqual(aliases[self.ASR_ALIAS], self.ASR_MODEL_TYPE)
        self.assertIs(registry.get(self.ASR_ALIAS), spec)
        self.assertEqual(tuple(registry), (self.ASR_MODEL_TYPE, ))
        with self.assertRaises(TypeError):
            specs["other"] = spec
        registry.clear()
        self.assertEqual(dict(specs), {})
        self.assertEqual(dict(aliases), {})

    def test_auto_factory_registers_and_dispatches_extension_models(self):
        model_type = _ExtensionASRConfig.model_type
        alias = "test-auto-register-stt"
        try:
            spec = AutoModelForSpeechRecognition.register(
                _ExtensionASRConfig,
                _FakeASRModel,
                default_model_path="acme/extension-asr",
                aliases=(alias, ),
            )
            model = AutoModel.from_pretrained(
                "acme/checkpoint",
                model_type=alias,
                config_kwargs={"sample_rate": 22_050},
                marker="extension",
            )
        finally:
            AutoModel.unregister(
                model_type,
                missing_ok=True,
            )

        self.assertEqual(spec.model_type, model_type)
        self.assertIsInstance(model, _FakeASRModel)
        self.assertEqual(model.init_kwargs["marker"], "extension")
        self.assertEqual(model.config.sample_rate, 22_050)

    def test_auto_factory_records_custom_processor_metadata(self):
        model_type = _ExtensionASRConfig.model_type
        try:
            spec = AutoModelForSpeechRecognition.register(
                _ExtensionASRConfig,
                _ExtensionASRModelWithProcessor,
                default_model_path="acme/extension-asr",
            )
            config = AutoConfig.for_model(model_type)
            processor = AutoProcessor.from_config(
                config,
                marker="extension-processor",
            )
        finally:
            AutoModel.unregister(model_type, missing_ok=True)

        self.assertEqual(spec.processor_module, __name__)
        self.assertEqual(spec.processor_class, "_ExtensionProcessor")
        self.assertIsInstance(processor, _ExtensionProcessor)
        self.assertEqual(
            processor.init_kwargs,
            {"marker": "extension-processor"},
        )

    def test_auto_factory_declares_components_without_a_shared_model_map(self):
        model_type = _ExtensionASRConfig.model_type
        try:
            spec = AutoModelForSpeechRecognition.register(
                _ExtensionASRConfig,
                _FakeASRModel,
                default_model_path="acme/extension-asr",
                components=("dac", ),
            )

            self.assertEqual(spec.components, ("dac", ))
            self.assertEqual(MODEL_COMPONENTS[model_type], ("dac", ))
            self.assertEqual(
                tuple(component.name for component in components_for_model(model_type)),
                ("dac", ),
            )
        finally:
            AutoModel.unregister(model_type, missing_ok=True)

        self.assertNotIn(model_type, MODEL_COMPONENTS)

    def test_unknown_declared_component_fails_when_resolved(self):
        model_type = _ExtensionASRConfig.model_type
        try:
            AutoModelForSpeechRecognition.register(
                _ExtensionASRConfig,
                _FakeASRModel,
                components=("future-codec", ),
            )
            with self.assertRaisesRegex(KeyError, "Unknown component 'future-codec'"):
                components_for_model(model_type)
        finally:
            AutoModel.unregister(model_type, missing_ok=True)

    def test_native_filter_resolves_owned_architecture_contracts(self):
        native_asr = list_model_specs(
            task="asr",
            native=True,
        )

        self.assertIn(
            "asr_whisper",
            {spec.model_type
             for spec in native_asr},
        )
        whisper = get_model_spec("asr_whisper")
        self.assertTrue(whisper.is_voicehub_native)
        self.assertEqual(
            whisper.native_architecture.architecture_id,
            "whisper",
        )
        generic = get_model_spec("asr_transformers")
        self.assertTrue(generic.is_voicehub_native)
        self.assertEqual(
            generic.native_architecture.architecture_id,
            "native-asr-dispatch",
        )
        sherpa = get_model_spec("vad_sherpa_onnx")
        self.assertTrue(sherpa.is_voicehub_native)
        self.assertEqual(
            sherpa.native_architecture.architecture_id,
            "native-vad-dispatch",
        )
        with self.assertRaisesRegex(TypeError, "native"):
            list_model_specs(native="yes")

    def test_alias_registration_is_idempotent_only_when_requested(self):
        register_model_spec(self._spec(
            self.ASR_MODEL_TYPE,
            "_FakeASRModel",
            "asr",
        ))
        register_model_alias(self.ASR_ALIAS, self.ASR_MODEL_TYPE)

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_model_alias(self.ASR_ALIAS, self.ASR_MODEL_TYPE)
        register_model_alias(
            self.ASR_ALIAS.upper(),
            self.ASR_MODEL_TYPE,
            exist_ok=True,
        )
        self.assertEqual(
            unregister_model_alias(self.ASR_ALIAS),
            self.ASR_MODEL_TYPE,
        )
        self.assertNotIn(self.ASR_ALIAS, MODEL_ALIASES)

    def test_task_factories_load_only_matching_backends(self):
        self._register_task_backends()
        fake_module = ModuleType(self.FAKE_MODULE)
        fake_module._FakeASRModel = _FakeASRModel
        fake_module._FakeVADModel = _FakeVADModel

        with patch.dict(sys.modules, {self.FAKE_MODULE: fake_module}):
            config = AutoConfig.for_model(
                self.ASR_ALIAS,
                sample_rate=16000,
            )
            asr_model = AutoModelForSpeechRecognition.from_config(config)
            vad_model = AutoModelForVoiceActivityDetection.from_pretrained(
                "acme/vad-checkpoint",
                model_type=self.VAD_ALIAS,
                marker="vad",
            )

        self.assertIsInstance(asr_model, _FakeASRModel)
        self.assertEqual(asr_model.config.sample_rate, 16000)
        self.assertIsInstance(vad_model, _FakeVADModel)
        self.assertEqual(
            vad_model.pretrained_model_name_or_path,
            "acme/vad-checkpoint",
        )
        self.assertEqual(vad_model.init_kwargs["marker"], "vad")

    def test_all_registered_models_round_trip_portable_metadata_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for spec in list_model_specs():
                with self.subTest(model_type=spec.model_type, task=spec.task.value):
                    config = AutoConfig.for_model(
                        spec.model_type,
                        name_or_path=spec.default_model_path,
                    )
                    model = AutoModel.from_config(
                        config,
                        device="cpu",
                        lazy_load=True,
                    )
                    destination = root / spec.model_type

                    model.save_pretrained(
                        destination,
                        include_native_export=False,
                    )
                    saved_config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
                    restored_config = AutoConfig.from_pretrained(
                        destination,
                        local_files_only=True,
                    )
                    restored_model = AutoModel.from_pretrained(
                        destination,
                        config=restored_config,
                        device="cpu",
                        lazy_load=True,
                    )

                    serialized_model_config = json.loads(model.config.to_json_string())
                    serialized_restored_config = json.loads(restored_config.to_json_string())
                    serialized_restored_config["name_or_path"] = (serialized_model_config["name_or_path"])

                    self.assertEqual(saved_config, serialized_model_config)
                    self.assertEqual(
                        serialized_restored_config,
                        serialized_model_config,
                    )
                    self.assertIs(type(restored_config), type(model.config))
                    self.assertEqual(restored_config.model_type, spec.model_type)
                    self.assertIs(type(restored_model), type(model))
                    self.assertFalse(model.is_loaded)
                    self.assertFalse(restored_model.is_loaded)
                    self.assertFalse((destination / "native_export").exists())

    def test_all_registered_configs_reject_secrets_added_after_construction(self):
        for spec in list_model_specs():
            with self.subTest(model_type=spec.model_type, task=spec.task.value):
                config = AutoConfig.for_model(spec.model_type)
                config.runtime_credentials = {
                    "token": "must-not-be-persisted",
                }

                with self.assertRaisesRegex(ValueError, "runtime secrets"):
                    config.to_dict()

    def test_all_registered_configs_reject_embedded_generation_secrets(self):
        for spec in list_model_specs():
            with self.subTest(model_type=spec.model_type, task=spec.task.value):
                with self.assertRaises(ValueError) as context:
                    AutoConfig.for_model(
                        spec.model_type,
                        generation_config={
                            "provider_options": {
                                "api_key": "must-not-be-persisted",
                            },
                        },
                    )

                message = str(context.exception).lower()
                self.assertTrue(
                    any(
                        marker in message for marker in (
                            "authentication",
                            "credential",
                            "secret",
                            "token",
                        )),
                    message,
                )

    def test_explicit_model_type_cannot_override_a_different_typed_config(self):
        self._register_task_backends()
        config = AutoConfig.for_model(self.ASR_MODEL_TYPE)

        with self.assertRaisesRegex(ValueError, "supplied config targets"):
            AutoModelForSpeechRecognition.from_pretrained(
                "acme/other-asr-checkpoint",
                model_type="asr_transformers",
                config=config,
            )

    def test_explicit_model_type_can_specialize_a_generic_config(self):
        register_model_spec(self._spec(
            self.ASR_MODEL_TYPE,
            "_FakeASRModel",
            "asr",
        ))
        fake_module = ModuleType(self.FAKE_MODULE)
        fake_module._FakeASRModel = _FakeASRModel
        config = VoiceHubConfig(sample_rate=16_000)

        with patch.dict(sys.modules, {self.FAKE_MODULE: fake_module}):
            model = AutoModelForSpeechRecognition.from_pretrained(
                "acme/asr-checkpoint",
                model_type=self.ASR_MODEL_TYPE,
                config=config,
            )

        self.assertIsInstance(model, _FakeASRModel)
        self.assertEqual(config.model_type, self.ASR_MODEL_TYPE)

    def test_task_mismatch_is_rejected_before_runtime_import(self):
        missing_module = "tests._voicehub_module_that_must_not_import"
        register_model_spec(
            self._spec(
                self.ASR_MODEL_TYPE,
                "_MissingASRModel",
                "asr",
                module=missing_module,
            ))
        config = VoiceHubConfig()
        config.model_type = self.ASR_MODEL_TYPE

        with self.assertRaisesRegex(
                ValueError,
                "Use AutoModelForSpeechRecognition",
        ):
            AutoModelForTextToSpeech.from_config(config)
        with self.assertRaisesRegex(
                ValueError,
                "Use AutoModelForSpeechRecognition",
        ):
            AutoModelForTextToSpeech.from_pretrained(model_type=self.ASR_MODEL_TYPE)
        self.assertNotIn(missing_module, sys.modules)

    def test_task_factory_discovery_remains_tts_only(self):
        self._register_task_backends()

        available = {spec.model_type for spec in AutoModelForTextToSpeech.available_models()}
        self.assertIn("dia", available)
        self.assertNotIn(self.ASR_MODEL_TYPE, available)
        self.assertNotIn(self.VAD_MODEL_TYPE, available)

    def test_empty_task_factory_sources_use_provider_defaults_without_hub_lookup(self):
        with patch.object(
                AutoConfig,
                "from_pretrained",
                side_effect=AssertionError("empty sources must not query config"),
        ):
            asr = AutoModelForSpeechRecognition.from_pretrained("   ")
            vad = AutoModelForVoiceActivityDetection.from_pretrained()

        self.assertEqual(asr.config.name_or_path, "openai/whisper-small")
        self.assertEqual(vad.config.name_or_path, "")
        self.assertFalse(asr.is_loaded)
        self.assertFalse(vad.is_loaded)
        with self.assertRaisesRegex(ValueError, "checkpoint-family provider"):
            vad.load()

    def test_tts_factories_share_the_registry_declared_compatibility_default(self):
        default = get_default_model_spec("tts")

        self.assertIsNotNone(default)
        self.assertEqual(default.model_type, "orpheustts")
        self.assertTrue(default.default_for_task)
        signature = inspect.signature(AutoModelForTextToSpeech.from_pretrained)
        self.assertIsNone(signature.parameters["model_type"].default)

    def test_tts_default_is_resolved_from_live_registry_metadata(self):
        builtin_default = get_default_model_spec("tts")
        extension_model_type = "test-default-tts"
        extension = ModelSpec(
            model_type=extension_model_type,
            module=self.FAKE_MODULE,
            class_name="_FakeDefaultTTSModel",
            default_model_path="acme/default-tts",
            task="tts",
            default_for_task=True,
        )
        fake_module = ModuleType(self.FAKE_MODULE)
        fake_module._FakeDefaultTTSModel = _FakeDefaultTTSModel

        try:
            register_model_spec(
                replace(builtin_default, default_for_task=False),
                exist_ok=True,
            )
            register_model_spec(extension)
            with patch.dict(sys.modules, {self.FAKE_MODULE: fake_module}):
                task_model = AutoModelForTextToSpeech.from_pretrained(marker="task")

            self.assertIs(get_default_model_spec("tts"), extension)
            self.assertEqual(task_model.config.name_or_path, extension.default_model_path)
            self.assertEqual(task_model.init_kwargs["marker"], "task")

            unregister_model_spec(extension_model_type)
            self.assertIsNone(get_default_model_spec("tts"))
            with self.assertRaisesRegex(
                    ValueError,
                    "no registry-declared default checkpoint",
            ):
                AutoModelForTextToSpeech.from_pretrained()
        finally:
            unregister_model_spec(extension_model_type, missing_ok=True)
            register_model_spec(builtin_default, exist_ok=True)

        self.assertIs(get_default_model_spec("tts"), builtin_default)

    def test_task_factory_default_is_resolved_from_registry_metadata(self):
        builtin_default = get_default_model_spec("asr")
        self.assertIsNotNone(builtin_default)
        extension_model_type = _ExtensionASRConfig.model_type

        try:
            register_model_spec(
                replace(builtin_default, default_for_task=False),
                exist_ok=True,
            )
            extension_spec = AutoModelForSpeechRecognition.register(
                _ExtensionASRConfig,
                _FakeASRModel,
                default_model_path="acme/extension-default-asr",
                default_for_task=True,
            )

            model = AutoModelForSpeechRecognition.from_pretrained()

            self.assertIs(get_default_model_spec("asr"), extension_spec)
            self.assertEqual(
                model.config.name_or_path,
                "acme/extension-default-asr",
            )
            self.assertIsInstance(model, _FakeASRModel)
        finally:
            AutoModel.unregister(extension_model_type, missing_ok=True)
            register_model_spec(builtin_default, exist_ok=True)

        self.assertIs(get_default_model_spec("asr"), builtin_default)

    def test_shared_auto_factories_contain_no_registered_model_literals(self):
        package_root = Path(__file__).resolve().parents[1] / "voicehub"
        registered_model_types = {spec.model_type for spec in list_model_specs(task=None)}
        for filename in ("auto.py", ):
            with self.subTest(filename=filename):
                tree = ast.parse((package_root / filename).read_text(encoding="utf-8"))
                string_literals = {
                    node.value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                }

                self.assertEqual(registered_model_types & string_literals, set())

    def test_upstream_model_type_aliases_are_resolved_in_task_context(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps({
                    "model_type": "wav2vec2",
                    'architectures': ["Wav2Vec2ForAudioFrameClassification"],
                }),
                encoding="utf-8",
            )

            asr = AutoModelForSpeechRecognition.from_pretrained(directory)
            vad = AutoModelForVoiceActivityDetection.from_pretrained(directory)

        self.assertEqual(asr.config.model_type, "asr_transformers")
        self.assertEqual(vad.config.model_type, "vad_transformers")
        resolved_directory = str(Path(directory).resolve())
        self.assertEqual(asr.config.name_or_path, resolved_directory)
        self.assertEqual(vad.config.name_or_path, resolved_directory)

    def test_factory_config_overrides_cannot_bypass_provider_validation(self):
        invalid_factories = (
            lambda: AutoModelForSpeechRecognition.from_pretrained(architecture_family="not-asr", ),
            lambda: AutoModelForVoiceActivityDetection.from_pretrained(architecture_family="not-vad", ),
            lambda: AutoModelForVoiceActivityDetection.from_pretrained(threshold=1.5, ),
            lambda: AutoModelForVoiceActivityDetection.from_pretrained(
                model_kwargs={
                    "api_key": "must-not-be-persisted",
                }, ),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory), self.assertRaises((TypeError, ValueError)):
                factory()

    def test_provider_overrides_rebuild_without_mutating_the_caller_config(self):
        config = AutoConfig.for_model("vad_transformers")

        model = AutoModelForVoiceActivityDetection.from_config(
            config,
            threshold=0.75,
        )

        self.assertIsNot(model.config, config)
        self.assertEqual(config.inference_config["threshold"], 0.5)
        self.assertEqual(model.config.inference_config["threshold"], 0.75)


if __name__ == "__main__":
    unittest.main()
