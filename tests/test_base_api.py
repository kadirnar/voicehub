import json
import tempfile
import unittest
from pathlib import Path

from voicehub import (
    AutoConfig,
    AutoModelForTextToSpeech,
    AutoProcessor,
    PreTrainedTTSModel,
    TTSGenerationConfig,
    TTSOutput,
    VoiceHubConfig,
)
from voicehub.processing.processor import AudioProcessor, VoiceHubProcessor


class DummyConfig(VoiceHubConfig):
    model_type = "dummy"


class UnsafeSerializationConfig(VoiceHubConfig):
    model_type = "unsafe-serialization"

    def to_dict(self):
        values = super().to_dict()
        values["provider_options"] = {
            "api_key": "must-not-be-persisted",
        }
        return values


class UnsafeGenerationSerializationConfig(TTSGenerationConfig):

    def to_dict(self):
        values = super().to_dict()
        values["provider_options"] = {
            "api_key": "must-not-be-persisted",
        }
        return values


class UnsafeProcessorSerializationConfig(VoiceHubProcessor):

    def to_dict(self):
        values = super().to_dict()
        values["provider_options"] = {
            "api_key": "must-not-be-persisted",
        }
        return values


class DummyForTextToSpeech(PreTrainedTTSModel):
    config_class = DummyConfig

    def _load_pretrained_model(self) -> None:
        self.model = object()

    def _generate(self, text: str, **kwargs) -> TTSOutput:
        return TTSOutput(
            audio=[0.0],
            sample_rate=self.sample_rate,
            metadata={
                "text": text,
                **kwargs
            },
        )


class BaseApiTests(unittest.TestCase):

    def test_config_round_trip(self):
        config = VoiceHubConfig(
            sample_rate=44100,
            architectures=["ExampleForTextToSpeech"],
            custom_value=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            loaded = VoiceHubConfig.from_pretrained(directory)

        self.assertEqual(loaded.sample_rate, 44100)
        self.assertEqual(loaded.custom_value, 7)
        self.assertEqual(loaded.architectures, ["ExampleForTextToSpeech"])

    def test_config_json_contains_model_type(self):
        config = AutoConfig.for_model("f5-tts", sample_rate=24000)
        with tempfile.TemporaryDirectory() as directory:
            config_path = config.save_pretrained(directory)
            payload = json.loads(Path(config_path).read_text(encoding="utf-8"))

        self.assertEqual(payload["model_type"], "f5tts")

    def test_config_rejects_nested_runtime_secrets_at_construction(self):
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            VoiceHubConfig(token="must-not-be-persisted")
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            VoiceHubConfig(
                provider_options={
                    "headers": {
                        "authorization": "Bearer must-not-be-persisted",
                    },
                }, )
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            VoiceHubConfig(
                generation_config={
                    "provider_options": {
                        "api_key": "must-not-be-persisted",
                    },
                }, )

    def test_config_rejects_secrets_added_after_construction_before_writing(self):
        config = VoiceHubConfig(pad_token_id=0)
        config.runtime_options = {"api_key": "must-not-be-persisted"}

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with self.assertRaisesRegex(ValueError, "runtime secrets"):
                config.save_pretrained(directory)

            self.assertFalse(config_path.exists())

    def test_config_serializers_validate_the_final_subclass_payload(self):
        config = UnsafeSerializationConfig()

        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            config.to_diff_dict()
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            config.to_json_string()
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            repr(config)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with self.assertRaisesRegex(ValueError, "runtime secrets"):
                config.save_pretrained(directory)

            self.assertFalse(config_path.exists())

    def test_config_loader_token_remains_runtime_only(self):
        with tempfile.TemporaryDirectory() as directory:
            VoiceHubConfig(pad_token_id=0).save_pretrained(directory)

            config = VoiceHubConfig.from_pretrained(
                directory,
                token="runtime-only-token",
                local_files_only=True,
            )

        self.assertEqual(config.pad_token_id, 0)
        self.assertFalse(hasattr(config, "token"))

        config.token = "runtime-only-token"
        self.assertNotIn("token", config.to_dict())

    def test_tts_output_tuple_protocol(self):
        audio = [0.0, 0.1]
        output = TTSOutput(audio=audio, sample_rate=24000)
        self.assertEqual(tuple(output), (audio, 24000))
        self.assertEqual(output[0], audio)
        self.assertEqual(output["sample_rate"], 24000)

    def test_generation_config_controls_uniform_generate_method(self):
        model = DummyForTextToSpeech(DummyConfig(generation_config={"speed": 1.1}, ))
        output = model.generate(
            "hello",
            generation_config=TTSGenerationConfig(seed=7),
            speed=1.25,
        )

        self.assertEqual(output.metadata["text"], "hello")
        self.assertEqual(output.metadata["speed"], 1.25)
        self.assertEqual(output.metadata["seed"], 7)

    def test_generation_config_round_trip(self):
        generation_config = TTSGenerationConfig(
            speed=1.2,
            seed=42,
            backend_option="value",
        )
        with tempfile.TemporaryDirectory() as directory:
            generation_config.save_pretrained(directory)
            loaded = TTSGenerationConfig.from_pretrained(directory)

        self.assertEqual(loaded.speed, 1.2)
        self.assertEqual(loaded.seed, 42)
        self.assertEqual(loaded.backend_option, "value")

    def test_generation_config_rejects_runtime_secrets_at_construction(self):
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            TTSGenerationConfig(token="must-not-be-persisted")
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            TTSGenerationConfig(
                provider_options={
                    "headers": {
                        "authorization": "Bearer must-not-be-persisted",
                    },
                }, )

    def test_generation_config_rejects_secrets_added_after_construction(self):
        config = TTSGenerationConfig(
            pad_token_id=0,
            eos_token_id=1,
        )
        config.provider_options = {
            "api_key": "must-not-be-persisted",
        }

        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            config.to_dict()
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            repr(config)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "generation_config.json"
            with self.assertRaisesRegex(ValueError, "runtime secrets"):
                config.save_pretrained(directory)

            self.assertFalse(config_path.exists())

    def test_generation_config_serializers_validate_final_subclass_payload(self):
        config = UnsafeGenerationSerializationConfig()

        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            repr(config)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "generation_config.json"
            with self.assertRaisesRegex(ValueError, "runtime secrets"):
                config.save_pretrained(directory)

            self.assertFalse(config_path.exists())

    def test_generation_config_loader_token_remains_runtime_only(self):
        with tempfile.TemporaryDirectory() as directory:
            TTSGenerationConfig(
                pad_token_id=0,
                eos_token_id=1,
            ).save_pretrained(directory)

            config = TTSGenerationConfig.from_pretrained(
                directory,
                token="runtime-only-token",
            )

        self.assertEqual(config.pad_token_id, 0)
        self.assertEqual(config.eos_token_id, 1)
        self.assertFalse(hasattr(config, "token"))

    def test_generation_config_rejects_secret_from_untrusted_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "generation_config.json"
            config_path.write_text(
                json.dumps({
                    "provider_options": {
                        "headers": {
                            "authorization": "Bearer must-not-be-persisted",
                        },
                    },
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "runtime secrets"):
                TTSGenerationConfig.from_pretrained(config_path)

    def test_auto_model_uses_architecture_specific_config(self):
        config = AutoConfig.for_model(
            "parler-tts",
            name_or_path="local-checkpoint",
            compile_model=True,
        )
        model = AutoModelForTextToSpeech.from_config(config)

        self.assertEqual(config.model_type, "parlertts")
        self.assertEqual(config.name_or_path, "local-checkpoint")
        self.assertTrue(config.compile_model)
        self.assertEqual(
            config.architectures,
            ["ParlerTTSForTextToSpeech"],
        )
        self.assertIsInstance(model, PreTrainedTTSModel)
        self.assertFalse(model.is_loaded)

    def test_auto_processor_uses_model_processor_class(self):
        config = AutoConfig.for_model("kokoro")
        processor = AutoProcessor.from_config(config)
        values = processor("Hello", voice="af_heart")

        self.assertEqual(values["text"], "Hello")
        self.assertEqual(values["voice"], "af_heart")

    def test_processors_reject_nested_runtime_secrets_at_construction(self):
        for processor_class in (VoiceHubProcessor, AudioProcessor):
            with self.subTest(processor_class=processor_class.__name__):
                with self.assertRaisesRegex(ValueError, "runtime secrets"):
                    processor_class(
                        provider_options={
                            "headers": {
                                "authorization": "Bearer must-not-be-persisted",
                            },
                        }, )

    def test_processors_reject_secrets_from_untrusted_checkpoints(self):
        for processor_class in (VoiceHubProcessor, AudioProcessor):
            with self.subTest(
                    processor_class=processor_class.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "processor_config.json"
                path.write_text(
                    json.dumps({
                        "provider_options": {
                            "headers": {
                                "authorization": "Bearer must-not-be-persisted",
                            },
                        },
                    }),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "runtime secrets"):
                    processor_class.from_pretrained(path)

    def test_processor_rejects_secrets_added_after_construction(self):
        processor = VoiceHubProcessor(normalization="nfc")
        processor.init_kwargs["provider_options"] = {
            "api_key": "must-not-be-persisted",
        }

        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            processor.to_dict()
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "processor_config.json"
            with self.assertRaisesRegex(ValueError, "runtime secrets"):
                processor.save_pretrained(directory)

            self.assertFalse(output_path.exists())

    def test_processor_loader_token_remains_runtime_only(self):
        with tempfile.TemporaryDirectory() as directory:
            VoiceHubProcessor(normalization="nfc").save_pretrained(directory)
            processor = VoiceHubProcessor.from_pretrained(
                directory,
                token="runtime-only-token",
                local_files_only=True,
            )

        self.assertEqual(processor.to_dict(), {"normalization": "nfc"})

    def test_processor_save_validates_final_subclass_payload_before_writing(self):
        processor = UnsafeProcessorSerializationConfig()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifact"
            with self.assertRaisesRegex(ValueError, "runtime secrets"):
                processor.save_pretrained(root)

            self.assertFalse((root / "processor_config.json").exists())
            self.assertFalse(root.exists())

    def test_save_pretrained_writes_complete_api_contract(self):
        config = AutoConfig.for_model("parlertts")
        model = AutoModelForTextToSpeech.from_config(config)
        model.generation_config = TTSGenerationConfig(seed=11)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            files = {path.name for path in Path(directory).iterdir()}
            loaded = AutoModelForTextToSpeech.from_pretrained(directory)

        self.assertEqual(
            files,
            {
                "config.json",
                "generation_config.json",
                "processor_config.json",
            },
        )
        self.assertEqual(loaded.generation_config.seed, 11)
        self.assertFalse(loaded.is_loaded)

    def test_tts_portable_state_config_rejects_ambiguous_json_before_construction(self):
        from unittest.mock import patch

        documents = {
            "duplicate": (
                '{"model_type":"dummy","name_or_path":"discarded-secret-value",'
                '"name_or_path":"saved/model"}',
                "Duplicate JSON object key 'name_or_path'",
            ),
            "constant": (
                '{"model_type":"dummy","name_or_path":"saved/model","metric":NaN}',
                "non-finite.*NaN",
            ),
            "overflow": (
                '{"model_type":"dummy","name_or_path":"saved/model","metric":1e400}',
                r"\$\.metric.*non-finite",
            ),
        }
        for name, (document, message) in documents.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = root / "config.json"
                config_path.write_text(document, encoding="utf-8")
                (root / "model_state.pt").write_bytes(b"unread-test-state")

                with patch.object(
                        DummyForTextToSpeech,
                        "__init__",
                        side_effect=AssertionError("model construction must not run"),
                ), self.assertRaisesRegex(ValueError, message) as raised:
                    DummyForTextToSpeech.from_pretrained(
                        root,
                        config=DummyConfig(name_or_path="caller/model"),
                    )

                self.assertIn(str(config_path.resolve()), str(raised.exception))
                self.assertNotIn("discarded-secret-value", str(raised.exception))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(
                '{"model_type":"dummy","name_or_path":"saved/model",'
                '"metadata":{"token_count":12}}',
                encoding="utf-8",
            )
            state_path = root / "model_state.pt"
            state_path.write_bytes(b"unread-test-state")

            model = DummyForTextToSpeech.from_pretrained(
                root,
                config=DummyConfig(name_or_path="caller/model"),
            )

        self.assertEqual(model.config.name_or_path, "saved/model")
        self.assertEqual(model._pending_model_state_path.resolve(), state_path.resolve())
        self.assertFalse(model.is_loaded)


if __name__ == "__main__":
    unittest.main()
