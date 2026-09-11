from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voicehub.auto import (
    AutoConfig,
    AutoModelForSpeechRecognition,
    AutoModelForTextToSpeech,
    AutoModelForVoiceActivityDetection,
    AutoProcessor,
)
from voicehub.processing.processor import VoiceHubProcessor


class AutoModelConfigKwargsTests(unittest.TestCase):

    def test_tts_config_kwargs_reach_explicit_model_type(self):
        model = AutoModelForTextToSpeech.from_pretrained(
            "local-or-remote-vits-checkpoint",
            model_type="vits",
            config_kwargs={
                "torch_dtype": "float32",
                "speaking_rate": 1.25,
            },
        )

        self.assertEqual(model.config.torch_dtype, "float32")
        self.assertEqual(model.config.speaking_rate, 1.25)
        self.assertFalse(model.is_loaded)

    def test_asr_config_kwargs_reach_explicit_model_type(self):
        model = AutoModelForSpeechRecognition.from_pretrained(
            "local-or-remote-asr-checkpoint",
            model_type="asr_transformers",
            config_kwargs={
                "architecture_family": "ctc",
                "torch_dtype": "float32",
            },
        )

        self.assertEqual(model.config.architecture_family, "ctc")
        self.assertEqual(model.config.torch_dtype, "float32")
        self.assertFalse(model.is_loaded)

    def test_vad_config_kwargs_reach_explicit_model_type(self):
        model = AutoModelForVoiceActivityDetection.from_pretrained(
            "local-or-remote-vad-checkpoint",
            model_type="vad_transformers",
            config_kwargs={
                "window_duration_s": 2.0,
                "hop_duration_s": 1.0,
            },
        )

        self.assertEqual(model.config.window_duration_s, 2.0)
        self.assertEqual(model.config.hop_duration_s, 1.0)
        self.assertFalse(model.is_loaded)

    def test_config_and_config_kwargs_are_mutually_exclusive(self):
        model = AutoModelForTextToSpeech.from_pretrained(
            "local-or-remote-vits-checkpoint",
            model_type="vits",
        )

        with self.assertRaisesRegex(TypeError, "either `config`"):
            AutoModelForTextToSpeech.from_pretrained(
                "local-or-remote-vits-checkpoint",
                config=model.config,
                config_kwargs={"torch_dtype": "float32"},
            )

    def test_config_kwargs_must_be_a_string_keyed_mapping(self):
        with self.assertRaisesRegex(TypeError, "mapping"):
            AutoModelForTextToSpeech.from_pretrained(
                "local-or-remote-vits-checkpoint",
                model_type="vits",
                config_kwargs=("torch_dtype", "float32"),
            )
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            AutoModelForTextToSpeech.from_pretrained(
                "local-or-remote-vits-checkpoint",
                model_type="vits",
                config_kwargs={"": "float32"},
            )

    def test_model_type_cannot_be_overridden_through_config_kwargs(self):
        with self.assertRaisesRegex(ValueError, "top-level factory argument"):
            AutoModelForTextToSpeech.from_pretrained(
                "local-or-remote-vits-checkpoint",
                model_type="vits",
                config_kwargs={"model_type": "parlertts"},
            )

    def test_auto_config_discovers_model_type_from_local_subfolder(self):
        with tempfile.TemporaryDirectory() as directory:
            nested_directory = Path(directory) / "nested"
            AutoConfig.for_model(
                "vits",
                discovery_marker="nested-config",
            ).save_pretrained(nested_directory)

            config = AutoConfig.from_pretrained(
                directory,
                subfolder="nested",
                local_files_only=True,
            )

        self.assertEqual(config.model_type, "vits")
        self.assertEqual(config.discovery_marker, "nested-config")

    def test_auto_config_forwards_remote_subfolder_to_both_resolution_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = AutoConfig.for_model(
                "vits",
                discovery_marker="remote-config",
            ).save_pretrained(Path(directory) / "nested")
            with (
                    patch(
                        "voicehub.auto.resolve_pretrained_file",
                        return_value=config_path,
                    ) as discovery_resolver,
                    patch(
                        "voicehub.configuration.resolve_pretrained_file",
                        return_value=config_path,
                    ) as concrete_resolver,
            ):
                config = AutoConfig.from_pretrained(
                    "organization/checkpoint",
                    subfolder="nested",
                    cache_dir="/tmp/voicehub-config-cache",
                    revision="stable-revision",
                    token=True,
                    local_files_only=True,
                )

        expected_loader_options = {
            "subfolder": "nested",
            "cache_dir": "/tmp/voicehub-config-cache",
            "revision": "stable-revision",
            "token": True,
            "local_files_only": True,
        }
        discovery_resolver.assert_called_once_with(
            "organization/checkpoint",
            "config.json",
            **expected_loader_options,
        )
        concrete_resolver.assert_called_once_with(
            "organization/checkpoint",
            "config.json",
            **expected_loader_options,
        )
        self.assertEqual(config.model_type, "vits")
        self.assertEqual(config.discovery_marker, "remote-config")

    def test_task_factory_propagates_config_subfolder_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            nested_directory = Path(directory) / "nested"
            AutoConfig.for_model(
                "vits",
                discovery_marker="task-factory",
            ).save_pretrained(nested_directory)

            model = AutoModelForTextToSpeech.from_pretrained(
                directory,
                config_kwargs={
                    "subfolder": "nested",
                    "local_files_only": True,
                },
            )

        self.assertEqual(model.config.model_type, "vits")
        self.assertEqual(model.config.discovery_marker, "task-factory")
        self.assertFalse(model.is_loaded)

    def test_auto_processor_separates_config_and_processor_kwargs(self):
        config = AutoConfig.for_model("kokoro")
        with patch.object(
                AutoConfig,
                "from_pretrained",
                return_value=config,
        ) as from_pretrained:
            processor = AutoProcessor.from_pretrained(
                "org/checkpoint",
                config_kwargs={
                    "revision": "stable-revision",
                    "local_files_only": True,
                },
                voice="af_heart",
            )

        from_pretrained.assert_called_once_with(
            "org/checkpoint",
            revision="stable-revision",
            local_files_only=True,
        )
        self.assertEqual(processor.init_kwargs, {"voice": "af_heart"})

    def test_auto_processor_config_kwargs_reach_explicit_model_type(self):
        processor = VoiceHubProcessor()
        with (
                patch.object(
                    AutoProcessor,
                    "_get_processor_class",
                    return_value=VoiceHubProcessor,
                ) as get_processor_class,
                patch.object(
                    VoiceHubProcessor,
                    "from_pretrained",
                    return_value=processor,
                ) as processor_loader,
        ):
            loaded = AutoProcessor.from_pretrained(
                "local-or-remote-kokoro-checkpoint",
                model_type="kokoro",
                config_kwargs={"processor_marker": "explicit-config"},
                voice="af_heart",
            )

        self.assertIs(loaded, processor)
        config = get_processor_class.call_args.args[0]
        self.assertEqual(config.processor_marker, "explicit-config")
        processor_loader.assert_called_once_with(
            "local-or-remote-kokoro-checkpoint",
            voice="af_heart",
        )

    def test_auto_processor_local_artifact_restores_only_processor_kwargs(self):
        with tempfile.TemporaryDirectory() as directory:
            AutoConfig.for_model("kokoro").save_pretrained(directory)
            (Path(directory) / "processor_config.json").write_text(
                json.dumps({
                    "speed": 1.1,
                    "voice": "stored-voice",
                }),
                encoding="utf-8",
            )
            processor = AutoProcessor.from_pretrained(
                directory,
                config_kwargs={"local_files_only": True},
                voice="override-voice",
            )

        self.assertEqual(
            processor.init_kwargs,
            {
                "speed": 1.1,
                "voice": "override-voice",
            },
        )

    def test_auto_processor_rejects_ambiguous_or_invalid_config_kwargs(self):
        config = AutoConfig.for_model("kokoro")
        with self.assertRaisesRegex(TypeError, "either `config`"):
            AutoProcessor.from_pretrained(
                "org/checkpoint",
                config=config,
                config_kwargs={"revision": "stable-revision"},
            )
        with self.assertRaisesRegex(TypeError, "mapping"):
            AutoProcessor.from_pretrained(
                "org/checkpoint",
                model_type="kokoro",
                config_kwargs=("revision", "stable-revision"),
            )
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            AutoProcessor.from_pretrained(
                "org/checkpoint",
                model_type="kokoro",
                config_kwargs={"": "stable-revision"},
            )
        with self.assertRaisesRegex(ValueError, "top-level factory argument"):
            AutoProcessor.from_pretrained(
                "org/checkpoint",
                model_type="kokoro",
                config_kwargs={"model_type": "vits"},
            )

    def test_auto_processor_restores_remote_artifact_with_shared_hub_options(self):
        config = AutoConfig.for_model("kokoro")
        restored = VoiceHubProcessor(voice="remote-voice")
        with (
                patch.object(
                    AutoConfig,
                    "from_pretrained",
                    return_value=config,
                ) as config_loader,
                patch.object(
                    VoiceHubProcessor,
                    "from_pretrained",
                    return_value=restored,
                ) as processor_loader,
        ):
            processor = AutoProcessor.from_pretrained(
                "org/checkpoint",
                config_kwargs={
                    "cache_dir": "/tmp/voicehub-processor-cache",
                    "revision": "stable-revision",
                    "token": True,
                    "local_files_only": True,
                },
                voice="override-voice",
            )

        config_loader.assert_called_once_with(
            "org/checkpoint",
            cache_dir="/tmp/voicehub-processor-cache",
            revision="stable-revision",
            token=True,
            local_files_only=True,
        )
        processor_loader.assert_called_once_with(
            "org/checkpoint",
            cache_dir="/tmp/voicehub-processor-cache",
            revision="stable-revision",
            token=True,
            local_files_only=True,
            voice="override-voice",
        )
        self.assertIs(processor, restored)

    def test_auto_processor_missing_remote_artifact_falls_back_to_constructor(self):
        config = AutoConfig.for_model("kokoro")
        with patch(
                "voicehub.processing.processor.resolve_pretrained_file",
                side_effect=FileNotFoundError("processor config is absent"),
        ) as resolve_file:
            processor = AutoProcessor.from_pretrained(
                "org/checkpoint",
                config=config,
                voice="fallback-voice",
            )

        resolve_file.assert_called_once_with(
            "org/checkpoint",
            "processor_config.json",
            subfolder="",
            cache_dir=None,
            revision=None,
            token=None,
            local_files_only=False,
        )
        self.assertEqual(processor.init_kwargs, {"voice": "fallback-voice"})

    def test_auto_processor_restores_direct_processor_config_file(self):
        config = AutoConfig.for_model("kokoro")
        with tempfile.TemporaryDirectory() as directory:
            processor_path = Path(directory) / "processor_config.json"
            processor_path.write_text(
                json.dumps({"voice": "direct-file-voice"}),
                encoding="utf-8",
            )
            processor = AutoProcessor.from_pretrained(
                processor_path,
                config=config,
                cache_dir="/tmp/voicehub-processor-cache",
                revision="stable-revision",
                token=True,
                local_files_only=True,
            )

        self.assertEqual(processor.init_kwargs, {"voice": "direct-file-voice"})


if __name__ == "__main__":
    unittest.main()
