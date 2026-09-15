from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from voicehub import AutoConfig, AutoModelForSpeechRecognition
from voicehub.architectures.whisper import WhisperConfig, WhisperModel
from voicehub.architectures.whisper.checkpoint import huggingface_whisper_tensor_mapping
from voicehub.architectures.whisper.tokenization import build_openai_whisper_special_tokens
from voicehub.checkpointing import save_safetensors
from voicehub.models.asr_native.configuration import FasterWhisperConfig, OpenAIWhisperConfig
from voicehub.models.asr_native.faster_whisper import FasterWhisperForSpeechRecognition
from voicehub.models.asr_native.openai_whisper import OpenAIWhisperForSpeechRecognition
from voicehub.models.asr_whisper_native import (
    NativeWhisperTrainingAdapter,
    WhisperASRConfig,
    WhisperForSpeechRecognition,
)
from voicehub.processing.waveform import save_pcm_wave
from voicehub.tokenization import encode_gpt2_token
from voicehub.training.auto import AutoTrainingAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tokenizer_document(timestamp_count=1_501):
    mergeable = {bytes((value, )): value for value in range(256)}
    mergeable.update({
        b"he": 256,
        b"hel": 257,
        b"hell": 258,
        b"hello": 259,
        b"": 260,
    })
    vocabulary = {encode_gpt2_token(token): token_id for token, token_id in mergeable.items()}
    special = dict(
        build_openai_whisper_special_tokens(
            len(mergeable),
            num_languages=3,
            timestamp_count=timestamp_count,
        ))
    no_speech = special.pop("<|nospeech|>")
    special["<|nocaptions|>"] = no_speech
    vocabulary["<|endoftext|>"] = special["<|endoftext|>"]
    added_tokens = [{
        "id": token_id,
        "content": token,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": token.startswith("<|0."),
        "special": not token.startswith("<|0."),
    } for token, token_id in sorted(
        special.items(),
        key=lambda item: item[1],
    )]
    return {
        "version": "1.0",
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocabulary,
            "merges": [
                ["h", "e"],
                ["he", "l"],
                ["hel", "l"],
                ["hell", "o"],
            ],
        },
    }


def _tiny_artifact(root: Path):
    torch.manual_seed(7)
    config = WhisperConfig(
        vocab_size=1_773,
        num_mel_bins=4,
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=16,
        max_source_positions=4,
        max_target_positions=16,
        pad_token_id=261,
        bos_token_id=261,
        eos_token_id=261,
        decoder_start_token_id=262,
    )
    reference = WhisperModel(config)
    values = config.to_dict()
    values["architectures"] = ["WhisperForConditionalGeneration"]
    (root / "config.json").write_text(
        json.dumps(values),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(_tokenizer_document()),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({
            "eos_token_id": 261,
            "decoder_start_token_id": 262,
            "no_timestamps_token_id": 271,
            "is_multilingual": True,
            "task_to_id": {
                "translate": 266,
                "transcribe": 267,
            },
            "lang_to_id": {
                "<|en|>": 263,
                "<|zh|>": 264,
                "<|de|>": 265,
            },
            "suppress_tokens": [],
            "begin_suppress_tokens": [220, 261],
        }),
        encoding="utf-8",
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps({
            "feature_size": 4,
            "sampling_rate": 16_000,
            "hop_length": 160,
            "n_fft": 400,
        }),
        encoding="utf-8",
    )
    native_state = reference.state_dict()
    source = {
        source_name: native_state[target_name]
        for source_name, target_name in huggingface_whisper_tensor_mapping(values)
    }
    save_safetensors(source, root / "model.safetensors")
    return config, reference


class NativeWhisperProviderTests(unittest.TestCase):

    def test_english_only_release_accepts_english_without_language_tokens(self):
        wrapper = WhisperForSpeechRecognition(device="cpu")
        wrapper.generation_adapter = SimpleNamespace(token_set=SimpleNamespace(is_multilingual=False))
        for language in (None, "en", "English", "<|en|>"):
            with self.subTest(language=language):
                self.assertEqual(wrapper._normalized_language(language), "en")
        with self.assertRaisesRegex(ValueError, "English-only"):
            wrapper._normalized_language("de")

    def test_provider_import_does_not_load_external_model_runtimes(self):
        code = """
import json
import sys
from voicehub.models.asr_whisper_native import WhisperForSpeechRecognition
names = ("transformers", "safetensors", "torchaudio", "whisper")
print(json.dumps({name: name in sys.modules for name in names}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(result.stdout),
            {
                "transformers": False,
                "safetensors": False,
                "torchaudio": False,
                "whisper": False,
            },
        )

    def test_safetensors_load_training_and_segment_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native_config, reference = _tiny_artifact(root)
            wrapper = WhisperForSpeechRecognition(
                WhisperASRConfig(name_or_path=root),
                device="cpu",
            )

            wrapper.load_for_training()

            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    wrapper.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )
            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": torch.zeros(800),
                    "sampling_rate": 16_000,
                    "text": "hello",
                    "language": "en",
                },
                phase="speech_recognition",
            )
            output = wrapper.model(
                prepared["input_features"].unsqueeze(0),
                labels=prepared["labels"].unsqueeze(0),
            )
            self.assertIsNotNone(output.loss)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(wrapper.model.decoder.token_embedding.weight.grad)

            token_set = wrapper.generation_adapter.token_set

            class FakeGenerationAdapter:

                def __init__(self):
                    self.token_set = token_set

                def generate(self, features, *, config):
                    self.features = features
                    self.config = config
                    return SimpleNamespace(
                        generated_sequences=torch.tensor([[272, 259, 274, 261]]),
                        language_token_ids=torch.tensor([263]),
                    )

            fake = FakeGenerationAdapter()
            wrapper.generation_adapter = fake
            result = wrapper.transcribe(
                torch.zeros(800),
                sampling_rate=16_000,
                return_timestamps=True,
            )

        self.assertEqual(native_config.expected_input_frames, 8)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.language, "en")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "hello")
        self.assertEqual(result.segments[0].start, 0.0)
        self.assertEqual(result.segments[0].end, 0.04)
        self.assertEqual(tuple(fake.features.shape), (1, 4, 8))
        self.assertTrue(fake.config.return_timestamps)

    def test_training_prepares_collated_paths_waveforms_and_row_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _tiny_artifact(root)
            audio_path = save_pcm_wave(
                root / "sample.wav",
                torch.zeros(400),
                8_000,
            )
            wrapper = WhisperForSpeechRecognition(
                WhisperASRConfig(name_or_path=root),
                device="cpu",
            )
            wrapper.load_for_training()

            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": [
                        audio_path,
                        torch.cat((torch.zeros(400), torch.ones(300))),
                    ],
                    "audio_lengths": torch.tensor([400, 400]),
                    "sampling_rate": torch.tensor([8_000, 16_000]),
                    "text": ["hello", "hello hello"],
                    "language": ["en", "de"],
                    "task": ["transcribe", "translate"],
                },
                phase="speech_recognition",
            )

            self.assertEqual(tuple(prepared["input_features"].shape), (2, 4, 8))
            self.assertEqual(tuple(prepared["labels"].shape), (2, 7))
            self.assertEqual(
                prepared["labels"][0].tolist(),
                [263, 267, 271, 259, 261, -100, -100],
            )
            self.assertEqual(
                prepared["labels"][1].tolist(),
                [265, 266, 271, 259, 32, 259, 261],
            )
            output = wrapper.model(
                prepared["input_features"],
                labels=prepared["labels"],
            )
            self.assertTrue(torch.isfinite(output.loss))

            with self.assertRaisesRegex(
                    ValueError,
                    r"row 1.*audio_lengths.*exceeds",
            ):
                wrapper.prepare_training_inputs(
                    {
                        "audio": torch.zeros(2, 800),
                        "audio_lengths": [800, 801],
                        "sampling_rate": 16_000,
                        "text": ["hello", "hello"],
                    },
                    phase="speech_recognition",
                )
            with self.assertRaisesRegex(ValueError, r"row 1.*audio context"):
                wrapper.prepare_training_inputs(
                    {
                        "audio": torch.zeros(2, 1_281),
                        "audio_lengths": [800, 1_281],
                        "sampling_rate": 16_000,
                        "text": ["hello", "hello"],
                    },
                    phase="speech_recognition",
                )
            with self.assertRaisesRegex(ValueError, r"row 1.*decoder context"):
                wrapper.prepare_training_inputs(
                    {
                        "audio": torch.zeros(2, 800),
                        "sampling_rate": 16_000,
                        "text": ["hello", "hello " * 20],
                    },
                    phase="speech_recognition",
                )

    def test_training_adapter_exports_reloadable_native_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference = _tiny_artifact(root)
            wrapper = WhisperForSpeechRecognition(
                WhisperASRConfig(name_or_path=root),
                device="cpu",
            )
            adapter = AutoTrainingAdapter.from_model(wrapper)
            self.assertIsInstance(adapter, NativeWhisperTrainingAdapter)

            export = root / "export"
            adapter.save_pretrained(export)
            exported_config = json.loads((export / "config.json").read_text(encoding="utf-8"))
            auto_config = AutoConfig.from_pretrained(export)
            reloaded = AutoModelForSpeechRecognition.from_pretrained(
                export,
                config=auto_config,
                device="cpu",
            )
            reloaded.load_for_training()

            self.assertEqual(exported_config["model_type"], "asr_whisper")
            self.assertIsInstance(auto_config, WhisperASRConfig)
            self.assertIsInstance(reloaded, WhisperForSpeechRecognition)
            self.assertEqual(
                exported_config["voicehub_checkpoint_format"],
                "native-whisper-v1",
            )
            self.assertTrue((export / "tokenizer.json").is_file())
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    reloaded.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )

            manifest = adapter.artifact_manifest()
            self.assertEqual(
                manifest["checkpoint_format"],
                "native-whisper-v1",
            )
            self.assertEqual(
                manifest["checkpoint_semantics"]["save_pretrained"],
                "voicehub-native-whisper-safetensors-and-processor",
            )

    def test_openai_compatibility_provider_fine_tunes_and_round_trips_natively(self, ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _tiny_artifact(root)
            wrapper = OpenAIWhisperForSpeechRecognition(
                OpenAIWhisperConfig(name_or_path=root),
                device="cpu",
            )
            adapter = wrapper.get_training_adapter()
            adapter.setup()
            export = root / "openai-compatible-export"
            adapter.save_pretrained(export)

            exported_config = AutoConfig.from_pretrained(export)
            reloaded = AutoModelForSpeechRecognition.from_pretrained(
                export,
                config=exported_config,
                device="cpu",
                lazy_load=False,
            )

            self.assertIsInstance(adapter, NativeWhisperTrainingAdapter)
            self.assertEqual(adapter.model_type, "asr_openai_whisper")
            self.assertIsInstance(exported_config, OpenAIWhisperConfig)
            self.assertIsInstance(
                reloaded,
                OpenAIWhisperForSpeechRecognition,
            )
            self.assertEqual(
                json.loads((export / "config.json").read_text(encoding="utf-8"))["model_type"],
                "asr_openai_whisper",
            )

    def test_faster_whisper_compatibility_round_trips_without_ctranslate2(self, ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _tiny_artifact(root)
            wrapper = FasterWhisperForSpeechRecognition(
                FasterWhisperConfig(
                    name_or_path=root,
                    compute_type="float32",
                ),
                device="cpu",
            )
            adapter = wrapper.get_training_adapter()
            adapter.setup()
            export = root / "faster-compatible-export"
            adapter.save_pretrained(export)

            auto_config = AutoConfig.from_pretrained(export)
            reloaded = AutoModelForSpeechRecognition.from_pretrained(
                export,
                config=auto_config,
                device="cpu",
                lazy_load=False,
            )

            self.assertIsInstance(adapter, NativeWhisperTrainingAdapter)
            self.assertEqual(adapter.model_type, "asr_faster_whisper")
            self.assertIsInstance(auto_config, FasterWhisperConfig)
            self.assertIsInstance(
                reloaded,
                FasterWhisperForSpeechRecognition,
            )
            self.assertEqual(auto_config.torch_dtype, "float32")


if __name__ == "__main__":
    unittest.main()
