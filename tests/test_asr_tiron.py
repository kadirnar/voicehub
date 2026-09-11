from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub import AutoConfig, AutoModelForSpeechRecognition
from voicehub.checkpointing import save_safetensors
from voicehub.models.asr_tiron import TironASRConfig, TironForSpeechRecognition
from voicehub.models.asr_tiron.metadata import SPEAKER_TOKEN_IDS, TIRON_CHECKPOINT_REVISION, TIRON_HARNESS_REVISION
from voicehub.models.asr_whisper_native import NativeWhisperTrainingAdapter
from voicehub.models.asr_whisper_native.native import WhisperConfig, WhisperModel
from voicehub.models.asr_whisper_native.native.checkpoint import huggingface_whisper_tensor_mapping
from voicehub.models.asr_whisper_native.native.tokenization import build_openai_whisper_special_tokens
from voicehub.processing.waveform import save_pcm_wave
from voicehub.tokenization import encode_gpt2_token
from voicehub.training.auto import AutoTrainingAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiron_tokenizer_document():
    mergeable = {bytes((value, )): value for value in range(256)}
    mergeable.update({
        b"he": 256,
        b"hel": 257,
        b"hell": 258,
        b"hello": 259,
        b"": 50_256,
    })
    vocabulary = {encode_gpt2_token(token): token_id for token, token_id in mergeable.items()}
    special = dict(build_openai_whisper_special_tokens(
        50_257,
        num_languages=100,
    ))
    special.update({
        f"<|speaker{index}|>": token_id
        for index, token_id in enumerate(SPEAKER_TOKEN_IDS, start=1)
    })
    vocabulary["<|endoftext|>"] = special["<|endoftext|>"]
    return {
        "version":
        "1.0",
        "added_tokens": [{
            "id": token_id,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": token.startswith("<|0."),
            "special": (not token.startswith("<|0.") or token.startswith("<|speaker")),
        } for token, token_id in sorted(
            special.items(),
            key=lambda item: item[1],
        )],
        "normalizer":
        None,
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


def _tiny_tiron_artifact(root: Path):
    torch.manual_seed(17)
    config = WhisperConfig(
        vocab_size=51_904,
        num_mel_bins=4,
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=16,
        max_source_positions=4,
        max_target_positions=24,
        use_cache=False,
        pad_token_id=50_256,
        bos_token_id=50_257,
        eos_token_id=50_257,
        decoder_start_token_id=50_258,
    )
    reference = WhisperModel(config)
    values = config.to_dict()
    values['architectures'] = ["WhisperForConditionalGeneration"]
    (root / "config.json").write_text(
        json.dumps(values),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(_tiron_tokenizer_document()),
        encoding="utf-8",
    )
    (root / "generation_config.json").write_text(
        json.dumps({
            "eos_token_id": 50_257,
            "decoder_start_token_id": 50_258,
            "no_timestamps_token_id": 50_364,
            "is_multilingual": True,
            "task_to_id": {
                "translate": 50_359,
                "transcribe": 50_360,
            },
            "lang_to_id": {
                "<|en|>": 50_259,
                "<|zh|>": 50_260,
            },
            "suppress_tokens": [],
            "begin_suppress_tokens": [],
            "max_new_tokens": 20,
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
    state = reference.state_dict()
    source = {
        source_name: state[target_name]
        for source_name, target_name in huggingface_whisper_tensor_mapping(values)
    }
    save_safetensors(source, root / "model.safetensors")
    return reference


class TironConfigurationTests(unittest.TestCase):

    def test_config_pins_public_artifacts_and_validates_grammar_controls(self):
        model = TironForSpeechRecognition()

        self.assertEqual(model.config.model_type, "asr_tiron")
        self.assertEqual(model.config.architecture_family, "speech-seq2seq")
        self.assertEqual(model.config.default_language, "en")
        self.assertEqual(model.config.name_or_path, "Trelis/tiron")
        self.assertEqual(model.config.revision, TIRON_CHECKPOINT_REVISION)
        self.assertTrue(model.config.constrained_decoding)
        with self.assertRaisesRegex(ValueError, "speech-seq2seq"):
            TironASRConfig(architecture_family="auto")
        with self.assertRaisesRegex(ValueError, "default_language"):
            TironASRConfig(default_language="")
        with self.assertRaisesRegex(ValueError, "pipeline_kwargs"):
            TironASRConfig(pipeline_kwargs={"batch_size": 2})
        with self.assertRaisesRegex(ValueError, "between 1 and 8"):
            TironASRConfig(max_speakers=9)

    def test_public_import_loads_no_external_runtime_or_torch(self):
        code = """
import json
import sys
from voicehub.models.asr_tiron import TironForSpeechRecognition
names = ("torch", "transformers", "safetensors", "torchaudio")
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
                "torch": False,
                "transformers": False,
                "safetensors": False,
                "torchaudio": False,
            },
        )


class TironNativeRuntimeTests(unittest.TestCase):

    def test_strict_safetensors_load_training_backward_and_target_grammar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = _tiny_tiron_artifact(root)
            wrapper = TironForSpeechRecognition(
                TironASRConfig(name_or_path=root),
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

            batch = wrapper.prepare_training_inputs(
                {
                    "audio": torch.zeros(800),
                    "sampling_rate": 16_000,
                    "language": "en",
                    "text": ("<|speaker1|><|0.00|>"
                             "hello"
                             "<|0.04|>"),
                },
                phase="speech_recognition",
            )
            self.assertEqual(
                batch["labels"].tolist(),
                [
                    50_259,
                    50_360,
                    51_866,
                    50_365,
                    259,
                    50_367,
                    50_257,
                ],
            )
            output = wrapper.model(
                batch["input_features"].unsqueeze(0),
                labels=batch["labels"].unsqueeze(0),
            )
            self.assertIsNotNone(output.loss)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(wrapper.model.decoder.token_embedding.weight.grad)

            with self.assertRaisesRegex(ValueError, "speaker1"):
                wrapper.prepare_training_inputs(
                    {
                        "audio": torch.zeros(800),
                        "sampling_rate": 16_000,
                        "text": ("<|speaker2|><|0.00|>"
                                 "hello"
                                 "<|0.04|>"),
                    },
                    phase="speech_recognition",
                )

    def test_training_prepares_collated_rows_and_validates_each_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _tiny_tiron_artifact(root)
            audio_path = save_pcm_wave(
                root / "sample.wav",
                torch.zeros(400),
                8_000,
            )
            wrapper = TironForSpeechRecognition(
                TironASRConfig(name_or_path=root),
                device="cpu",
            )
            wrapper.load_for_training()
            first_target = "<|speaker1|><|0.00|>hello<|0.04|>"

            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": [
                        audio_path,
                        torch.cat((torch.zeros(800), torch.ones(100))),
                    ],
                    "audio_lengths": [400, 800],
                    "sampling_rate": [8_000, 16_000],
                    "language": ["en", "zh"],
                    "task": ["transcribe", "transcribe"],
                    "text": [first_target, "<|nospeech|>"],
                },
                phase="speech_recognition",
            )

            self.assertEqual(tuple(prepared["input_features"].shape), (2, 4, 8))
            self.assertEqual(tuple(prepared["labels"].shape), (2, 7))
            self.assertEqual(
                prepared["labels"][0].tolist(),
                [
                    50_259,
                    50_360,
                    51_866,
                    50_365,
                    259,
                    50_367,
                    50_257,
                ],
            )
            self.assertEqual(
                prepared["labels"][1].tolist(),
                [
                    50_260,
                    50_360,
                    50_363,
                    50_257,
                    -100,
                    -100,
                    -100,
                ],
            )
            output = wrapper.model(
                prepared["input_features"],
                labels=prepared["labels"],
            )
            self.assertTrue(torch.isfinite(output.loss))

            with self.assertRaisesRegex(ValueError, r"row 1.*speaker1"):
                wrapper.prepare_training_inputs(
                    {
                        "audio": torch.zeros(2, 800),
                        "audio_lengths": [800, 800],
                        "sampling_rate": 16_000,
                        "language": ["en", "zh"],
                        "text": [
                            first_target,
                            "<|speaker2|><|0.00|>hello<|0.04|>",
                        ],
                    },
                    phase="speech_recognition",
                )
            with self.assertRaisesRegex(ValueError, r"row 1.*audio context"):
                wrapper.prepare_training_inputs(
                    {
                        "audio": torch.zeros(2, 1_281),
                        "audio_lengths": [800, 1_281],
                        "sampling_rate": 16_000,
                        "language": ["en", "zh"],
                        "text": [first_target, first_target],
                    },
                    phase="speech_recognition",
                )
            with self.assertRaisesRegex(ValueError, r"row 1.*decoder context"):
                wrapper.prepare_training_inputs(
                    {
                        "audio": torch.zeros(2, 800),
                        "sampling_rate": 16_000,
                        "language": ["en", "zh"],
                        "text": [
                            first_target,
                            ("<|speaker1|><|0.00|>" + "hello" * 30 + "<|0.04|>"),
                        ],
                    },
                    phase="speech_recognition",
                )

    def test_inference_preserves_speakers_timestamps_and_trailing_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _tiny_tiron_artifact(root)
            wrapper = TironForSpeechRecognition(
                TironASRConfig(name_or_path=root),
                device="cpu",
            )
            wrapper.load()
            generated = [
                51_866,
                50_365,
                259,
                50_367,
                51_867,
                50_368,
                259,
                50_257,
            ]
            calls = []

            def generate_window(
                input_features,
                *,
                language,
                max_new_tokens,
                max_speakers,
                constrained_decoding,
            ):
                calls.append({
                    "shape": tuple(input_features.shape),
                    "language": language,
                    "max_new_tokens": max_new_tokens,
                    "max_speakers": max_speakers,
                    "constrained_decoding": constrained_decoding,
                })
                return generated, language

            wrapper._generate_window = generate_window
            result = wrapper.transcribe(
                torch.zeros(1_600),
                sampling_rate=16_000,
                return_timestamps=True,
                max_speakers=2,
            )

        self.assertEqual(result.text, "hello hello")
        self.assertEqual(
            [(
                segment.speaker,
                segment.start,
                segment.end,
                segment.text,
            ) for segment in result.segments],
            [
                ("SPEAKER_00", 0.0, 0.04, "hello"),
                ("SPEAKER_01", 0.06, 0.1, "hello"),
            ],
        )
        self.assertEqual(result.metadata["backend"], "voicehub-native")
        self.assertEqual(
            result.metadata["reference_harness_revision"],
            TIRON_HARNESS_REVISION,
        )
        self.assertEqual(
            calls,
            [{
                "shape": (1, 4, 8),
                "language": "en",
                "max_new_tokens": 444,
                "max_speakers": 2,
                "constrained_decoding": True,
            }],
        )

    def test_invalid_inference_controls_fail_before_generation(self):
        wrapper = TironForSpeechRecognition(device="cpu")
        cases = (
            ({
                "task": "translate"
            }, "not translation"),
            ({
                "return_timestamps": "word"
            }, "word-level"),
            ({
                "chunk_length_s": 15.0
            }, "one window"),
            ({
                "batch_size": 2
            }, "one audio window"),
            ({
                "num_beams": 2
            }, "num_beams=1"),
            ({
                "hotwords": ("VoiceHub", )
            }, "hotword"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    wrapper._transcribe(
                        torch.zeros(800),
                        sampling_rate=16_000,
                        **kwargs,
                    )
        for invalid_limit in (True, 1.5):
            with self.subTest(max_new_tokens=invalid_limit):
                with self.assertRaisesRegex(TypeError, "must be an integer"):
                    wrapper._transcribe(
                        torch.zeros(800),
                        sampling_rate=16_000,
                        max_new_tokens=invalid_limit,
                    )

    def test_training_rejects_serving_artifacts_before_loading(self):
        TironForSpeechRecognition(
            TironASRConfig(name_or_path="publisher/tiron.safetensors"))._validate_training_runtime()
        with self.assertRaisesRegex(ValueError, "inference-only"):
            TironForSpeechRecognition(
                TironASRConfig(name_or_path="publisher/tiron.gguf"))._validate_training_runtime()

    def test_native_training_adapter_exports_and_reloads_complete_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = _tiny_tiron_artifact(root)
            wrapper = TironForSpeechRecognition(
                TironASRConfig(name_or_path=root),
                device="cpu",
            )
            adapter = AutoTrainingAdapter.from_model(wrapper)
            self.assertIsInstance(adapter, NativeWhisperTrainingAdapter)

            export = root / "export"
            adapter.save_pretrained(export)
            exported_values = json.loads((export / "config.json").read_text(encoding="utf-8"))
            auto_config = AutoConfig.from_pretrained(export)
            reloaded = AutoModelForSpeechRecognition.from_pretrained(
                export,
                config=auto_config,
                device="cpu",
                lazy_load=False,
            )

            self.assertEqual(exported_values["model_type"], "asr_tiron")
            self.assertEqual(
                exported_values["tiron_token_grammar"],
                "speaker_blocks-v1",
            )
            self.assertIsInstance(auto_config, TironASRConfig)
            self.assertIsInstance(reloaded, TironForSpeechRecognition)
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(
                    reloaded.model.state_dict()[name],
                    expected,
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
