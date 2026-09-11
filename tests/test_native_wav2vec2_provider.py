from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from voicehub import AutoConfig, AutoModelForSpeechRecognition, get_model_spec
from voicehub.checkpointing import save_safetensors
from voicehub.models.asr_wav2vec2 import NativeWav2Vec2TrainingAdapter, Wav2Vec2ASRConfig, Wav2Vec2ForSpeechRecognition
from voicehub.models.asr_wav2vec2.native import Wav2Vec2Config, Wav2Vec2ForCTC
from voicehub.models.asr_wav2vec2.native.tokenization import Wav2Vec2CTCTokenizer
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.specs import get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_artifact(root: Path):
    torch.manual_seed(71)
    config = Wav2Vec2Config(
        vocab_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_dropout=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        final_dropout=0.0,
        layerdrop=0.0,
        conv_dim=(4, 8),
        conv_stride=(2, 2),
        conv_kernel=(4, 2),
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        apply_spec_augment=False,
        mask_time_prob=0.0,
        mask_time_min_masks=0,
        mask_feature_prob=0.0,
        mask_feature_min_masks=0,
        ctc_loss_reduction="sum",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    reference = Wav2Vec2ForCTC(config)
    values = config.to_dict()
    values['architectures'] = ["Wav2Vec2ForCTC"]
    (root / "config.json").write_text(
        json.dumps(values),
        encoding="utf-8",
    )
    (root / "vocab.json").write_text(
        json.dumps({
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "|": 4,
            "A": 5,
            "B": 6,
            "'": 7,
        }),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({
            "do_lower_case": True,
            "word_delimiter_token": "|",
        }),
        encoding="utf-8",
    )
    (root / "special_tokens_map.json").write_text(
        json.dumps({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }),
        encoding="utf-8",
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps({
            "do_normalize": True,
            "feature_size": 1,
            "padding_value": 0.0,
            "return_attention_mask": False,
            "sampling_rate": 16_000,
        }),
        encoding="utf-8",
    )
    save_safetensors(
        reference.state_dict(),
        root / "model.safetensors",
    )
    return config, reference


class _DeterministicCTCModel(torch.nn.Module):

    def __init__(self, token_ids: tuple[int, ...], vocab_size: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.token_ids = token_ids
        self.vocab_size = vocab_size

    def forward(self, input_values, attention_mask=None):
        del attention_mask
        logits = torch.full(
            (input_values.shape[0], len(self.token_ids), self.vocab_size),
            -10.0,
            dtype=input_values.dtype,
            device=input_values.device,
        )
        for index, token_id in enumerate(self.token_ids):
            logits[:, index, token_id] = 10.0 + self.anchor
        return SimpleNamespace(
            logits=logits,
            input_lengths=torch.full(
                (input_values.shape[0], ),
                len(self.token_ids),
                dtype=torch.long,
                device=input_values.device,
            ),
        )


class NativeWav2Vec2ProviderTests(unittest.TestCase):

    def test_runtime_language_override_does_not_leak_into_default_calls(self):
        vocabulary = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "|": 4,
            "A": 5,
            "B": 6,
            "'": 7,
        }
        tokenizer = Wav2Vec2CTCTokenizer(
            {
                "en": vocabulary,
                "fr": vocabulary,
            },
            target_language="en",
        )
        wrapper = Wav2Vec2ForSpeechRecognition(
            Wav2Vec2ASRConfig(target_language="en"),
            device="cpu",
        )
        wrapper.ctc_processor = SimpleNamespace(
            tokenizer=tokenizer,
            sampling_rate=16_000,
        )
        wrapper.native_config = SimpleNamespace(
            sampling_rate=16_000,
            vocab_size=len(vocabulary),
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

        self.assertEqual(wrapper._select_runtime_language("fr"), "fr")
        self.assertEqual(tokenizer.target_language, "fr")
        self.assertEqual(wrapper._select_runtime_language(None), "en")
        self.assertEqual(tokenizer.target_language, "en")

    def test_provider_import_does_not_load_external_model_runtimes(self):
        code = """
import json
import sys
from voicehub.models.asr_wav2vec2 import Wav2Vec2ForSpeechRecognition
names = ("transformers", "tokenizers", "safetensors", "torchaudio")
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
                "tokenizers": False,
                "safetensors": False,
                "torchaudio": False,
            },
        )

    def test_local_safetensors_load_training_loss_and_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference = _tiny_artifact(root)
            wrapper = Wav2Vec2ForSpeechRecognition(
                Wav2Vec2ASRConfig(name_or_path=root),
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
                    "audio": torch.linspace(-1.0, 1.0, 30),
                    "sampling_rate": 16_000,
                    "text": "AB",
                },
                phase="speech_recognition",
            )
            self.assertAlmostEqual(
                float(prepared["input_values"].mean()),
                0.0,
                places=6,
            )
            output = wrapper.model(
                prepared["input_values"].unsqueeze(0),
                attention_mask=prepared["attention_mask"].unsqueeze(0),
                labels=prepared["labels"].unsqueeze(0),
            )
            self.assertIsNotNone(output.loss)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(wrapper.model.lm_head.weight.grad)

    def test_inference_decodes_ctc_and_reports_word_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _ = _tiny_artifact(root)
            wrapper = Wav2Vec2ForSpeechRecognition(
                Wav2Vec2ASRConfig(name_or_path=root),
                device="cpu",
            )
            wrapper.load_for_training()
            wrapper.model = _DeterministicCTCModel(
                (5, 5, 0, 4, 6, 6, 0),
                vocab_size=8,
            )

            result = wrapper.transcribe(
                torch.linspace(-0.5, 0.5, 30),
                sampling_rate=16_000,
                return_timestamps=True,
            )

        self.assertEqual(result.text, "a b")
        self.assertEqual(result.metadata["backend"], "voicehub-native")
        self.assertEqual(len(result.segments), 1)
        words = result.segments[0].words
        self.assertEqual(tuple(word.text for word in words), ("a", "b"))
        self.assertAlmostEqual(words[0].start, 0.0)
        self.assertAlmostEqual(words[0].end, 0.0005)
        self.assertAlmostEqual(words[1].start, 0.001)
        self.assertAlmostEqual(words[1].end, 0.0015)
        self.assertGreater(words[0].confidence, 0.99)

    def test_training_adapter_exports_reloadable_native_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, reference = _tiny_artifact(root)
            wrapper = Wav2Vec2ForSpeechRecognition(
                Wav2Vec2ASRConfig(name_or_path=root),
                device="cpu",
            )
            adapter = AutoTrainingAdapter.from_model(wrapper)
            self.assertIsInstance(
                adapter,
                NativeWav2Vec2TrainingAdapter,
            )

            context = adapter.create_training_context({
                "audio": torch.randn(30),
                "sampling_rate": 16_000,
                "text": "AB",
            })
            training_output = adapter.execute_training_phase(context)
            self.assertTrue(torch.isfinite(training_output.loss))

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

            self.assertEqual(
                exported_config["model_type"],
                "asr_wav2vec2",
            )
            self.assertEqual(
                exported_config["voicehub_checkpoint_format"],
                "native-wav2vec2-ctc-v1",
            )
            self.assertIsInstance(auto_config, Wav2Vec2ASRConfig)
            self.assertIsInstance(
                reloaded,
                Wav2Vec2ForSpeechRecognition,
            )
            for filename in (
                    "config.json",
                    "model.safetensors",
                    "preprocessor_config.json",
                    "special_tokens_map.json",
                    "tokenizer_config.json",
                    "vocab.json",
            ):
                self.assertTrue((export / filename).is_file(), filename)
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
                "native-wav2vec2-ctc-v1",
            )

    def test_registry_and_training_spec_use_native_components(self):
        model_spec = get_model_spec("asr_wav2vec2")
        training_spec = get_training_spec("asr_wav2vec2")

        self.assertEqual(
            model_spec.module,
            "voicehub.models.asr_wav2vec2.modeling",
        )
        self.assertEqual(
            model_spec.config_module,
            "voicehub.models.asr_wav2vec2.configuration",
        )
        self.assertIn("voicehub-native", model_spec.capabilities)
        self.assertEqual(model_spec.architecture, "wav2vec2")
        self.assertEqual(
            training_spec.source_entrypoints,
            ("voicehub.models.asr_wav2vec2.native.Wav2Vec2ForCTC", ),
        )

    def test_external_runtime_configuration_is_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "never executes"):
            Wav2Vec2ASRConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors only"):
            Wav2Vec2ASRConfig(use_safetensors=False)
        with self.assertRaisesRegex(ValueError, "does not delegate"):
            Wav2Vec2ASRConfig(model_kwargs={"device_map": "auto"})

        wrapper = Wav2Vec2ForSpeechRecognition(device="cpu")
        with self.assertRaisesRegex(ValueError, "generative decoding"):
            wrapper._pipeline_call_options(
                language=None,
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=None,
                max_new_tokens=10,
                hotwords=None,
            )


if __name__ == "__main__":
    unittest.main()
