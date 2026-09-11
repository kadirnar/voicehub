from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is VoiceHub's compute runtime")
class NativeDiaTests(unittest.TestCase):

    @staticmethod
    def dia_config():
        from voicehub.models.dia.native.configuration import DiaArchitectureConfig, DiaDecoderConfig, DiaEncoderConfig

        return DiaArchitectureConfig(
            encoder_config=DiaEncoderConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=4,
                max_position_embeddings=64,
                vocab_size=256,
            ),
            decoder_config=DiaDecoderConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                cross_hidden_size=8,
                cross_num_attention_heads=2,
                cross_num_key_value_heads=2,
                cross_head_dim=4,
                max_position_embeddings=64,
                vocab_size=12,
                num_channels=2,
                bos_token_id=10,
                eos_token_id=8,
                pad_token_id=9,
            ),
            delay_pattern=(0, 1),
        )

    @staticmethod
    def dac_config():
        from voicehub.models.dac.native.configuration import DacConfig

        return DacConfig(
            encoder_hidden_size=4,
            downsampling_ratios=(2, 2),
            decoder_hidden_size=16,
            n_codebooks=2,
            codebook_size=8,
            codebook_dim=2,
            sampling_rate=16_000,
        )

    @classmethod
    def write_artifact(cls, root: Path):
        import torch

        from voicehub.checkpointing import save_safetensors
        from voicehub.models.dac.native.modeling import DacModel
        from voicehub.models.dia.native.modeling import DiaForConditionalGeneration

        root.mkdir(parents=True, exist_ok=True)
        dia_config = cls.dia_config()
        model = DiaForConditionalGeneration(dia_config)
        (root / "config.json").write_text(
            json.dumps(dia_config.to_dict()),
            encoding="utf-8",
        )
        save_safetensors(
            model.state_dict(),
            root / "model.safetensors",
            metadata={"format": "pt"},
        )
        (root / "preprocessor_config.json").write_text(
            json.dumps({
                "sampling_rate": 16_000,
                "hop_length": 4,
            }),
            encoding="utf-8",
        )
        (root / "tokenizer_config.json").write_text(
            json.dumps({"max_length": 64}),
            encoding="utf-8",
        )
        (root / "audio_tokenizer_config.json").write_text(
            json.dumps({
                "audio_tokenizer_name_or_path": "./audio_tokenizer",
            }),
            encoding="utf-8",
        )
        codec_root = root / "audio_tokenizer"
        codec_root.mkdir()
        dac_config = cls.dac_config()
        codec = DacModel(dac_config)
        codec_config = {
            **dac_config.to_dict(),
            "voicehub_checkpoint_format": "native-state-dict-v1",
        }
        (codec_root / "config.json").write_text(
            json.dumps(codec_config),
            encoding="utf-8",
        )
        save_safetensors(
            codec.state_dict(),
            codec_root / "model.safetensors",
            metadata={"format": "pt"},
        )
        return model, codec

    def test_modules_do_not_import_provider_frameworks(self):
        code = """
import sys
import voicehub.models.dia
import voicehub.models.dia.training
print(*[
    name in sys.modules
    for name in ("transformers", "huggingface_hub", "torchaudio")
])
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False False False")

    def test_public_checkpoint_inventory_is_exact(self):
        from voicehub.models.dia.native.checkpoint import dia_header_fingerprint, native_dia_tensor_shapes
        from voicehub.models.dia.native.configuration import DiaArchitectureConfig
        from voicehub.models.dia.native.metadata import (
            NARI_DIA_HEADER_FINGERPRINT,
            NARI_DIA_PARAMETER_COUNT,
            NARI_DIA_TENSOR_COUNT,
        )

        shapes = native_dia_tensor_shapes(DiaArchitectureConfig())

        self.assertEqual(len(shapes), NARI_DIA_TENSOR_COUNT)
        self.assertEqual(
            sum(math.prod(shape) for shape in shapes.values()),
            NARI_DIA_PARAMETER_COUNT,
        )
        self.assertEqual(
            dia_header_fingerprint(shapes),
            NARI_DIA_HEADER_FINGERPRINT,
        )

    def test_tokenizer_and_delay_protocol_are_deterministic(self):
        import torch

        from voicehub.models.dia.native.processing import DiaByteTokenizer, DiaProcessor

        tokenizer = DiaByteTokenizer(max_length=32)
        self.assertEqual(
            tokenizer.encode("[S1]Hi [S2]"),
            [1, 72, 105, 32, 2],
        )
        self.assertEqual(
            tokenizer.decode([1, 72, 105, 32, 2]),
            "[S1]Hi [S2]",
        )
        audio = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
        indices = DiaProcessor.build_indices(1, 3, 2, (0, 1))
        delayed = DiaProcessor.apply_audio_delay(
            audio,
            pad_token_id=9,
            bos_token_id=10,
            precomputed_indices=indices,
        )
        self.assertEqual(
            delayed.tolist(),
            [[[1, 10], [3, 2], [5, 4]]],
        )
        revert = DiaProcessor.build_indices(
            1,
            3,
            2,
            (0, 1),
            revert=True,
        )
        restored = DiaProcessor.apply_audio_delay(
            delayed,
            pad_token_id=-1,
            bos_token_id=-1,
            precomputed_indices=revert,
        )
        self.assertEqual(
            restored.tolist(),
            [[[1, 2], [3, 4], [5, -1]]],
        )

    def test_full_teacher_forced_backward_reaches_every_parameter(self):
        import torch

        from voicehub.models.dia.native.modeling import DiaForConditionalGeneration

        model = DiaForConditionalGeneration(self.dia_config())
        output = model(
            input_ids=torch.tensor([[1, 72, 2]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            decoder_input_ids=torch.tensor([[[10, 10], [1, 10], [2, 3], [8, 4]]]),
            decoder_attention_mask=torch.ones(1, 4, dtype=torch.long),
            labels=torch.tensor([[1, 2, 8, -100], [3, 4, 5, 8]]),
        )
        output.loss.backward()

        self.assertEqual(output.logits.shape, (2, 4, 12))
        self.assertEqual(output.loss.ndim, 0)
        self.assertTrue(torch.isfinite(output.loss))
        trainable = list(model.named_parameters())
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.grad is not None for _, parameter in trainable))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for _, parameter in trainable))

    def test_strict_safetensors_load_export_and_reload(self):
        import torch

        from voicehub.models.dia.native.runtime import load_dia_runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            expected_model, expected_codec = self.write_artifact(root)
            runtime = load_dia_runtime(
                root,
                device="cpu",
                compute_dtype="float32",
                for_training=True,
            )

            self.assertFalse(
                any(parameter.requires_grad for parameter in runtime.processor.audio_tokenizer.parameters()))
            for name, expected in expected_model.state_dict().items():
                torch.testing.assert_close(
                    runtime.model.state_dict()[name],
                    expected,
                )
            for name, expected in expected_codec.state_dict().items():
                torch.testing.assert_close(
                    runtime.processor.audio_tokenizer.state_dict()[name],
                    expected,
                )

            export = Path(directory) / "export"
            runtime.save_pretrained(export)
            codec_config = json.loads(
                (export / "audio_tokenizer" / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                codec_config["voicehub_checkpoint_format"],
                "native-state-dict-v1",
            )
            restored = load_dia_runtime(
                export,
                device="cpu",
                compute_dtype="float32",
            )
            for name, expected in runtime.model.state_dict().items():
                torch.testing.assert_close(
                    restored.model.state_dict()[name],
                    expected,
                )

            batch = restored.processor(
                text=["[S1] Hello."],
                generation=True,
            )
            tokens = restored.model.generate(
                **batch,
                do_sample=False,
                guidance_scale=None,
                max_new_tokens=2,
                top_k=None,
                top_p=1.0,
            )
            self.assertEqual(tokens.ndim, 3)
            self.assertEqual(tokens.shape[-1], 2)

    def test_training_collator_encodes_audio_and_freezes_codec(self):
        import torch

        from voicehub.models.dia.native.runtime import load_dia_runtime
        from voicehub.models.dia.training import DiaTrainingCollator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifact(root)
            runtime = load_dia_runtime(
                root,
                device="cpu",
                compute_dtype="float32",
                for_training=True,
            )
            collator = DiaTrainingCollator(runtime.processor)
            batch = collator([{
                "text": "[S1] Hello.",
                "audio": {
                    "array": torch.linspace(-0.2, 0.2, 64),
                    "sampling_rate": 16_000,
                },
            }])
            columnar_batch = runtime.prepare_inputs({
                "text": ["[S1] Hello."],
                "audio": [{
                    "array": torch.linspace(-0.2, 0.2, 64),
                    "sampling_rate": 16_000,
                }],
            })
            loss = runtime.forward_loss(batch)
            loss.backward()

            self.assertEqual(
                set(columnar_batch),
                set(batch),
            )
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(parameter.grad is not None for parameter in runtime.model.parameters()))
            self.assertTrue(
                all(parameter.grad is None for parameter in runtime.processor.audio_tokenizer.parameters()))

    def test_public_wrapper_is_native_and_rejects_legacy(self):
        import torch

        from voicehub.models.dia.modeling import DiaConfig, DiaForTextToSpeech

        with self.assertRaisesRegex(ValueError, "native"):
            DiaConfig(backend="transformers")
        with self.assertRaisesRegex(ValueError, "InferenceStrategy"):
            DiaConfig(use_torch_compile=True)
        legacy = DiaForTextToSpeech(
            model_path="nari-labs/Dia-1.6B",
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "Dia-1.6B-0626"):
            legacy._validate_training_runtime()

        class FakeProcessor:

            def __call__(self, **kwargs):
                self.prepared = kwargs
                return SimpleNamespace(
                    to=lambda device: {
                        "input_ids": torch.tensor([[1]]),
                        "decoder_input_ids": torch.tensor([[[10, 10]]]),
                    })

            def batch_decode(self, values, *, audio_prompt_len=None):
                self.decoded = (values, audio_prompt_len)
                return [torch.tensor([0.25, -0.25])]

        class FakeModel:

            def generate(self, **kwargs):
                self.options = kwargs
                return torch.tensor([[[10, 10], [8, 8]]])

        processor = FakeProcessor()
        source_model = FakeModel()
        runtime = SimpleNamespace(
            artifacts=SimpleNamespace(revision="test-revision"),
            model=source_model,
            processor=processor,
            sample_rate=44_100,
        )
        wrapper = DiaForTextToSpeech(device="cpu")
        wrapper.model = source_model
        wrapper._dia_runtime = runtime
        wrapper._loaded_backend = "native"
        with patch(
                "voicehub.models.dia.modeling.seeded_inference",
                return_value=nullcontext(17),
        ):
            output = wrapper._generate(
                "[S1] Test.",
                max_tokens=4,
                cfg_scale=2.5,
            )

        self.assertEqual(output.metadata["backend"], "voicehub-native")
        self.assertEqual(output.metadata["seed"], 17)
        self.assertEqual(source_model.options["max_new_tokens"], 4)
        self.assertEqual(source_model.options["guidance_scale"], 2.5)

    def test_malformed_checkpoint_is_rejected_before_assignment(self):
        import torch

        from voicehub.checkpointing.errors import CheckpointCompatibilityError
        from voicehub.models.dia.native.checkpoint import HuggingFaceDiaCheckpointAdapter
        from voicehub.models.dia.native.modeling import DiaForConditionalGeneration

        config = self.dia_config()
        source = DiaForConditionalGeneration(config)
        checkpoint = dict(source.state_dict())
        checkpoint.pop(next(iter(checkpoint)))
        with torch.device("meta"):
            target = DiaForConditionalGeneration(config)
        with self.assertRaises(CheckpointCompatibilityError):
            HuggingFaceDiaCheckpointAdapter().load_assign_streaming(
                target,
                checkpoint,
                config.to_dict(),
                device="cpu",
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()
