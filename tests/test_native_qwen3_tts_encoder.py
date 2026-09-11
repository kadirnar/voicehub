from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.qwen3tts.native.checkpoint import (
    export_qwen3_tts_speech_tokenizer,
    load_qwen3_tts_decoder_checkpoint,
    load_qwen3_tts_encoder_checkpoint,
)
from voicehub.models.qwen3tts.native.codec import Qwen3TTSSpeechDecoder
from voicehub.models.qwen3tts.native.configuration import (
    Qwen3TTSDecoderConfig,
    Qwen3TTSEncoderConfig,
    Qwen3TTSTokenizerConfig,
)
from voicehub.models.qwen3tts.native.encoder import Qwen3TTSSpeechEncoder


def _tiny_encoder_config() -> Qwen3TTSEncoderConfig:
    return Qwen3TTSEncoderConfig.from_dict({
        "sampling_rate": 240,
        "_frame_rate": 30.0,
        "audio_channels": 1,
        "hidden_size": 8,
        "num_filters": 2,
        "num_residual_layers": 1,
        "upsampling_ratios": [2, 2],
        "kernel_size": 3,
        "last_kernel_size": 3,
        "residual_kernel_size": 3,
        "dilation_growth_rate": 2,
        "use_causal_conv": True,
        "pad_mode": "constant",
        "compress": 2,
        "trim_right_ratio": 1.0,
        "codebook_size": 8,
        "codebook_dim": 4,
        "num_quantizers": 4,
        "use_conv_shortcut": False,
        "vector_quantization_hidden_dimension": 4,
        "num_semantic_quantizers": 1,
        "upsample_groups": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "hidden_act": "gelu",
        "max_position_embeddings": 128,
        "initializer_range": 0.02,
        "norm_eps": 1e-5,
        "use_cache": False,
        "use_streaming": False,
        "rope_theta": 10_000.0,
        "sliding_window": 8,
        "attention_dropout": 0.0,
        "layer_scale_initial_scale": 0.01,
        "attention_bias": False,
        "normalize": False,
    })


def _tiny_decoder_config() -> Qwen3TTSDecoderConfig:
    return Qwen3TTSDecoderConfig.from_dict({
        "latent_dim": 8,
        "codebook_dim": 8,
        "codebook_size": 8,
        "decoder_dim": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "max_position_embeddings": 128,
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_quantizers": 4,
        "num_semantic_quantizers": 1,
        "sliding_window": 8,
        "upsample_rates": [2, 2],
        "upsampling_ratios": [2],
        "vector_quantization_hidden_dimension": 8,
    })


class NativeQwen3TTSEncoderTests(unittest.TestCase):

    def test_published_encoder_namespace_is_checkpoint_exact(self):
        encoder = Qwen3TTSSpeechEncoder(
            Qwen3TTSEncoderConfig(),
            valid_num_quantizers=16,
            initialize=False,
        )
        state = encoder.state_dict()
        self.assertEqual(len(state), 225)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            56_234_304,
        )
        rows = [(f"encoder.{name}|F32|" + "x".join(str(item) for item in value.shape))
                for name, value in sorted(state.items())]
        self.assertEqual(
            hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest(),
            "5e8d7354d9c30a170d083126d8f73ed15e48395cc12814f1eb9cabe8a320e6e2",
        )
        self.assertEqual(
            state["encoder.layers.14.conv.weight"].shape,
            (512, 1024, 3),
        )
        self.assertEqual(
            state["downsample.conv.weight"].shape,
            (512, 512, 4),
        )
        self.assertEqual(
            state["quantizer.acoustic_residual_vector_quantizer."
                  "layers.30.codebook.embed_sum"].shape,
            (2048, 256),
        )

    def test_encoder_config_roundtrips_published_private_frame_rate(self):
        values = _tiny_encoder_config().to_dict()
        values["transformers_version"] = "4.57.3"
        restored = Qwen3TTSEncoderConfig.from_dict(values)
        self.assertEqual(restored.total_downsample, 8)
        self.assertEqual(restored.to_dict()["_frame_rate"], 30.0)
        self.assertEqual(
            restored.to_dict()["transformers_version"],
            "4.57.3",
        )

        tokenizer = Qwen3TTSTokenizerConfig.from_dict({
            "encoder_config": restored.to_dict(),
            "decoder_config": _tiny_decoder_config().to_dict(),
            "encoder_valid_num_quantizers": 4,
            "input_sample_rate": 240,
            "output_sample_rate": 24_000,
            "encode_downsample_rate": 8,
            "decode_upsample_rate": 8,
        })
        self.assertIsInstance(
            tokenizer.encoder_config,
            Qwen3TTSEncoderConfig,
        )

    def test_encode_trims_right_padded_batches_at_12hz_boundary(self):
        torch.manual_seed(7)
        encoder = Qwen3TTSSpeechEncoder(
            _tiny_encoder_config(),
            valid_num_quantizers=4,
        ).eval()
        audio = torch.randn(2, 17)
        mask = torch.ones(2, 17, dtype=torch.bool)
        mask[1, 10:] = False
        audio[1, 10:] = 0
        with torch.no_grad():
            batched = encoder(audio)
            codes = encoder.encode(audio, mask)
        self.assertEqual(batched.shape, (2, 4, 3))
        self.assertEqual(codes[0].shape, (3, 4))
        self.assertEqual(codes[1].shape, (2, 4))
        for code in codes:
            self.assertGreaterEqual(int(code.min()), 0)
            self.assertLess(int(code.max()), 8)

    def test_full_tokenizer_checkpoint_roundtrip_is_strict(self):
        torch.manual_seed(11)
        encoder_config = _tiny_encoder_config()
        decoder_config = _tiny_decoder_config()
        source_encoder = Qwen3TTSSpeechEncoder(
            encoder_config,
            valid_num_quantizers=4,
        )
        source_decoder = Qwen3TTSSpeechDecoder(decoder_config)
        with tempfile.TemporaryDirectory() as directory:
            path = export_qwen3_tts_speech_tokenizer(
                source_encoder,
                source_decoder,
                Path(directory) / "model.safetensors",
            )
            target_encoder = Qwen3TTSSpeechEncoder(
                encoder_config,
                valid_num_quantizers=4,
                initialize=False,
            )
            target_decoder = Qwen3TTSSpeechDecoder(
                decoder_config,
                initialize=False,
            )
            encoder_report = load_qwen3_tts_encoder_checkpoint(
                target_encoder,
                path,
                device="cpu",
                dtype=torch.float32,
                verify_official=False,
            )
            decoder_report = load_qwen3_tts_decoder_checkpoint(
                target_decoder,
                path,
                device="cpu",
                dtype=torch.float32,
                verify_official=False,
            )
        self.assertEqual(
            encoder_report.tensor_count,
            len(source_encoder.state_dict()),
        )
        self.assertEqual(
            decoder_report.tensor_count,
            len(source_decoder.state_dict()),
        )
        for name, expected in source_encoder.state_dict().items():
            torch.testing.assert_close(
                target_encoder.state_dict()[name],
                expected,
            )
        for name, expected in source_decoder.state_dict().items():
            torch.testing.assert_close(
                target_decoder.state_dict()[name],
                expected,
            )

    def test_encoder_loader_fails_closed_on_incomplete_namespace(self):
        source = Qwen3TTSSpeechEncoder(
            _tiny_encoder_config(),
            valid_num_quantizers=4,
        )
        state = {
            "encoder." + name: value
            for name, value in source.state_dict().items() if name != "downsample.conv.weight"
        }
        with tempfile.TemporaryDirectory() as directory:
            path = save_safetensors(
                state,
                Path(directory) / "incomplete.safetensors",
            )
            target = Qwen3TTSSpeechEncoder(
                _tiny_encoder_config(),
                valid_num_quantizers=4,
                initialize=False,
            )
            with self.assertRaises(CheckpointCompatibilityError):
                load_qwen3_tts_encoder_checkpoint(
                    target,
                    path,
                    device="cpu",
                    dtype=torch.float32,
                    verify_official=False,
                )


if __name__ == "__main__":
    unittest.main()
