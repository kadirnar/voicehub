"""Regressions found by the edwko/OuteTTS upstream parity audit.

Reference values were produced with the author package (outetts 0.4.4 at
f5eac6e70d792844c6a6959d900a47af2c061a5b, pyloudnorm 0.2.0, NumPy
2.4.6).
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from voicehub.architectures.causal_lm.configuration import LlamaConfig
from voicehub.architectures.outetts.artifacts import resolve_outetts_artifacts
from voicehub.architectures.outetts.checkpoint import load_outetts_language_model
from voicehub.architectures.outetts.modeling import OuteTTSForCausalLM
from voicehub.architectures.outetts.postprocessing import (
    _biquad,
    _lfilter,
    chunk_text,
    integrated_loudness,
    normalize_loudness,
)
from voicehub.architectures.outetts.prompting import normalize_outetts_text
from voicehub.architectures.outetts.runtime import OuteTTSRuntime
from voicehub.models.outetts.inference import OuteTTSForTextToSpeech


def _tiny_llama3_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=True,
        rope_theta=500_000.0,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8,
            "rope_type": "llama3",
        },
    )


def _reference_sine(sample_rate: int = 24_000) -> torch.Tensor:
    time = torch.arange(2 * sample_rate, dtype=torch.float64) / sample_rate
    audio = (0.5 * torch.sin(2 * math.pi * 1000 * time)).to(torch.float32)
    audio[:sample_rate // 2] *= 0.05
    return audio


class OuteTTSCheckpointLoadingTests(unittest.TestCase):

    def _export(self, root: Path) -> OuteTTSForCausalLM:
        torch.manual_seed(3)
        original = OuteTTSForCausalLM(_tiny_llama3_config()).eval()
        original.save_pretrained(root)
        (root / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        return original

    def test_loaded_language_model_runs_with_materialized_rotary_frequencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self._export(root)
            artifacts = resolve_outetts_artifacts(root)
            restored, _ = load_outetts_language_model(artifacts, device="cpu", dtype=torch.float32)

        self.assertFalse([name for name, value in restored.named_buffers() if value.device.type == "meta"])
        input_ids = torch.tensor([[1, 5, 6, 7, 9, 11]], dtype=torch.long)
        with torch.no_grad():
            expected = original(input_ids).logits
            observed = restored.eval()(input_ids).logits
        self.assertTrue(torch.equal(observed, expected))

    def test_hugging_face_snapshot_symlinks_keep_safetensors_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            export = Path(temporary) / "export"
            export.mkdir()
            self._export(export)
            blobs = Path(temporary) / "blobs"
            snapshot = Path(temporary) / "snapshots" / "0123"
            blobs.mkdir()
            snapshot.mkdir(parents=True)
            for index, path in enumerate(sorted(export.iterdir())):
                blob = blobs / f"blob{index}"
                path.rename(blob)
                os.symlink(blob, snapshot / path.name)

            artifacts = resolve_outetts_artifacts(snapshot)
            self.assertEqual(artifacts.checkpoint.name, "model.safetensors")
            model, config = load_outetts_language_model(artifacts, device="cpu", dtype=torch.float32)

        self.assertEqual(config.model_type, "llama")
        self.assertIsInstance(model, OuteTTSForCausalLM)


class OuteTTSPostprocessingTests(unittest.TestCase):

    def test_biquad_filter_matches_direct_recursion(self):
        torch.manual_seed(0)
        signal = torch.randn(1_000, dtype=torch.float64)
        for numerator, denominator in (
                _biquad("high_shelf", 4.0, 1 / math.sqrt(2), 1500.0, 24_000),
                _biquad("high_pass", 0.0, 0.5, 38.0, 24_000),
        ):
            expected = []
            values = signal.tolist()
            for index, value in enumerate(values):
                output = numerator[0] * value
                if index >= 1:
                    output += numerator[1] * values[index - 1] - denominator[1] * expected[index - 1]
                if index >= 2:
                    output += numerator[2] * values[index - 2] - denominator[2] * expected[index - 2]
                expected.append(output)
            observed = _lfilter(signal, numerator, denominator, block=64)
            torch.testing.assert_close(
                observed, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=1e-10)

    def test_integrated_loudness_matches_pyloudnorm_reference(self):
        self.assertAlmostEqual(integrated_loudness(_reference_sine(), 24_000), -9.53387507039484, places=5)

    def test_normalization_matches_author_process_audio_tensor(self):
        normalized = normalize_loudness(_reference_sine()[None, None], 24_000)

        self.assertEqual(tuple(normalized.shape), (1, 1, 48_000))
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertAlmostEqual(float(normalized.abs().max()), 0.18865302205085754, places=6)
        self.assertAlmostEqual(float(normalized[0, 0, 12_345]), 0.13339783251285553, places=6)
        self.assertAlmostEqual(integrated_loudness(normalized, 24_000), -18.0, places=4)

    def test_short_audio_is_measured_padded_to_one_block(self):
        time = torch.arange(3_000, dtype=torch.float64) / 24_000
        audio = (0.1 * torch.sin(2 * math.pi * 440 * time)).to(torch.float32)

        normalized = normalize_loudness(audio[None, None], 24_000)

        self.assertEqual(tuple(normalized.shape), (1, 1, 3_000))
        self.assertAlmostEqual(float(normalized[0, 0, 1_000]), 0.29997020959854126, places=6)

    def test_silence_is_returned_unchanged(self):
        silence = torch.zeros(1, 1, 24_000)
        self.assertTrue(torch.equal(normalize_loudness(silence, 24_000), silence))

    def test_chunking_matches_author_chunk_text(self):
        repeated = "Hello there. How are you doing today? I am fine, thanks for asking. " * 3
        self.assertEqual(
            chunk_text(repeated),
            [
                "Hello there. How are you doing today? I am fine, thanks for asking. Hello there. How are you "
                "doing today? I am fine, thanks for asking. Hello there.",
                "How are you doing today? I am fine, thanks for asking.",
            ],
        )
        long_sentence = " ".join(["word"] * 65) + "."
        self.assertEqual(
            chunk_text(long_sentence),
            [" ".join(["word"] * 30), " ".join(["word"] * 30), "word word word word word."],
        )
        split = "Short one here. " + " ".join(f"w{i}" for i in range(29)) + ". Tail words end!"
        self.assertEqual(
            chunk_text(split),
            [
                "Short one here. " + " ".join(f"w{i}" for i in range(27)),
                "w27 w28. Tail words end!",
            ],
        )
        self.assertEqual(
            chunk_text("Hello. How are you? Fine."),
            ["Hello. How are you? Fine."],
        )


class _RecordingCodec(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls: list[torch.Tensor] = []

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(codes.clone())
        samples = codes.shape[-1] * 320
        time = torch.arange(samples, dtype=torch.float32) / 24_000
        return (0.3 * torch.sin(2 * math.pi * 300 * time))[None, None]


class OuteTTSChunkedDecodingTests(unittest.TestCase):

    def test_chunk_codes_are_decoded_once_and_loudness_normalized(self):
        runtime = OuteTTSRuntime.__new__(OuteTTSRuntime)
        nn.Module.__init__(runtime)
        runtime.codec = _RecordingCodec()
        chunk_codes = iter((([1, 2, 3], [4, 5, 6]), ([7, 8], [9, 10])))
        prompts = []

        def generate_one(text, **kwargs):
            prompts.append(text)
            return next(chunk_codes)

        runtime._generate_one = generate_one
        text = (
            "One two three four five six seven eight nine ten eleven twelve thirteen. "
            "Fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree "
            "twentyfour twentyfive twentysix twentyseven twentyeight twentynine thirty thirtyone thirtytwo.")

        audio = runtime.generate(
            text,
            speaker=None,
            generation_type="CHUNKED",
            max_length=8_192,
            sampler={},
            seed=0,
        )

        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts, chunk_text(text))
        self.assertEqual(len(runtime.codec.calls), 1)
        self.assertEqual(runtime.codec.calls[0].tolist(), [[[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]]])
        self.assertEqual(tuple(audio.shape), (1, 1, 5 * 320))
        padded = torch.nn.functional.pad(audio.reshape(-1), (0, 9_600 - audio.shape[-1]))
        self.assertAlmostEqual(integrated_loudness(padded, 24_000), -18.0, places=4)
        self.assertEqual(float(audio[0, 0, 0]), 0.0)


class OuteTTSDefaultsTests(unittest.TestCase):

    def test_text_normalization_joins_n_contractions_like_upstream(self):
        self.assertEqual(normalize_outetts_text("Rock' n roll is fine."), "Rock'n roll is fine.")
        self.assertEqual(normalize_outetts_text("I can' t go , ok ?"), "I can't go, ok?")

    def test_auto_dtype_runs_cpu_in_float32_like_author_auto_config(self):
        model = OuteTTSForTextToSpeech(device="cpu")
        sentinel = RuntimeError("stop after dtype resolution")
        with mock.patch(
                "voicehub.architectures.outetts.runtime.load_outetts_runtime",
                side_effect=sentinel,
        ) as loader:
            with self.assertRaises(RuntimeError):
                model._load_pretrained_model()
        self.assertIs(loader.call_args.kwargs["dtype"], torch.float32)

    def test_auto_dtype_uses_bfloat16_on_capable_cuda(self):
        model = OuteTTSForTextToSpeech(device="cuda")
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            self.assertIs(model._auto_dtype(torch), torch.bfloat16)
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            self.assertIs(model._auto_dtype(torch), torch.float16)


if __name__ == "__main__":
    unittest.main()
