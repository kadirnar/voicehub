"""Regression tests for Fish S2 behaviour that must match the source recipe.

Each test pins one difference found by comparing VoiceHub with the
original ``fish_speech`` inference path on the published S2-Pro checkpoint.
"""

from __future__ import annotations

import math
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from voicehub.architectures.fishtts import sampling
from voicehub.architectures.fishtts.checkpoint import convert_legacy_fish_codec
from voicehub.architectures.fishtts.codec import FishModifiedDAC, _codec_rotary
from voicehub.architectures.fishtts.configuration import FishS2Config
from voicehub.architectures.fishtts.modeling import (
    FishAttention,
    FishQKRMSNorm,
    FishS2ForConditionalGeneration,
    _rotary_frequencies,
)
from voicehub.architectures.fishtts.prompting import build_fish_prompt, group_speaker_turns
from voicehub.architectures.fishtts.runtime import FishS2Runtime
from voicehub.architectures.fishtts.sampling import generate_fish_codes, logits_to_probabilities
from voicehub.architectures.fishtts.tokenization import FishTokenizer
from voicehub.checkpointing import SafeTensorReader
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.processing.waveform import resample_waveform_hann

from tests.test_native_fishtts import _tiny_codec_config, _tokenizer_test_config, _write_test_tokenizer


def _source_rotary_table(length: int, dimension: int, base: float) -> torch.Tensor:
    """``fish_speech``'s ``precompute_freqs_cis`` (bfloat16 by default)."""
    frequencies = 1.0 / (base**(torch.arange(0, dimension, 2)[:(dimension // 2)].float() / dimension))
    angles = torch.outer(torch.arange(length), frequencies)
    table = torch.polar(torch.ones_like(angles), angles)
    return torch.stack([table.real, table.imag], dim=-1).to(dtype=torch.bfloat16)


def _source_apply_rotary(values: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    shaped = values.float().reshape(*values.shape[:-1], -1, 2)
    table = table.view(1, shaped.size(1), 1, shaped.size(3), 2)
    output = torch.stack(
        [
            shaped[..., 0] * table[..., 0] - shaped[..., 1] * table[..., 1],
            shaped[..., 1] * table[..., 0] + shaped[..., 0] * table[..., 1],
        ],
        -1,
    )
    return output.flatten(3).type_as(values)


def _tokenizer(directory: str) -> FishTokenizer:
    config = _tokenizer_test_config()
    path = Path(directory) / "tokenizer.json"
    _write_test_tokenizer(path, config)
    return FishTokenizer.from_tokenizer_json(path, config=config)


class _RecordingCodec(nn.Module):
    sample_rate = 44_100

    def __init__(self, num_codebooks: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.num_codebooks = num_codebooks
        self.seen: torch.Tensor | None = None

    def encode(self, waveform, audio_lengths):
        del audio_lengths
        self.seen = waveform.detach().clone()
        return (
            torch.zeros((1, self.num_codebooks, 2), dtype=torch.long),
            torch.tensor([2]),
        )

    def from_indices(self, indices):
        return torch.zeros((1, 1, indices.shape[-1]))


class FishSourceParityTests(unittest.TestCase):

    def test_reference_system_text_is_tokenized_as_separate_parts(self):
        # The source encodes each TextPart on its own; merging the
        # reference transcript with "\n\nSpeech:\n" changed BPE tokens
        # ("." + "\n\n" -> ".\n\n") on the official tokenizer.
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = _tokenizer(directory)
            original = FishTokenizer.encode
            with patch.object(
                    FishTokenizer,
                    "encode",
                    autospec=True,
                    side_effect=original,
            ) as encode:
                build_fish_prompt(
                    "hello",
                    tokenizer,
                    reference_text="sample.",
                    reference_codes=torch.tensor([[1, 2], [2, 3], [3, 1]]),
                )
            texts = [call.args[1] for call in encode.call_args_list]
        self.assertIn(
            "convert the provided text to speech reference to the following:\n\nText:\n",
            texts,
        )
        self.assertIn("<|speaker:0|>sample.", texts)
        self.assertIn("\n\nSpeech:\n", texts)
        self.assertFalse(any("sample.\n\nSpeech" in text for text in texts))

    def test_chunk_byte_budget_ignores_join_separators(self):
        self.assertEqual(
            group_speaker_turns(("ab", "cd"), maximum_utf8_bytes=4),
            ("ab\ncd", ),
        )
        self.assertEqual(
            group_speaker_turns(("ab", "cd", "e"), maximum_utf8_bytes=4),
            ("ab\ncd", "e"),
        )

    def test_legacy_codec_conversion_drops_source_derived_buffers(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            codec = FishModifiedDAC(_tiny_codec_config())
        state = {"generator." + name: value for name, value in codec.state_dict().items()}
        # The official codec.pth carries these non-persistent transformer
        # buffers; the source loads it with strict=False.
        state["generator.quantizer.pre_module.freqs_cis"] = torch.zeros(4, 2, 2)
        state["generator.encoder.block.4.block.5.causal_mask"] = torch.ones(4, 4, dtype=torch.bool)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"state_dict": state}, root / "codec.pth")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                converted = convert_legacy_fish_codec(
                    root / "codec.pth",
                    root / "converted",
                    trust_legacy_pickle=True,
                    verify_official_integrity=False,
                    config=codec.config,
                )
            with SafeTensorReader(converted) as stored:
                names = set(stored.keys())
            self.assertEqual(names, set(codec.state_dict()))

            state["generator.quantizer.unreviewed_weight"] = torch.zeros(1)
            torch.save({"state_dict": state}, root / "unknown.pth")
            with self.assertRaisesRegex(CheckpointCompatibilityError, "unreviewed_weight"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    convert_legacy_fish_codec(
                        root / "unknown.pth",
                        root / "rejected",
                        trust_legacy_pickle=True,
                        verify_official_integrity=False,
                        config=codec.config,
                    )

    def test_reference_audio_uses_torchaudio_functional_resampler(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = _tokenizer(directory)
            codec = _RecordingCodec(_tokenizer_test_config().num_codebooks)
            runtime = FishS2Runtime(
                semantic_model=FishS2ForConditionalGeneration(_tokenizer_test_config()),
                tokenizer=tokenizer,
                codec=codec,
            )
            waveform = torch.randn(1_600, generator=torch.Generator().manual_seed(3)) * 0.1
            runtime.encode_reference({"array": waveform.numpy(), "sampling_rate": 16_000})
        expected = resample_waveform_hann(waveform, 16_000, 44_100)
        torch.testing.assert_close(codec.seen.view(-1), expected, rtol=0, atol=0)

    def test_temperature_divides_in_the_logits_dtype(self):
        logits = torch.randn(64, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16) * 4
        actual = logits_to_probabilities(logits, temperature=0.7, top_p=1.0, top_k=64)
        expected = torch.softmax(logits / torch.tensor(0.7, dtype=torch.bfloat16), dim=-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_exponential_race_keeps_the_only_surviving_token(self):
        probabilities = torch.zeros(8, dtype=torch.bfloat16)
        probabilities[3] = 1
        with patch.object(sampling.torch, "rand_like", side_effect=torch.zeros_like):
            self.assertEqual(int(sampling.sample_exponential_race(probabilities)), 3)
        uniform = torch.rand(8, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16) + 0.01
        mixed = torch.softmax(torch.randn(8, generator=torch.Generator().manual_seed(6)), -1).to(torch.bfloat16)
        with patch.object(sampling.torch, "rand_like", return_value=uniform):
            self.assertEqual(
                int(sampling.sample_exponential_race(mixed)),
                int(torch.argmax(mixed / -torch.log(uniform))),
            )

    def _tiny_generation(self, steps: int):
        torch.manual_seed(0)
        config = FishS2Config.tiny()
        model = FishS2ForConditionalGeneration(config).eval()
        prompt = torch.zeros(config.num_codebooks + 1, 3, dtype=torch.long)
        prompt[0] = torch.tensor([4, 5, 6])
        tokens = iter(range(config.semantic_begin_id, config.semantic_end_id + 1))
        windows = []

        def main_token(logits, *, previous_tokens, **unused):
            del logits, unused
            windows.append(previous_tokens.clone())
            return torch.tensor(next(tokens))

        def codebooks(slow_hidden, semantic_token, *, model, **unused):
            del slow_hidden, semantic_token, unused
            return torch.zeros(model.config.num_codebooks, dtype=torch.long)

        return model, prompt, steps, windows, main_token, codebooks

    def test_repetition_window_never_contains_the_prefill_token(self):
        model, prompt, steps, windows, main_token, codebooks = self._tiny_generation(3)
        with patch.object(sampling, "_sample_main_token", side_effect=main_token), \
                patch.object(sampling, "_sample_codebooks", side_effect=codebooks):
            generate_fish_codes(model, prompt, max_new_tokens=steps)
        first = model.config.semantic_begin_id
        self.assertTrue(windows[0].eq(0).all())
        self.assertTrue(windows[1].eq(0).all())
        self.assertEqual(windows[2].tolist(), [0] * (sampling.REPETITION_WINDOW - 1) + [first + 1])

    def test_decode_steps_after_prefill_use_math_attention(self):
        model, prompt, steps, _, main_token, codebooks = self._tiny_generation(3)
        backends = []
        original = model.forward_generate

        def forward_generate(*args, **kwargs):
            backends.append((
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(),
            ))
            return original(*args, **kwargs)

        with patch.object(model, "forward_generate", side_effect=forward_generate), \
                patch.object(sampling, "_sample_main_token", side_effect=main_token), \
                patch.object(sampling, "_sample_codebooks", side_effect=codebooks):
            generate_fish_codes(model, prompt, max_new_tokens=steps)
        self.assertEqual(backends[0], (True, True, True))
        self.assertEqual(backends[1:], [(False, False, True)] * (steps - 1))

    def test_prefill_normalizes_and_projects_only_the_last_position(self):
        torch.manual_seed(1)
        config = FishS2Config.tiny()
        model = FishS2ForConditionalGeneration(config).eval()
        prompt = torch.zeros(1, config.num_codebooks + 1, 5, dtype=torch.long)
        prompt[0, 0] = torch.tensor([4, 5, 6, 7, 8])
        lengths = []
        model.norm.register_forward_hook(lambda module, inputs, output: lengths.append(inputs[0].shape[1]))
        with torch.no_grad():
            model.setup_caches(max_batch_size=1, max_seq_len=config.text.max_position_embeddings)
            output = model.forward_generate(prompt, torch.arange(5))
        self.assertEqual(lengths, [1])
        self.assertEqual(tuple(output.logits.shape[:2]), (1, 1))

    def test_query_key_norm_matches_torch_rms_norm(self):
        values = (torch.randn(2, 3, 4, 16, generator=torch.Generator().manual_seed(7)) * 3).to(torch.bfloat16)
        norm = FishQKRMSNorm(16, epsilon=1e-6).to(torch.bfloat16)
        reference = nn.RMSNorm(16, eps=1e-6).to(torch.bfloat16)
        with torch.no_grad():
            weight = (1 + 0.1 * torch.randn(16, generator=torch.Generator().manual_seed(8))).to(torch.bfloat16)
            norm.weight.copy_(weight)
            reference.weight.copy_(weight)
            torch.testing.assert_close(norm(values), reference(values), rtol=0, atol=0)

    def test_fast_attention_matches_source_eq_scaled_dot_product(self):
        config = FishS2Config.tiny().audio_decoder
        attention = FishAttention(config, manual_attention=True)
        generator = torch.Generator().manual_seed(9)
        query, key, value = (torch.randn(1, 4, 3, 8, generator=generator).to(torch.bfloat16) for _ in range(3))
        mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))[None, None]
        bias = torch.zeros(1, 1, 3, 3, dtype=query.dtype)
        bias = torch.where(mask.logical_not(), float("-inf"), bias)
        weights = query @ key.transpose(-2, -1) * (1 / math.sqrt(query.size(-1)))
        weights += bias
        expected = torch.softmax(weights, dim=-1) @ value
        torch.testing.assert_close(
            attention._manual_attention(query, key, value, mask),
            expected,
            rtol=0,
            atol=0,
        )

    def test_rotary_tables_are_the_source_bfloat16_tables(self):
        positions = torch.tensor([0, 1, 7, 33, 63])
        cosine, sine = _rotary_frequencies(positions, 16, base=1_000_000.0, length=64)
        table = _source_rotary_table(64, 16, 1_000_000)[positions]
        torch.testing.assert_close(cosine, table[..., 0], rtol=0, atol=0)
        torch.testing.assert_close(sine, table[..., 1], rtol=0, atol=0)

        values = torch.randn(1, 5, 2, 16, generator=torch.Generator().manual_seed(4))
        sequence = torch.arange(5)
        torch.testing.assert_close(
            _codec_rotary(values, sequence, base=10_000.0),
            _source_apply_rotary(values, _source_rotary_table(5, 16, 10_000)),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
