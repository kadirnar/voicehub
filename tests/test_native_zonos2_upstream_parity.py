"""Regression tests for ZONOS2 behaviors verified against upstream Zyphra/ZONOS2.

Each case pins a difference found by the upstream parity audit (upstream
commit 194c0a3, ``TTSLLM`` offline path). No network or GPU is needed.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from voicehub.architectures.zonos2.configuration import Zonos2ArchitectureConfig
from voicehub.architectures.zonos2.modeling import Zonos2ForCausalLM
from voicehub.architectures.zonos2.prompting import build_zonos2_prompt
from voicehub.architectures.zonos2.runtime import (
    ZONOS2_DEFAULT_QUALITY_BUCKETS,
    NativeZonos2Runtime,
    resolve_zonos2_quality_buckets,
)
from voicehub.architectures.zonos2.sampling import (
    Zonos2SamplingOptions,
    apply_repetition_penalty,
    generate_zonos2_codes,
)
from voicehub.architectures.zonos2.speaker import (
    extract_zonos2_speaker_embedding,
    resample_zonos2_speaker_audio,
    zonos2_speaker_mel,
)


def _reference_hann_resample(waveform, source_rate, target_rate, width=6, rolloff=0.99):
    """Direct (non-polyphase) evaluation of torchaudio's sinc_interp_hann."""
    from math import ceil, gcd, pi

    divisor = gcd(source_rate, target_rate)
    original, target = source_rate // divisor, target_rate // divisor
    base = min(original, target) * rolloff
    source = waveform.double()
    output_length = ceil(target * source.numel() / original)
    samples = torch.arange(source.numel(), dtype=torch.float64)
    outputs = []
    for index in range(output_length):
        times = (samples / original - index / target) * base
        times = times.clamp(-width, width)
        window = torch.cos(times * pi / width / 2)**2
        radians = times * pi
        sinc = torch.where(radians == 0, torch.ones_like(radians), radians.sin() / radians)
        outputs.append((source * sinc * window).sum() * base / original)
    return torch.stack(outputs).float()


class Zonos2SpeakerResamplingParityTests(unittest.TestCase):

    def test_resampler_matches_upstream_hann_sinc_filter(self):
        generator = torch.Generator().manual_seed(3)
        waveform = torch.randn(1_601, generator=generator)
        for source_rate in (16_000, 22_050, 48_000):
            with self.subTest(source_rate=source_rate):
                expected = _reference_hann_resample(waveform, source_rate, 24_000)
                actual = resample_zonos2_speaker_audio(waveform, source_rate)
                self.assertEqual(actual.shape, expected.shape)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)

    def test_resampler_matches_torchaudio_transform_when_available(self):
        try:
            import torchaudio
        except ImportError:  # pragma: no cover - torchaudio is a test-only extra
            self.skipTest("torchaudio is not installed")
        waveform = torch.randn(9_000, generator=torch.Generator().manual_seed(5))
        for source_rate in (8_000, 16_000, 44_100, 48_000):
            with self.subTest(source_rate=source_rate):
                expected = torchaudio.transforms.Resample(source_rate, 24_000)(waveform)
                actual = resample_zonos2_speaker_audio(waveform, source_rate)
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)

    def test_speaker_embedding_uses_upstream_resampler(self):
        waveform = torch.randn(8_000, generator=torch.Generator().manual_seed(7)) * 0.1
        seen = []

        def encoder(features):
            seen.append(features)
            return features.mean(dim=1)

        encoder_module = torch.nn.Module()
        encoder_module.weight = torch.nn.Parameter(torch.zeros(1))
        encoder_module.forward = encoder
        extract_zonos2_speaker_embedding(
            encoder_module,
            {
                "array": waveform,
                "sampling_rate": 16_000
            },
        )
        expected = zonos2_speaker_mel(resample_zonos2_speaker_audio(waveform, 16_000))
        torch.testing.assert_close(seen[0], expected)


class Zonos2QualityConditioningParityTests(unittest.TestCase):

    def test_default_quality_conditioning_matches_upstream(self):
        config = Zonos2ArchitectureConfig()
        resolved = resolve_zonos2_quality_buckets(config, None)
        expected = [None] * len(config.quality_features)
        expected[config.quality_features.index("trailing_silence_s")] = 3
        self.assertEqual(ZONOS2_DEFAULT_QUALITY_BUCKETS, {"trailing_silence_s": 3})
        self.assertEqual(resolved, expected)
        prompt, _ = build_zonos2_prompt(
            config,
            "The quick brown fox jumps over the lazy dog.",
            quality_buckets=resolved,
        )
        # Upstream TTSLLM prompt for this text starts with the quality row
        # [pad]*9 + [511], followed by BOS (2).
        self.assertEqual(prompt[0, 0].tolist(), [config.audio_pad_id] * 9 + [511])
        self.assertEqual(prompt[0, 1, -1].item(), 2)
        self.assertEqual(prompt.shape[1], 64)

    def test_quality_mapping_and_explicit_disable(self):
        config = Zonos2ArchitectureConfig()
        self.assertEqual(
            resolve_zonos2_quality_buckets(config, {"lufs": 1}),
            [1, None, None, None, None, None],
        )
        self.assertIsNone(resolve_zonos2_quality_buckets(config, {}))
        self.assertIsNone(resolve_zonos2_quality_buckets(config, []))
        with self.assertRaisesRegex(ValueError, "Unknown ZONOS2 quality"):
            resolve_zonos2_quality_buckets(config, {"loudness": 1})

    def test_runtime_generate_applies_default_quality_tokens(self):
        config = Zonos2ArchitectureConfig()
        runtime = NativeZonos2Runtime.__new__(NativeZonos2Runtime)
        runtime.config = config
        runtime.device = torch.device("cpu")
        captured = {}

        def fake_build(_config, _text, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop after prompt construction")

        with patch("voicehub.architectures.zonos2.runtime.build_zonos2_prompt", side_effect=fake_build):
            with self.assertRaisesRegex(RuntimeError, "stop after prompt"):
                runtime.generate("hello", decode_audio=False)
        self.assertEqual(captured["quality_buckets"], [None, None, None, None, None, 3])


class Zonos2RepetitionPenaltyParityTests(unittest.TestCase):

    def test_end_of_audio_and_padding_history_are_not_penalized(self):
        logits = torch.full((1, 2, 6), 2.0)
        history = [torch.tensor([4, 5]), torch.tensor([1, 4])]
        options = Zonos2SamplingOptions(repetition_penalty=2.0, repetition_codebooks=-1)
        penalized = apply_repetition_penalty(logits, history, options, codebook_size=4)
        expected = logits.clone()
        expected[0, 0, 1] = 1.0  # only the real codec id 1 is penalized
        torch.testing.assert_close(penalized, expected)

    def test_penalty_at_or_below_one_is_disabled(self):
        logits = torch.randn(1, 2, 6)
        history = [torch.tensor([1, 2])]
        options = Zonos2SamplingOptions(repetition_penalty=0.5)
        self.assertIs(apply_repetition_penalty(logits, history, options), logits)

    def test_generation_passes_codebook_size_to_the_penalty(self):
        config = Zonos2ArchitectureConfig(
            n_layers=1,
            dim=16,
            head_dim=8,
            n_kv_heads=1,
            ffn_dim_multiplier=1.0,
            multiple_of=8,
            max_seqlen=64,
            n_codebooks=2,
            codebook_size=4,
            eoa_id=4,
            audio_pad_id=5,
            text_vocab=448,
            speaker_enabled=False,
            speaker_lda_dim=None,
            speaker_background_token_enabled=False,
            accurate_mode_token_enabled=False,
            speaking_rate_num_buckets=0,
            speaking_rate_buckets=(),
            quality_num_buckets=0,
            quality_features=(),
            quality_buckets={},
            quality_dropout={},
            moe_n_experts=1,
            special_topk_layers={},
        )
        model = Zonos2ForCausalLM(config).eval()
        prompt = torch.full((1, 3, config.frame_width), config.audio_pad_id)
        prompt[..., -1] = config.text_vocab
        with patch(
                "voicehub.architectures.zonos2.sampling.apply_repetition_penalty",
                side_effect=lambda logits, generated, options, codebook_size=None: logits,
        ) as penalty:
            generate_zonos2_codes(
                model,
                prompt,
                options=Zonos2SamplingOptions(max_new_tokens=2, seed=0),
            )
        self.assertEqual(penalty.call_args.kwargs["codebook_size"], 4)


if __name__ == "__main__":
    unittest.main()
