"""Regression tests for CosyVoice 3 source parity (no network, CPU only).

Each test pins one behavior that the upstream-parity audit found to differ
from the source ``FunAudioLLM/CosyVoice`` inference path. Constants marked
"source" were captured from the source repository at revision
074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc.
"""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.nn import functional

from voicehub.architectures.cosyvoice_native import flow as flow_module
from voicehub.architectures.cosyvoice_native import vocoder as vocoder_module
from voicehub.architectures.cosyvoice_native.audio import _prompt_mel_filters, prompt_mel_features
from voicehub.architectures.cosyvoice_native.configuration import CosyVoiceArchitectureConfig
from voicehub.architectures.cosyvoice_native.flow import _apply_leading_rotary, fixed_flow_noise
from voicehub.architectures.cosyvoice_native.language_model import CosyVoiceLanguageModel, nucleus_keep_count
from voicehub.architectures.cosyvoice_native.modeling import CosyVoiceNativeModel, suppress_long_silences
from voicehub.architectures.cosyvoice_native.tokenization import (
    COSYVOICE3_SPECIAL_TOKENS,
    END_OF_PROMPT,
    END_OF_TEXT,
    IM_END,
    IM_START,
    CosyVoiceTextTokenizer,
)
from voicehub.architectures.cosyvoice_native.vocoder import CosyVoiceSourceNoise
from voicehub.models.cosyvoice_native.configuration_cosyvoice import CosyVoiceConfig
from voicehub.processing import resample_waveform_hann
from voicehub.tokenization import encode_gpt2_token


def _source_nucleus(weighted_scores, top_p=0.8, top_k=25):
    """Transcription of the source ``cosyvoice.utils.common.nucleus_sampling``."""
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    return indices[prob.multinomial(1, replacement=True)].item()


def _source_ras(weighted_scores, decoded_tokens, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    """Transcription of the source ``ras_sampling``/``random_sampling``."""
    top_ids = _source_nucleus(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        weighted_scores[top_ids] = -float("inf")
        top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item()
    return top_ids


def _write_tokenizer(directory: Path, *, extra_vocab: dict[bytes, int] | None = None):
    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    for raw, token_id in (extra_vocab or {}).items():
        vocabulary[encode_gpt2_token(raw)] = token_id
    vocabulary[encode_gpt2_token(b"hi")] = max(vocabulary.values()) + 1
    merges = "h i\n"
    published = (END_OF_TEXT, IM_START, IM_END)
    first_special = max(vocabulary.values()) + 1
    (directory / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n" + merges, encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(
        json.dumps({
            "add_prefix_space": False,
            "added_tokens_decoder": {
                str(token_id): {
                    "content": spelling,
                    "special": True
                }
                for token_id, spelling in enumerate(published, first_special)
            },
        }),
        encoding="utf-8",
    )
    return CosyVoiceTextTokenizer.from_files(
        directory / "vocab.json",
        directory / "merges.txt",
        directory / "tokenizer_config.json",
        validate_published_ids=False,
        register_source_special_tokens=True,
    ), first_special


class CosyVoiceTokenizerParityTests(unittest.TestCase):

    def test_source_special_tokens_are_registered_after_the_asset_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer, first_special = _write_tokenizer(Path(temporary))
        # The published BlankEN assets end at <|im_end|>; the source
        # tokenizer adds <|endofprompt|>, [breath], ... in list order.
        new_tokens = [token for token in COSYVOICE3_SPECIAL_TOKENS if token not in (IM_START, IM_END)]
        self.assertEqual(new_tokens[0], END_OF_PROMPT)
        self.assertEqual(len(COSYVOICE3_SPECIAL_TOKENS), 280)
        for offset, spelling in enumerate(new_tokens):
            self.assertEqual(tokenizer.encode(spelling).input_ids, (first_special + 3 + offset, ))
        self.assertEqual(
            tokenizer.encode("a[breath]b").input_ids,
            (ord("a"), first_special + 3 + new_tokens.index("[breath]"), ord("b")),
        )

    def test_ordinary_bracket_vocabulary_entries_are_not_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            # "[x" exists in the vocabulary, but without a merge rule BPE
            # must not produce it; it is not a special token.
            tokenizer, _ = _write_tokenizer(Path(temporary), extra_vocab={b"[x": 256})
        self.assertEqual(tokenizer.encode("[x").input_ids, (ord("["), ord("x")))


class CosyVoiceSamplingParityTests(unittest.TestCase):

    def test_top_p_uses_the_full_distribution_not_the_top_k_renormalized_one(self):
        ordered = torch.tensor([0.5, 0.2, 0.1, 0.1, 0.1])
        # Source: running sums 0, .5, .7 are all below .8 -> three kept.
        # (Top-k-renormalized top-p would keep only two.)
        self.assertEqual(nucleus_keep_count(ordered[:3], top_p=0.8, top_k=3), 3)
        # float32(0.5 + 0.2 + 0.1) rounds above 0.8, so the fourth stays out.
        self.assertEqual(nucleus_keep_count(ordered, top_p=0.8, top_k=25), 3)
        self.assertEqual(nucleus_keep_count(ordered, top_p=0.81, top_k=25), 4)
        self.assertEqual(nucleus_keep_count(ordered, top_p=0.8, top_k=2), 2)
        self.assertEqual(nucleus_keep_count(torch.tensor([0.9, 0.1]), top_p=0.8, top_k=25), 1)

    def test_repetition_aware_sampling_matches_the_source_draw_for_draw(self):
        model = CosyVoiceLanguageModel(CosyVoiceArchitectureConfig.tiny().language)
        scores_generator = torch.Generator().manual_seed(3)
        steps = [torch.randn(40, generator=scores_generator) * 3 for _ in range(200)]
        steps = [torch.cat((step[:20] + 4, step[20:])).log_softmax(dim=0) for step in steps]

        torch.manual_seed(1234)
        expected: list[int] = []
        for scores in steps:
            expected.append(_source_ras(scores.clone(), expected))

        generator = torch.Generator().manual_seed(1234)
        actual: list[int] = []
        for scores in steps:
            actual.append(
                model._sample_token(
                    scores.clone(),
                    actual,
                    top_k=25,
                    top_p=0.8,
                    repetition_window=10,
                    repetition_threshold=0.1,
                    generator=generator,
                ))
        self.assertEqual(actual, expected)
        self.assertGreater(len(set(actual)), 5)

    def test_repeated_token_triggers_a_resample_without_it(self):
        model = CosyVoiceLanguageModel(CosyVoiceArchitectureConfig.tiny().language)
        scores = torch.full((40, ), -30.0)
        scores[7] = 0.0
        scores[9] = -1.0
        scores = scores.log_softmax(dim=0)
        token = model._sample_token(
            scores,
            [7],
            top_k=25,
            top_p=0.8,
            repetition_window=10,
            repetition_threshold=0.1,
            generator=torch.Generator().manual_seed(0),
        )
        self.assertNotEqual(token, 7)
        self.assertEqual(token, 9)

    def test_default_length_cap_is_twenty_tokens_per_text_token(self):
        torch.manual_seed(0)
        model = CosyVoiceLanguageModel(CosyVoiceArchitectureConfig.tiny().language).eval()
        tokens = model.generate(
            torch.tensor([[3, 4]]),
            min_new_tokens=40,
            generator=torch.Generator().manual_seed(0),
        )
        self.assertEqual(tokens.shape, (1, 40))
        self.assertIsNone(CosyVoiceConfig().generation_config["max_new_tokens"])

    def test_long_silences_are_capped_at_five_tokens(self):
        tokens = torch.tensor([[7, 1, 2, 28, 29, 55, 248, 494, 8, 1, 1]])
        self.assertEqual(
            suppress_long_silences(tokens).tolist(),
            [[7, 1, 2, 28, 29, 55, 8, 1, 1]],
        )


class CosyVoiceFlowParityTests(unittest.TestCase):

    def test_fixed_noise_is_the_source_seed_zero_table(self):
        noise = fixed_flow_noise(80, 3)
        # source CausalConditionalCFM.rand_noise[0, :2, :3]
        expected = torch.tensor([
            [-1.1258398294448853, -1.152360200881958, -0.2505785822868347],
            [0.869179368019104, 0.5761968493461609, -0.41195306181907654],
        ])
        self.assertTrue(torch.equal(noise[0, :2], expected))

    def test_only_the_first_head_is_rotated(self):
        torch.manual_seed(0)
        head_dim, heads, length = 8, 3, 5
        values = torch.randn(1, length, heads * head_dim)
        inv_freq = 1.0 / (10_000.0**(torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(length).float(), inv_freq)
        rotated = _apply_leading_rotary(values, (angles.cos(), angles.sin()))
        self.assertTrue(torch.equal(rotated[..., head_dim:], values[..., head_dim:]))
        # x-transformers apply_rotary_pos_emb on the leading channels.
        freqs = torch.stack((angles, angles), dim=-1).flatten(-2)
        pairs = values[..., :head_dim].unflatten(-1, (-1, 2))
        rotate_half = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        expected = values[..., :head_dim] * freqs.cos() + rotate_half * freqs.sin()
        torch.testing.assert_close(rotated[..., :head_dim], expected, rtol=0, atol=1e-6)

    def test_prompt_tokens_condition_the_flow_and_are_removed_from_the_output(self):
        torch.manual_seed(0)
        model = CosyVoiceNativeModel(CosyVoiceArchitectureConfig.tiny()).eval()
        config = model.config.flow
        tokens = torch.tensor([[1, 2, 3, 4]])
        prompt = torch.tensor([[5, 6, 7]])
        features = torch.randn(1, prompt.shape[1] * config.token_mel_ratio, config.mel_channels)
        speaker = torch.randn(1, config.speaker_embedding_dim)
        with torch.inference_mode():
            plain = model.flow.generate(tokens, torch.tensor([4]), speaker, steps=2)
            prompted = model.flow.generate(
                tokens,
                torch.tensor([4]),
                speaker,
                prompt_speech_tokens=prompt,
                prompt_features=features,
                steps=2,
            )
            again = model.flow.generate(
                tokens,
                torch.tensor([4]),
                speaker,
                prompt_speech_tokens=prompt,
                prompt_features=features,
                steps=2,
            )
        self.assertEqual(prompted.shape, (1, config.mel_channels, 4 * config.token_mel_ratio))
        self.assertEqual(plain.shape, prompted.shape)
        self.assertFalse(torch.allclose(plain, prompted))
        self.assertTrue(torch.equal(prompted, again))
        with self.assertRaisesRegex(ValueError, "prompt_speech_tokens"):
            model.flow.generate(tokens, torch.tensor([4]), speaker, prompt_features=features)
        with self.assertRaisesRegex(ValueError, "Prompt mel frames"):
            model.flow.generate(
                tokens,
                torch.tensor([4]),
                speaker,
                prompt_speech_tokens=prompt[:, :2],
                prompt_features=features,
            )


class CosyVoiceVocoderParityTests(unittest.TestCase):

    def test_source_noise_replays_the_source_fixed_tables(self):
        noise = CosyVoiceSourceNoise(
            harmonics=8,
            mel_channels=80,
            speech_vocab_size=6_561,
            speaker_embedding_dim=192,
            f0_hidden_size=512,
        )
        # source CausalHiFTGenerator.m_source.l_sin_gen.rand_ini / sine_waves
        expected_phase = torch.tensor([[
            0.0, 0.17098891735076904, 0.1653319001197815, 0.14378005266189575, 0.7629473805427551,
            0.6589300632476807, 0.027519047260284424, 0.43482542037963867, 0.05383557081222534
        ]])
        self.assertTrue(torch.equal(noise.phase_offsets, expected_phase))
        table = noise.noise(123_457, device=torch.device("cpu"))
        self.assertEqual(table.shape, (1, 123_457, 9))
        self.assertTrue(
            torch.equal(
                table[0, 0],
                torch.tensor([
                    0.9228259921073914, 0.477447509765625, 0.8270518183708191, 0.2834571599960327,
                    0.7934815883636475, 0.1734347939491272, 0.8523750305175781, 0.9228546023368835,
                    0.7310199737548828
                ]),
            ))
        self.assertTrue(
            torch.equal(
                table[0, 123_456],
                torch.tensor([
                    0.27953243255615234, 0.47952115535736084, 0.6389868855476379, 0.21858936548233032,
                    0.18366748094558716, 0.3289433717727661, 0.8943942189216614, 0.4206131100654602,
                    0.15828430652618408
                ]),
            ))
        longer = noise.noise(1_000_000, device=torch.device("cpu"))
        self.assertTrue(torch.equal(longer[:, :123_457], table))

    def test_inference_is_deterministic_and_uses_source_numerics(self):
        torch.manual_seed(0)
        model = CosyVoiceNativeModel(CosyVoiceArchitectureConfig.tiny()).eval()
        mel = torch.randn(1, model.config.hift.mel_channels, 6)
        predictor_dtypes = []
        handle = model.hift.f0_predictor.condnet[0].register_forward_pre_hook(
            lambda module, inputs: predictor_dtypes.append(inputs[0].dtype))
        slopes = []
        original = functional.leaky_relu

        def recording_leaky_relu(values, negative_slope=0.01, inplace=False):
            slopes.append(negative_slope)
            return original(values, negative_slope, inplace)

        with mock.patch.object(vocoder_module.functional, "leaky_relu", recording_leaky_relu):
            with torch.inference_mode():
                first, _ = model.hift(mel)
                second, _ = model.hift(mel)
        handle.remove()
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(predictor_dtypes, [torch.float64, torch.float64])
        stages = len(model.config.hift.upsample_rates)
        # Each pass: one 0.1 slope per upsample stage, then the default 0.01.
        self.assertEqual(slopes, ([0.1] * stages + [0.01]) * 2)

    def test_istft_clips_the_linear_magnitude(self):
        model = CosyVoiceNativeModel(CosyVoiceArchitectureConfig.tiny())
        bins = model.config.hift.istft_n_fft // 2 + 1
        values = torch.zeros(1, 2 * bins, 6)
        values[:, :bins] = 5.0  # exp(5) = 148.4 > 100
        values[:, bins:] = 0.3
        actual = model.hift._istft(values)
        magnitude = torch.full((1, bins, 6), 100.0)
        phase = torch.full((1, bins, 6), 0.3).sin()
        expected = torch.istft(
            torch.complex(magnitude * phase.cos(), magnitude * phase.sin()),
            model.config.hift.istft_n_fft,
            model.config.hift.istft_hop_length,
            model.config.hift.istft_n_fft,
            window=model.hift.stft_window,
        )
        self.assertTrue(torch.equal(actual, expected))


@unittest.skipUnless(importlib.util.find_spec("torchaudio"), "torchaudio reference unavailable")
class CosyVoicePromptFrontendParityTests(unittest.TestCase):

    def test_resampling_matches_torchaudio_bit_for_bit(self):
        import torchaudio

        torch.manual_seed(0)
        waveform = torch.rand(16_003) * 1.6 - 0.8
        for source_rate, target_rate in ((16_000, 24_000), (16_000, 16_000), (44_100, 16_000), (22_050,
                                                                                                24_000)):
            expected = torchaudio.transforms.Resample(source_rate, target_rate)(waveform[None])[0]
            actual = resample_waveform_hann(waveform, source_rate, target_rate, match="transform")
            self.assertTrue(torch.equal(actual, expected), (source_rate, target_rate))

    @unittest.skipUnless(importlib.util.find_spec("librosa"), "librosa reference unavailable")
    def test_prompt_mel_matches_the_matcha_recipe(self):
        from librosa.filters import mel as librosa_mel

        basis = torch.from_numpy(librosa_mel(sr=24_000, n_fft=1_920, n_mels=80, fmin=0, fmax=None))
        torch.testing.assert_close(_prompt_mel_filters(), basis, rtol=0, atol=1e-12)
        torch.manual_seed(1)
        waveform = torch.rand(24_000) * 1.2 - 0.6
        padded = functional.pad(waveform[None, None], (720, 720), mode="reflect")[:, 0]
        spectrum = torch.view_as_real(
            torch.stft(
                padded,
                1_920,
                hop_length=480,
                win_length=1_920,
                window=torch.hann_window(1_920),
                center=False,
                return_complex=True,
            ))
        magnitude = torch.sqrt(spectrum.pow(2).sum(-1) + 1e-9)
        expected = torch.log(torch.clamp(torch.matmul(basis, magnitude), min=1e-5))[0].T
        actual = prompt_mel_features(waveform)
        self.assertEqual(actual.shape, (50, 80))
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
        self.assertTrue(math.isfinite(float(actual.sum())))


if __name__ == "__main__":
    unittest.main()
