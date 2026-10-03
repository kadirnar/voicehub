"""Regression tests for Qwen3-TTS numerics that must follow upstream exactly.

The official runtime decodes through Transformers with SDPA attention.
Its bf16 results depend on which SDPA kernel runs, the GEMM shapes,
where reductions happen, and where RoPE frequencies are computed. Each
test pins one of those choices so autoregressive generation stays bit-
identical to the upstream recipe on real checkpoints.
"""

from __future__ import annotations

import hashlib
import unittest
from unittest import mock

import torch

from tests.test_native_qwen3_tts import _tiny_architecture, _tiny_decoder
from voicehub.architectures.qwen3_tts import codec as qwen3_tts_codec
from voicehub.architectures.qwen3_tts import modeling as qwen3_tts_modeling
from voicehub.architectures.qwen3_tts.codec import Qwen3TTSSpeechDecoder, materialize_qwen3_tts_decoder_buffers
from voicehub.architectures.qwen3_tts.modeling import Qwen3TTSForConditionalGeneration, materialize_qwen3_tts_buffers


def _transformers_inverse_frequency(base: float, dimension: int) -> torch.Tensor:
    # Transformers' default RoPE init, evaluated on the CPU at construction.
    return 1.0 / (base**(torch.arange(0, dimension, 2, dtype=torch.int64).float() / dimension))


class _Recorder:

    def __init__(self, function):
        self.function = function
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.function(*args, **kwargs)


class Qwen3TTSUpstreamNumericsTests(unittest.TestCase):

    def _greedy_talker_run(self, *, tts_pad_embed=None, dtype=torch.float32):
        torch.manual_seed(21)
        talker = Qwen3TTSForConditionalGeneration(_tiny_architecture()).eval().talker.to(dtype)
        backbone = talker.model
        recorder = _Recorder(backbone.forward_with_cache)
        heads = []
        original_head = talker.codec_head.forward

        def head(hidden_states):
            heads.append(tuple(hidden_states.shape))
            return original_head(hidden_states)

        sampled = []
        original_sample = qwen3_tts_modeling._sample_token

        def sample(logits, **kwargs):
            sampled.append(logits.dtype)
            return original_sample(logits, **kwargs)

        with (
                mock.patch.object(backbone, "forward_with_cache", recorder),
                mock.patch.object(talker.codec_head, "forward", head),
                mock.patch.object(qwen3_tts_modeling, "_sample_token", sample),
                torch.no_grad(),
        ):
            codes = talker.generate_codes(
                prompt_embeds=torch.randn(1, 3, 8, dtype=dtype),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
                trailing_text_hidden=torch.randn(1, 1, 8, dtype=dtype),
                max_new_tokens=4,
                do_sample=False,
                subtalker_dosample=False,
                repetition_penalty=1.05,
                tts_pad_embed=tts_pad_embed,
            )
        return talker, codes, recorder.calls, heads, sampled

    def test_all_ones_mask_is_dropped_like_transformers_sdpa(self):
        _, _, calls, _, _ = self._greedy_talker_run()
        self.assertTrue(calls)
        for _, kwargs in calls:
            self.assertIsNone(kwargs.get("attention_mask"))

    def test_prefill_logits_project_every_position_and_process_in_float32(self):
        _, _, _, heads, sampled = self._greedy_talker_run(dtype=torch.bfloat16)
        # Prefill: the whole [batch, time, hidden] block, as upstream does.
        self.assertEqual(heads[0], (1, 3, 8))
        self.assertTrue(all(shape == (1, 8) for shape in heads[1:]))
        # Talker logits processors run on a float32 copy (Transformers).
        self.assertEqual(sampled[0], torch.float32)

    def test_codec_feedback_is_one_reduction_plus_given_pad_embedding(self):
        pad = torch.randn(1, 1, 8, dtype=torch.bfloat16)
        talker, codes, calls, _, _ = self._greedy_talker_run(
            tts_pad_embed=pad,
            dtype=torch.bfloat16,
        )
        self.assertGreaterEqual(codes.shape[0], 2)
        # Step 0 consumed the single trailing text position; step 1 must use
        # the caller's batched TTS pad embedding.
        step_input = calls[2][0][0][:, 0]
        frame = codes[1]
        terms = [talker.get_input_embeddings()(frame[:1])]
        terms.extend(
            table(frame[index:index + 1])
            for index, table in enumerate(talker.code_predictor.get_input_embeddings(), start=1))
        expected = torch.stack(terms, dim=1).sum(dim=1) + pad.reshape(1, -1)
        self.assertTrue(torch.equal(step_input, expected))

    def test_single_query_attention_uses_unmasked_sdpa(self):
        query = torch.randn(1, 4, 1, 8)
        key = torch.randn(1, 2, 5, 8)
        value = torch.randn(1, 2, 5, 8)
        recorder = _Recorder(torch.nn.functional.scaled_dot_product_attention)
        with mock.patch.object(qwen3_tts_modeling.functional, "scaled_dot_product_attention", recorder):
            output = qwen3_tts_modeling._scaled_dot_product_attention(
                query,
                key,
                value,
                attention_bias=None,
                scale=8**-0.5,
                dropout_p=0.0,
                groups=2,
                is_causal=True,
            )
        self.assertEqual(len(recorder.calls), 1)
        kwargs = recorder.calls[0][1]
        self.assertIsNone(kwargs["attn_mask"])
        self.assertFalse(kwargs["is_causal"])
        expected = torch.softmax(
            query @ qwen3_tts_modeling._expand_kv(key, 2).transpose(-1, -2) * 8**-0.5,
            dim=-1,
        ) @ qwen3_tts_modeling._expand_kv(value, 2)
        torch.testing.assert_close(output, expected)

    def test_speech_decoder_attention_uses_boolean_sliding_mask_sdpa(self):
        torch.manual_seed(5)
        decoder = Qwen3TTSSpeechDecoder(_tiny_decoder()).eval()
        window = decoder.config.sliding_window
        recorder = _Recorder(torch.nn.functional.scaled_dot_product_attention)
        codes = torch.randint(0, decoder.config.codebook_size, (1, decoder.config.num_quantizers, window + 3))
        with (
                mock.patch.object(qwen3_tts_codec.functional, "scaled_dot_product_attention", recorder),
                torch.no_grad(),
        ):
            decoder(codes)
        self.assertEqual(len(recorder.calls), decoder.config.num_hidden_layers)
        for _, kwargs in recorder.calls:
            mask = kwargs["attn_mask"]
            self.assertEqual(mask.dtype, torch.bool)
            self.assertFalse(kwargs.get("is_causal", False))
        positions = torch.arange(window + 3)
        expected = ((positions[None, :] <= positions[:, None])
                    & (positions[None, :] > positions[:, None] - window))
        self.assertTrue(torch.equal(mask[0, 0], expected))

    def test_rope_inverse_frequency_matches_transformers_cpu_init(self):
        for base, dimension in ((10_000.0, 64), (1_000_000.0, 128), (10_000.0, 4)):
            with self.subTest(base=base, dimension=dimension):
                rotary = qwen3_tts_modeling.RotaryEmbedding(dimension, base=base, device="meta")
                materialize_qwen3_tts_buffers(rotary, device="cpu")
                self.assertTrue(
                    torch.equal(
                        rotary.inverse_frequency,
                        _transformers_inverse_frequency(base, dimension),
                    ))

    def test_speaker_mel_filters_match_librosa_slaney_bank_bitwise(self):
        from voicehub.architectures.qwen3_tts.runtime import _qwen3_tts_speaker_mel_filters

        filters = _qwen3_tts_speaker_mel_filters()
        self.assertEqual(filters.dtype, torch.float32)
        self.assertEqual(tuple(filters.shape), (128, 513))
        # SHA-256 of librosa 1.0.0 `filters.mel(sr=24000, n_fft=1024,
        # n_mels=128, fmin=0, fmax=12000)` as little-endian float32 bytes.
        digest = hashlib.sha256(filters.numpy().astype("<f4").tobytes()).hexdigest()
        self.assertEqual(
            digest,
            "634664518adebcf0e280349d8e199250cb2e195d4216f2e7ab725a2d585f4e02",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "Needs CUDA to check the device-independent frontend.")
    def test_speaker_mel_is_computed_on_cpu_like_upstream(self):
        from voicehub.architectures.qwen3_tts.runtime import qwen3_tts_speaker_mel

        torch.manual_seed(3)
        waveform = torch.randn(24_000) * 0.1
        on_cpu = qwen3_tts_speaker_mel(waveform)
        on_cuda = qwen3_tts_speaker_mel(waveform.cuda())
        self.assertEqual(on_cuda.device.type, "cuda")
        self.assertTrue(torch.equal(on_cuda.cpu(), on_cpu))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA pow differs from the CPU reference only on GPU.")
    def test_materialized_cuda_rope_buffers_match_transformers_cpu_init(self):
        model = Qwen3TTSForConditionalGeneration(_tiny_architecture()).eval()
        decoder = Qwen3TTSSpeechDecoder(_tiny_decoder()).eval()
        materialize_qwen3_tts_buffers(model, device="cuda")
        materialize_qwen3_tts_decoder_buffers(decoder, device="cuda")
        modules = [
            module for root in (model, decoder) for module in root.modules()
            if isinstance(module, qwen3_tts_modeling.RotaryEmbedding)
        ]
        self.assertTrue(modules)
        for module in modules:
            self.assertTrue(
                torch.equal(
                    module.inverse_frequency.cpu(),
                    _transformers_inverse_frequency(module.base, module.dimension),
                ))


if __name__ == "__main__":
    unittest.main()
