"""Regressions found by the nari-labs/dia upstream parity audit."""

from __future__ import annotations

import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is VoiceHub's compute runtime")
class NativeDiaUpstreamParityTests(unittest.TestCase):

    @staticmethod
    def components():
        from tests.test_dia_training import NativeDiaTests
        from voicehub.architectures.dac.modeling import DacModel
        from voicehub.architectures.dia.modeling import DiaForConditionalGeneration
        from voicehub.architectures.dia.processing import DiaProcessor

        config = NativeDiaTests.dia_config()
        model = DiaForConditionalGeneration(config).eval()
        codec = DacModel(NativeDiaTests.dac_config()).eval()
        processor = DiaProcessor(config, audio_tokenizer=codec, sampling_rate=16_000, hop_length=4)
        return model, codec, processor

    def test_audio_prompt_is_resampled_like_torchaudio(self):
        # Upstream Dia.load_audio resamples with torchaudio's default Hann
        # sinc; the generic resampler changed most DAC prompt codes.
        import torch
        from torch.nn import functional as F

        from voicehub.processing.waveform import resample_waveform_hann

        _, codec, processor = self.components()
        waveform = torch.sin(torch.arange(90, dtype=torch.float32) * 0.3)
        captured = []
        original = codec.encode_output

        def capture(values, **kwargs):
            captured.append(values.detach().clone())
            return original(values, **kwargs)

        with patch.object(codec, "encode_output", side_effect=capture):
            processor(text=["[S1] Hi."], audio={"array": waveform, "sampling_rate": 24_000}, generation=True)
        expected = resample_waveform_hann(waveform, 24_000, 16_000)
        expected = F.pad(expected, (0, -expected.shape[-1] % 4))
        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(captured[0].reshape(-1), expected, rtol=0, atol=0)

    def test_audio_prompt_matches_torchaudio_functional_resample(self):
        # Dia.load_audio calls torchaudio.functional.resample on [C, T]; for a
        # mono prompt that is bit-exact to the shared resampler's default mode.
        import torch
        from torch.nn import functional as F

        try:
            import torchaudio
        except ImportError:
            self.skipTest("torchaudio is only a test-time reference")

        _, codec, processor = self.components()
        waveform = torch.sin(torch.arange(2_205, dtype=torch.float32) * 0.07)
        captured = []
        original = codec.encode_output

        def capture(values, **kwargs):
            captured.append(values.detach().clone())
            return original(values, **kwargs)

        with patch.object(codec, "encode_output", side_effect=capture):
            processor(text=["[S1] Hi."], audio={"array": waveform, "sampling_rate": 22_050}, generation=True)
        expected = torchaudio.functional.resample(waveform[None], 22_050, 16_000)[0]
        expected = F.pad(expected, (0, -expected.shape[-1] % 4))
        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(captured[0].reshape(-1), expected))

    def test_temperature_is_not_clamped_and_zero_means_greedy(self):
        import torch

        model, _, processor = self.components()
        batch = processor(text=["Hello"], generation=True)
        logits = torch.zeros(2, 12)
        logits[:, 3] = 1.0

        def fixed_logits(**kwargs):
            sequence_length = kwargs["decoder_input_ids"].shape[1]
            return SimpleNamespace(logits=logits[:, None].expand(2, sequence_length, 12).clone())

        observed = []
        multinomial = torch.multinomial

        def record(probabilities, *args, **kwargs):
            observed.append(probabilities.clone())
            return multinomial(probabilities, *args, **kwargs)

        options = dict(max_new_tokens=4, guidance_scale=None, top_k=None, top_p=1.0)
        with patch.object(model, "forward", side_effect=fixed_logits):
            with patch("voicehub.architectures.dia.modeling.torch.multinomial", side_effect=record):
                model.generate(**batch, do_sample=True, temperature=0.5, **options)
                self.assertTrue(observed)
                # Channel 0 keeps codes below EOS (8): 3 has logit 1/0.5.
                probabilities = observed[0][0, :8]
                expected = torch.softmax(torch.tensor([0, 0, 0, 2.0, 0, 0, 0, 0]), dim=-1)
                torch.testing.assert_close(probabilities, expected)
                observed.clear()
                tokens = model.generate(**batch, do_sample=True, temperature=0.0, **options)
            self.assertEqual(observed, [])
        self.assertEqual(tokens[0, 1, 0].item(), 3)

    def test_default_generation_budget_matches_upstream(self):
        # Upstream Dia.generate and the checkpoint generation_config use 3072;
        # 256 frames (~3 s) silently truncated ordinary sentences.
        from voicehub.architectures.dia.modeling import DiaForConditionalGeneration
        from voicehub.models.dia.inference import DiaConfig
        from voicehub.models.dia.model import Dia

        self.assertEqual(DiaConfig().generation_config["max_new_tokens"], 3072)
        default = inspect.signature(DiaForConditionalGeneration.generate).parameters["max_new_tokens"].default
        self.assertEqual(default, 3072)
        self.assertEqual(inspect.signature(Dia.generate).parameters["max_tokens"].default, 3072)

    def test_long_audio_prompt_generation_stays_inside_decoder_positions(self):
        import torch

        model, _, processor = self.components()
        # 38 DAC frames -> 40 decoder positions of the 64 available.
        audio = {"array": torch.linspace(-0.3, 0.3, 4 * 38), "sampling_rate": 16_000}
        batch = processor(text=["Hello"], audio=audio, generation=True)
        forward = model.forward

        def never_eos(**kwargs):
            # Real decoder (keeps its positional check) without a natural EOS,
            # so generation must stop through the length budget.
            output = forward(**kwargs)
            output.logits[..., 8] = -torch.inf
            return output

        with torch.no_grad(), patch.object(model, "forward", side_effect=never_eos):
            tokens = model.generate(**batch, max_new_tokens=3072, do_sample=False, guidance_scale=None)
        prompt_positions = batch["decoder_attention_mask"].shape[1] - 1
        self.assertLessEqual(tokens.shape[1], 64 + 1)
        self.assertGreater(tokens.shape[1], prompt_positions)
        # Every channel reaches its delayed EOS before the positional limit.
        self.assertTrue((tokens[0, :, 0] == 8).any())
        self.assertTrue((tokens[0, :, 1] == 8).any())

    @staticmethod
    def decisive_components():
        import torch

        model, codec, processor = NativeDiaUpstreamParityTests.components()
        # Wide float64 weights give clear argmax margins, so the cached and
        # recomputed decoders must agree token for token.
        torch.manual_seed(0)
        model = model.double()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not name.endswith("norm.weight"):
                    parameter.normal_(0.0, 0.5)
        return model, codec, processor

    def test_cached_generation_matches_full_decoder_recompute(self):
        # Upstream Dia reuses self-attention K/V and projects cross-attention
        # K/V once; the native loop recomputed the whole prefix every step.
        import torch

        model, _, processor = self.decisive_components()
        audio = [
            {
                "array": torch.linspace(-0.3, 0.3, 4 * 6),
                "sampling_rate": 16_000
            },
            {
                "array": torch.linspace(0.2, -0.2, 4 * 3),
                "sampling_rate": 16_000
            },
        ]
        batches = {
            "text": processor(text=["Hello", "A longer line."], generation=True),
            "prompt": processor(text=["Hello", "A longer line."], audio=audio, generation=True),
        }
        for name, batch in batches.items():
            for guidance_scale in (None, 3.0):
                for do_sample in (False, True):
                    with self.subTest(batch=name, guidance_scale=guidance_scale, do_sample=do_sample):
                        options = dict(
                            max_new_tokens=24,
                            do_sample=do_sample,
                            guidance_scale=guidance_scale,
                            top_k=5,
                        )
                        torch.manual_seed(7)
                        cached = model.generate(**batch, use_cache=True, **options)
                        torch.manual_seed(7)
                        recomputed = model.generate(**batch, use_cache=False, **options)
                        self.assertTrue(torch.equal(cached, recomputed))

    def test_cached_generation_feeds_one_frame_per_step_with_batched_guidance(self):
        import torch

        model, _, processor = self.decisive_components()
        batch = processor(text=["Hello"], generation=True)
        forward = model.forward
        calls = []

        def record(**kwargs):
            calls.append(tuple(kwargs["decoder_input_ids"].shape[:2]))
            return forward(**kwargs)

        with patch.object(model, "forward", side_effect=record):
            model.generate(**batch, max_new_tokens=12, do_sample=False, guidance_scale=3.0)
        self.assertGreater(len(calls), 2)
        # Conditional and unconditional rows share one decoder call; after
        # the prefill only the newest frame is decoded.
        self.assertEqual(calls[0], (2, 1))
        self.assertEqual(set(calls[1:]), {(2, 1)})

    def test_decoder_cache_logits_match_teacher_forcing(self):
        import torch

        from voicehub.architectures.dia.modeling import DiaDecoderCache

        model, _, _ = self.decisive_components()
        input_ids = torch.randint(0, 256, (2, 7))
        attention_mask = torch.ones_like(input_ids)
        attention_mask[1, :2] = 0
        decoder_ids = torch.randint(0, 8, (2, 9, 2))
        with torch.no_grad():
            encoded = model.model.encoder(input_ids, attention_mask).last_hidden_state
            full = model(
                attention_mask=attention_mask, decoder_input_ids=decoder_ids, encoder_outputs=encoded)
            cache = DiaDecoderCache(decoder_ids.shape[1])
            steps = [
                model(
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_ids[:, start:end],
                    encoder_outputs=encoded,
                    decoder_cache=cache,
                ).logits for start, end in ((0, 4), (4, 5), (5, 6), (6, 9))
            ]
        self.assertEqual(cache.length, decoder_ids.shape[1])
        # RoPE and softmax run in float32 like the reference, over different
        # key lengths here, so some CPUs differ by an ulp (up to ~8e-7).
        torch.testing.assert_close(torch.cat(steps, dim=1), full.logits, rtol=1e-5, atol=1e-5)

    def test_codec_stays_float32_for_half_precision_compute(self):
        import torch

        from tests.test_dia_training import NativeDiaTests
        from voicehub.architectures.dia import runtime as dia_runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            NativeDiaTests.write_artifact(root)
            with patch.object(dia_runtime, "resolve_dia_dtype", return_value=torch.bfloat16):
                loaded = dia_runtime.load_dia_runtime(root, device="cpu", compute_dtype="bfloat16")
        self.assertEqual(next(loaded.model.parameters()).dtype, torch.bfloat16)
        self.assertTrue(all(p.dtype == torch.float32 for p in loaded.processor.audio_tokenizer.parameters()))


if __name__ == "__main__":
    unittest.main()
