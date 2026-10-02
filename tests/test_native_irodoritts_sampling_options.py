"""Runtime option semantics that must follow the released Irodori runtime."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.test_native_irodoritts import _tiny_config, _write_tokenizer
from voicehub.architectures.irodoritts.modeling import TextToLatentRFDiT
from voicehub.architectures.irodoritts.runtime import InferenceRuntime, SamplingRequest

HOP = 1920


class IrodoriSamplingOptionTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(3)
        self.decode_batches = []

        def decode_latent(latent):
            self.decode_batches.append(latent.shape[0])
            return latent[:, :, 0].repeat_interleave(HOP, dim=1)

        self.config = _tiny_config(use_duration_predictor=False)
        self.codec = SimpleNamespace(
            device=torch.device("cpu"),
            sample_rate=48_000,
            hop_length=HOP,
            decode_latent=decode_latent,
        )
        self.directory = tempfile.TemporaryDirectory()
        self.runtime = InferenceRuntime(
            model=TextToLatentRFDiT(self.config).eval(),
            model_cfg=self.config,
            tokenizer=_write_tokenizer(Path(self.directory.name)),
            codec=self.codec,
            model_device="cpu",
        )

    def tearDown(self):
        self.directory.cleanup()

    def _request(self, **overrides):
        values = {"text": "known", "seconds": 0.53, "num_steps": 2, "seed": 5, "trim_tail": False}
        values.update(overrides)
        return SamplingRequest(**values)

    def test_manual_seconds_trim_to_requested_sample_count(self):
        result = self.runtime.synthesize(self._request(no_ref=True))
        # int(0.53 * 48 kHz) samples, decoded from ceil(25440 / 1920) = 14 frames.
        self.assertEqual(result.audio.numel(), 25_440)

    def test_speaker_kv_scaling_defaults_to_released_time_threshold(self):
        captured = []

        def sampler(**kwargs):
            captured.append(kwargs)
            return torch.zeros(1, kwargs["sequence_length"], self.config.patched_latent_dim)

        reference = torch.randn(1, 6, self.config.latent_dim)
        with patch("voicehub.architectures.irodoritts.runtime.sample_euler_rf_cfg", side_effect=sampler):
            self.runtime.synthesize(self._request(ref_latent=reference, speaker_kv_scale=1.5))
            self.runtime.synthesize(
                self._request(ref_latent=reference, speaker_kv_scale=1.5, speaker_kv_min_t=0.7))
            self.runtime.synthesize(self._request(no_ref=True, speaker_kv_scale=1.5))
        self.assertEqual([(call["speaker_kv_scale"], call["speaker_kv_min_t"]) for call in captured],
                         [(1.5, 0.9), (1.5, 0.7), (None, None)])

    def test_decode_mode_controls_candidate_batching(self):
        self.runtime.synthesize(self._request(no_ref=True, num_candidates=2))
        self.assertEqual(self.decode_batches, [1, 1])
        self.decode_batches.clear()
        self.runtime.synthesize(self._request(no_ref=True, num_candidates=2, decode_mode="batch"))
        self.assertEqual(self.decode_batches, [2])
        with self.assertRaisesRegex(ValueError, "decode_mode"):
            self.runtime.synthesize(self._request(no_ref=True, decode_mode="parallel"))


if __name__ == "__main__":
    unittest.main()
