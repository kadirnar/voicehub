from __future__ import annotations

import math
import unittest

import torch

from voicehub.models.vad_auditok.native import (
    EnergyVoiceActivityDetector,
    create_energy_vad_architecture_spec,
    estimate_energy_threshold,
)


class NativeEnergyVADTests(unittest.TestCase):

    def test_percentile_threshold_uses_noise_floor_plus_six_db(self):
        energies = torch.tensor([-200.0, 10.0, 20.0, 30.0])

        threshold = estimate_energy_threshold(
            energies,
            method="p50",
        )

        self.assertAlmostEqual(threshold, 26.0)

    def test_digital_silence_estimates_infinite_threshold(self):
        threshold = estimate_energy_threshold(
            torch.full((3, ), -200.0),
            method="otsu",
        )

        self.assertTrue(math.isinf(threshold))

    def test_duration_join_padding_and_strict_maximum_split(self):
        waveform = torch.zeros(1_600)
        waveform[160:640] = 0.2
        waveform[800:1_440] = 0.2
        detector = EnergyVoiceActivityDetector()

        result = detector.detect(
            waveform,
            sampling_rate=16_000,
            energy_threshold_db=50,
            threshold_method="fixed",
            analysis_window_s=0.01,
            minimum_energy_threshold_db=40,
            min_speech_duration_ms=20,
            min_silence_duration_ms=0,
            speech_pad_ms=10,
            max_speech_duration_s=0.025,
            strict_min_duration=True,
        )

        self.assertEqual(
            tuple((region.start_sample, region.end_sample) for region in result.regions),
            (
                (0, 400),
                (400, 800),
                (640, 1_040),
                (1_040, 1_440),
            ),
        )

    def test_partial_final_window_is_not_energy_diluted_by_padding(self):
        result = EnergyVoiceActivityDetector().detect(
            torch.ones(50),
            sampling_rate=1_000,
            energy_threshold_db=89,
            threshold_method="fixed",
            analysis_window_s=0.1,
            minimum_energy_threshold_db=40,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
            max_speech_duration_s=None,
            strict_min_duration=False,
        )

        self.assertGreater(
            float(result.frame_energies_db[0].item()),
            90,
        )
        self.assertEqual(result.regions, ())

    def test_architecture_declares_algorithmic_non_trainable_contract(self):
        spec = create_energy_vad_architecture_spec()

        self.assertEqual(spec.architecture_id, "energy-vad")
        self.assertFalse(spec.capabilities.training)
        self.assertTrue(spec.capabilities.has_feature("algorithmic"))


if __name__ == "__main__":
    unittest.main()
