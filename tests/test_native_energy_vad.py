from __future__ import annotations

import math
import unittest

import torch

from voicehub.architectures.energy_vad import (
    EnergyRegion,
    EnergyVoiceActivityDetector,
    create_energy_vad_architecture_spec,
    estimate_energy_threshold,
)


class NativeEnergyVADTests(unittest.TestCase):

    def test_auditok_retains_trailing_silence_by_default_and_can_trim(self):
        from voicehub.models.vad_auditok import AuditokVADConfig, AuditokVADForVoiceActivityDetection

        waveform = torch.zeros(16_000)
        waveform[1600:8000] = 0.2
        for trim, expected_end in ((False, 0.6), (True, 0.5)):
            with self.subTest(drop_trailing_silence=trim):
                config = AuditokVADConfig(drop_trailing_silence=trim)
                restored = AuditokVADConfig.from_dict(config.to_dict())
                self.assertEqual(restored.drop_trailing_silence, trim)
                model = AuditokVADForVoiceActivityDetection(config, device="cpu")
                result = model.detect(waveform, sampling_rate=16_000, speech_pad_ms=0)
                self.assertEqual([(s.start, s.end) for s in result.segments], [(0.1, expected_end)])
        self.assertFalse(AuditokVADConfig().drop_trailing_silence)
        with self.assertRaisesRegex(TypeError, "drop_trailing_silence"):
            AuditokVADConfig(drop_trailing_silence="false")

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

        # Auditok 833ae72 tokens (no padding): (160, 480), (800, 1120),
        # (1120, 1440); the strict remainder (480, 640) is dropped. The
        # 160-sample pads then meet halfway between overlapping tokens.
        self.assertEqual(
            tuple((region.start_sample, region.end_sample) for region in result.regions),
            (
                (0, 640),
                (640, 1_120),
                (1_120, 1_600),
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
        # Auditok counts the partial final window as one analysis window.
        self.assertEqual(result.regions, (EnergyRegion(0, 50), ))

    def test_architecture_declares_algorithmic_non_trainable_contract(self):
        spec = create_energy_vad_architecture_spec()

        self.assertEqual(spec.architecture_id, "energy-vad")
        self.assertFalse(spec.capabilities.training)
        self.assertTrue(spec.capabilities.has_feature("algorithmic"))


def _auditok_regions(waveform, *, sampling_rate=16_000, **overrides):
    options = {
        "energy_threshold_db": 50,
        "threshold_method": "fixed",
        "analysis_window_s": 0.05,
        "minimum_energy_threshold_db": None,
        "min_speech_duration_ms": 200,
        "min_silence_duration_ms": 0,
        "speech_pad_ms": 0,
        "max_speech_duration_s": None,
        "strict_min_duration": False,
        "drop_trailing_silence": False,
    }
    options.update(overrides)
    result = EnergyVoiceActivityDetector().detect(
        torch.as_tensor(waveform, dtype=torch.float32),
        sampling_rate=sampling_rate,
        **options,
    )
    return [(region.start_sample, region.end_sample) for region in result.regions]


class AuditokParityRegressionTests(unittest.TestCase):
    """Expected regions come from ``auditok.split`` at the pinned revision."""

    def test_maximum_duration_truncates_tokens_like_auditok(self):
        waveform = torch.zeros(16_000)
        waveform[1_600:8_000] = 0.2
        for strict, expected in (
            (False, [(1_600, 6_400), (6_400, 8_000)]),
            (True, [(1_600, 6_400)]),
        ):
            with self.subTest(strict_min_duration=strict):
                self.assertEqual(
                    _auditok_regions(waveform, max_speech_duration_s=0.3, strict_min_duration=strict),
                    expected,
                )
        waveform[8_000:9_600] = 0.2
        self.assertEqual(
            _auditok_regions(waveform, max_speech_duration_s=0.3, min_silence_duration_ms=100),
            [(1_600, 6_400), (6_400, 11_200)],
        )

    def test_wrapper_keeps_truncated_pieces_separate(self):
        from voicehub.models.vad_auditok import AuditokVADConfig, AuditokVADForVoiceActivityDetection

        waveform = torch.zeros(16_000)
        waveform[1_600:8_000] = 0.2
        model = AuditokVADForVoiceActivityDetection(AuditokVADConfig(), device="cpu")
        output = model.detect(
            waveform,
            sampling_rate=16_000,
            min_speech_duration_ms=200,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
            max_speech_duration_s=0.3,
        )

        self.assertEqual([(s.start, s.end) for s in output.segments], [(0.1, 0.4), (0.4, 0.5)])

    def test_wrapper_measures_energy_at_the_input_rate(self):
        from voicehub.models.vad_auditok import AuditokVADConfig, AuditokVADForVoiceActivityDetection

        # 48 kHz white noise at ~52 dB: auditok.split detects (0.0, 1.0);
        # resampling to 16 kHz first would drop it below 50 dB.
        generator = torch.Generator().manual_seed(0)
        waveform = torch.randn(48_000, generator=generator) * 400 / 32_768
        model = AuditokVADForVoiceActivityDetection(AuditokVADConfig(), device="cpu")
        output = model.detect(waveform, sampling_rate=48_000, speech_pad_ms=0)

        self.assertEqual([(s.start, s.end) for s in output.segments], [(0.0, 1.0)])
        self.assertEqual(output.sample_rate, 48_000)

    def test_durations_convert_to_windows_with_auditok_float_rules(self):
        waveform = torch.zeros(16_000)
        waveform[1_600:2_720] = 0.2  # seven 10 ms windows
        # Auditok: ceil(0.07 / 0.01) == 8 windows, ceil(0.06 / 0.01) == 6.
        self.assertEqual(_auditok_regions(waveform, analysis_window_s=0.01, min_speech_duration_ms=70), [])
        self.assertEqual(
            _auditok_regions(waveform, analysis_window_s=0.01, min_speech_duration_ms=60),
            [(1_600, 2_720)],
        )

    def test_analysis_window_truncates_to_whole_samples(self):
        waveform = torch.zeros(16_000)
        waveform[5_320:10_640] = 0.2
        # int(0.0333 * 16000) == 532, not round(...) == 533.
        self.assertEqual(
            _auditok_regions(waveform, analysis_window_s=0.0333, min_speech_duration_ms=100),
            [(5_320, 10_640)],
        )

    def test_offline_calibration_has_no_energy_floor_by_default(self):
        from voicehub.models.vad_auditok import AuditokVADConfig

        generator = torch.Generator().manual_seed(0)
        waveform = torch.randn(16_000, generator=generator, dtype=torch.float64) * 10 / 32_768
        waveform[4_000:12_000] *= 300
        result = EnergyVoiceActivityDetector().detect(
            waveform.float(),
            sampling_rate=16_000,
            energy_threshold_db=50,
            threshold_method="percentile",
            analysis_window_s=0.05,
            minimum_energy_threshold_db=None,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            speech_pad_ms=0,
            max_speech_duration_s=None,
            strict_min_duration=False,
            drop_trailing_silence=False,
        )

        self.assertLess(result.threshold_db, 40)
        self.assertEqual([(r.start_sample, r.end_sample) for r in result.regions], [(4_000, 13_600)])
        self.assertIsNone(AuditokVADConfig().minimum_energy_threshold_db)
        clamped = EnergyVoiceActivityDetector().detect(
            waveform.float(),
            sampling_rate=16_000,
            energy_threshold_db=50,
            threshold_method="percentile",
            analysis_window_s=0.05,
            minimum_energy_threshold_db=40,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            speech_pad_ms=0,
            max_speech_duration_s=None,
            strict_min_duration=False,
        )
        self.assertEqual(clamped.threshold_db, 40)

    def test_threshold_estimates_are_bit_identical_to_auditok(self):
        # Expected values: auditok.signal.estimate_energy_threshold (NumPy
        # histogram/percentile) on the same float64 energies.
        cases = (
            (
                [
                    24.282, 31.841, 60.064, 49.108, 24.706, 41.656, 43.953, 27.987, 56.729, 25.684, 39.561,
                    45.837, 41.531, 49.34, 56.892, 67.813, 34.21, 52.427, 54.811, 34.636, 20.075, 68.673,
                    34.92, 35.699
                ],
                44.753671874999995,
                36.299400000000006,
            ),
            (
                [
                    31.984, 59.435, 56.102, 22.289, 36.473, 33.106, 48.214, 64.341, 35.278, 68.606, 44.934,
                    60.411, 63.663, 27.041, 43.676, 37.292, 45.633, 28.043, 28.6, 36.642, 50.841, 27.085,
                    66.011, 45.235
                ],
                46.89490625,
                36.6304,
            ),
        )
        for values, otsu, p20 in cases:
            energies = torch.tensor(values, dtype=torch.float64)
            self.assertEqual(estimate_energy_threshold(energies, method="otsu"), otsu)
            self.assertEqual(estimate_energy_threshold(energies, method="p20"), p20)

    def test_constant_energy_threshold_has_no_percentile_margin(self):
        self.assertEqual(
            estimate_energy_threshold(torch.full((5, ), 42.0, dtype=torch.float64), method="percentile"),
            42.0,
        )

    def test_frame_energies_use_float64_on_the_int16_scale(self):
        waveform = torch.linspace(-0.5, 0.5, 1_000, dtype=torch.float32)
        reference = waveform.double().reshape(10, 100) * 32_768.0
        expected = 20 * torch.log10(reference.square().mean(dim=-1).sqrt())
        energies = EnergyVoiceActivityDetector().detect(
            waveform,
            sampling_rate=1_000,
            energy_threshold_db=50,
            threshold_method="fixed",
            analysis_window_s=0.1,
            minimum_energy_threshold_db=None,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
            max_speech_duration_s=None,
            strict_min_duration=False,
        ).frame_energies_db

        self.assertEqual(energies.dtype, torch.float64)
        torch.testing.assert_close(energies, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
