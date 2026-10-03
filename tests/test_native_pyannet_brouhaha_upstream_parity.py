"""Regression tests pinning Brouhaha VAD to its upstream pipeline semantics.

``marianne-m/brouhaha-vad@9132cbe62ac78f90abdbc21bcf6ec6cfe9bb4891``
requires ``pyannote.audio>=3.1,<3.3.1``.  Golden values were produced with
pyannote.audio 3.3.0 / pyannote.core 5.0.0: ``Inference.aggregate`` plus the
loose crop of ``BrouhahaInference.slide`` on the model receptive field
``SlidingWindow(start=0, duration=991 / 16000, step=270 / 16000)``, and
``Binarize`` on that same grid.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from voicehub.architectures.pyannet.configuration import PyanNetConfig
from voicehub.architectures.pyannet.inference import PyanNetFrameInference, PyanNetFrameOutput
from voicehub.architectures.pyannet.modeling import PyanNet, SincNet
from voicehub.checkpointing import save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_pyannote_brouhaha import (
    PyannoteBrouhahaVADConfig,
    PyannoteBrouhahaVADForVoiceActivityDetection,
)

RECEPTIVE_FIELD_STEP = 270 / 16_000
RECEPTIVE_FIELD_DURATION = 991 / 16_000
# Inference.aggregate(chunk c filled with (c + 1) * [1, 10, 100]) on the
# receptive-field grid, then the loose Segment(0, duration) crop.
UPSTREAM_AGGREGATE = {
    105_600: (
        392,
        {
            35: 1.0,
            36: 1.3171377182006836,
            71: 1.2928584814071655,
            72: 1.294819951057434,
            351: 1.6828622817993164,
            352: 2.0,
            387: 2.0,
            388: 0.0,
            391: 0.0,
        },
    ),
    110_000: (
        408,
        {
            35: 1.0,
            36: 1.3171377182006836,
            71: 1.5024844408035278,
            72: 1.500344157218933,
            351: 2.4938108921051025,
            352: 2.7030656337738037,
            387: 2.6765964031219482,
            388: 3.0,
            407: 3.0,
        },
    ),
}
SCORES = np.asarray(
    [
        0.5, 0.9, 1.0, 0.5, 0.4, 0.0, 0.0, 0.8, 0.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.79, 0.8, 0.5, 0.78,
        0.3, 0.0, 1.0, 1.0, 1.0
    ],
    dtype=np.float32,
).tolist()
# pyannote.audio.utils.signal.Binarize(**options)(SlidingWindowFeature(
#     SCORES, SlidingWindow(start=0, duration=991 / 16000, step=270 / 16000)))
UPSTREAM_DEFAULT_REGIONS = (
    (0.047843750000000004, 0.08159375),
    (0.14909375000000002, 0.16596875),
    (0.18284375000000003, 0.21659375000000003),
    (0.30096875, 0.33471875),
    (0.40221875, 0.43596875),
)
UPSTREAM_MIN_DURATION_OFF_REGIONS = ((0.047843750000000004, 0.43596875), )
UPSTREAM_MIN_DURATION_ON_REGIONS = (
    (0.047843750000000004, 0.08159375),
    (0.18284375000000003, 0.21659375000000003),
    (0.30096875, 0.33471875),
    (0.40221875, 0.43596875),
)
UPSTREAM_HYSTERESIS_REGIONS = (
    (0.047843750000000004, 0.11534375000000001),
    (0.14909375000000002, 0.16596875),
    (0.18284375000000003, 0.21659375000000003),
    (0.30096875, 0.38534375),
    (0.40221875, 0.43596875),
)


def _tiny_brouhaha_config():
    return PyanNetConfig(
        variant="brouhaha",
        num_classes=3,
        lstm_hidden_size=8,
        lstm_num_layers=1,
        lstm_dropout=0.0,
        linear_hidden_size=8,
        linear_num_layers=1,
        chunk_duration_s=6.0,
        chunk_step_s=0.6,
        repeat_final_chunk=True,
    )


class BrouhahaFrameGridTests(unittest.TestCase):

    def test_receptive_field_matches_pyannote_sincnet(self):
        self.assertEqual(SincNet.receptive_field_size(1), 991)
        self.assertEqual(SincNet.receptive_field_size(2) - SincNet.receptive_field_size(1), 270)

    def test_overlap_add_matches_pyannote_receptive_field_aggregate(self):
        # pyannote >= 3.1 places chunk c at rint(c * 0.6 / step) = 36, 71,
        # ... frames; the pyannote 3.0 rule (start - step / 2) gives 35, 71.
        model = PyanNet(_tiny_brouhaha_config()).eval()
        for num_samples, (frames, expected) in UPSTREAM_AGGREGATE.items():
            calls = []

            def fake_forward(batch):
                frames_per_chunk = model.frame_count(batch.shape[-1])
                values = []
                for _ in range(batch.shape[0]):
                    calls.append(None)
                    values.append(
                        torch.full(
                            (frames_per_chunk, 3), float(len(calls))) * torch.tensor([1.0, 10.0, 100.0]))
                return torch.stack(values)

            model.forward = fake_forward
            output = PyanNetFrameInference(model)(torch.zeros(num_samples))
            self.assertEqual(output.scores.shape, (frames, 3))
            self.assertEqual(output.frame_step_s, RECEPTIVE_FIELD_STEP)
            self.assertEqual(output.frame_duration_s, RECEPTIVE_FIELD_DURATION)
            for frame, value in expected.items():
                self.assertEqual(output.scores[frame, 0].item(), value, (num_samples, frame))

    def test_last_chunk_runs_as_its_own_batch_like_pyannote(self):
        # BrouhahaInference.slide: complete chunks in batch_size batches, then
        # infer(last_chunk[None]); GPU kernels are batch-size dependent.
        model = PyanNet(_tiny_brouhaha_config()).eval()
        for num_samples, expected in (
            (376_320, [30, 1]),
            (96_000 + 40 * 9_600 + 1, [32, 9, 1]),
            (470_400, [32, 8]),
            (50_000, [1]),
        ):
            sizes = []

            def fake_forward(batch):
                sizes.append(batch.shape[0])
                return torch.zeros(batch.shape[0], model.frame_count(batch.shape[-1]), 3)

            model.forward = fake_forward
            PyanNetFrameInference(model, batch_size=32)(torch.zeros(num_samples))
            self.assertEqual(sizes, expected, num_samples)


class BrouhahaProviderPostprocessingTests(unittest.TestCase):

    def _provider(self, directory):
        config = _tiny_brouhaha_config()
        model = PyanNet(config)
        root = Path(directory)
        save_safetensors(
            model.state_dict(), root / "model.safetensors", metadata={"format": "voicehub-pyannet-v1"})
        write_json_file(root / "config.json", config.to_dict())
        provider = PyannoteBrouhahaVADForVoiceActivityDetection(
            PyannoteBrouhahaVADConfig(name_or_path=root),
            device="cpu",
        ).load()
        scores = torch.tensor(SCORES, dtype=torch.float32)
        provider._frame_output = lambda waveform: PyanNetFrameOutput(
            scores=torch.stack(
                (scores, torch.full_like(scores, 20.0), torch.full_like(scores, 30.0)), dim=-1),
            frame_hop_samples=270,
            frame_length_samples=270,
            frame_start_samples=0,
            valid_samples=waveform.numel(),
            frame_step_s=RECEPTIVE_FIELD_STEP,
            frame_duration_s=RECEPTIVE_FIELD_DURATION,
        )
        return provider

    def _regions(self, **options):
        with tempfile.TemporaryDirectory() as directory:
            output = self._provider(directory).detect(torch.zeros(16_000), sampling_rate=16_000, **options)
        return tuple((segment.start, segment.end) for segment in output.segments)

    def test_default_parameters_report_upstream_binarize_regions(self):
        # RegressiveActivityDetectionPipeline.default_parameters():
        # onset = offset = 0.78, min_duration_on = min_duration_off = 0.
        self.assertEqual(self._regions(), UPSTREAM_DEFAULT_REGIONS)

    def test_durations_and_hysteresis_map_to_pipeline_parameters(self):
        self.assertEqual(self._regions(min_silence_duration_ms=100), UPSTREAM_MIN_DURATION_OFF_REGIONS)
        self.assertEqual(self._regions(min_speech_duration_ms=30), UPSTREAM_MIN_DURATION_ON_REGIONS)
        self.assertEqual(self._regions(onset=0.6, offset=0.3), UPSTREAM_HYSTERESIS_REGIONS)

    def test_non_native_rates_resample_like_torchaudio_functional(self):
        from voicehub.processing.waveform import resample_waveform_hann

        waveform = torch.randn(48_000, generator=torch.Generator().manual_seed(3))
        with tempfile.TemporaryDirectory() as directory:
            provider = self._provider(directory)
            seen = []
            frame_output = provider._frame_output

            def capture(values):
                seen.append(values)
                return frame_output(values)

            provider._frame_output = capture
            output = provider.detect(waveform, sampling_rate=48_000)
        torch.testing.assert_close(seen[0], resample_waveform_hann(waveform, 48_000, 16_000), rtol=0, atol=0)
        self.assertEqual(output.duration, 1.0)


if __name__ == "__main__":
    unittest.main()
