"""Regression tests pinning PyanNet VAD to pyannote.audio 3.0.0 semantics.

Golden values were produced with ``pyannote.audio`` at
``795b92ab265888c58d160f90ae4d91b7bcc6aa2c`` (``Binarize`` and
``Inference.aggregate``) and ``pyannote.core`` 5.0.0.
"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from voicehub.architectures.pyannet.binarize import binarize_frame_scores
from voicehub.architectures.pyannet.configuration import PyanNetConfig
from voicehub.architectures.pyannet.inference import PyanNetFrameInference, PyanNetFrameOutput, _closest_frame
from voicehub.architectures.pyannet.modeling import PyanNet
from voicehub.checkpointing import save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_pyannote_segmentation import (
    PyannoteSegmentationVADConfig,
    PyannoteSegmentationVADForVoiceActivityDetection,
)

SEGMENTATION_3_STEP = 10.0 / 589
SCORES = np.asarray(
    [
        0.5, 0.9, 1.0, 0.5, 0.4, 0.0, 0.0, 0.6, 0.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8, 0.5, 0.5, 0.3,
        0.0, 1.0, 1.0, 1.0
    ],
    dtype=np.float32,
).tolist()
# pyannote.audio.utils.signal.Binarize(**options)(SlidingWindowFeature(
#     SCORES, SlidingWindow(start=0, duration=10 / 589, step=10 / 589)))
UPSTREAM_BINARIZE = (
    (
        {},
        (
            (0.025466893039049237, 0.07640067911714771),
            (0.1273344651952462, 0.14431239388794567),
            (0.16129032258064516, 0.19524617996604415),
            (0.2801358234295416, 0.3480475382003395),
            (0.3820033955857386, 0.41595925297113756),
        ),
    ),
    (
        {
            "min_duration_off": 0.06
        },
        (
            (0.025466893039049237, 0.19524617996604415),
            (0.2801358234295416, 0.41595925297113756),
        ),
    ),
    (
        {
            "min_duration_on": 0.04
        },
        (
            (0.025466893039049237, 0.07640067911714771),
            (0.2801358234295416, 0.3480475382003395),
        ),
    ),
    (
        {
            "onset": 0.7,
            "offset": 0.25
        },
        (
            (0.025466893039049237, 0.09337860780984719),
            (0.16129032258064516, 0.19524617996604415),
            (0.2971137521222411, 0.36502546689303905),
            (0.3820033955857386, 0.41595925297113756),
        ),
    ),
    (
        {
            "pad_onset": 0.02,
            "pad_offset": 0.02
        },
        (
            (0.0054668930390492365, 0.09640067911714771),
            (0.10733446519524618, 0.21524617996604414),
            (0.26013582342954156, 0.4359592529711376),
        ),
    ),
)


def _pyannote_aggregate(chunk_scores, *, duration, step, num_samples, sample_rate):
    """Transcription of pyannote.audio 3.0.0 ``Inference.slide`` aggregation."""
    num_chunks, frames_per_chunk, _ = chunk_scores.shape
    frame = duration / frames_per_chunk

    def closest_frame(time):
        return int(np.rint((time - 0.0 - 0.5 * frame) / frame))

    num_frames = closest_frame(0.0 + duration + (num_chunks - 1) * step) + 1
    output = np.zeros((num_frames, 1), dtype=np.float32)
    count = np.zeros((num_frames, 1), dtype=np.float32)
    hamming = np.hamming(frames_per_chunk).reshape(-1, 1)
    for index in range(num_chunks):
        start = closest_frame(0.0 + index * step)
        output[start:start + frames_per_chunk] += chunk_scores[index] * 1 * hamming * np.ones_like(hamming)
        count[start:start + frames_per_chunk] += 1 * hamming * np.ones_like(hamming)
    average = output / np.maximum(count, 1e-12)
    last = int(np.floor((num_samples / sample_rate - 0.0) / frame))
    return average[:last + 1]


def _powerset_config(**overrides):
    values = {
        "variant": "powerset-segmentation",
        "num_classes": 3,
        "max_active_classes": 2,
        "lstm_hidden_size": 8,
        "lstm_num_layers": 1,
        "lstm_dropout": 0.0,
        "linear_hidden_size": 8,
        "linear_num_layers": 1,
        "chunk_duration_s": 10.0,
        "chunk_step_s": 1.0,
    }
    values.update(overrides)
    return PyanNetConfig(**values)


class PyanNetUpstreamFrameGridTests(unittest.TestCase):

    def test_chunk_placement_rounds_ties_like_pyannote_in_seconds(self):
        # pyannote: SlidingWindow(0, 10/589, 10/589).closest_frame(t).
        # Evaluating the same rule in samples rounds 30 s and 50 s up.
        self.assertEqual(_closest_frame(30.0, frame_step_s=SEGMENTATION_3_STEP), 1766)
        self.assertEqual(_closest_frame(50.0, frame_step_s=SEGMENTATION_3_STEP), 2944)
        self.assertEqual(_closest_frame(20.0, frame_step_s=SEGMENTATION_3_STEP), 1178)
        self.assertEqual(_closest_frame(0.0, frame_step_s=SEGMENTATION_3_STEP), 0)

    def test_overlap_add_matches_pyannote_aggregate_beyond_thirty_seconds(self):
        torch.manual_seed(0)
        model = PyanNet(_powerset_config()).eval()
        generator = torch.Generator().manual_seed(1234)
        produced = []

        def fake_forward(batch):
            frames = 589
            labels = torch.randint(0, 7, (batch.shape[0], frames), generator=generator)
            values = torch.nn.functional.one_hot(labels, 7).to(batch.dtype)
            produced.append(model.speech_probabilities(values))
            return values

        model.forward = fake_forward
        sample_count = 16_000 * 53 + 4_321
        output = PyanNetFrameInference(model, batch_size=7)(torch.zeros(sample_count))
        chunk_scores = torch.cat(produced).unsqueeze(-1).numpy()
        expected = _pyannote_aggregate(
            chunk_scores,
            duration=10.0,
            step=1.0,
            num_samples=sample_count,
            sample_rate=16_000,
        )
        self.assertEqual(tuple(output.scores.shape), expected.shape)
        np.testing.assert_array_equal(output.scores.numpy(), expected)
        self.assertEqual(output.frame_step_s, SEGMENTATION_3_STEP)
        self.assertEqual(output.frame_duration_s, SEGMENTATION_3_STEP)


class PyannoteBinarizeTests(unittest.TestCase):

    def test_binarize_matches_upstream_golden_regions(self):
        for options, expected in UPSTREAM_BINARIZE:
            with self.subTest(options=options):
                arguments = {"onset": 0.5, "offset": 0.5, **options}
                self.assertEqual(
                    binarize_frame_scores(
                        SCORES,
                        frame_step_s=SEGMENTATION_3_STEP,
                        frame_duration_s=SEGMENTATION_3_STEP,
                        **arguments,
                    ),
                    expected,
                )

    def test_scores_equal_to_the_threshold_neither_open_nor_close_regions(self):
        regions = binarize_frame_scores(
            [0.5, 0.5, 1.0, 0.5, 0.5, 0.0, 0.5],
            frame_step_s=1.0,
            frame_duration_s=1.0,
        )
        self.assertEqual(regions, ((2.5, 5.5), ))


class SegmentationProviderPostprocessingTests(unittest.TestCase):

    def _provider(self, directory, scores, frame_step_s):
        config = _powerset_config()
        model = PyanNet(config)
        root = Path(directory)
        save_safetensors(model.state_dict(), root / "model.safetensors", metadata={"format": "voicehub-pyannet-v1"})
        write_json_file(root / "config.json", config.to_dict())
        provider = PyannoteSegmentationVADForVoiceActivityDetection(
            PyannoteSegmentationVADConfig(name_or_path=root),
            device="cpu",
        ).load()
        provider._frame_output = lambda waveform: PyanNetFrameOutput(
            scores=torch.tensor(scores, dtype=torch.float32).unsqueeze(-1),
            frame_hop_samples=max(1, round(frame_step_s * 16_000)),
            frame_length_samples=max(1, round(frame_step_s * 16_000)),
            frame_start_samples=0,
            valid_samples=waveform.numel(),
            frame_step_s=frame_step_s,
            frame_duration_s=frame_step_s,
        )
        return provider

    def test_preset_reports_pyannote_frame_middle_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = self._provider(directory, SCORES, SEGMENTATION_3_STEP)
            output = provider.detect(torch.zeros(16_000), sampling_rate=16_000)
        self.assertEqual(
            tuple((segment.start, segment.end) for segment in output.segments),
            UPSTREAM_BINARIZE[0][1],
        )

    def test_durations_map_to_pipeline_hyper_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = self._provider(directory, SCORES, SEGMENTATION_3_STEP)
            output = provider.detect(
                torch.zeros(16_000),
                sampling_rate=16_000,
                min_speech_duration_ms=40,
                min_silence_duration_ms=60,
            )
        self.assertEqual(
            tuple((segment.start, segment.end) for segment in output.segments),
            UPSTREAM_BINARIZE[1][1],
        )

    def test_non_native_rates_resample_like_torchaudio_functional(self):
        # pyannote.audio's Audio uses torchaudio.functional.resample defaults,
        # which resample_waveform_hann reproduces bit for bit.
        from voicehub.processing.waveform import resample_waveform_hann

        waveform = torch.randn(48_000, generator=torch.Generator().manual_seed(3))
        with tempfile.TemporaryDirectory() as directory:
            provider = self._provider(directory, SCORES, SEGMENTATION_3_STEP)
            seen = []
            frame_output = provider._frame_output

            def capture(values):
                seen.append(values)
                return frame_output(values)

            provider._frame_output = capture
            output = provider.detect(waveform, sampling_rate=48_000)
        torch.testing.assert_close(seen[0], resample_waveform_hann(waveform, 48_000, 16_000), rtol=0, atol=0)
        self.assertEqual(output.duration, 1.0)

    def test_regions_are_clipped_to_the_recording(self):
        # The last frame middle (0.4160 s) lies beyond a 0.41 s recording and
        # padding pushes the first region before zero; VADOutput needs both
        # inside [0, duration].
        with tempfile.TemporaryDirectory() as directory:
            provider = self._provider(directory, SCORES, SEGMENTATION_3_STEP)
            output = provider.detect(
                torch.zeros(6_560),
                sampling_rate=16_000,
                speech_pad_ms=30,
            )
        self.assertEqual(output.segments[0].start, 0.0)
        self.assertEqual(output.segments[-1].end, 0.41)
        self.assertTrue(math.isclose(output.segments[-1].start, 0.2801358234295416 - 0.03))


if __name__ == "__main__":
    unittest.main()
