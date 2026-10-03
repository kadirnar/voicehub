"""Upstream-parity regressions for the ``vad_pyannote`` provider.

The reference is ``pyannote.audio`` 3.0.0
(``795b92ab265888c58d160f90ae4d91b7bcc6aa2c``) running the
``pyannote/voice-activity-detection`` pipeline (revision
``8d678d7647d9c991934b2270b21fe390f9a8fea5``) on
``pyannote/segmentation@Interspeech2021``.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.architectures.pyannet.configuration import PyanNetConfig
from voicehub.architectures.pyannet.inference import PyanNetFrameOutput
from voicehub.architectures.pyannet.modeling import PyanNet
from voicehub.checkpointing import save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_pyannote import PyannoteVADConfig, PyannoteVADForVoiceActivityDetection
from voicehub.processing.waveform import resample_waveform_hann

# pyannote/voice-activity-detection config.yaml at the pinned revision.
PIPELINE_PARAMETERS = {
    "onset": 0.8104268538848918,
    "offset": 0.4806866463041527,
    "min_duration_on": 0.05537587440407595,
    "min_duration_off": 0.09791355693027545,
}
# pyannote.audio 3.0.0 frame step for the 5 s / 293-frame segmentation chunk.
FRAME_STEP_S = 5 / 293
ONSET = PIPELINE_PARAMETERS["onset"]
# Covers: a score equal to onset (does not open), sustain between offset and
# onset, a 5-frame gap (filled by min_duration_off), 6-frame gaps (kept), a
# 3-frame region (removed by min_duration_on), a 4-frame region (kept), and
# a region still active on the last frame.
SCORES = ([0.1] * 3 + [ONSET] + [0.9] * 6 + [0.6] * 3 + [0.1] * 5 + [0.95] * 8 + [0.1] * 6 + [0.9] * 3 +
          [0.1] * 7 + [0.9] * 4 + [0.2] * 6 + [0.9] * 8)
# pyannote.audio 3.0.0 ``Binarize`` on SCORES (float32) with the pipeline
# parameters, then ``get_timeline().support()`` as in the model card.
UPSTREAM_PIPELINE_REGIONS = (
    (0.07679180887372014, 0.45221843003412965),
    (0.7252559726962458, 0.7935153583617748),
    (0.8959044368600683, 1.0153583617747441),
)
UPSTREAM_HYSTERESIS_REGIONS = (
    (0.07679180887372014, 0.2303754266211604),
    (0.3156996587030717, 0.45221843003412965),
    (0.5546075085324232, 0.60580204778157),
    (0.7252559726962458, 0.7935153583617748),
    (0.8959044368600683, 1.0153583617747441),
)


def _provider(directory, *, seed=0):
    torch.manual_seed(seed)
    config = PyanNetConfig(
        lstm_hidden_size=8,
        lstm_num_layers=1,
        lstm_dropout=0.0,
        linear_hidden_size=8,
        linear_num_layers=1,
        chunk_duration_s=0.1,
        chunk_step_s=0.05,
    )
    model = PyanNet(config)
    save_safetensors(
        model.state_dict(),
        Path(directory) / "model.safetensors",
        metadata={"format": "voicehub-pyannet-v1"},
    )
    write_json_file(Path(directory) / "config.json", config.to_dict())
    return PyannoteVADForVoiceActivityDetection(
        PyannoteVADConfig(name_or_path=directory, batch_size=2),
        device="cpu",
    ).load()


def _stub_scores(provider, scores):
    provider._frame_output = lambda waveform: PyanNetFrameOutput(
        scores=torch.tensor(scores, dtype=torch.float32).unsqueeze(-1),
        frame_hop_samples=round(FRAME_STEP_S * 16_000),
        frame_length_samples=round(FRAME_STEP_S * 16_000),
        frame_start_samples=0,
        valid_samples=waveform.numel(),
        frame_step_s=FRAME_STEP_S,
        frame_duration_s=FRAME_STEP_S,
    )


class PyannoteVADPipelineDefaultsTests(unittest.TestCase):

    def test_defaults_are_the_pinned_pipeline_hyper_parameters(self):
        options = PyannoteVADConfig().inference_config
        self.assertEqual(options["onset"], PIPELINE_PARAMETERS["onset"])
        self.assertEqual(options["offset"], PIPELINE_PARAMETERS["offset"])
        self.assertEqual(options["speech_pad_ms"], 0)
        # Integer milliseconds are equivalent to the pipeline's float
        # seconds: unpadded region and gap lengths are whole multiples of the
        # frame step, and no multiple falls between the two values.
        for name, key in (
            ("min_speech_duration_ms", "min_duration_on"),
            ("min_silence_duration_ms", "min_duration_off"),
        ):
            low, high = sorted((options[name] / 1000.0, PIPELINE_PARAMETERS[key]))
            self.assertFalse(any(low <= frames * FRAME_STEP_S <= high for frames in range(1, 1_000)))

    def test_default_detection_matches_the_upstream_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = _provider(directory)
            _stub_scores(provider, SCORES)
            output = provider.detect(torch.zeros(16_800), sampling_rate=16_000)
        self.assertEqual(
            tuple((segment.start, segment.end) for segment in output.segments),
            UPSTREAM_PIPELINE_REGIONS,
        )

    def test_hysteresis_only_matches_upstream_binarize(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = _provider(directory)
            _stub_scores(provider, SCORES)
            output = provider.detect(
                torch.zeros(16_800),
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
            )
        self.assertEqual(
            tuple((segment.start, segment.end) for segment in output.segments),
            UPSTREAM_HYSTERESIS_REGIONS,
        )


class PyannoteVADResamplingTests(unittest.TestCase):

    def test_detect_scores_resampled_input_with_the_pyannote_filter(self):
        waveform = torch.randn(4_000, generator=torch.Generator().manual_seed(3))
        options = {
            "min_speech_duration_ms": 0,
            "min_silence_duration_ms": 0,
            "return_frames": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            provider = _provider(directory)
            resampled = provider.detect(waveform, sampling_rate=8_000, **options)
            reference = provider.detect(
                resample_waveform_hann(waveform, 8_000, 16_000),
                sampling_rate=16_000,
                **options,
            )
        self.assertEqual(resampled.probabilities, reference.probabilities)
        self.assertEqual(resampled.segments, reference.segments)


if __name__ == "__main__":
    unittest.main()
