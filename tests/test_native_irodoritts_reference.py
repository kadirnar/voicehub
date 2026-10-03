"""Reference-audio frontend parity checks for native Irodori-TTS.

The expected loudness values were measured with the original runtime's
``audiotools`` 0.7.2 ``AudioSignal.loudness()`` (pyloudnorm K-weighting)
on the analytic signals built below.
"""

from __future__ import annotations

import math
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from tests.test_native_irodoritts import _tiny_config, _write_tokenizer
from voicehub.architectures.irodoritts.codec import IrodoriDACVAECodec
from voicehub.processing.loudness import integrated_loudness, normalize_loudness
from voicehub.architectures.irodoritts.modeling import TextToLatentRFDiT
from voicehub.architectures.irodoritts.runtime import InferenceRuntime, SamplingRequest
from voicehub.processing.waveform import resample_waveform_hann

RATE = 48_000


def _sine(frequency: float, seconds: float, amplitude: float) -> torch.Tensor:
    time = torch.arange(int(seconds * RATE), dtype=torch.float64) / RATE
    return amplitude * torch.sin(2 * math.pi * frequency * time)


def _reference_signals() -> dict[str, tuple[torch.Tensor, float]]:
    chirp_time = torch.arange(int(1.37 * RATE), dtype=torch.float64) / RATE
    return {
        "sine_then_silence": (
            torch.cat([_sine(1000.0, 1.0, 0.3),
                       torch.zeros(2 * RATE, dtype=torch.float64)]).float(),
            -14.208349227905273,
        ),
        # The last 400 ms gating block is partial and must be zero-padded.
        "chirp_partial_block": (
            (0.2 * torch.sin(2 * math.pi * (50 * chirp_time + 4000 * chirp_time**2))).float(),
            -13.97457218170166,
        ),
        # Shorter than 0.5 s: audiotools zero-pads before metering.
        "short_0p2s": (_sine(440.0, 0.2, 0.05).float(), -34.016090393066406),
        "quiet_and_loud": (
            torch.cat([_sine(200.0, 1.0, 0.001), _sine(3000.0, 1.0, 0.5)]).float(),
            -6.662816047668457,
        ),
    }


class _RecordingCodecGraph(nn.Module):
    """Minimal DACVAE stand-in that records the encoder input."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen: list[torch.Tensor] = []
        self.quantizer = nn.Module()
        self.quantizer.in_proj = lambda encoded: torch.cat([encoded, encoded], dim=1)

    def encoder(self, waveform: torch.Tensor) -> torch.Tensor:
        self.seen.append(waveform.detach().clone())
        frames = waveform.shape[-1] // IrodoriDACVAECodec.hop_length
        return waveform[..., :frames].expand(-1, 32, -1).contiguous()


def _recording_codec() -> tuple[IrodoriDACVAECodec, _RecordingCodecGraph]:
    codec = IrodoriDACVAECodec.__new__(IrodoriDACVAECodec)
    graph = _RecordingCodecGraph()
    codec.model = graph
    codec.normalize_db = -16.0
    return codec, graph


def _write_float_wave(path: Path, samples: torch.Tensor, sample_rate: int) -> None:
    payload = samples.to(torch.float32).numpy().tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    path.write_bytes(
        b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(payload)) + b"WAVE" + b"fmt " +
        struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload)


class IrodoriReferenceFrontendTests(unittest.TestCase):

    def test_loudness_matches_released_audiotools_meter(self):
        for name, (signal, expected) in _reference_signals().items():
            with self.subTest(signal=name):
                measured = integrated_loudness(signal[None, None], RATE)
                self.assertEqual(measured.dtype, torch.float32)
                self.assertAlmostEqual(measured.item(), expected, delta=2e-5)
        silence = integrated_loudness(torch.zeros(1, 1, RATE), RATE)
        self.assertEqual(silence.item(), -70.0)

    def test_loudness_normalization_reaches_target_and_limits_peak(self):
        signal, _ = _reference_signals()["quiet_and_loud"]
        normalized = normalize_loudness(signal, RATE, -16.0)
        self.assertAlmostEqual(integrated_loudness(normalized[None, None], RATE).item(), -16.0, delta=1e-3)
        boosted = normalize_loudness(signal, RATE, 6.0)
        self.assertLessEqual(boosted.abs().max().item(), 1.0)

    def test_codec_frontend_follows_released_normalization_options(self):
        codec, graph = _recording_codec()
        signal, _ = _reference_signals()["sine_then_silence"]
        loud = signal * 5.0  # peak 1.5

        codec.encode_waveform(loud)
        normalized = graph.seen[-1][0, 0, :loud.numel()]
        self.assertAlmostEqual(integrated_loudness(normalized[None, None], RATE).item(), -16.0, delta=1e-3)

        # `None` disables normalization instead of selecting the default.
        codec.encode_waveform(loud, normalize_db=None)
        torch.testing.assert_close(graph.seen[-1][0, 0, :loud.numel()], loud, rtol=0, atol=0)

        codec.encode_waveform(loud, normalize_db=None, ensure_max=True)
        limited = loud * (1.0 / float(loud.abs().max()))
        torch.testing.assert_close(graph.seen[-1][0, 0, :loud.numel()], limited, rtol=0, atol=0)

        # Normalization already limits peaks, so `ensure_max` is ignored then.
        codec.encode_waveform(loud, normalize_db=-16.0, ensure_max=True)
        torch.testing.assert_close(graph.seen[-1], graph.seen[0], rtol=0, atol=0)

    def test_codec_resamples_references_like_torchaudio_functional(self):
        codec, graph = _recording_codec()
        reference = _sine(220.0, 0.75, 0.2).float()[::3].contiguous()  # 16 kHz
        codec.encode_waveform(reference, sample_rate=16_000, normalize_db=None)
        expected = resample_waveform_hann(reference[None, None], 16_000, RATE)
        torch.testing.assert_close(graph.seen[-1][..., :expected.shape[-1]], expected, rtol=0, atol=0)

    def test_runtime_reads_float_wave_references_and_forwards_frontend_options(self):
        calls = []

        class _Codec:
            sample_rate = RATE
            hop_length = 1920
            device = torch.device("cpu")

            def encode_waveform(self, waveform, **kwargs):
                calls.append((waveform, kwargs))
                return torch.zeros(1, 40, 4)

        config = _tiny_config(use_duration_predictor=False)
        with tempfile.TemporaryDirectory() as directory:
            runtime = InferenceRuntime(
                model=TextToLatentRFDiT(config).eval(),
                model_cfg=config,
                tokenizer=_write_tokenizer(Path(directory)),
                codec=_Codec(),
                model_device="cpu",
            )
            path = Path(directory) / "reference.wav"
            samples = torch.linspace(-0.5, 0.5, 16_000)
            _write_float_wave(path, samples, 16_000)
            latent, mask, _, _ = runtime._reference(
                SamplingRequest(
                    text="known",
                    ref_wav=str(path),
                    ref_normalize_db=None,
                    ref_ensure_max=False,
                    max_ref_seconds=0.5,
                ),
                batch_size=1,
            )
        waveform, options = calls[0]
        torch.testing.assert_close(waveform, samples[:8_000], rtol=0, atol=0)
        self.assertEqual(options, {"sample_rate": 16_000, "normalize_db": None, "ensure_max": False})
        # The reference latent is bounded by max_ref_seconds, like the original runtime.
        self.assertEqual(latent.shape[1], math.ceil(0.5 * RATE / 1920))
        self.assertTrue(mask.all())


if __name__ == "__main__":
    unittest.main()
