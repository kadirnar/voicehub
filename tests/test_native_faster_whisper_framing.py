"""Regression tests for faster-whisper transcription framing.

Reference behavior: SYSTRAN/faster-whisper ``feature_extractor.py`` and
``WhisperModel.generate_segments`` / ``_split_segments_by_timestamps`` at
7b99be5376b41cd481dfc52e8caa00a497c3294e.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tests.test_native_whisper_provider import _tiny_artifact
from voicehub.models.asr_native.configuration import FasterWhisperConfig
from voicehub.models.asr_native.faster_whisper import FasterWhisperForSpeechRecognition

# Tiny-artifact token IDs (see ``_tokenizer_document``).
HELLO = 259
EOT = 261
ZH = 264
TIMESTAMP_BEGIN = 272


def _faster_whisper_log_mel(audio: np.ndarray, n_mels: int) -> np.ndarray:
    """Upstream ``FeatureExtractor.__call__`` (padding=160) in NumPy."""
    sr, n_fft, hop = 16_000, 400, 160
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mels = np.linspace(0.0, 45.245640471924965, n_mels + 2)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_mel = 1000.0 / f_sp
    log_t = mels >= min_log_mel
    freqs[log_t] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (mels[log_t] - min_log_mel))
    fdiff = np.diff(freqs)
    ramps = freqs.reshape(-1, 1) - fftfreqs.reshape(1, -1)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (freqs[2:n_mels + 2] - freqs[:n_mels]))[:, None]
    filters = weights.astype(np.float32)

    waveform = np.pad(audio.astype(np.float32), (0, hop))
    waveform = np.pad(waveform, (n_fft // 2, n_fft // 2), mode="reflect")
    frames = 1 + (len(waveform) - n_fft) // hop
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    columns = np.stack([waveform[i * hop:i * hop + n_fft] for i in range(frames)]) * window
    magnitudes = np.abs(np.fft.rfft(columns, axis=-1).T.astype(np.complex64)[:, :-1])**2
    log_spec = np.log10(np.clip(filters @ magnitudes, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


class _ScriptedGenerationAdapter:
    """Return scripted token windows and record each decoding request."""

    def __init__(self, token_set, windows):
        self.token_set = token_set
        self.windows = list(windows)
        self.calls = []
        self.detections = 0

    def _detect_languages(self, encoded, *, encoder_attention_mask):
        del encoded, encoder_attention_mask
        self.detections += 1
        return (ZH, )

    def generate(self, features, *, config):
        self.calls.append(SimpleNamespace(features=features.clone(), config=config))
        return SimpleNamespace(generated_sequences=torch.tensor([self.windows.pop(0) + [EOT, EOT]]))


class FasterWhisperFramingTests(unittest.TestCase):

    def _wrapper(self, root: Path) -> FasterWhisperForSpeechRecognition:
        _tiny_artifact(root)
        wrapper = FasterWhisperForSpeechRecognition(
            FasterWhisperConfig(name_or_path=root, compute_type="float32"),
            device="cpu",
        )
        wrapper.load()
        return wrapper

    def test_features_use_whole_recording_log_mel_and_log_mel_zero_padding(self):
        generator = torch.Generator().manual_seed(3)
        # A loud first half and a quiet second half: faster-whisper clamps
        # every window against the maximum of the whole recording.
        audio = torch.randn(1_500, generator=generator)
        audio[750:] *= 1e-4
        with tempfile.TemporaryDirectory() as directory:
            wrapper = self._wrapper(Path(directory))
            features = wrapper._recording_features(audio)
            window = wrapper._window_features(features, 4, 5)

        expected = _faster_whisper_log_mel(audio.numpy(), n_mels=4)
        self.assertEqual(tuple(features.shape), expected.shape)
        np.testing.assert_allclose(features.numpy(), expected, rtol=0, atol=2e-5)
        self.assertEqual(tuple(window.shape), (1, 4, 8))
        torch.testing.assert_close(window[0, :, :5], features[:, 4:9], rtol=0, atol=0)
        # ``pad_or_trim`` pads log-mel frames with 0.0, not with silence.
        self.assertTrue(torch.equal(window[0, :, 5:], torch.zeros(4, 3)))

    def test_timestamped_window_resumes_from_last_complete_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = self._wrapper(Path(directory))
            # 1,500 samples -> 9 content frames, decoded in 8-frame windows.
            # Window 1 closes "hello" at 0.04 s, then opens an unfinished
            # segment: upstream drops it and decodes again from 0.04 s.
            adapter = _ScriptedGenerationAdapter(
                wrapper.generation_adapter.token_set,
                [
                    [TIMESTAMP_BEGIN, HELLO, TIMESTAMP_BEGIN + 2, TIMESTAMP_BEGIN + 2, HELLO],
                    [TIMESTAMP_BEGIN, HELLO, TIMESTAMP_BEGIN + 1],
                ],
            )
            wrapper.generation_adapter = adapter
            audio = torch.linspace(-0.5, 0.5, 1_500)
            features = wrapper._recording_features(audio)
            result = wrapper.transcribe(
                audio,
                sampling_rate=16_000,
                language="en",
                return_timestamps=True,
            )

        self.assertEqual(len(adapter.calls), 2)
        torch.testing.assert_close(adapter.calls[0].features[0], features[:, :8], rtol=0, atol=0)
        torch.testing.assert_close(adapter.calls[1].features[0, :, :5], features[:, 4:9], rtol=0, atol=0)
        self.assertEqual(result.text, "hellohello")
        self.assertEqual(
            [(segment.text, round(segment.start, 9), round(segment.end, 9)) for segment in result.segments],
            [("hello", 0.0, 0.04), ("hello", 0.04, 0.06)],
        )
        self.assertEqual(result.language, "en")

    def test_untimestamped_windows_advance_by_window_and_reuse_detected_language(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = self._wrapper(Path(directory))
            adapter = _ScriptedGenerationAdapter(
                wrapper.generation_adapter.token_set,
                [[HELLO], [HELLO]],
            )
            wrapper.generation_adapter = adapter
            result = wrapper.transcribe(torch.linspace(-0.5, 0.5, 1_500), sampling_rate=16_000)

        self.assertEqual(adapter.detections, 1)
        self.assertEqual([call.config.language for call in adapter.calls], ["zh", "zh"])
        self.assertEqual([call.config.return_timestamps for call in adapter.calls], [False, False])
        self.assertEqual(result.text, "hellohello")
        self.assertEqual(result.segments, ())
        self.assertEqual(result.language, "zh")

    def test_split_window_matches_upstream_seek_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = self._wrapper(Path(directory))
            single_ending = wrapper._split_window(
                [TIMESTAMP_BEGIN, HELLO, TIMESTAMP_BEGIN + 3],
                seek=100,
                segment_size=8,
            )
            closed_pairs = wrapper._split_window(
                [TIMESTAMP_BEGIN, HELLO, TIMESTAMP_BEGIN + 1, TIMESTAMP_BEGIN + 1, HELLO, TIMESTAMP_BEGIN + 3],
                seek=100,
                segment_size=8,
            )
            no_timestamps = wrapper._split_window([HELLO], seek=100, segment_size=8)

        def rounded(split):
            return [(round(start, 9), round(end, 9), tokens) for start, end, tokens in split[0]], split[1]

        self.assertEqual(
            rounded(single_ending),
            ([(1.0, 1.06, [TIMESTAMP_BEGIN, HELLO, TIMESTAMP_BEGIN + 3])], 108),
        )
        # Closed pairs ending with a timestamp keep everything and still
        # advance by the whole window; an unfinished tail resumes earlier.
        self.assertEqual([segment[:2] for segment in rounded(closed_pairs)[0]], [(1.0, 1.02), (1.02, 1.06)])
        self.assertEqual(closed_pairs[1], 108)
        self.assertEqual(rounded(no_timestamps), ([(1.0, 1.08, [HELLO])], 108))


    def test_faster_whisper_model_names_resolve_to_safetensors_sources(self):
        expected = {
            "small": "openai/whisper-small",
            "large": "openai/whisper-large-v3",
            "Systran/faster-whisper-small": "openai/whisper-small",
            "Systran/faster-whisper-large-v1": "openai/whisper-large",
            "mobiuslabsgmbh/faster-whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
            "openai/whisper-small": "openai/whisper-small",
        }
        for name, source in expected.items():
            with self.subTest(name=name):
                model = FasterWhisperForSpeechRecognition(
                    FasterWhisperConfig(name_or_path=name, compute_type="float32"),
                    device="cpu",
                )
                self.assertEqual(model.config.name_or_path, source)
                self.assertEqual(
                    FasterWhisperForSpeechRecognition(model_path=name, device="cpu").config.name_or_path,
                    source,
                )


if __name__ == "__main__":
    unittest.main()
