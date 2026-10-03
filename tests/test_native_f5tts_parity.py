"""Regression tests pinning native F5-TTS to the released inference recipe.

Expected values were produced with the original repository
(SWivid/F5-TTS 9c614e9, pydub 0.25.1, rjieba 0.2.1, torchaudio 2.8).
"""

from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voicehub.architectures.f5tts.audio import (
    F5MelSpectrogram,
    _ratecv_frames,
    normalize_reference_rms,
    preprocess_reference_audio,
    pydub_sample_width,
    remove_generated_silence,
)
from voicehub.architectures.f5tts.configuration import F5TTSArchitectureConfig
from voicehub.architectures.f5tts.frontend import F5Vocabulary, NativeF5TextFrontend, character_tokens
from voicehub.architectures.f5tts.modeling import F5ConditionalFlowMatcher
from voicehub.architectures.f5tts.modules import TextEmbedding
from voicehub.architectures.f5tts.runtime import (
    NativeF5TTSRuntime,
    chunk_text,
    normalize_reference_text,
    prompt_reference_text,
)
from voicehub.models.f5tts.inference import F5TTSConfig, F5TTSForTextToSpeech
from voicehub.processing.waveform import resample_waveform_hann

try:
    import torchaudio
except ImportError:  # pragma: no cover - optional reference implementation
    torchaudio = None


def _tiny_config() -> F5TTSArchitectureConfig:
    return F5TTSArchitectureConfig(
        model_name="test-f5",
        mel_dim=8,
        dim=32,
        depth=2,
        heads=4,
        dim_head=8,
        text_dim=16,
        text_num_embeds=12,
        conv_layers=1,
        n_fft=32,
        win_length=32,
        hop_length=8,
        sample_rate=8_000,
        dropout=0.0,
    )


class TinyVocoder(torch.nn.Module):

    def __init__(self, input_channels: int = 8, hop_length: int = 8):
        super().__init__()
        self.backbone = SimpleNamespace(input_channels=input_channels)
        self.projection = torch.nn.Conv1d(input_channels, hop_length, kernel_size=1)

    def decode(self, features):
        return self.projection(features).transpose(1, 2).reshape(features.shape[0], -1)


def _synthetic_reference() -> torch.Tensor:
    """16 kHz 16-bit PCM: silence, loud tone, pause, tone, -48 dBFS tail."""
    rate = 16_000

    def tone(seconds, amplitude, frequency):
        positions = torch.arange(int(rate * seconds), dtype=torch.float64) / rate
        return amplitude * torch.sin(2 * math.pi * frequency * positions)

    signal = torch.cat((
        torch.zeros(int(rate * 0.3), dtype=torch.float64),
        tone(1.0, 0.2, 220.0),
        torch.zeros(int(rate * 1.5), dtype=torch.float64),
        tone(2.0, 0.05, 330.0),
        tone(0.2, 0.004, 500.0),
        torch.zeros(int(rate * 0.25), dtype=torch.float64),
    ))
    # np.round(x * 32767) as written by the reference, decoded as int / 32768.
    return (signal * 32_767).round().float() / 32_768


def _write_pcm16(path: Path, waveform: torch.Tensor, rate: int) -> None:
    samples = (waveform * 32_768).round().clamp(-32_768, 32_767).to(torch.int16)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.numpy().tobytes())


class ReleasedTextFrontendTests(unittest.TestCase):

    def test_word_boundary_spaces_match_convert_char_to_pinyin(self):
        expected = {
            "The quick brown fox jumps over the lazy dog.": "The quick brown fox jumps over the lazy dog.",
            "well-known state-of-the-art": "well- known state- of- the- art",
            "Hello,world!How are you?": "Hello, world! How are you?",
            "(C++) and c#": "( C++) and c#",
            "Wait... what": "Wait ... what",
            "Café naïve": "Café naï ve",
            "It costs $3.50, or 3.5% off": "It costs $ 3.50, or 3.5% off",
            "end.Start": "end. Start",
            "a\r\nb": "a \r\nb",
            "AT&Tx": "AT&Tx",
            "__init__": "__ init __",
            "I don't really care what you call me.": "I don't really care what you call me.",
        }
        for text, released in expected.items():
            with self.subTest(text=text):
                self.assertEqual("".join(character_tokens(text)), released)

    def test_frontend_applies_released_spacing_and_rejects_cjk_ideographs(self):
        vocabulary = F5Vocabulary((" ", "-", "a", "b", "c"))
        frontend = NativeF5TextFrontend(vocabulary)
        self.assertEqual(frontend.normalize("ab-ca"), ("a", "b", "-", " ", "c", "a"))
        for text in ("你好", "豈", "\U00020000"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "pinyin-with-tone"):
                frontend.normalize(text)


class ReleasedTextPreparationTests(unittest.TestCase):

    def test_reference_text_gets_the_released_trailing_spaces(self):
        self.assertEqual(prompt_reference_text("his matter."), "his matter. ")
        self.assertEqual(normalize_reference_text("his matter."), "his matter.  ")
        self.assertEqual(normalize_reference_text("no stop"), "no stop.  ")
        self.assertEqual(normalize_reference_text("already. "), "already.  ")
        self.assertEqual(normalize_reference_text("你好。"), "你好。")
        with self.assertRaisesRegex(ValueError, "reference_text"):
            normalize_reference_text("   ")

    def test_chunk_budget_ignores_the_joining_space(self):
        # 6 + 4 bytes fit a 10-byte budget; the joining space is not counted.
        self.assertEqual(chunk_text("aaaa, bbbb", maximum_bytes=10), ("aaaa, bbbb", ))
        self.assertEqual(chunk_text("aaaa, bbbb", maximum_bytes=9), ("aaaa,", "bbbb"))


class ReleasedReferenceAudioTests(unittest.TestCase):

    def test_reference_is_trimmed_like_pydub(self):
        waveform = _synthetic_reference()
        prepared = preprocess_reference_audio(waveform, 16_000, sample_width=2)
        # pydub keeps both phrases (the 1.5 s pause is shorter than 2 x 1 s of
        # kept silence), trims 300 ms / 450 ms below -42 dBFS, and appends
        # 50 ms of 11025 Hz silence that ratecv turns into 799 frames.
        self.assertEqual(prepared.numel(), 72_799)
        self.assertTrue(torch.equal(prepared[:-799], waveform[4_800:76_800]))
        self.assertTrue(torch.equal(prepared[-799:], torch.zeros(799)))

    def test_channels_are_measured_together_and_kept(self):
        waveform = _synthetic_reference()
        stereo = torch.stack((waveform, waveform))
        prepared = preprocess_reference_audio(stereo, 16_000, sample_width=2)
        self.assertEqual(tuple(prepared.shape), (2, 72_799))

    def test_appended_silence_matches_audioop_ratecv(self):
        expected = {12_000: 599, 16_000: 799, 22_050: 1_101, 24_000: 1_198, 44_100: 2_201, 48_000: 2_395}
        for rate, frames in expected.items():
            with self.subTest(rate=rate):
                self.assertEqual(_ratecv_frames(551, 11_025, rate), frames)

    def test_generated_silence_removal_matches_pydub(self):
        waveform = _synthetic_reference()
        trimmed = remove_generated_silence(waveform, 16_000)
        # pydub.split_on_silence(..., keep_silence=500) yields 1800 + 2950 ms.
        self.assertEqual(trimmed.numel(), (1_800 + 2_950) * 16)
        self.assertTrue(torch.equal(trimmed[:28_800], waveform[:28_800]))

    def test_rms_normalization_uses_the_released_arithmetic(self):
        waveform = _synthetic_reference()
        normalized, rms = normalize_reference_rms(waveform)
        rms_tensor = torch.sqrt(torch.mean(torch.square(waveform.unsqueeze(0))))
        self.assertEqual(rms, rms_tensor.item())
        self.assertTrue(torch.equal(normalized, waveform * 0.1 / rms_tensor))

    def test_sample_width_follows_pydub_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.wav"
            _write_pcm16(path, _synthetic_reference()[:1_000], 16_000)
            self.assertEqual(pydub_sample_width(path), 2)
            float_path = Path(directory) / "float.wav"
            payload = path.read_bytes()
            # Rewrite the header as 32-bit IEEE float (format tag 3).
            float_path.write_bytes(
                payload[:20] + (3).to_bytes(2, "little") + payload[22:34] + (32).to_bytes(2, "little") +
                payload[36:])
            self.assertEqual(pydub_sample_width(float_path), 4)

    @unittest.skipIf(torchaudio is None, "torchaudio is not installed")
    def test_resampler_matches_torchaudio_default(self):
        generator = torch.Generator().manual_seed(0)
        waveform = torch.randn(1, 7_919, generator=generator)
        for source, target in ((16_000, 24_000), (44_100, 24_000), (22_050, 24_000), (48_000, 24_000)):
            with self.subTest(source=source):
                expected = torchaudio.transforms.Resample(source, target)(waveform)
                resampled = resample_waveform_hann(waveform, source, target, match="transform")
                self.assertTrue(torch.equal(resampled, expected))


class ReleasedSamplingTests(unittest.TestCase):

    def test_cpu_built_buffers_do_not_depend_on_the_device_context(self):
        reference_mel = F5MelSpectrogram()
        reference_text = TextEmbedding(
            12, 16, mask_padding=True, average_upsampling=False, conv_layers=1, conv_mult=2)
        with torch.device("meta"):
            mel = F5MelSpectrogram()
            text = TextEmbedding(
                12, 16, mask_padding=True, average_upsampling=False, conv_layers=1, conv_mult=2)
        self.assertEqual(mel.window.device.type, "cpu")
        self.assertTrue(torch.equal(mel.window, reference_mel.window))
        self.assertTrue(torch.equal(mel.filter_bank, reference_mel.filter_bank))
        self.assertTrue(torch.equal(text.freqs_cis, reference_text.freqs_cis))

    def test_runtime_conditions_on_every_prompt_mel_frame(self):
        torch.manual_seed(0)
        flow = F5ConditionalFlowMatcher(_tiny_config()).eval()
        runtime = NativeF5TTSRuntime(
            flow_model=flow,
            vocoder=TinyVocoder(),
            frontend=NativeF5TextFrontend(
                F5Vocabulary((" ", ".", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"))),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.wav"
            _write_pcm16(path, _synthetic_reference()[:16_000] * 4, 8_000)
            with patch.object(flow, "sample", wraps=flow.sample) as sample:
                runtime.infer(ref_file=path, ref_text="abc.", gen_text="abcdefghij", seed=1, nfe_step=2)
        reference, *_ = sample.call_args.args
        # The released recipe passes no ``lens``: the mask covers all
        # ``samples // hop + 1`` mel frames of the centred STFT.
        self.assertIsNone(sample.call_args.kwargs.get("lengths"))
        self.assertEqual(reference.dtype, torch.float32)
        self.assertEqual(reference.ndim, 2)


class ReducedPrecisionLoadingTests(unittest.TestCase):

    def test_half_precision_keeps_released_float32_frontend_and_vocoder(self):
        architecture = _tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            vocabulary = Path(directory) / "vocab.txt"
            vocabulary.write_text("".join(f"{token}\n" for token in " .abcdefghij"), encoding="utf-8")
            artifacts = SimpleNamespace(
                checkpoint=Path(directory) / "model.safetensors",
                vocabulary=vocabulary,
                vocoder=Path(directory) / "vocos.safetensors",
            )
            model = F5TTSForTextToSpeech(
                F5TTSConfig(
                    architecture=architecture.to_dict(),
                    model_name=architecture.model_name,
                    torch_dtype="float16",
                    allow_unvalidated_reduced_precision_inference=True,
                ),
                device="cpu",
            )
            # CPU maps half precision to float32; force the CUDA resolution.
            with (
                    patch("voicehub.models.f5tts.inference.resolve_torch_dtype", return_value=torch.float16),
                    patch("voicehub.models.f5tts.inference.resolve_f5tts_artifacts", return_value=artifacts),
                    patch("voicehub.models.f5tts.inference.load_f5tts_checkpoint"),
                    patch("voicehub.models.f5tts.inference.load_vocos_checkpoint"),
                    patch("voicehub.models.f5tts.inference.NativeVocos", TinyVocoder),
            ):
                model._load_pretrained_model()
        runtime = model.model
        self.assertEqual(next(runtime.ema_model.transformer.parameters()).dtype, torch.float16)
        self.assertEqual(runtime.ema_model.mel_spec.window.dtype, torch.float32)
        self.assertEqual(runtime.ema_model.mel_spec.filter_bank.dtype, torch.float32)
        # Not merely re-widened from half precision.
        self.assertTrue(
            torch.equal(runtime.ema_model.mel_spec.window,
                        F5MelSpectrogram(n_fft=32, win_length=32).window))
        self.assertTrue(
            torch.equal(
                runtime.ema_model.mel_spec.filter_bank,
                F5MelSpectrogram(sample_rate=8_000, n_fft=32, hop_length=8, win_length=32,
                                 n_mels=8).filter_bank))
        self.assertEqual(next(runtime.vocoder.parameters()).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
