from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import torch

from voicehub.models.llasa.modeling import LlasaForTextToSpeech
from voicehub.processing import (
    NativeAudio,
    decode_pcm_wave,
    load_native_audio,
    load_pcm_wave,
    normalize_waveform,
    resample_waveform,
    resample_waveform_kaiser,
    save_pcm_wave,
)
from voicehub.training.data import load_audio_tensor


class NativeWaveformTests(unittest.TestCase):

    def test_integer_audio_is_scaled_and_downmixed(self):
        stereo = torch.tensor(
            [[-32768, 32767], [0, 16384], [32767, -32768]],
            dtype=torch.int16,
        )

        waveform = normalize_waveform(stereo)

        self.assertEqual(tuple(waveform.shape), (3, ))
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertTrue(torch.isfinite(waveform).all())
        self.assertLess(waveform.abs().max().item(), 0.26)

    def test_resampling_is_differentiable_and_request_local(self):
        waveform = torch.linspace(-1.0, 1.0, 101, requires_grad=True)

        result = resample_waveform(
            waveform,
            10_000,
            16_000,
            filter_width=8,
            chunk_size=17,
        )

        self.assertEqual(tuple(result.shape), (162, ))
        self.assertTrue(torch.isfinite(result).all())
        result.square().mean().backward()
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())

    def test_kaiser_resampling_matches_the_upstream_polyphase_recipe(self):
        waveform = torch.arange(
            12,
            dtype=torch.float32,
        ).reshape(2, 6).requires_grad_(True)

        result = resample_waveform_kaiser(
            waveform,
            6,
            4,
            lowpass_filter_width=4,
            rolloff=0.9,
            beta=8.0,
        )

        expected = torch.tensor([
            [0.1298657507, 1.4262738228, 3.0711078644, 4.5626463890],
            [4.9353113174, 7.7400221825, 9.0446834564, 10.6009693146],
        ])
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
        result.square().mean().backward()
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())

    def test_pcm_wave_loading_handles_24_bit_and_downmixes(self):
        frames = (
            (-(2**23), 2**23 - 1),
            (0, 2**22),
            (2**23 - 1, -(2**23)),
        )

        def encode_24(value):
            return int(value % 2**24).to_bytes(3, "little", signed=False)

        payload = b"".join(encode_24(sample) for frame in frames for sample in frame)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(2)
                stream.setsampwidth(3)
                stream.setframerate(8_000)
                stream.writeframes(payload)

            audio = load_native_audio(
                path,
                target_sampling_rate=16_000,
            )

        self.assertIsInstance(audio, NativeAudio)
        self.assertEqual(audio.sampling_rate, 16_000)
        self.assertEqual(audio.waveform.shape[-1], 6)
        self.assertTrue(torch.isfinite(audio.waveform).all())

    def test_mapping_rates_are_validated(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            load_native_audio(
                {
                    "array": [0.0, 1.0],
                    "sampling_rate": 8_000,
                },
                sampling_rate=16_000,
            )

    def test_valid_sample_prefix_is_trimmed_before_resampling_for_files(self):
        source = torch.linspace(-0.5, 0.5, 80)
        with tempfile.TemporaryDirectory() as directory:
            path = save_pcm_wave(
                Path(directory) / "trimmed.wav",
                source,
                8_000,
            )
            restored = load_native_audio(
                path,
                num_samples=40,
                target_sampling_rate=16_000,
            )
            with self.assertRaisesRegex(ValueError, "exceeds"):
                load_native_audio(
                    path,
                    num_samples=81,
                    target_sampling_rate=16_000,
                )

        self.assertEqual(restored.sampling_rate, 16_000)
        self.assertEqual(restored.waveform.numel(), 80)

    def test_unsupported_file_formats_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.mp3"
            path.write_bytes(b"not an mp3")

            with self.assertRaisesRegex(ValueError, "PCM WAVE"):
                load_native_audio(path)

    def test_pcm_wave_writer_round_trips_channel_first_audio(self):
        waveform = torch.tensor(
            [
                [-1.0, -0.5, 0.0, 0.5, 1.0],
                [1.0, 0.5, 0.0, -0.5, -1.0],
            ],
            dtype=torch.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_pcm_wave(
                Path(directory) / "nested" / "stereo.wav",
                waveform,
                16_000,
            )
            with wave.open(str(path), "rb") as stream:
                self.assertEqual(stream.getnchannels(), 2)
                self.assertEqual(stream.getsampwidth(), 2)
                self.assertEqual(stream.getframerate(), 16_000)
                self.assertEqual(stream.getnframes(), 5)
            # Native loading intentionally downmixes files to mono.
            restored = load_native_audio(path)
            channels, sample_rate = load_pcm_wave(
                path,
                preserve_channels=True,
            )

        torch.testing.assert_close(
            restored.waveform,
            torch.zeros(5),
            rtol=0,
            atol=1 / 32767,
        )
        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(tuple(channels.shape), (2, 5))
        torch.testing.assert_close(
            channels,
            waveform,
            rtol=0,
            atol=1 / 32767,
        )

    def test_in_memory_pcm_wave_decoder_is_bounded(self):
        source = torch.linspace(-0.75, 0.75, 12)
        with tempfile.TemporaryDirectory() as directory:
            path = save_pcm_wave(
                Path(directory) / "memory.wav",
                source,
                12_000,
            )
            payload = path.read_bytes()
            restored, sample_rate = decode_pcm_wave(payload)

        self.assertEqual(sample_rate, 12_000)
        torch.testing.assert_close(
            restored,
            source,
            rtol=0,
            atol=1 / 32767,
        )
        with self.assertRaisesRegex(ValueError, "limit"):
            decode_pcm_wave(payload, max_bytes=len(payload) - 1)

    def test_training_audio_loader_reuses_the_native_waveform_boundary(self):
        source = torch.linspace(-0.5, 0.5, 80)
        with tempfile.TemporaryDirectory() as directory:
            path = save_pcm_wave(
                Path(directory) / "training.wav",
                source,
                8_000,
            )
            loaded = load_audio_tensor(
                str(path),
                sample_rate=16_000,
                model_type="test",
                install_extra="training",
            )

        self.assertEqual(loaded.shape, (160, ))
        self.assertEqual(loaded.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loaded).all())

    def test_llasa_reference_loading_uses_the_native_pcm_boundary(self):
        model = LlasaForTextToSpeech(device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = save_pcm_wave(
                Path(directory) / "reference.wav",
                torch.linspace(-0.5, 0.5, 8),
                8_000,
            )
            loaded = model._load_reference(str(path))

        self.assertEqual(tuple(loaded.shape), (1, 16))
        self.assertEqual(loaded.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loaded).all())

    def test_pcm_wave_channel_preservation_flag_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_pcm_wave(
                Path(directory) / "mono.wav",
                torch.zeros(4),
                8_000,
            )
            with self.assertRaisesRegex(TypeError, "boolean"):
                load_pcm_wave(path, preserve_channels=1)

    def test_pcm_wave_writer_rejects_nonfinite_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "finite"):
                save_pcm_wave(
                    Path(directory) / "invalid.wav",
                    torch.tensor([0.0, float("nan")]),
                    16_000,
                )


if __name__ == "__main__":
    unittest.main()
