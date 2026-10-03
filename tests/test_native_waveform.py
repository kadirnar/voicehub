from __future__ import annotations

import importlib.util
import itertools
import tempfile
import unittest
import wave
from pathlib import Path

import torch

from voicehub.models.llasa.inference import LlasaForTextToSpeech
from voicehub.processing import (
    NativeAudio,
    decode_pcm_wave,
    load_native_audio,
    load_pcm_wave,
    normalize_waveform,
    resample_waveform,
    resample_waveform_hann,
    resample_waveform_kaiser,
    resample_waveform_kaiser_best,
    save_pcm_wave,
)
from voicehub.training.data import load_audio_tensor

TORCHAUDIO_AVAILABLE = importlib.util.find_spec("torchaudio") is not None
HANN_RATES = (8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000)


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

    def test_hann_resampling_reproduces_torchaudio_reference_values(self):
        # Reference values: torchaudio 2.8 functional.resample and
        # transforms.Resample(24_000, 16_000) on the same input.
        waveform = torch.sin(torch.arange(40, dtype=torch.float32) * 0.7).reshape(2, 20)
        waveform = waveform * torch.tensor([[1.0], [0.5]])
        expected_down = torch.tensor([
            [
                0.0717478022, 0.8524141908, 0.8707162142, -0.0125372773, -0.8699551821, -0.8599711657,
                0.0168276150, 0.8764830232, 0.8563733697, -0.0327278748, -0.8594605923, -0.8959908485,
                0.1374603957, 0.3783452809
            ],
            [
                0.4162121117, 0.3320973814, -0.2035598010, -0.4904620647, -0.3044177592, 0.1952179372,
                0.4967935085, 0.2993372679, -0.1987964362, -0.4983800352, -0.2914739549, 0.1921815574,
                0.5212510228, 0.2016496360
            ],
        ])
        expected_up = torch.tensor([
            0.0018438583, 0.4023151994, 0.8291160464, 0.9869054556, 0.9461240172, 0.7283859849, 0.3354935646,
            -0.1252679676, -0.5574275851, -0.8716647625, -0.9972223043, -0.9190559387, -0.6293591857,
            -0.1907602698, 0.2034675181, 0.6615770459, 1.0498452187, 0.7015190125
        ])
        up_source = torch.sin(torch.arange(12, dtype=torch.float32) * 0.7)
        for match in ("functional", "transform"):
            with self.subTest(match=match):
                down = resample_waveform_hann(waveform, 24_000, 16_000, match=match)
                torch.testing.assert_close(down, expected_down, rtol=0, atol=1e-6)
                up = resample_waveform_hann(up_source, 16_000, 24_000, match=match)
                torch.testing.assert_close(up, expected_up, rtol=0, atol=1e-6)
        self.assertIs(resample_waveform_hann(waveform, 16_000, 16_000), waveform)

    def test_hann_resampling_keeps_torchaudio_output_length_rounding(self):
        # torchaudio computes ceil(new * length / orig) through a float32
        # tensor: 441 * 47561 / 320 = 65545.003125 becomes 65545, not 65546.
        # One extra reference sample changed XTTS greedy output entirely.
        for match in ("functional", "transform"):
            with self.subTest(match=match):
                self.assertEqual(
                    resample_waveform_hann(torch.zeros(47_561), 16_000, 22_050, match=match).shape[-1],
                    65_545,
                )
                self.assertEqual(
                    resample_waveform_hann(torch.zeros(1, 90_569), 22_050, 16_000, match=match).shape[-1],
                    65_719,
                )
        self.assertEqual(resample_waveform_hann(torch.zeros(77_040), 16_000, 22_050).shape[-1], 106_171)

    def test_hann_resampling_keeps_leading_dimensions_and_gradients(self):
        waveform = torch.randn(2, 3, 37, dtype=torch.float64, requires_grad=True)
        result = resample_waveform_hann(waveform, 44_100, 16_000)
        self.assertEqual(tuple(result.shape), (2, 3, 14))
        self.assertEqual(result.dtype, torch.float64)
        torch.testing.assert_close(
            result[1, 2],
            resample_waveform_hann(waveform[1:2, 2:3], 44_100, 16_000)[0, 0])
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(waveform.grad).all())

    def test_hann_resampling_rejects_invalid_arguments(self):
        waveform = torch.zeros(8)
        with self.assertRaisesRegex(ValueError, "match"):
            resample_waveform_hann(waveform, 16_000, 24_000, match="kaiser")
        with self.assertRaisesRegex(ValueError, "lowpass_filter_width"):
            resample_waveform_hann(waveform, 16_000, 24_000, lowpass_filter_width=0)
        with self.assertRaisesRegex(ValueError, "rolloff"):
            resample_waveform_hann(waveform, 16_000, 24_000, rolloff=1.5)
        with self.assertRaisesRegex(ValueError, "source_rate"):
            resample_waveform_hann(waveform, 16_000.0, 24_000)
        with self.assertRaisesRegex(TypeError, "floating-point"):
            resample_waveform_hann(torch.zeros(8, dtype=torch.int16), 16_000, 24_000)
        with self.assertRaisesRegex(ValueError, "empty"):
            resample_waveform_hann(torch.zeros(0), 16_000, 24_000)

    def _assert_hann_matches_torchaudio(self, device, dtypes):
        from torchaudio.functional import resample
        from torchaudio.transforms import Resample

        generator = torch.Generator().manual_seed(0)
        lengths = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 160, 1_001)
        shapes = ((), (2, ), (2, 3))
        settings = ((6, 0.99), (16, 0.85), (64, 0.9475937167399596))
        rate_pairs = tuple(itertools.permutations(HANN_RATES, 2))
        for dtype, (source_rate, target_rate) in itertools.product(dtypes, rate_pairs):
            for index, length in enumerate(lengths):
                lowpass_filter_width, rolloff = settings[index % len(settings)]
                waveform = torch.randn(
                    *shapes[index % len(shapes)], length, generator=generator, dtype=torch.float64)
                waveform = waveform.to(device=device, dtype=dtype)
                options = {"lowpass_filter_width": lowpass_filter_width, "rolloff": rolloff}
                transform = Resample(source_rate, target_rate, **options).to(device=device, dtype=dtype)
                for match, expected in (
                    ("functional", resample(waveform, source_rate, target_rate, **options)),
                    ("transform", transform(waveform)),
                ):
                    actual = resample_waveform_hann(
                        waveform, source_rate, target_rate, match=match, **options)
                    if actual.shape != expected.shape or not torch.equal(actual, expected):
                        self.fail(
                            f"{match} {dtype} {source_rate}->{target_rate} shape={tuple(waveform.shape)} "
                            f"options={options}: {tuple(actual.shape)} vs {tuple(expected.shape)}")

    @unittest.skipUnless(TORCHAUDIO_AVAILABLE, "torchaudio is a test-only reference")
    def test_hann_resampling_is_bit_exact_to_torchaudio_on_cpu(self):
        self._assert_hann_matches_torchaudio("cpu", (torch.float32, torch.float64))

    @unittest.skipUnless(TORCHAUDIO_AVAILABLE and torch.cuda.is_available(), "needs torchaudio and CUDA")
    def test_hann_resampling_is_bit_exact_to_torchaudio_on_cuda(self):
        self._assert_hann_matches_torchaudio("cuda", (torch.float32, torch.float64))

    def test_kaiser_best_resampling_matches_librosa_load_defaults(self):
        # Golden values: librosa 0.9.1 ``resample(x, orig_sr, 22050)``
        # (resampy 0.4.3 ``kaiser_best``, then ``fix_length`` to ceil).
        steps = torch.arange(64, dtype=torch.float64)
        waveform = (0.5 * torch.sin(steps * 0.37) + 0.25 * torch.cos(steps * 1.91)).float()
        cases = {
            16_000: (89, {
                0: 0.24356313049793243,
                1: 0.17908760905265808,
                5: 0.6862472891807556,
                17: -0.4950007200241089,
                30: 0.3047464191913605,
                87: -0.2954469323158264,
                88: 0.0,
            }),
            44_100: (32, {
                0: 0.09166640043258667,
                1: 0.3042701482772827,
                5: -0.2680314779281616,
                17: 0.005466431379318237,
                30: -0.12004460394382477,
                31: -0.3977660834789276,
            }),
        }
        for source_rate, (length, values) in cases.items():
            with self.subTest(source_rate=source_rate):
                result = resample_waveform_kaiser_best(waveform, source_rate, 22_050)
                self.assertEqual(result.dtype, torch.float32)
                self.assertEqual(tuple(result.shape), (length, ))
                for index, value in values.items():
                    self.assertAlmostEqual(result[index].item(), value, delta=1e-7)
        self.assertIs(resample_waveform_kaiser_best(waveform, 22_050, 22_050), waveform)

    def test_kaiser_best_vectorization_matches_the_resampy_loop(self):
        from voicehub.processing.waveform import _resampy_kaiser_best_table

        def reference(values, source_rate, target_rate):
            # Line-by-line transcription of resampy 0.4.3 ``_resample_loop``
            # (float64 weights, float32 output accumulation) + fix_length.
            ratio = float(target_rate) / source_rate
            table = _resampy_kaiser_best_table().tolist()
            if ratio < 1:
                table = [ratio * value for value in table]
            delta = [table[index + 1] - table[index] for index in range(len(table) - 1)] + [0.0]
            scale = min(1.0, ratio)
            precision = 2**13
            step = int(scale * precision)
            samples = values.tolist()
            output = torch.zeros(int(len(samples) * float(target_rate) / float(source_rate)))
            for index in range(output.numel()):
                time = index * (1.0 / ratio)
                whole = int(time)
                fraction = scale * (time - whole)
                accumulator = output[index]
                for wing_fraction, count, sign, origin in (
                    (fraction, whole + 1, -1, whole),
                    (scale - fraction, len(samples) - whole - 1, 1, whole + 1),
                ):
                    position = wing_fraction * precision
                    offset = int(position)
                    eta = position - offset
                    for tap in range(min(count, (len(table) - offset) // step)):
                        weight = table[offset + tap * step] + eta * delta[offset + tap * step]
                        accumulator = torch.tensor(
                            float(accumulator) + weight * samples[origin + sign * tap],
                            dtype=torch.float32,
                        )
                output[index] = accumulator
            length = -(-len(samples) * target_rate // source_rate)
            return torch.nn.functional.pad(output, (0, length - output.numel()))[:length]

        generator = torch.Generator().manual_seed(0)
        waveform = torch.rand(57, generator=generator) * 2.0 - 1.0
        for source_rate in (16_000, 44_100, 48_000, 8_000):
            target_rate = 22_050
            with self.subTest(source_rate=source_rate):
                self.assertTrue(
                    torch.equal(
                        resample_waveform_kaiser_best(waveform, source_rate, target_rate),
                        reference(waveform, source_rate, target_rate),
                    ))

    def test_kaiser_best_table_matches_the_published_resampy_filter(self):
        from voicehub.processing.waveform import _resampy_kaiser_best_table

        table = _resampy_kaiser_best_table()
        self.assertEqual(tuple(table.shape), (409_601, ))
        published = {
            0: 0.9173473712608761,
            1: 0.9173473523046404,
            4_096: 0.6308682827433317,
            8_192: 0.08152330317945146,
            100_000: -0.010423628222795207,
            409_600: -5.2887095330227954e-08,
        }
        for index, value in published.items():
            self.assertAlmostEqual(table[index].item(), value, delta=1e-11)

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

    def test_ieee_float_wave_files_are_decoded(self):
        import struct

        def float_wave(samples, *, channels, rate, extensible):
            data = struct.pack(f"<{len(samples)}f", *samples)
            block = channels * 4
            fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else 3, channels, rate, rate * block, block, 32)
            if extensible:
                guid = struct.pack("<H", 3) + bytes.fromhex("000000001000800000aa00389b71")
                fmt += struct.pack("<HHI", 22, 32, 0) + guid
            else:
                fmt += struct.pack("<H", 0)
            fact = b"fact" + struct.pack("<II", 4, len(samples) // channels)
            body = (
                b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + fact + b"data" +
                struct.pack("<I", len(data)) + data)
            return b"RIFF" + struct.pack("<I", len(body)) + body

        stereo = [0.5, -0.25, 1.5, 0.5, -1.0, 0.0]
        mono = [0.125, -0.875, 0.3]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "float.wav"
            path.write_bytes(float_wave(stereo, channels=2, rate=16_000, extensible=False))
            waveform, rate = load_pcm_wave(path)
            channels, _ = load_pcm_wave(path, preserve_channels=True)
            audio = load_native_audio(path)
        self.assertEqual(rate, 16_000)
        torch.testing.assert_close(waveform, torch.tensor([0.125, 1.0, -0.5]), rtol=0, atol=0)
        torch.testing.assert_close(
            channels, torch.tensor([[0.5, 1.5, -1.0], [-0.25, 0.5, 0.0]]), rtol=0, atol=0)
        torch.testing.assert_close(audio.waveform, waveform, rtol=0, atol=0)

        restored, rate = decode_pcm_wave(float_wave(mono, channels=1, rate=24_000, extensible=True))
        self.assertEqual(rate, 24_000)
        torch.testing.assert_close(restored, torch.tensor(mono), rtol=0, atol=0)

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
