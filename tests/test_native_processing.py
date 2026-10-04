from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from voicehub.processing import LogMelSpectrogram, ModelBatch, PadOrTrimAudio, ProcessorGraph


class ProcessingImportTests(unittest.TestCase):

    def test_package_discovery_keeps_dsp_modules_lazy(self):
        code = (
            "import sys; "
            "import voicehub.processing as processing; "
            "assert 'torch' not in sys.modules; "
            "assert 'KaldiFbank' in processing.__all__; "
            "assert 'load_native_audio' in processing.__all__")
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )


@unittest.skipUnless(torch is not None, "Native processing uses PyTorch")
class ProcessorGraphTests(unittest.TestCase):

    def test_graph_round_trip_runs_identical_audio_operations(self):
        graph = ProcessorGraph(
            inputs=("waveform", ),
            outputs=("input_features", "waveform_length"),
            operations=(
                PadOrTrimAudio(length=1600),
                LogMelSpectrogram(
                    input_key="padded_waveform",
                    n_mels=16,
                    whisper_scaling=True,
                ),
            ),
            metadata={"architecture": "test"},
        )
        restored = ProcessorGraph.from_dict(graph.to_dict())
        waveform = torch.linspace(-0.5, 0.5, 800)

        expected = graph.run({"waveform": waveform})
        actual = restored.run({"waveform": waveform})

        torch.testing.assert_close(
            actual["input_features"],
            expected["input_features"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(actual["waveform_length"].item(), 800)

    def test_graph_rejects_implicit_overwrites(self):
        with self.assertRaisesRegex(ValueError, "overwrites"):
            ProcessorGraph(
                inputs=("waveform", ),
                outputs=("waveform", ),
                operations=(PadOrTrimAudio(
                    length=100,
                    output_key="waveform",
                ), ),
            )

    def test_graph_requires_an_exact_input_contract(self):
        graph = ProcessorGraph(
            inputs=("waveform", ),
            outputs=("padded_waveform", ),
            operations=(PadOrTrimAudio(length=100), ),
        )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            graph.run({"waveform": torch.zeros(100), "secret": "ignored"})

    def test_model_batch_recursively_moves_tensor_values(self):
        batch = ModelBatch(
            data={
                "input": torch.ones(1, dtype=torch.float32),
                "nested": (torch.ones(1, dtype=torch.float32), ),
            },
            batch_size=1,
        )

        converted = batch.to(dtype=torch.float64)

        self.assertEqual(converted["input"].dtype, torch.float64)
        self.assertEqual(converted["nested"][0].dtype, torch.float64)


@unittest.skipUnless(torch is not None, "Native processing uses PyTorch")
class LibrosaMelFilterTests(unittest.TestCase):
    # SHA-256 of the float32 arrays in OpenAI Whisper's
    # ``whisper/assets/mel_filters.npz`` (librosa.filters.mel(sr=16000,
    # n_fft=400, n_mels=N)) at openai/whisper 04f449b8.
    WHISPER_FILTER_SHA256 = {
        80: "4f2701b1d287d74a0dc9871026e9519d98cb76426615f2539b0d151a0ae4ec2e",
        128: "2a5f9822897750e047c85dea37cc268d3be0ecfd23c28a5f10da129d99d05afe",
    }

    def test_match_librosa_reproduces_whisper_filter_assets_bit_for_bit(self):
        import hashlib

        from voicehub.processing.audio import mel_filter_bank

        for n_mels, expected in self.WHISPER_FILTER_SHA256.items():
            filters = mel_filter_bank(
                sample_rate=16_000,
                n_fft=400,
                n_mels=n_mels,
                match_librosa=True,
            )
            self.assertEqual(filters.dtype, torch.float32)
            self.assertEqual(tuple(filters.shape), (n_mels, 201))
            digest = hashlib.sha256(filters.contiguous().numpy().tobytes()).hexdigest()
            self.assertEqual(digest, expected)
            # The default float32 evaluation is close but not bit-exact.
            default = mel_filter_bank(sample_rate=16_000, n_fft=400, n_mels=n_mels)
            torch.testing.assert_close(default, filters, rtol=0, atol=1e-6)

    def test_log_mel_operation_forwards_and_serializes_librosa_filters(self):
        from voicehub.processing.audio import mel_filter_bank

        operation = LogMelSpectrogram(
            n_mels=80,
            whisper_scaling=True,
            librosa_filters=True,
        )
        self.assertTrue(operation.to_config()["librosa_filters"])
        self.assertFalse(LogMelSpectrogram().to_config()["librosa_filters"])
        with self.assertRaises(TypeError):
            LogMelSpectrogram(librosa_filters=1)
        waveform = torch.linspace(-0.5, 0.5, 4000)
        actual = operation.process({"waveform": waveform})["input_features"]
        power = torch.stft(
            waveform,
            n_fft=400,
            hop_length=160,
            window=torch.hann_window(400),
            center=True,
            return_complex=True,
        )[..., :-1].abs().square()
        filters = mel_filter_bank(
            sample_rate=16_000,
            n_fft=400,
            n_mels=80,
            match_librosa=True,
        )
        expected = torch.clamp(filters @ power, min=1e-10).log10()
        expected = torch.maximum(expected, expected.max() - 8.0)
        expected = (expected + 4.0) / 4.0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
