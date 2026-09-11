import importlib.util
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(
    TORCH_AVAILABLE,
    "The native codec compatibility layer requires PyTorch",
)
class CodecCompatibilityTests(unittest.TestCase):

    def test_native_dac_archive_round_trip_needs_no_numpy(self):
        import torch

        from voicehub.components.audio.codecs.dac.model.base import DACFile

        source = DACFile(
            codes=torch.arange(24).reshape(1, 3, 8),
            chunk_length=4,
            original_length=2_048,
            input_db=torch.tensor([-23.5]),
            channels=1,
            sample_rate=44_100,
            padding=True,
            dac_version="1.0.0",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = source.save(Path(directory) / "sample")
            restored = DACFile.load(path)

        self.assertEqual(path.suffix, ".dac")
        self.assertEqual(restored.chunk_length, source.chunk_length)
        self.assertEqual(restored.original_length, source.original_length)
        self.assertEqual(restored.sample_rate, source.sample_rate)
        self.assertTrue(torch.equal(restored.codes, source.codes))
        self.assertTrue(torch.equal(restored.input_db, source.input_db))

    def test_weights_checkpoint_round_trip_preserves_constructor_metadata(self):
        import torch

        from voicehub.components.audio.codecs._compat import BaseModel

        class TinyCodec(BaseModel):

            def __init__(self, width: int = 2):
                super().__init__()
                self.width = width
                self.projection = torch.nn.Linear(width, width)

        source = TinyCodec(width=3)
        with torch.no_grad():
            source.projection.weight.fill_(0.25)
            source.projection.bias.fill_(-0.5)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pth"
            source.save(
                checkpoint,
                metadata={"family": "test-codec"},
                package=False,
            )
            restored = TinyCodec.load(checkpoint)

        self.assertEqual(restored.width, 3)
        self.assertEqual(restored.metadata["family"], "test-codec")
        self.assertTrue(torch.equal(
            restored.projection.weight,
            source.projection.weight,
        ))
        self.assertTrue(torch.equal(
            restored.projection.bias,
            source.projection.bias,
        ))

    def test_loudness_normalization_preserves_shape_and_target(self):
        import torch

        from voicehub.components.audio.codecs._compat import integrated_loudness, normalize_loudness

        sample_rate = 16_000
        time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        waveform = 0.01 * torch.sin(2 * torch.pi * 220 * time)

        normalized = normalize_loudness(
            waveform,
            sample_rate,
            -20.0,
        )

        self.assertEqual(normalized.shape, waveform.shape)
        self.assertEqual(normalized.dtype, waveform.dtype)
        self.assertLessEqual(float(normalized.abs().max()), 1.0)
        measured = integrated_loudness(normalized, sample_rate)
        self.assertAlmostEqual(float(measured.item()), -20.0, places=2)

    def test_short_audio_signal_keeps_its_original_length(self):
        import torch

        from voicehub.components.audio.codecs._compat import AudioSignal

        sample_rate = 16_000
        waveform = torch.linspace(-0.01, 0.01, sample_rate // 10)
        signal = AudioSignal(waveform, sample_rate)

        signal.normalize(-24.0).ensure_max_of_audio()

        self.assertEqual(signal.audio_data.shape, (1, 1, waveform.numel()))
        self.assertEqual(signal.signal_length, waveform.numel())
        self.assertLessEqual(float(signal.audio_data.abs().max()), 1.0)

    def test_irodori_uses_internal_loudness_normalization(self):
        required = ("einops", )
        if any(importlib.util.find_spec(name) is None for name in required):
            self.skipTest("Irodori runtime dependencies are not installed")

        import torch

        from voicehub.components.audio.codecs._compat import integrated_loudness
        from voicehub.models.irodoritts.source.irodori_tts.codec import DACVAECodec

        sample_rate = 16_000
        time = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        waveform = 0.005 * torch.sin(2 * torch.pi * 440 * time)

        normalized = DACVAECodec._normalize_loudness(
            waveform,
            sample_rate,
            -18.0,
        )

        self.assertEqual(normalized.shape, waveform.shape)
        self.assertAlmostEqual(
            float(integrated_loudness(normalized, sample_rate).item()),
            -18.0,
            places=2,
        )

    def test_active_codec_imports_do_not_resolve_legacy_dependencies(self):
        script = textwrap.dedent(
            """
            import importlib
            import importlib.abc
            import sys

            class RejectLegacyCodecDependency(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".", 1)[0] in {
                        "audiotools",
                        "loguru",
                        "vector_quantize_pytorch",
                    }:
                        raise AssertionError(
                            f"Active codec import attempted to load {fullname}")
                    return None

            sys.meta_path.insert(0, RejectLegacyCodecDependency())
            modules = (
                "voicehub.components.audio.codecs.dac",
                "voicehub.models.fishtts.source.fish_speech.models.dac.modded_dac",
                "voicehub.models.higgstts.native.tokenizer",
                "voicehub.models.irodoritts.source.dacvae",
                "voicehub.models.irodoritts.source.irodori_tts.codec",
            )
            for module_name in modules:
                importlib.import_module(module_name)

            forbidden_eager_modules = (
                "voicehub.components.audio.codecs.dac.model.discriminator",
                "voicehub.components.audio.codecs.dac.nn.loss",
                "voicehub.models.irodoritts.source.dacvae.model.discriminator",
                "voicehub.models.irodoritts.source.dacvae.nn.loss",
            )
            loaded = [name for name in forbidden_eager_modules if name in sys.modules]
            if loaded:
                raise AssertionError(f"Training-only codec modules loaded eagerly: {loaded}")
        """)

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
