from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from voicehub.architectures import get_architecture_spec
from voicehub.architectures.openvoice.artifacts import OpenVoiceArtifacts, resolve_openvoice_artifacts
from voicehub.architectures.openvoice.checkpoint import (
    load_openvoice_checkpoint,
    read_openvoice_checkpoint,
    save_openvoice_checkpoint,
)
from voicehub.architectures.openvoice.configuration import OpenVoiceConverterConfig
from voicehub.architectures.openvoice.metadata import (
    OPENVOICE_CHECKPOINT_REVISION,
    OPENVOICE_CONVERTER_CHECKPOINT,
    OPENVOICE_SOURCE_REVISION,
)
from voicehub.architectures.openvoice.modeling import OpenVoiceToneColorConverter
from voicehub.architectures.openvoice.processing import OpenVoiceAudioProcessor
from voicehub.architectures.openvoice.runtime import OpenVoiceRuntime, load_openvoice_runtime
from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.openvoice import (
    OpenVoiceConfig,
    OpenVoiceForTextToSpeech,
    OpenVoiceTrainingAdapter,
    OpenVoiceTrainingCollator,
)
from voicehub.training import AutoTrainingAdapter, TrainingSupport


def _tiny_config() -> OpenVoiceConverterConfig:
    return OpenVoiceConverterConfig(
        n_fft=64,
        hop_length=4,
        win_length=16,
        inter_channels=8,
        hidden_channels=8,
        filter_channels=16,
        n_heads=2,
        n_layers=1,
        resblock_kernel_sizes=(3, ),
        resblock_dilation_sizes=((1, 3, 5), ),
        upsample_rates=(2, 2),
        upsample_initial_channel=16,
        upsample_kernel_sizes=(4, 4),
        speaker_embedding_size=8,
    )


def _artifacts(
    root: Path,
    config: OpenVoiceConverterConfig,
    checkpoint: Path,
    *,
    legacy: bool = False,
) -> OpenVoiceArtifacts:
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    return OpenVoiceArtifacts(
        source=str(root),
        revision=None,
        config_path=config_path,
        config=config,
        checkpoint_path=checkpoint,
        legacy_pytorch=legacy,
        expected_checkpoint_sha256=None,
    )


class OpenVoiceArchitectureTests(unittest.TestCase):

    def test_provenance_and_exact_released_inventory_are_immutable(self):
        self.assertEqual(
            OPENVOICE_SOURCE_REVISION,
            "74a1d147b17a8c3092dd5430504bd83ef6c7eb23",
        )
        self.assertEqual(
            OPENVOICE_CHECKPOINT_REVISION,
            "f36e7edfe1684461a8343844af60babc2efbb727",
        )
        self.assertEqual(OPENVOICE_CONVERTER_CHECKPOINT["tensors"], 486)
        self.assertEqual(
            OPENVOICE_CONVERTER_CHECKPOINT["parameters"],
            32_792_226,
        )
        self.assertEqual(
            OPENVOICE_CONVERTER_CHECKPOINT["sha256"],
            "9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e",
        )

        source = (
            Path(__file__).resolve().parents[1] / "voicehub" / "architectures" / "openvoice" / "SOURCE.json")
        document = json.loads(source.read_text(encoding="utf-8"))
        self.assertFalse(
            any(
                "upstream parity" in limitation.lower() and "no " not in limitation.lower()
                for limitation in document["verified_scope"]["limitations"]))

    def test_default_meta_graph_matches_the_audited_checkpoint(self):
        with torch.device("meta"):
            model = OpenVoiceToneColorConverter(OpenVoiceConverterConfig())

        state = model.state_dict()
        self.assertEqual(len(state), 486)
        self.assertEqual(
            sum(tensor.numel() for tensor in state.values()),
            32_792_226,
        )

    def test_tiny_raw_audio_training_reaches_the_entire_converter(self):
        torch.manual_seed(7)
        config = _tiny_config()
        model = OpenVoiceToneColorConverter(config)
        processor = OpenVoiceAudioProcessor(config)
        source = processor.spectrogram((torch.randn(192), torch.randn(176)))
        target = processor.waveform_batch((torch.randn(192), torch.randn(180)))
        target_reference = processor.spectrogram((torch.randn(192), torch.randn(180)))

        output = model(
            source_spectrogram=source.values,
            source_lengths=source.lengths,
            source_reference_spectrogram=source.values,
            source_reference_lengths=source.lengths,
            target_reference_spectrogram=target_reference.values,
            target_reference_lengths=target_reference.lengths,
            target_waveform=target.values,
            target_lengths=target.lengths,
            tau=0.0,
        )
        self.assertEqual(output.waveform.shape, (2, 1, 192))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))

        output.loss.backward()
        parameters = tuple(model.parameters())
        self.assertTrue(parameters)
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))

    def test_reference_lengths_crop_padding_inside_autograd(self):
        torch.manual_seed(11)
        config = _tiny_config()
        model = OpenVoiceToneColorConverter(config)
        processor = OpenVoiceAudioProcessor(config)
        references = processor.spectrogram((torch.randn(192), torch.randn(176)))
        padded = references.values.detach().clone().requires_grad_(True)

        embeddings = model.extract_speaker_embeddings(
            padded,
            lengths=references.lengths,
        )
        embeddings.sum().backward()

        self.assertEqual(embeddings.shape, (2, 8, 1))
        self.assertIsNotNone(padded.grad)
        second_length = int(references.lengths[1])
        self.assertEqual(
            torch.count_nonzero(padded.grad[1, :, second_length:]).item(),
            0,
        )

    def test_native_checkpoint_round_trip_and_strict_inventory(self):
        config = _tiny_config()
        model = OpenVoiceToneColorConverter(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_openvoice_checkpoint(
                model,
                root / "model.safetensors",
            )
            artifacts = _artifacts(root, config, checkpoint)
            restored = OpenVoiceToneColorConverter(config)
            load_openvoice_checkpoint(restored, artifacts)

            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]))

            state = dict(model.state_dict())
            state.pop(next(iter(state)))
            bad_checkpoint = save_safetensors(
                state,
                root / "bad.safetensors",
            )
            bad_artifacts = _artifacts(
                root,
                config,
                bad_checkpoint,
            )
            with self.assertRaises(CheckpointCompatibilityError):
                read_openvoice_checkpoint(restored, bad_artifacts)

    def test_legacy_pickle_import_is_explicitly_trusted(self):
        config = _tiny_config()
        model = OpenVoiceToneColorConverter(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pth"
            torch.save({"model": model.state_dict()}, checkpoint)
            artifacts = _artifacts(
                root,
                config,
                checkpoint,
                legacy=True,
            )

            with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
                read_openvoice_checkpoint(model, artifacts)
            state = read_openvoice_checkpoint(
                model,
                artifacts,
                trust_pickle_checkpoint=True,
            )
            self.assertEqual(set(state), set(model.state_dict()))

    def test_runtime_export_accepts_only_an_empty_destination(self):
        config = _tiny_config()
        model = OpenVoiceToneColorConverter(config)
        processor = OpenVoiceAudioProcessor(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_checkpoint = save_openvoice_checkpoint(
                model,
                root / "source.safetensors",
            )
            artifacts = _artifacts(root, config, source_checkpoint)
            runtime = OpenVoiceRuntime(
                model=model,
                processor=processor,
                artifacts=artifacts,
            )
            destination = root / "export"
            destination.mkdir()
            runtime.save_pretrained(destination)

            resolved = resolve_openvoice_artifacts(destination)
            self.assertFalse(resolved.legacy_pytorch)
            reloaded = load_openvoice_runtime(destination)
            self.assertEqual(
                set(reloaded.model.state_dict()),
                set(model.state_dict()),
            )
            self.assertIsNotNone(reloaded._weight_norm_cache)
            self.assertGreater(reloaded._weight_norm_cache.module_count, 0)
            reloaded.train()
            self.assertIsNone(reloaded._weight_norm_cache)
            reloaded.eval()
            self.assertIsNotNone(reloaded._weight_norm_cache)
            with self.assertRaises(FileExistsError):
                runtime.save_pretrained(destination)

    def test_training_collator_preserves_variable_length_audio(self):
        collator = OpenVoiceTrainingCollator()
        batch = collator([
            {
                "source_audio": torch.randn(180),
                "target_audio": torch.randn(192),
                "sampling_rate": 22_050,
            },
            {
                "source_audio": torch.randn(160),
                "target_audio": torch.randn(176),
                "sampling_rate": 22_050,
            },
        ])

        self.assertEqual(
            tuple(value.numel() for value in batch["source_audio"]),
            (180, 160),
        )
        self.assertEqual(
            tuple(value.numel() for value in batch["target_audio"]),
            (192, 176),
        )
        self.assertEqual(batch["sampling_rate"], 22_050)

    def test_public_training_requires_an_explicit_reconstructed_recipe_opt_in(self):
        disabled = OpenVoiceForTextToSpeech(
            OpenVoiceConfig(enable_reconstructed_finetuning=False),
            lazy_load=True,
        )
        enabled = OpenVoiceForTextToSpeech(
            OpenVoiceConfig(enable_reconstructed_finetuning=True),
            lazy_load=True,
        )

        with self.assertRaisesRegex(ValueError, "does not publish"):
            disabled._validate_training_runtime()
        enabled._validate_training_runtime()
        adapter = AutoTrainingAdapter.from_model(enabled)
        self.assertIsInstance(adapter, OpenVoiceTrainingAdapter)
        self.assertIs(adapter.spec.support, TrainingSupport.CUSTOM)

    def test_public_inference_fails_closed_at_external_boundaries(self):
        model = OpenVoiceForTextToSpeech(lazy_load=True)
        with self.assertRaisesRegex(ValueError, "base_audio"):
            model._validate_generation_inputs({
                "speaker_audio_path": torch.zeros(1_000),
                "speaker_audio_sampling_rate": 22_050,
            })
        with self.assertRaisesRegex(ValueError, "external VAD"):
            model._validate_generation_inputs({
                "base_audio": torch.zeros(1_000),
                "base_audio_sampling_rate": 22_050,
                "speaker_audio_path": torch.zeros(1_000),
                "speaker_audio_sampling_rate": 22_050,
                "vad": True,
            })

    def test_inference_audio_is_resampled_like_upstream_librosa_load(self):
        from voicehub.processing.waveform import resample_waveform_kaiser_best

        class RecordingRuntime:

            def __init__(self):
                self.processor = OpenVoiceAudioProcessor(OpenVoiceConverterConfig())
                self.references = []
                self.sources = []

            def extract_segment_embedding(self, segments):
                self.references.append(segments)
                return torch.zeros(1, 256, 1)

            def convert(self, waveform, *, source_embedding, target_embedding, tau):
                del source_embedding, target_embedding, tau
                self.sources.append(waveform)
                return torch.zeros(32)

        model = OpenVoiceForTextToSpeech(device="cpu", lazy_load=True)
        model.runtime = RecordingRuntime()
        steps = torch.arange(441, dtype=torch.float32)
        base = torch.sin(steps * 0.05)
        reference = torch.cos(steps[:160] * 0.3)
        model._generate(
            "unused",
            base_audio=base,
            base_audio_sampling_rate=44_100,
            source_embedding=torch.zeros(1, 256, 1),
            speaker_audio_path=reference,
            speaker_audio_sampling_rate=16_000,
            seed=0,
        )

        # Upstream ``ToneColorConverter.convert``/``extract_se`` read audio
        # with ``librosa.load(sr=22050)``: resampy kaiser_best, ceil length.
        expected_base = resample_waveform_kaiser_best(base, 44_100, 22_050)
        expected_reference = resample_waveform_kaiser_best(reference, 16_000, 22_050)
        self.assertEqual(expected_base.numel(), 221)
        self.assertEqual(expected_reference.numel(), 221)
        self.assertTrue(torch.equal(model.runtime.sources[0], expected_base))
        self.assertEqual(len(model.runtime.references[0]), 1)
        self.assertTrue(torch.equal(model.runtime.references[0][0], expected_reference))

    def test_native_base_model_receives_loading_options(self):
        from voicehub.models.melotts import MeloTTSForTextToSpeech

        class Loaded(Exception):
            pass

        class RecordingRuntime:

            def __init__(self):
                self.processor = OpenVoiceAudioProcessor(OpenVoiceConverterConfig())

            def convert(self, waveform, *, source_embedding, target_embedding, tau):
                del source_embedding, target_embedding, tau
                return waveform

        resolved = []
        loaded = []

        def resolve_model_directory(name_or_path, **kwargs):
            resolved.append((name_or_path, kwargs))
            return Path(directory)

        def runtime(source, **kwargs):
            loaded.append((source, kwargs))
            raise Loaded

        with tempfile.TemporaryDirectory() as directory:
            for name in ("config.json", "model.safetensors"):
                (Path(directory) / name).touch()
            model = OpenVoiceForTextToSpeech(
                device="cpu",
                lazy_load=True,
                token="hf_secret",
                base_model_name_or_path="example/melotts",
                base_model_revision="0123abcd",
                cache_dir=directory,
                local_files_only=True,
                trust_pickle_checkpoint=True,
                dtype="float64",
            )
            model.runtime = RecordingRuntime()
            generated = mock.Mock(audio=torch.zeros(441), sample_rate=44_100)
            with mock.patch.object(MeloTTSForTextToSpeech, "generate", return_value=generated) as generate:
                for language in ("EN", "ZH"):
                    model._generate(
                        "unused",
                        language=language,
                        source_embedding=torch.zeros(1, 256, 1),
                        target_embedding=torch.zeros(1, 256, 1),
                        seed=0,
                    )
            self.assertEqual(generate.call_count, 2)
            with (
                    mock.patch(
                        "voicehub.models.melotts.inference.resolve_model_directory",
                        side_effect=resolve_model_directory,
                    ),
                    mock.patch(
                        "voicehub.architectures.melotts.runtime.MeloTTSRuntime",
                        side_effect=runtime,
                    ),
                    self.assertRaises(Loaded),
            ):
                model._base_model.load()

        hub_options = {
            "revision": "0123abcd",
            "cache_dir": directory,
            "token": "hf_secret",
            "local_files_only": True,
        }
        self.assertEqual(resolved, [("example/melotts", {"model_type": "melotts", **hub_options})])
        ((source, options), ) = loaded
        self.assertEqual(source, "example/melotts")
        self.assertEqual(
            {name: options[name]
             for name in hub_options},
            hub_options,
        )
        self.assertIs(options["dtype"], torch.float64)
        self.assertTrue(options["trust_pickle_checkpoint"])

    def test_reference_segments_follow_upstream_millisecond_splitting(self):
        # Golden (start, length) pairs from upstream ``split_audio_vad`` on
        # fully active audio (pydub 0.25.1 slicing of the concatenation).
        cases = {
            (2_593_917, 44_100): (
                (0, 432_312),
                (432_312, 432_312),
                (864_624, 432_312),
                (1_296_936, 432_313),
                (1_729_249, 432_312),
                (2_161_561, 432_312),
            ),
            (245_000, 16_000): ((0, 122_496), (122_496, 122_496)),
            (1_000_003, 22_050): (
                (0, 199_993),
                (199_993, 199_994),
                (399_987, 199_993),
                (599_980, 200_016),
                (799_996, 199_993),
            ),
            # Upstream asserts on audio shorter than half a segment; the
            # native path encodes the whole file like ``extract_se([path])``
            # (no millisecond truncation of the last sample).
            (77_040, 16_000): ((0, 77_040), ),
            (106_171, 22_050): ((0, 106_171), ),
        }
        for (samples, rate), expected in cases.items():
            with self.subTest(samples=samples, rate=rate):
                waveform = torch.arange(samples, dtype=torch.float32)
                segments = OpenVoiceAudioProcessor.upstream_reference_segments(
                    waveform,
                    sampling_rate=rate,
                )
                self.assertEqual(
                    tuple((int(segment[0].item()), segment.numel()) for segment in segments),
                    expected,
                )

    def test_documented_usage_opts_into_the_official_pickle_checkpoint(self):
        # The default checkpoint is a legacy pickle that only loads with
        # ``trust_pickle_checkpoint=True``; the published example must run.
        root = Path(__file__).resolve().parents[1]
        page = (root / "docs/models/providers/openvoice.md").read_text(encoding="utf-8")
        usage = page.split("```python", 1)[1].split("```", 1)[0]
        self.assertIn("from_pretrained(\n    'myshell-ai/OpenVoiceV2'", usage)
        self.assertIn('AutoConfig.for_model("openvoice", trust_pickle_checkpoint=True)', usage)
        notebook = (root / "notebooks/models/openvoice.ipynb").read_text(encoding="utf-8")
        self.assertIn("trust_pickle_checkpoint=True", notebook)

    def test_architecture_registration_is_lazy_and_truthful(self):
        spec = get_architecture_spec("openvoice")
        self.assertEqual(spec.architecture_id, "openvoice-v2-converter")
        self.assertTrue(spec.capabilities.training)
        self.assertEqual(spec.capabilities.dtypes, ("float32", ))
        self.assertFalse(spec.metadata["official_training_recipe_available"])
        self.assertFalse(spec.metadata["full_upstream_finetuning_parity"])

        code = """
import json
import sys
from voicehub.registry import get_model_spec
spec = get_model_spec("openvoice")
print(json.dumps({
    "native": spec.is_voicehub_native,
    "architecture": spec.architecture,
    "torch": "torch" in sys.modules,
    "provider": "voicehub.models.openvoice.inference" in sys.modules,
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["native"])
        self.assertEqual(
            payload["architecture"],
            "openvoice-v2-converter",
        )
        self.assertFalse(payload["torch"])
        self.assertFalse(payload["provider"])


if __name__ == "__main__":
    unittest.main()
