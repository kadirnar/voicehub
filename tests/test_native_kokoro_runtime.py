from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - optional test environment
    torch = None
    nn = None

if torch is not None:
    from voicehub.checkpointing.errors import CheckpointCompatibilityError
    from voicehub.models.kokoro.artifacts import resolve_kokoro_artifacts
    from voicehub.models.kokoro.model import KModel
    from voicehub.models.kokoro.modeling import KOKORO_SAMPLE_RATE, KokoroConfig, KokoroForTextToSpeech
    from voicehub.models.kokoro.native.albert import KokoroAlbertModel
    from voicehub.models.kokoro.native.checkpoint import (
        KOKORO_CHECKPOINT_REVISION,
        KOKORO_LEGACY_PARAMETER_COUNT,
        KOKORO_LEGACY_TENSOR_COUNT,
        KOKORO_PYTORCH_SHA256,
        import_legacy_kokoro_voice,
        load_native_kokoro_checkpoint,
        load_native_kokoro_voice,
        save_native_kokoro_checkpoint,
    )
    from voicehub.models.kokoro.native.configuration import KokoroAlbertConfig
    from voicehub.models.kokoro.native.registration import KOKORO_SOURCE_REVISION, create_kokoro_architecture_spec
    from voicehub.models.kokoro.pipeline import (
        GraphemeFallbackFrontend,
        KokoroFrontendError,
        KPipeline,
        PhonemeFrontend,
        _call_frontend,
    )
    from voicehub.models.kokoro.training import KokoroPreprocessedTrainingModel
    from voicehub.registry import get_model_spec
    from voicehub.training.contracts import TrainingSupport
    from voicehub.training.specs import get_training_spec


@unittest.skipUnless(torch is not None, "PyTorch is required for native Kokoro")
class NativeKokoroRuntimeTests(unittest.TestCase):

    class _InferenceModel(KModel):

        def __init__(self):
            nn.Module.__init__(self)
            self.anchor = nn.Parameter(torch.ones(()))
            self.vocab = {
                " ": 1,
                "h": 2,
                "ə": 3,
                "l": 4,
                "o": 5,
            }
            self.context_length = 12

        @property
        def device(self):
            return self.anchor.device

        @property
        def dtype(self):
            return self.anchor.dtype

        def tokenize_phonemes(self, phonemes):
            unknown = set(phonemes) - set(self.vocab)
            if unknown:
                raise ValueError(f"unsupported: {unknown}")
            return [self.vocab[symbol] for symbol in phonemes]

        def forward(
            self,
            phonemes,
            ref_s,
            speed=1.0,
            return_output=False,
        ):
            del ref_s, speed
            output = self.Output(
                audio=self.anchor.expand(len(phonemes) * 4).detach().cpu(),
                pred_dur=torch.ones(len(phonemes) + 2, dtype=torch.long),
            )
            return output if return_output else output.audio

    class _TrainingModel(KModel):

        def __init__(self):
            nn.Module.__init__(self)
            self.anchor = nn.Parameter(torch.tensor(0.25))

        @property
        def device(self):
            return self.anchor.device

        @property
        def dtype(self):
            return self.anchor.dtype

        def encode_text(self, input_ids, *, input_lengths=None, ref_s):
            del input_lengths, ref_s
            batch, text = input_ids.shape
            text_mask = torch.zeros(
                batch,
                text,
                device=self.device,
                dtype=torch.bool,
            )
            return {
                "duration_encoding": self.anchor.expand(batch, text, 4),
                "duration_logits": self.anchor.expand(batch, text, 3),
                "input_lengths": torch.full(
                    (batch, ),
                    text,
                    device=self.device,
                    dtype=torch.long,
                ),
                "predictor_style": self.anchor.expand(batch, 2),
                "decoder_style": self.anchor.expand(batch, 2),
                "text_encoding": self.anchor.expand(batch, 4, text),
                "text_mask": text_mask,
            }

        def decode_aligned(self, encoded, alignment):
            batch = alignment.shape[0]
            frames = alignment.shape[-1]
            value = self.anchor + encoded["duration_encoding"].mean()
            return {
                "waveform": value.expand(batch, 1, frames * 64),
                "f0": value.expand(batch, frames),
                "energy": value.expand(batch, frames),
                "alignment": alignment,
            }

    def test_albert_is_native_differentiable_and_checkpoint_shaped(self):
        config = KokoroAlbertConfig(
            vocab_size=16,
            embedding_size=8,
            hidden_size=16,
            num_hidden_layers=2,
            num_hidden_groups=1,
            num_attention_heads=4,
            intermediate_size=32,
            max_position_embeddings=16,
        )
        model = KokoroAlbertModel(config)
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

        output = model(input_ids, attention_mask=attention_mask)
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (2, 4, 16))
        self.assertIsNotNone(model.embeddings.word_embeddings.weight.grad)
        self.assertIn(
            ("encoder.albert_layer_groups.0.albert_layers.0."
             "attention.query.weight"),
            model.state_dict(),
        )
        self.assertIn("pooler.weight", model.state_dict())

    def test_native_checkpoint_export_is_strict_and_round_trips(self):
        source = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
        target = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters(), start=1):
                parameter.fill_(index / 10)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_native_kokoro_checkpoint(source, checkpoint)
            report = load_native_kokoro_checkpoint(target, checkpoint)

            self.assertEqual(
                tuple(sorted(source.state_dict())),
                report.loaded,
            )
            for name, tensor in source.state_dict().items():
                self.assertTrue(torch.equal(tensor, target.state_dict()[name]))

            incompatible = nn.Sequential(nn.Linear(5, 4), nn.LayerNorm(4))
            with self.assertRaises(CheckpointCompatibilityError):
                load_native_kokoro_checkpoint(incompatible, checkpoint)

    def test_direct_checkpoint_requires_its_coherent_artifact_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.touch()
            with self.assertRaisesRegex(
                    NotADirectoryError,
                    "containing artifact directory",
            ):
                resolve_kokoro_artifacts(checkpoint)

    def test_legacy_voice_conversion_is_weights_only_and_portable(self):
        voice = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 1, 256)
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "voice.pt"
            native = Path(directory) / "voice.safetensors"
            torch.save(voice, legacy)

            with patch.object(torch, "load", wraps=torch.load) as load:
                import_legacy_kokoro_voice(legacy, output_path=native)

            self.assertTrue(load.call_args.kwargs["weights_only"])
            restored = load_native_kokoro_voice(native)
            self.assertTrue(torch.equal(restored, voice))

    def test_frontend_boundary_is_explicit_and_does_not_hide_errors(self):
        vocabulary = {
            " ": 1,
            "h": 2,
            "e": 3,
            "l": 4,
            "o": 5,
            "w": 6,
            "r": 7,
            "d": 8,
            "ə": 9,
        }
        fallback = GraphemeFallbackFrontend(vocabulary)
        phonemes = PhonemeFrontend(vocabulary)

        self.assertEqual(
            fallback("  HELLO\tworld  ", language_code="a"),
            "hello world",
        )
        self.assertEqual(
            phonemes("həlo", language_code="a"),
            "həlo",
        )
        with self.assertRaises(KokoroFrontendError):
            phonemes("hello!", language_code="a")

        calls = []

        def broken(text, *, language_code):
            calls.append((text, language_code))
            raise TypeError("frontend implementation failed")

        with self.assertRaisesRegex(TypeError, "implementation failed"):
            _call_frontend(broken, "hello", language_code="a")
        self.assertEqual(calls, [("hello", "a")])

    def test_pipeline_preserves_graphemes_with_explicit_phonemes(self):
        model = self._InferenceModel()
        pipeline = KPipeline("en-us", model=model)
        voice = torch.ones(10, 1, 256)

        results = list(pipeline(
            "Hello",
            voice=voice,
            phonemes="həlo",
            split_pattern=None,
        ))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].graphemes, "Hello")
        self.assertEqual(results[0].phonemes, "həlo")
        self.assertEqual(
            results[0].frontend_id,
            "caller-supplied-phonemes",
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            list(pipeline(
                "first\nsecond",
                voice=voice,
                phonemes="həlo",
            ))

    def test_preprocessed_objectives_backpropagate_and_report_scope(self):
        native_model = self._TrainingModel()
        training_model = KokoroPreprocessedTrainingModel(native_model)
        input_ids = torch.tensor([[1, 2, 3, 4]])
        durations = torch.tensor([[1, 1, 1, 1]])
        style = torch.zeros(1, 4)

        result = training_model.acoustic_objective(
            input_ids,
            ref_s=style,
            durations=durations,
            audio_values=torch.zeros(1, 256),
            audio_lengths=torch.tensor([128]),
            f0_targets=torch.zeros(1, 4),
            energy_targets=torch.zeros(1, 4),
        )
        result["loss"].backward()

        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertGreater(abs(float(native_model.anchor.grad)), 0)
        self.assertEqual(
            result["metadata"]["recipe_status"],
            "reconstructed-not-author-verified",
        )
        self.assertFalse(result["metadata"]["raw_audio_training"])
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            training_model.acoustic_objective(
                input_ids,
                ref_s=style,
                durations=durations,
                audio_values=torch.zeros(1, 256),
                audio_lengths=torch.tensor([128.0]),
            )

    def test_wrapper_requires_preprocessed_training_and_never_applies_g2p(self):
        disabled = KokoroForTextToSpeech(device="cpu")
        with self.assertRaisesRegex(ValueError, "disabled"):
            disabled._validate_training_runtime()

        wrapper = KokoroForTextToSpeech(
            device="cpu",
            enable_preprocessed_training=True,
        )
        wrapper.model = self._InferenceModel()
        wrapper.pipeline = SimpleNamespace()
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            wrapper.prepare_training_inputs(
                {
                    "input_ids": torch.ones(1, 4),
                    "ref_s": torch.zeros(1, 256),
                    "durations": torch.ones(1, 4, dtype=torch.long),
                },
                phase="duration",
            )
        with self.assertRaisesRegex(ValueError, "Precompute"):
            wrapper.prepare_training_inputs(
                {
                    "text": "Hello",
                    "ref_s": torch.zeros(1, 256),
                    "durations": torch.ones(1, 4, dtype=torch.long),
                },
                phase="duration",
            )

        prepared = wrapper.prepare_training_inputs(
            {
                "phonemes": "həlo",
                "ref_s": torch.zeros(1, 256),
                "durations": torch.ones(1, 6, dtype=torch.long),
            },
            phase="duration",
        )
        self.assertEqual(tuple(prepared["input_ids"].shape), (1, 6))
        self.assertEqual(prepared["input_ids"][0, 0].item(), 0)
        self.assertEqual(prepared["input_ids"][0, -1].item(), 0)

        voice_pack = torch.arange(
            10 * 256,
            dtype=torch.float32,
        ).reshape(10, 1, 256)
        wrapper.pipeline = SimpleNamespace(load_voice=lambda voice: voice_pack, )
        prepared_voice = wrapper.prepare_training_inputs(
            {
                "phonemes": "həlo",
                "voice": "af_heart",
                # Positive durations include both boundary tokens. Voice
                # styles are indexed by four phonemes, not by six durations.
                "durations": torch.ones(1, 6, dtype=torch.long),
            },
            phase="duration",
        )
        self.assertTrue(torch.equal(
            prepared_voice["ref_s"][0],
            voice_pack[3, 0],
        ))

    def test_registry_source_lock_and_training_boundary_are_truthful(self):
        architecture = create_kokoro_architecture_spec()
        model_spec = get_model_spec("kokoro")
        training_spec = get_training_spec("kokoro")

        self.assertEqual(architecture.upstream_revision, KOKORO_SOURCE_REVISION)
        self.assertFalse(architecture.metadata["full_finetuning_ready"])
        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "kokoro")
        self.assertIn("fine-tuning", model_spec.capabilities)
        self.assertIs(training_spec.support, TrainingSupport.PREPROCESSED)
        self.assertTrue(training_spec.native_training)
        self.assertEqual(
            tuple(phase.name for phase in training_spec.phases),
            ("duration", "acoustic"),
        )
        self.assertFalse(training_spec.separate_optimizers)
        self.assertEqual(
            tuple(phase.optimizer_names for phase in training_spec.phases),
            (("model", ), ("model", )),
        )
        self.assertEqual(
            tuple(phase.name for phase in training_spec.phases if phase.is_scheduled(0)),
            ("duration", ),
        )
        self.assertEqual(
            tuple(phase.name for phase in training_spec.phases if phase.is_scheduled(1)),
            ("acoustic", ),
        )

    def test_source_manifest_pins_the_measured_checkpoint_and_parity(self):
        source_path = (
            Path(__file__).resolve().parents[1] / "voicehub" / "models" / "kokoro" / "source" / "SOURCE.json")
        manifest = json.loads(source_path.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["checkpoint"]["revision"],
            KOKORO_CHECKPOINT_REVISION,
        )
        self.assertEqual(
            manifest["checkpoint"]["sha256"],
            KOKORO_PYTORCH_SHA256,
        )
        self.assertEqual(
            manifest["checkpoint"]["tensor_count"],
            KOKORO_LEGACY_TENSOR_COUNT,
        )
        self.assertEqual(
            manifest["checkpoint"]["parameter_count"],
            KOKORO_LEGACY_PARAMETER_COUNT,
        )
        self.assertTrue(manifest["parity"]["albert"]["exact"])
        self.assertTrue(manifest["parity"]["full_decoder"]["waveform_exact"])
        self.assertEqual(
            manifest["training"]["status"],
            "reconstructed-not-author-verified",
        )

    def test_runtime_controls_reject_external_execution_and_fixed_rate(self):
        config = KokoroConfig(sample_rate=8_000)
        self.assertEqual(config.sample_rate, KOKORO_SAMPLE_RATE)
        with self.assertRaisesRegex(ValueError, "never executes"):
            KokoroConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            KokoroConfig(use_safetensors=False)
        with self.assertRaisesRegex(ValueError, "does not delegate"):
            KokoroConfig(model_kwargs={"device_map": "auto"})


if __name__ == "__main__":
    unittest.main()
