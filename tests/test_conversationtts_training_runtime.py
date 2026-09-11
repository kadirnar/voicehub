import importlib.util
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.conversationtts.inference import ConversationTTSConfig, ConversationTTSForTextToSpeech
from voicehub.models.conversationtts.native.checkpoint import (
    export_conversationtts_checkpoint,
    load_conversationtts_checkpoint,
)
from voicehub.models.conversationtts.native.processing import (
    ConversationTTSProtocol,
    build_conversationtts_sequence,
    collate_conversationtts_sequences,
)
from voicehub.models.conversationtts.runtime import resume_for_inference
from voicehub.registry import get_model_spec
from voicehub.training.arguments import TrainingArguments
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import get_training_spec
from voicehub.training.trainer import Trainer

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class ConversationTTSCheckpointTests(unittest.TestCase):

    def setUp(self):
        import torch

        self.torch = torch

    def _model(self):
        return self.torch.nn.Linear(1, 1, bias=False)

    def test_checkpoint_uses_restricted_torch_loader(self):
        model = self._model()
        loader = Mock(return_value={
            "model": {
                "module.weight": self.torch.tensor([[3.0]]),
            },
        })
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            with patch(
                    "voicehub.models.conversationtts.native.checkpoint.torch.load",
                    loader,
            ):
                report = resume_for_inference(
                    checkpoint,
                    None,
                    model,
                    "cpu",
                )

        loader.assert_called_once_with(
            checkpoint.resolve(),
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(report.format, "pytorch-weights-only")
        self.assertEqual(model.weight.item(), 3.0)

    def test_legacy_checkpoint_is_validated_on_cpu_before_device_copy(self):
        model = self._model()
        loader = Mock(return_value={
            "model": {
                "weight": self.torch.tensor([[5.0]]),
            },
        })
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            with patch(
                    "voicehub.models.conversationtts.native.checkpoint.torch.load",
                    loader,
            ):
                report = load_conversationtts_checkpoint(
                    model,
                    checkpoint,
                    device="cuda",
                )

        loader.assert_called_once_with(
            checkpoint.resolve(),
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(report.parameter_count, 1)
        self.assertEqual(model.weight.item(), 5.0)

    def test_checkpoint_refuses_unsafe_fallback_when_weights_only_is_unsupported(self, ):
        model = self._model()
        loader = Mock(side_effect=TypeError("weights_only is unsupported"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            with (
                    patch(
                        "voicehub.models.conversationtts.native.checkpoint."
                        "torch.load",
                        loader,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "cannot load.*safely",
                    ),
            ):
                resume_for_inference(
                    checkpoint,
                    None,
                    model,
                    "cpu",
                )

        self.assertEqual(loader.call_count, 1)

    def test_checkpoint_does_not_retry_unsafe_content_failures(self):
        loader = Mock(side_effect=RuntimeError("unsafe checkpoint"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            with (
                    patch(
                        "voicehub.models.conversationtts.native.checkpoint."
                        "torch.load",
                        loader,
                    ),
                    self.assertRaisesRegex(RuntimeError, "unsafe checkpoint"),
            ):
                resume_for_inference(
                    checkpoint,
                    None,
                    self._model(),
                    "cpu",
                )
        self.assertEqual(loader.call_count, 1)

    def test_checkpoint_does_not_retry_unrelated_type_errors(self):
        loader = Mock(side_effect=TypeError("invalid map_location"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            with (
                    patch(
                        "voicehub.models.conversationtts.native.checkpoint."
                        "torch.load",
                        loader,
                    ),
                    self.assertRaisesRegex(TypeError, "invalid map_location"),
            ):
                resume_for_inference(
                    checkpoint,
                    None,
                    self._model(),
                    "cpu",
                )
        self.assertEqual(loader.call_count, 1)

    def test_checkpoint_rejects_prefix_collisions(self):
        loader = Mock(
            return_value={
                "model": {
                    "weight": self.torch.tensor([[1.0]]),
                    "module.weight": self.torch.tensor([[2.0]]),
                },
            })
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            with (
                    patch(
                        "voicehub.models.conversationtts.native.checkpoint."
                        "torch.load",
                        loader,
                    ),
                    self.assertRaisesRegex(
                        CheckpointCompatibilityError,
                        "colliding keys",
                    ),
            ):
                resume_for_inference(
                    checkpoint,
                    None,
                    self._model(),
                    "cpu",
                )

    def test_safetensors_export_round_trips_exactly(self):
        model = self._model()
        model.weight.data.fill_(7.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            exported = export_conversationtts_checkpoint(model, checkpoint)
            model.weight.data.zero_()
            report = load_conversationtts_checkpoint(
                model,
                exported,
                device="cpu",
            )

        self.assertEqual(report.format, "safetensors")
        self.assertEqual(report.tensor_count, 1)
        self.assertEqual(model.weight.item(), 7.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class ConversationTTSProcessingTests(unittest.TestCase):

    def setUp(self):
        import torch

        self.torch = torch
        self.protocol = ConversationTTSProtocol(
            audio_num_codebooks=2,
            audio_codebook_size=8,
            audio_vocab_size=11,
            text_vocab_size=32,
            text_padding_token_id=30,
            audio_padding_token_id=10,
            maximum_sequence_length=16,
        )

    def test_sequence_matches_published_text_audio_framing(self):
        sequence, mask = build_conversationtts_sequence(
            [1, 2],
            [[3, 4], [5, 6]],
            protocol=self.protocol,
        )

        self.assertTrue(
            self.torch.equal(
                sequence,
                self.torch.tensor([
                    [0, 0, 1],
                    [0, 0, 2],
                    [3, 5, 0],
                    [4, 6, 0],
                    [0, 0, 0],
                ]),
            ))
        self.assertTrue(
            self.torch.equal(
                mask,
                self.torch.tensor([
                    [False, False, True],
                    [False, False, True],
                    [True, True, False],
                    [True, True, False],
                    [True, True, False],
                ]),
            ))

    def test_collator_shifts_labels_and_uses_source_padding_ids(self):
        first = build_conversationtts_sequence(
            [1, 2],
            [[3, 4], [5, 6]],
            protocol=self.protocol,
        )
        second = build_conversationtts_sequence(
            [7],
            [[1], [2]],
            protocol=self.protocol,
        )
        batch = collate_conversationtts_sequences(
            [first, second],
            protocol=self.protocol,
        )

        self.assertEqual(tuple(batch["tokens"].shape), (2, 4, 3))
        self.assertEqual(tuple(batch["labels"].shape), (2, 4, 2))
        self.assertEqual(batch["ignore_id"], 10)
        self.assertEqual(batch["residual_ignore_id"], 10)
        self.assertEqual(batch["tokens"][1, 3].tolist(), [10, 10, 30])
        self.assertEqual(batch["labels"][1, 3].tolist(), [10, 10])

    def test_framed_batch_runs_through_the_native_two_level_objective(self):
        from voicehub.models.conversationtts.native.decoder import build_llama32_decoder
        from voicehub.models.conversationtts.source.conversationtts.models import model_new

        def tiny_decoder():
            return build_llama32_decoder(
                vocabulary_size=1,
                number_of_layers=1,
                number_of_heads=2,
                number_of_kv_heads=1,
                embedding_dimension=8,
                maximum_sequence_length=16,
                intermediate_dimension=16,
            )

        sequence = build_conversationtts_sequence(
            [1, 2],
            [[3, 4], [5, 6]],
            protocol=self.protocol,
        )
        batch = collate_conversationtts_sequences(
            [sequence],
            protocol=self.protocol,
        )

        real_empty = self.torch.empty

        def poisoned_empty(*args, **kwargs):
            tensor = real_empty(*args, **kwargs)
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
            return tensor

        with self.torch.random.fork_rng():
            self.torch.manual_seed(0)
            with (
                    patch.object(
                        self.torch,
                        "empty",
                        side_effect=poisoned_empty,
                    ),
                    patch.dict(model_new.FLAVORS, {"tiny": tiny_decoder}),
            ):
                model = model_new.Model(
                    model_new.ModelArgs(
                        backbone_flavor="tiny",
                        decoder_flavor="tiny",
                        text_vocab_size=32,
                        audio_vocab_size=11,
                        audio_num_codebooks=2,
                    ))
            self.assertTrue(self.torch.isfinite(model.audio_head).all())
            model.random_type = "none"
            c0_logits, residual_logits, residual_labels = model(
                tokens=batch["tokens"],
                labels=batch["labels"],
                tokens_mask=batch["tokens_mask"],
            )
            zero_loss, _ = model_new.CrossEntropyAndAccuracy_zero(
                c0_logits,
                batch["labels"][..., 0],
                batch["loss_mask"],
                ignore_id=10,
            )
            residual_loss, _ = model_new.CrossEntropyAndAccuracy_residual(
                residual_logits,
                residual_labels,
                loss_weights=[1.0],
                ignore_id=10,
            )
            loss = zero_loss + residual_loss
            loss.backward()

        self.assertTrue(self.torch.isfinite(loss))
        self.assertIsNotNone(model.codebook0_head.weight.grad)
        self.assertIsNotNone(model.audio_head.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class ConversationTTSTrainingRuntimeTests(unittest.TestCase):

    def setUp(self):
        import torch

        self.torch = torch
        self.events = []
        events = self.events

        class FakeCache(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.register_buffer("values", torch.ones(1))

        class FakeAttention(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.kv_cache = None
                self.cache_enabled = False

        class FakeLayer(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.attn = FakeAttention()

        class FakeTransformer(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([FakeLayer()])

            def setup_caches(self, *_args, **_kwargs):
                self.layers[0].attn.kv_cache = FakeCache()
                self.layers[0].attn.cache_enabled = True

            def caches_are_setup(self):
                return self.layers[0].attn.kv_cache is not None

            def caches_are_enabled(self):
                return self.layers[0].attn.cache_enabled

        class FakeModel(torch.nn.Module):

            def __init__(self, config):
                super().__init__()
                self.config = config
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.backbone = FakeTransformer()
                self.decoder = FakeTransformer()

            def setup_caches(self, max_batch_size):
                events.append(("setup_caches", max_batch_size))
                self.backbone.setup_caches(max_batch_size)
                self.decoder.setup_caches(max_batch_size)
                self.register_buffer(
                    "backbone_causal_mask",
                    torch.ones(1, dtype=torch.bool),
                )
                self.register_buffer(
                    "decoder_causal_mask",
                    torch.ones(1, dtype=torch.bool),
                )

            def forward(
                self,
                tokens,
                labels,
                tokens_mask,
                input_pos=None,
            ):
                del tokens_mask, input_pos
                batch_size, sequence_length, _ = tokens.shape
                c0_logits = self.weight.expand(
                    batch_size,
                    sequence_length - 1,
                    4,
                )
                residual_labels = labels[..., 1:].reshape(-1, 1)
                residual_logits = self.weight.expand(
                    residual_labels.shape[0],
                    1,
                    4,
                )
                return c0_logits, residual_logits, residual_labels

        class FakeGenerator:

            def __init__(
                self,
                model,
                text_tokenizer_path,
                audio_tokenizer_path,
            ):
                events.append((
                    "generator",
                    text_tokenizer_path,
                    audio_tokenizer_path,
                ))
                self._model = model
                model.setup_caches(1)
                self._text_tokenizer = object()
                self._audio_tokenizer = object()
                self.sample_rate = 24_000

            def generate_v1(self, **_kwargs):
                return self._model.weight.detach().reshape(1)

        self.model_module = SimpleNamespace(
            ConversationTTSModel=FakeModel,
            ConversationTTSArchitectureConfig=(lambda **values: SimpleNamespace(**values)),
        )
        self.generator_module = SimpleNamespace(
            Generator=FakeGenerator,
            prepare_prompt=Mock(),
        )
        self.loss_module = SimpleNamespace(
            CrossEntropyAndAccuracy_zero=(
                lambda logits, labels, mask, ignore_id=0: (
                    logits.mean(),
                    {
                        "zero_loss": logits.detach().mean(),
                    },
                )),
            CrossEntropyAndAccuracy_residual=(
                lambda logits, labels, loss_weights, ignore_id=0: (
                    logits.mean(),
                    {
                        "residual_loss": logits.detach().mean(),
                    },
                )),
        )

    def _model(self):
        return ConversationTTSForTextToSpeech(
            ConversationTTSConfig(
                name_or_path="test/conversationtts",
                model_args={
                    "backbone_flavor": "test",
                    "decoder_flavor": "test",
                    "text_vocab_size": 8,
                    "audio_vocab_size": 4,
                    "audio_num_codebooks": 2,
                },
                torch_dtype="float32",
            ),
            device="cpu",
        )

    @contextmanager
    def _patched_runtime(self):
        imported = []

        def import_runtime(name, **_kwargs):
            imported.append(name)
            if name == "torch":
                return self.torch
            if name == "voicehub.models.conversationtts.native.modeling":
                return self.model_module
            if name.endswith("inference.generator"):
                return self.generator_module
            raise AssertionError(f"Unexpected ConversationTTS import: {name}")

        def restore_checkpoint(_checkpoint, _experiment, model, _device):
            model.weight.data.fill_(1.0)

        with (
                patch(
                    "voicehub.models.conversationtts."
                    "modeling.import_optional",
                    side_effect=import_runtime,
                ),
                patch(
                    "voicehub.models.conversationtts."
                    "modeling.resume_for_inference",
                    side_effect=restore_checkpoint,
                ),
                patch.object(
                    ConversationTTSForTextToSpeech,
                    "_checkpoint_path",
                    return_value=Path("/unused/checkpoint.pt"),
                ),
                patch.object(
                    ConversationTTSForTextToSpeech,
                    "_text_tokenizer_path",
                    return_value=Path("/unused/text-tokenizer"),
                ) as text_tokenizer,
                patch.object(
                    ConversationTTSForTextToSpeech,
                    "_audio_tokenizer_path",
                    return_value=Path("/unused/audio-tokenizer.safetensors"),
                ) as audio_tokenizer,
        ):
            yield SimpleNamespace(
                imported=imported,
                text_tokenizer=text_tokenizer,
                audio_tokenizer=audio_tokenizer,
            )

    @contextmanager
    def _patched_training_loss(self):

        def import_source(name, **_kwargs):
            if name.endswith("models.model_new"):
                return self.loss_module
            raise AssertionError(f"Unexpected ConversationTTS recipe import: {name}")

        with patch(
                "voicehub.models.conversationtts.training.import_optional",
                side_effect=import_source,
        ):
            yield

    def _batch(self):
        return {
            "tokens": self.torch.zeros((1, 4, 3), dtype=self.torch.long),
            "labels": self.torch.ones((1, 3, 2), dtype=self.torch.long),
            "tokens_mask": self.torch.ones((1, 4, 3), dtype=self.torch.bool),
        }

    def assert_training_graph_is_cache_free(self, model):
        self.assertFalse(model.backbone.caches_are_setup())
        self.assertFalse(model.backbone.caches_are_enabled())
        self.assertFalse(model.decoder.caches_are_setup())
        self.assertFalse(model.decoder.caches_are_enabled())
        self.assertFalse(hasattr(model, "backbone_causal_mask"))
        self.assertFalse(hasattr(model, "decoder_causal_mask"))

    def test_cold_training_load_skips_all_serving_allocations(self):
        model = self._model()
        with self._patched_runtime() as runtime:
            model.load_for_training()

        self.assertTrue(model._loaded_for_training)
        self.assertTrue(model.model.training)
        self.assertIsNone(model._generator)
        self.assertIsNone(model._generator_module)
        self.assert_training_graph_is_cache_free(model.model)
        self.assertNotIn(
            "voicehub.models.conversationtts.source.conversationtts."
            "inference.generator",
            runtime.imported,
        )
        runtime.text_tokenizer.assert_not_called()
        runtime.audio_tokenizer.assert_not_called()
        self.assertNotIn(("setup_caches", 1), self.events)

    def test_registry_exposes_native_raw_and_preencoded_training(self):
        model_spec = get_model_spec("conversationtts")
        training_spec = get_training_spec("conversationtts")

        self.assertTrue(model_spec.is_voicehub_native)
        self.assertEqual(model_spec.architecture, "conversationtts")
        self.assertIn(
            "raw-audio-fine-tuning",
            model_spec.capabilities,
        )
        self.assertIn(
            "preencoded-code-fine-tuning",
            model_spec.capabilities,
        )
        self.assertTrue(training_spec.native_training)
        self.assertIs(training_spec.support, TrainingSupport.NATIVE)

    def test_precomputed_ids_are_framed_without_loading_tokenizers(self):
        model = self._model()
        prepared = model.prepare_training_inputs(
            {
                "text_token_ids": [[1, 2], [1]],
                "audio_codes": [
                    [[1, 2], [2, 1]],
                    [[1], [2]],
                ],
            },
            phase="codec_language_model",
        )

        self.assertEqual(tuple(prepared["tokens"].shape), (2, 4, 3))
        self.assertEqual(tuple(prepared["labels"].shape), (2, 4, 2))
        self.assertEqual(prepared["ignore_id"], 3)

    def test_collated_raw_audio_uses_each_unpadded_waveform_length(self):
        encoded_lengths = []
        torch = self.torch

        class FakeCodecModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def encode(self, waveform):
                encoded_lengths.append(int(waveform.shape[-1]))
                return torch.zeros(
                    (1, 2, 1),
                    dtype=torch.long,
                    device=waveform.device,
                )

        codec = SimpleNamespace(
            device=torch.device("cpu"),
            model=FakeCodecModel(),
        )
        tokenizer = SimpleNamespace(tokenize=lambda _text: [1, 2])
        model = self._model()
        adapter = model.get_training_adapter()
        collated = adapter.data_collator([
            {
                "text": "first",
                "audio": {
                    "waveform": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                    "sampling_rate": 24_000,
                },
            },
            {
                "text": "second",
                "audio": {
                    "waveform": torch.tensor([0.1, 0.2]),
                    "sampling_rate": 24_000,
                },
            },
        ])
        with (
                patch.object(
                    model,
                    "_get_training_text_tokenizer",
                    return_value=tokenizer,
                ),
                patch.object(
                    model,
                    "_get_training_audio_tokenizer",
                    return_value=codec,
                ),
        ):
            prepared = model.prepare_training_inputs(
                collated,
                phase="codec_language_model",
            )

        self.assertEqual(encoded_lengths, [4, 2])
        self.assertEqual(tuple(prepared["tokens"].shape), (2, 3, 3))

    def test_lifecycle_preserves_weights_gradients_and_adapter_identity(self):
        model = self._model()
        with self._patched_runtime(), self._patched_training_loss():
            adapter = model.get_training_adapter()
            first_output = adapter(**self._batch())
            first_output.loss.backward()
            self.assertIsNotNone(model.model.weight.grad)

            training_model = model.model
            model.model.weight.data.fill_(7.0)
            model.model.weight.grad = None

            model.load()
            self.assertIs(model.model, training_model)
            self.assertFalse(model._loaded_for_training)
            self.assertIsNotNone(model._generator)
            self.assertTrue(model.model.backbone.caches_are_setup())
            self.assertTrue(model.model.decoder.caches_are_setup())
            self.assertEqual(model.model.weight.item(), 7.0)

            model.load_for_training()
            self.assertIs(model.model, training_model)
            self.assertIs(adapter.primary_model, training_model)
            self.assertTrue(model._loaded_for_training)
            self.assertIsNone(model._generator)
            self.assertTrue(model.model.training)
            self.assert_training_graph_is_cache_free(model.model)
            self.assertEqual(model.model.weight.item(), 7.0)

            second_output = adapter(**self._batch())
            second_output.loss.backward()
            self.assertIsNotNone(model.model.weight.grad)

            generated = model.generate("preserve my weights")

        self.assertEqual(float(generated.audio[0]), 7.0)
        self.assertEqual(
            [event for event in self.events if event[0] == "setup_caches"],
            [
                ("setup_caches", 1),
                ("setup_caches", 1),
            ],
        )

    def test_portable_artifact_restores_then_builds_inference_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._model()
            with self._patched_runtime():
                trainer = Trainer(
                    model=source,
                    args=TrainingArguments(
                        output_dir=directory,
                        use_cpu=True,
                    ),
                )
                trainer._ensure_model_loaded()
                trainer._move_model_to_device()
                source.model.weight.data.fill_(5.0)
                trainer.save_model(directory)

                restored = ConversationTTSForTextToSpeech.from_pretrained(
                    directory,
                    device="cpu",
                    lazy_load=False,
                )
                generated = restored.generate("portable artifact")

                self.assertFalse(restored._loaded_for_training)
                self.assertIsNotNone(restored._generator)
                self.assertTrue(restored.model.backbone.caches_are_setup())
                self.assertEqual(restored.model.weight.item(), 5.0)
                self.assertEqual(float(generated.audio[0]), 5.0)

                restored.load_for_training()
                self.assertTrue(restored._loaded_for_training)
                self.assert_training_graph_is_cache_free(restored.model)
                self.assertEqual(restored.model.weight.item(), 5.0)


if __name__ == "__main__":
    unittest.main()
