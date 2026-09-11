import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from voicehub.models.fishtts.training import FishSemanticDataset, FishSpeechTrainingAdapter, FishTextDataCollator
from voicehub.training.arguments import TrainingArguments
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.contracts import TrainingPhaseSpec, TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily, get_training_spec

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class FishSpeechTrainingTests(unittest.TestCase):

    @staticmethod
    def _labels():
        import torch

        # Primary labels 10 and 11 are semantic positions. Residual targets
        # outside those positions remain ignored by the source objective.
        return torch.tensor(
            [[
                [-100, 10, 4, 11],
                [-100, 1, -100, 2],
                [-100, 3, -100, 4],
            ]],
            dtype=torch.long,
        )

    @staticmethod
    def _spec(*, include_codec=False):
        phases = [
            TrainingPhaseSpec(
                name="semantic",
                component_paths=("model", ),
                optimizer_names=("semantic", ),
            )
        ]
        if include_codec:
            phases.append(
                TrainingPhaseSpec(
                    name="codec",
                    component_paths=("_codec", ),
                    optimizer_names=("codec", ),
                ))
        return ModelTrainingSpec(
            model_type="fishtts",
            family=TrainingFamily.COMPOSITE,
            module_paths=("model", ),
            component_paths=("model", "_codec"),
            native_training=True,
            support=TrainingSupport.CUSTOM,
            phases=tuple(phases),
            default_phase="semantic",
        )

    @staticmethod
    def _tokenizer():

        class Tokenizer:

            def get_token_id(self, token):
                if token != "<|end_of_text|>":
                    raise KeyError(token)
                return 99

            def save_pretrained(self, path):
                Path(path, "tokenizer.marker").write_text(
                    "saved",
                    encoding="utf-8",
                )

        return Tokenizer()

    @classmethod
    def _wrapper(cls, *, filtered_logits=True):
        import torch

        tokenizer = cls._tokenizer()

        class SemanticConfig:
            num_codebooks = 2
            semantic_begin_id = 10
            semantic_end_id = 11

            @staticmethod
            def save(path):
                Path(path).write_text(
                    json.dumps({
                        "model_type": "dual_ar",
                        "num_codebooks": 2,
                    }),
                    encoding="utf-8",
                )

        class SemanticModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.token_seed = torch.nn.Parameter(torch.randn(16))
                self.codebook_seed = torch.nn.Parameter(torch.randn(2, 8))
                self.config = SemanticConfig()
                self.tokenizer = tokenizer

            def forward(self, inp, labels, key_padding_mask=None):
                batch_size, _, sequence_length = inp.shape
                token_logits = self.token_seed.view(1, 1, -1).expand(
                    batch_size,
                    sequence_length,
                    -1,
                )
                full_logits = self.codebook_seed.view(
                    1,
                    1,
                    2,
                    -1,
                ).expand(
                    batch_size,
                    sequence_length,
                    -1,
                    -1,
                )
                semantic_mask = (labels[:, 0].ge(10) & labels[:, 0].le(11))
                codebook_logits = (full_logits[semantic_mask] if filtered_logits else full_logits)
                return SimpleNamespace(
                    token_logits=token_logits,
                    codebook_logits=codebook_logits,
                )

        class Codec(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

        class Wrapper:

            def __init__(self):
                self.model = SemanticModel()
                self._codec = Codec()
                self.config = SimpleNamespace(
                    model_type="fishtts",
                    training_base_loss_weight=1.0,
                    training_semantic_loss_weight=1.0,
                )

            def load_for_training(self):
                return self

        return Wrapper()

    def test_source_losses_accept_full_sequence_and_dual_ar_logits(self):
        import torch
        import torch.nn.functional as functional

        torch.manual_seed(7)
        labels = self._labels()
        token_logits = torch.randn(1, 4, 16, requires_grad=True)
        full_logits = torch.randn(1, 4, 2, 8, requires_grad=True)
        semantic_mask = labels[:, 0].ge(10) & labels[:, 0].le(11)

        full_losses, selected, targets = (
            FishSpeechTrainingAdapter.compute_source_losses(
                token_logits=token_logits,
                codebook_logits=full_logits,
                labels=labels,
                semantic_begin_id=10,
                semantic_end_id=11,
                num_codebooks=2,
            ))
        expected_base = functional.cross_entropy(
            token_logits.reshape(-1, 16),
            labels[:, 0].reshape(-1),
            ignore_index=-100,
        )
        expected_semantic = functional.cross_entropy(
            full_logits[semantic_mask].reshape(-1, 8),
            labels[:, 1:].permute(0, 2, 1)[semantic_mask].reshape(-1),
            ignore_index=-100,
        )
        self.assertTrue(torch.allclose(full_losses["base_loss"], expected_base))
        self.assertTrue(torch.allclose(
            full_losses["semantic_loss"],
            expected_semantic,
        ))
        self.assertEqual(tuple(selected.shape), (2, 2, 8))
        self.assertEqual(tuple(targets.shape), (2, 2))

        dual_losses, _, _ = (
            FishSpeechTrainingAdapter.compute_source_losses(
                token_logits=token_logits,
                codebook_logits=full_logits[semantic_mask],
                labels=labels,
                semantic_begin_id=10,
                semantic_end_id=11,
                num_codebooks=2,
            ))
        self.assertTrue(torch.allclose(
            dual_losses["semantic_loss"],
            expected_semantic,
        ))

    def test_builtin_profile_and_recipe_are_voicehub_native(self):
        wrapper = self._wrapper()
        spec = get_training_spec("fishtts")

        self.assertEqual(
            spec.training_default_model_name_or_path,
            "fishaudio/s2-pro",
        )
        self.assertIs(spec.support, TrainingSupport.PREPROCESSED)
        self.assertTrue(all(entrypoint.startswith("voicehub.") for entrypoint in spec.source_entrypoints))
        self.assertEqual(
            spec.get_phase().required_inputs,
            ("inputs", "labels"),
        )
        self.assertIsInstance(
            AutoTrainingAdapter.from_model(wrapper),
            FishSpeechTrainingAdapter,
        )

    def test_adapter_runs_dual_ar_recipe_and_freezes_codec(self):
        import torch

        wrapper = self._wrapper(filtered_logits=True)
        adapter = FishSpeechTrainingAdapter(wrapper, self._spec())
        labels = self._labels()
        output = adapter(
            inputs=torch.zeros_like(labels),
            labels=labels,
            attention_masks=torch.tensor([[False, False, False, False]], ),
        )

        self.assertEqual(output.training_phase, "semantic")
        self.assertEqual(
            set(output.losses),
            {"loss", "base_loss", "semantic_loss"},
        )
        self.assertFalse(wrapper._codec.weight.requires_grad)
        self.assertFalse(output.metadata["codec_trainable"])
        output.loss.backward()
        self.assertIsNotNone(wrapper.model.token_seed.grad)
        self.assertIsNotNone(wrapper.model.codebook_seed.grad)

    def test_adapter_accepts_full_sequence_codebook_head(self):
        import torch

        wrapper = self._wrapper(filtered_logits=False)
        adapter = FishSpeechTrainingAdapter(wrapper, self._spec())
        labels = self._labels()
        output = adapter(
            tokens=torch.zeros_like(labels),
            labels=labels,
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.metadata["semantic_positions"], 2)

    def test_codec_phase_is_rejected(self):
        import torch

        wrapper = self._wrapper()
        adapter = FishSpeechTrainingAdapter(
            wrapper,
            self._spec(include_codec=True),
        )
        labels = self._labels()
        with self.assertRaisesRegex(ValueError, "offline tokenizer"):
            adapter(
                inputs=torch.zeros_like(labels),
                labels=labels,
                training_phase="codec",
            )

    def test_collator_matches_source_padding_convention(self):
        import torch

        collator = FishTextDataCollator(
            tokenizer=self._tokenizer(),
            max_length=8,
        )
        batch = collator([
            {
                "tokens": torch.tensor([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                ]),
                "labels": torch.tensor([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                ]),
            },
            {
                "tokens": torch.tensor([
                    [7, 8, 9],
                    [1, 2, 3],
                    [4, 5, 6],
                ]),
                "labels": torch.tensor([
                    [7, 8, 9],
                    [1, 2, 3],
                    [4, 5, 6],
                ]),
            },
        ])
        self.assertEqual(tuple(batch["inputs"].shape), (2, 3, 3))
        self.assertEqual(batch["inputs"][0, 0, -1].item(), 99)
        self.assertEqual(batch["inputs"][0, 1, -1].item(), 0)
        self.assertEqual(batch["labels"][0, :, -1].tolist(), [-100] * 3)
        self.assertEqual(
            batch["attention_masks"].tolist(),
            [[False, False, True], [False, False, False]],
        )

    def test_preprocessed_dataset_validates_channels(self):
        import torch

        dataset = FishSemanticDataset(
            [{
                "tokens": torch.zeros(3, 4),
                "labels": torch.zeros(3, 4),
            }],
            tokenizer=self._tokenizer(),
            num_codebooks=2,
        )
        self.assertEqual(tuple(dataset[0]["tokens"].shape), (3, 4))
        with self.assertRaisesRegex(ValueError, "same shape"):
            invalid = FishSemanticDataset(
                [{
                    "tokens": torch.zeros(3, 4),
                    "labels": torch.zeros(3, 3),
                }],
                tokenizer=self._tokenizer(),
                num_codebooks=2,
            )
            invalid[0]

    def test_optimizer_and_scheduler_preserve_source_defaults(self):
        wrapper = self._wrapper()
        adapter = FishSpeechTrainingAdapter(wrapper, self._spec()).setup()
        arguments = TrainingArguments(
            learning_rate=2e-4,
            weight_decay=0.1,
            warmup_steps=2,
        )
        optimizer = adapter.create_optimizer(
            "semantic",
            list(adapter.named_parameters()),
            arguments,
        )
        self.assertEqual(optimizer.defaults["lr"], 2e-4)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))
        self.assertEqual(optimizer.defaults["eps"], 1e-5)

        scheduler = adapter.create_scheduler(
            "semantic",
            optimizer,
            num_training_steps=10,
            training_args=arguments,
        )
        schedule = scheduler.lr_lambdas[0]
        self.assertEqual(schedule(0), 0.0)
        self.assertEqual(schedule(1), 0.5)
        self.assertEqual(schedule(2), 1.0)

    def test_save_pretrained_exports_source_safetensors(self):
        wrapper = self._wrapper()
        adapter = FishSpeechTrainingAdapter(wrapper, self._spec())
        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            destination = Path(directory)
            self.assertTrue((destination / "model.safetensors").is_file())
            self.assertTrue((destination / "config.json").is_file())
            self.assertTrue((destination / "tokenizer.marker").is_file())
            self.assertTrue((destination / "codec" / "model.safetensors").is_file())
            self.assertTrue((destination / "NOTICE").is_file())
            self.assertTrue((destination / "THIRD_PARTY_LICENSE").is_file())
            self.assertFalse((destination / "codec.pth").exists())


if __name__ == "__main__":
    unittest.main()
