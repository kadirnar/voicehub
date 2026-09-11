import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from voicehub.models.cosyvoice.training import CosyVoiceTrainingAdapter
from voicehub.models.higgstts.training import HiggsTrainingAdapter
from voicehub.models.orpheustts.training import OrpheusSFTDataset
from voicehub.models.xtts.training import XTTSTrainingAdapter
from voicehub.training.arguments import TrainingArguments
from voicehub.training.recipes import F5TTSTrainingAdapter, Qwen3TTSTrainingAdapter
from voicehub.training.specs import get_training_spec
from voicehub.training.trainer import Trainer

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class SourceTrainingRecipeTests(unittest.TestCase):

    def test_f5_source_mel_layout_is_converted_to_time_major(self):
        import torch

        adapter = F5TTSTrainingAdapter(
            object(),
            get_training_spec("f5tts"),
        )
        prepared = adapter.prepare_training_inputs(
            {
                "mel": torch.randn(2, 100, 31),
                "mel_lengths": torch.tensor([31, 24]),
                "text": ["one", "two"],
            },
            None,
        )
        self.assertEqual(tuple(prepared["inp"].shape), (2, 31, 100))
        self.assertEqual(prepared["lens"].tolist(), [31, 24])

    def test_orpheus_keeps_provenance_source_out_of_spoken_text(self):

        class Tokenizer:

            def __init__(self):
                self.texts = []

            def encode(self, text, *, add_special_tokens):
                self.texts.append((text, add_special_tokens))
                return [7]

        tokenizer = Tokenizer()
        dataset = OrpheusSFTDataset(
            [{
                "text": "Hello",
                "source": "studio-corpus",
                "voice": "narrator",
                "audio_codes": ([0], [0, 0], [0, 0, 0, 0]),
            }],
            tokenizer=tokenizer,
            codec=None,
        )

        dataset[0]

        self.assertEqual(tokenizer.texts, [("narrator: Hello", True)])
        self.assertNotIn("studio-corpus", tokenizer.texts[0][0])

    def test_xtts_native_adapter_requires_preencoded_gpt_inputs(self):
        adapter = XTTSTrainingAdapter(object(), get_training_spec("xtts"))
        batch = {
            "text_inputs": [[1, 2, 3]],
            "text_lengths": [3],
            "audio_codes": [[4, 5]],
            "wav_lengths": [44_100],
            "cond_latents": [[[0.1, 0.2]]],
            "metadata": {
                "source": "fixture"
            },
        }

        prepared = adapter.prepare_training_inputs(batch, None)

        self.assertEqual(
            set(prepared),
            {
                "text_inputs",
                "text_lengths",
                "audio_codes",
                "wav_lengths",
                "cond_latents",
            },
        )
        with self.assertRaisesRegex(ValueError, "precomputed text/audio tokens"):
            adapter.prepare_training_inputs(
                {
                    "text": ["hello"],
                    "audio": ["clips/example.wav"],
                    "language": ["en"],
                },
                None,
            )

    def test_qwen_residual_codebook_loss_uses_aligned_labels(self):
        source_path = (
            Path(__file__).parents[1] / "voicehub" / "models" / "qwen3tts" / "source" / "qwen_tts" / "core" /
            "models" / "modeling_qwen3_tts.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        predictor = next(
            node for node in tree.body if (
                isinstance(node, ast.ClassDef) and
                node.name == "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration"))
        forward_finetune = next(
            node for node in predictor.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward_finetune")
        loss_call = next(
            node for node in ast.walk(forward_finetune) if (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr == "loss_function"))
        shift_labels = next(keyword.value for keyword in loss_call.keywords if keyword.arg == "shift_labels")
        self.assertIsInstance(shift_labels, ast.Name)
        self.assertEqual(shift_labels.id, "labels")

    @staticmethod
    def _qwen_training_fixture():
        import torch
        from torch.nn import functional

        class ProjectionSpy(torch.nn.Linear):

            def forward(self, inputs):
                self.input_shape = tuple(inputs.shape)
                outputs = super().forward(inputs)
                self.output_shape = tuple(outputs.shape)
                return outputs

        class CodePredictor(torch.nn.Module):

            def __init__(self, codebook_count, hidden_size):
                super().__init__()
                self.embeddings = torch.nn.ModuleList(
                    [torch.nn.Embedding(32, hidden_size) for _ in range(codebook_count - 1)])

            def get_input_embeddings(self):
                return self.embeddings

        class SpeakerEncoder(torch.nn.Module):

            def __init__(self, hidden_size):
                super().__init__()
                self.projection = torch.nn.Linear(3, hidden_size)

            def forward(self, reference_mels):
                return self.projection(reference_mels.mean(dim=1))

        class TalkerSpy(torch.nn.Module):

            def __init__(self, codebook_count):
                super().__init__()
                text_hidden_size = 3
                talker_hidden_size = 5
                self.model = torch.nn.Module()
                self.model.text_embedding = torch.nn.Embedding(
                    32,
                    text_hidden_size,
                )
                self.model.codec_embedding = torch.nn.Embedding(
                    32,
                    talker_hidden_size,
                )
                self.text_projection = ProjectionSpy(
                    text_hidden_size,
                    talker_hidden_size,
                    bias=False,
                )
                self.code_predictor = CodePredictor(
                    codebook_count,
                    talker_hidden_size,
                )
                self.codec_head = torch.nn.Linear(
                    talker_hidden_size,
                    32,
                    bias=False,
                )

            def get_text_embeddings(self):
                return self.model.text_embedding

            def forward(
                self,
                *,
                inputs_embeds,
                attention_mask,
                labels,
                output_hidden_states,
            ):
                self.received_input_shape = tuple(inputs_embeds.shape)
                self.received_attention_mask = attention_mask.detach().clone()
                self.received_labels = labels.detach().clone()
                self.received_output_hidden_states = output_hidden_states

                positions = torch.arange(
                    inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                ).view(1, -1, 1)
                hidden_states = inputs_embeds + positions
                logits = self.codec_head(hidden_states)
                causal_logits = logits[:, :-1].contiguous()
                causal_labels = labels[:, 1:].contiguous()
                self.causal_labels = causal_labels.detach().clone()
                loss = functional.cross_entropy(
                    causal_logits.view(-1, causal_logits.shape[-1]),
                    causal_labels.view(-1),
                    ignore_index=-100,
                )
                self.last_hidden_states = hidden_states.detach().clone()
                return SimpleNamespace(
                    loss=loss,
                    logits=logits,
                    last_hidden_state=hidden_states,
                    hidden_states=(((hidden_states, ), None) if output_hidden_states else None),
                )

            def forward_sub_talker_finetune(
                self,
                codec_ids,
                talker_hidden_states,
            ):
                self.sub_talker_codec_ids = codec_ids.detach().clone()
                self.sub_talker_hidden_states = (talker_hidden_states.detach().clone())
                logits = talker_hidden_states.unsqueeze(1)
                return logits, talker_hidden_states.square().mean()

        class FakeQwen(torch.nn.Module):

            def __init__(self, codebook_count):
                super().__init__()
                self.talker = TalkerSpy(codebook_count)
                self.speaker_encoder = SpeakerEncoder(hidden_size=5)
                self.config = SimpleNamespace(tts_model_type="base")

            @property
            def device(self):
                return next(self.parameters()).device

            @property
            def dtype(self):
                return next(self.parameters()).dtype

        class ReadyQwenAdapter(Qwen3TTSTrainingAdapter):

            def setup(self):
                return self

        sequence_length = 10
        codebook_count = 4
        model = FakeQwen(codebook_count)
        wrapper = SimpleNamespace(
            config=SimpleNamespace(
                model_type="qwen3tts",
                sub_talker_loss_weight=0.3,
            ), )
        adapter = ReadyQwenAdapter(
            wrapper,
            get_training_spec("qwen3tts"),
        )
        adapter.primary_model = model

        input_ids = torch.zeros(
            (1, sequence_length, 2),
            dtype=torch.long,
        )
        input_ids[0, :, 0] = torch.arange(sequence_length) % 32
        input_ids[0, :, 1] = (torch.arange(sequence_length) + 3) % 32
        codec_ids = torch.zeros(
            (1, sequence_length, codebook_count),
            dtype=torch.long,
        )
        codec_ids[0, 7] = torch.tensor([1, 2, 3, 4])
        codec_ids[0, 8] = torch.tensor([2, 3, 4, 5])
        codec_mask = torch.zeros((1, sequence_length), dtype=torch.bool)
        codec_mask[0, 7:9] = True
        labels = torch.full((1, sequence_length), -100, dtype=torch.long)
        labels[0, 7:] = torch.tensor([1, 2, 3])
        codec_embedding_mask = torch.ones(
            (1, sequence_length, 1),
            dtype=torch.bool,
        )
        codec_embedding_mask[:, 6] = False
        batch = {
            "input_ids": input_ids,
            "codec_ids": codec_ids,
            "ref_mels": torch.arange(
                6,
                dtype=torch.float32,
            ).reshape(1, 2, 3),
            "text_embedding_mask": torch.ones(
                (1, sequence_length, 1),
                dtype=torch.bool,
            ),
            "codec_embedding_mask": codec_embedding_mask,
            "attention_mask": torch.ones(
                (1, sequence_length),
                dtype=torch.long,
            ),
            "codec_0_labels": labels,
            "codec_mask": codec_mask,
        }
        return adapter, model, batch

    def test_higgs_adapter_combines_native_text_and_audio_losses(self):
        import torch

        adapter = HiggsTrainingAdapter(
            SimpleNamespace(
                config=SimpleNamespace(
                    model_type="higgstts",
                    training_audio_loss_weight=3.0,
                    training_text_loss_weight=2.0,
                ), ),
            get_training_spec("higgstts"),
        )
        text_loss = torch.tensor(2.0, requires_grad=True)
        audio_loss = torch.tensor(5.0, requires_grad=True)

        loss = adapter._aggregate_losses({
            "loss": text_loss + audio_loss,
            "text_loss": text_loss,
            "audio_loss": audio_loss,
        })

        self.assertEqual(loss.item(), 19.0)
        loss.backward()
        self.assertEqual(text_loss.grad.item(), 2.0)
        self.assertEqual(audio_loss.grad.item(), 3.0)

    def test_xtts_preserves_the_published_loss_weighting(self):
        import torch

        class ReadyXTTSAdapter(XTTSTrainingAdapter):

            def setup(self):
                return self

        wrapper = SimpleNamespace(
            config=SimpleNamespace(
                model_type="xtts",
                training_text_loss_weight=0.01,
                training_mel_loss_weight=1.0,
            ), )
        adapter = ReadyXTTSAdapter(wrapper, get_training_spec("xtts"))
        parameter = torch.nn.Parameter(torch.tensor(1.0))

        def forward(**_inputs):
            text_loss = parameter * 2.0
            mel_loss = parameter * 3.0
            return text_loss, mel_loss, parameter.reshape(1)

        adapter.primary_model = forward
        inputs = {
            "text_inputs": torch.ones(1, 2, dtype=torch.long),
            "text_lengths": torch.tensor([2]),
            "audio_codes": torch.ones(1, 2, dtype=torch.long),
            "wav_lengths": torch.tensor([2]),
            "cond_mels": torch.ones(1, 1, 80, 2),
            "cond_idxs": torch.zeros(1, 2, dtype=torch.long),
            "cond_lens": torch.tensor([2]),
        }
        output = adapter.execute_training_phase(
            adapter.create_training_context(
                inputs,
                training_phase="language_model",
            ), )

        self.assertAlmostEqual(output.loss.item(), 3.02, places=6)
        self.assertAlmostEqual(output.losses["loss_text_ce"].item(), 0.02)
        self.assertAlmostEqual(output.losses["loss_mel_ce"].item(), 3.0)
        output.loss.backward()
        self.assertAlmostEqual(parameter.grad.item(), 3.02, places=6)
        self.assertEqual(
            adapter.artifact_manifest()["checkpoint_semantics"]["save_pretrained"],
            "voicehub-native-xtts2-safetensors",
        )

    def test_xtts_preencoded_evaluation_reports_native_loss(self):
        import torch

        class GPT(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, **inputs):
                self.last_inputs = inputs
                text_loss = self.scale * 2.0
                mel_loss = self.scale * 3.0
                return text_loss, mel_loss, self.scale.reshape(1)

        class PreencodedEvalAdapter(XTTSTrainingAdapter):

            def setup(self):
                if self.primary_model is None:
                    self.primary_model = GPT()
                    self.primary_path = "model.gpt"
                    self._components = [
                        ("model.gpt", self.primary_model),
                    ]
                return self

        wrapper = SimpleNamespace(
            config=SimpleNamespace(
                model_type="xtts",
                training_text_loss_weight=0.01,
                training_mel_loss_weight=1.0,
            ), )
        adapter = PreencodedEvalAdapter(wrapper, get_training_spec("xtts"))
        preencoded_record = {
            "text_inputs": torch.tensor([1, 2], dtype=torch.long),
            "text_lengths": torch.tensor(2),
            "audio_codes": torch.ones(2, dtype=torch.long),
            "wav_lengths": torch.tensor(8),
            "cond_mels": torch.ones(1, 80, 2),
            "cond_idxs": torch.tensor([0, 1], dtype=torch.long),
            "cond_lens": torch.tensor(1),
        }
        trainer = Trainer(
            model=wrapper,
            args=TrainingArguments(
                per_device_eval_batch_size=1,
                use_cpu=True,
            ),
            eval_dataset=[preencoded_record],
            training_adapter=adapter,
        )

        metrics = trainer.evaluate()

        self.assertAlmostEqual(metrics["eval_loss"], 3.02, places=6)
        self.assertFalse(adapter.primary_model.training)
        self.assertIn("audio_codes", adapter.primary_model.last_inputs)

    def test_cosyvoice_executes_only_the_selected_source_component(self):
        import torch

        class SourceComponent(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, values):
                loss = values.mean() * self.scale
                return SimpleNamespace(
                    loss=loss,
                    logits=values * self.scale,
                    accuracy=torch.tensor(0.75, device=values.device),
                )

        class NativeGraph(torch.nn.Module):

            def __init__(self, component):
                super().__init__()
                self.llm = component

            def forward(self, *, component, **inputs):
                self.selected_component = component
                return self.llm(**inputs)

        class Wrapper:

            def __init__(self, graph):
                self.config = SimpleNamespace(
                    model_type="cosyvoice",
                    training_component="llm",
                )
                self.model = graph

            @staticmethod
            def prepare_training_inputs(inputs, *, phase):
                return dict(inputs)

        class ReadyCosyVoiceAdapter(CosyVoiceTrainingAdapter):

            def setup(self):
                return self

        component = SourceComponent()
        wrapper = Wrapper(NativeGraph(component))
        adapter = ReadyCosyVoiceAdapter(
            wrapper,
            get_training_spec("cosyvoice"),
        )
        output = adapter.execute_training_phase(
            adapter.create_training_context(
                {"values": torch.tensor([1.0, 3.0])},
                training_phase="language_model",
            ), )

        self.assertEqual(output.loss.item(), 4.0)
        self.assertEqual(output.optimizer_names, ("llm", ))
        self.assertEqual(wrapper.model.selected_component, "llm")
        self.assertEqual(output.metadata["accuracy"].item(), 0.75)
        output.loss.backward()
        self.assertEqual(component.scale.grad.item(), 2.0)
        self.assertEqual(
            adapter.artifact_manifest()["checkpoint_semantics"]["save_pretrained"],
            "inference-ready-voicehub-native-cosyvoice-safetensors",
        )

    def test_qwen_export_does_not_mutate_the_live_training_model(self):
        import torch

        class Saveable:

            def save_pretrained(self, destination, **kwargs):
                self.destination = Path(destination)
                self.kwargs = kwargs

        class FakeTalker(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.codec_embedding = torch.nn.Embedding(4, 3)

        class FakeQwen(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.talker = FakeTalker()
                self.speaker_encoder = torch.nn.Linear(3, 3)
                self.config = SimpleNamespace(
                    talker_config=SimpleNamespace(
                        spk_id={"original": 1},
                        spk_is_dialect={"original": False},
                    ),
                    tts_model_type="base",
                )
                self.speech_tokenizer = SimpleNamespace(
                    model=Saveable(),
                    feature_extractor=Saveable(),
                )
                self.saved_state = None
                self.saved_model_type = None

            def save_pretrained(
                self,
                destination,
                *,
                state_dict,
                safe_serialization,
            ):
                self.destination = Path(destination)
                self.saved_state = {name: value.detach().clone() for name, value in state_dict.items()}
                self.saved_model_type = self.config.tts_model_type
                self.safe_serialization = safe_serialization

        class ReadyQwenAdapter(Qwen3TTSTrainingAdapter):

            def setup(self):
                return self

        model = FakeQwen()
        processor = Saveable()
        wrapper = SimpleNamespace(
            config=SimpleNamespace(
                model_type="qwen3tts",
                training_speaker_id=2,
                training_speaker_name="new_voice",
            ),
            model=SimpleNamespace(processor=processor),
        )
        adapter = ReadyQwenAdapter(
            wrapper,
            get_training_spec("qwen3tts"),
        )
        adapter.primary_model = model
        adapter._target_speaker_embedding = torch.tensor([7.0, 8.0, 9.0])
        original_embedding = (model.talker.model.codec_embedding.weight.detach().clone())

        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)

        torch.testing.assert_close(
            model.talker.model.codec_embedding.weight,
            original_embedding,
        )
        torch.testing.assert_close(
            model.saved_state["talker.model.codec_embedding.weight"][2],
            adapter._target_speaker_embedding,
        )
        self.assertNotIn("speaker_encoder.weight", model.saved_state)
        self.assertEqual(model.saved_model_type, "custom_voice")
        self.assertEqual(model.config.tts_model_type, "base")
        self.assertEqual(model.config.talker_config.spk_id, {"original": 1})
        self.assertTrue(model.safe_serialization)

    def test_qwen_projects_text_embeddings_to_talker_hidden_size(self):
        adapter, model, batch = self._qwen_training_fixture()

        output = adapter.execute_training_phase(adapter.create_training_context(batch), )

        projection = model.talker.text_projection
        self.assertEqual(projection.input_shape, (1, 10, 3))
        self.assertEqual(projection.output_shape, (1, 10, 5))
        self.assertEqual(model.talker.received_input_shape, (1, 10, 5))
        self.assertFalse(model.talker.received_output_hidden_states)
        output.loss.backward()
        self.assertIsNotNone(projection.weight.grad)

    def test_qwen_applies_one_shift_and_aligns_sub_talker_targets(self):
        import torch

        adapter, model, batch = self._qwen_training_fixture()

        adapter.execute_training_phase(adapter.create_training_context(batch), )

        talker = model.talker
        torch.testing.assert_close(
            talker.received_attention_mask,
            batch["attention_mask"],
        )
        torch.testing.assert_close(
            talker.received_labels,
            batch["codec_0_labels"],
        )
        torch.testing.assert_close(
            talker.causal_labels,
            batch["codec_0_labels"][:, 1:],
        )
        self.assertEqual(
            torch.nonzero(
                talker.causal_labels[0] != -100,
                as_tuple=False,
            ).flatten().tolist(),
            [6, 7, 8],
        )

        next_codec_mask = batch["codec_mask"][:, 1:]
        torch.testing.assert_close(
            talker.sub_talker_hidden_states,
            talker.last_hidden_states[:, :-1][next_codec_mask],
        )
        torch.testing.assert_close(
            talker.sub_talker_codec_ids,
            batch["codec_ids"][:, 1:][next_codec_mask],
        )


if __name__ == "__main__":
    unittest.main()
