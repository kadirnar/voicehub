from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily


def _training_spec(model_type: str, family: TrainingFamily) -> ModelTrainingSpec:
    return ModelTrainingSpec(
        model_type=model_type,
        family=family,
        module_paths=("model", ),
        support=TrainingSupport.CUSTOM,
        source_entrypoints=(f"voicehub.models.{model_type}.training", ),
    )


class _Wrapper:

    def __init__(self, model, **attributes):
        self.model = model
        self.config = attributes.pop(
            "config",
            SimpleNamespace(name_or_path="test"),
        )
        for name, value in attributes.items():
            setattr(self, name, value)

    def load_for_training(self):
        return self


@unittest.skipUnless(torch is not None, "PyTorch is required")
class RemainingTTSTrainingTests(unittest.TestCase):

    def test_echo_rectified_flow_has_gradients_and_exports(self):
        from voicehub.models.echo.model import EchoDiT
        from voicehub.models.echo.sampling import PCAState
        from voicehub.models.echo.training import EchoTrainingAdapter

        model = EchoDiT(
            latent_size=4,
            model_size=8,
            num_layers=1,
            num_heads=2,
            intermediate_size=16,
            norm_eps=1e-5,
            text_vocab_size=32,
            text_model_size=8,
            text_num_layers=1,
            text_num_heads=2,
            text_intermediate_size=16,
            speaker_patch_size=2,
            speaker_model_size=8,
            speaker_num_layers=1,
            speaker_num_heads=2,
            speaker_intermediate_size=16,
            timestep_embed_size=4,
            adaln_rank=2,
        )
        codec = nn.Linear(1, 1)
        wrapper = _Wrapper(
            model,
            fish_ae=codec,
            pca_state=PCAState(
                pca_components=torch.eye(4),
                pca_mean=torch.zeros(4),
                latent_scale=1.25,
            ),
        )
        adapter = EchoTrainingAdapter(
            wrapper,
            _training_spec("echo", TrainingFamily.FLOW_MATCHING),
        )
        output = adapter(
            target_latents=torch.randn(2, 4, 4),
            text_input_ids=torch.randint(0, 32, (2, 3)),
            text_mask=torch.ones(2, 3, dtype=torch.bool),
            speaker_latents=torch.randn(2, 4, 4),
            speaker_mask=torch.ones(2, 4, dtype=torch.bool),
            noise=torch.randn(2, 4, 4),
            timesteps=torch.tensor([0.25, 0.75]),
            latent_mask=torch.tensor([[True, True, True, False], [True, True, False, False]], ),
        )
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(model.out_proj.weight.grad)
        self.assertFalse(any(parameter.requires_grad for parameter in codec.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            from voicehub.checkpointing import SafeTensorReader

            with SafeTensorReader(Path(directory) / "pytorch_model.safetensors") as reader:
                state = reader.state_dict()
            self.assertTrue(torch.equal(state["out_proj.weight"], model.out_proj.weight))
            self.assertTrue((Path(directory) / "pca_state.safetensors").is_file())

    def test_vui_delayed_objective_masks_padding_and_exports(self):
        from voicehub.models.vui.patterns import DelayedPatternProvider
        from voicehub.models.vui.training import VuiTrainingAdapter

        class TinyDecoder(nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(8, 8)
                self.max_seqlen = 64
                self.cache_cleared = False

            def deallocate_kv_cache(self):
                self.cache_cleared = True

            def forward(self, values, positions, attn_mask=None):
                del positions
                if attn_mask is None:
                    raise AssertionError("Training must provide an explicit mask.")
                return self.projection(values)

        @dataclass
        class TinyVuiConfig:
            model: object

            def model_dump(self):
                return {"model": vars(self.model)}

        class TinyVui(nn.Module):

            def __init__(self):
                super().__init__()
                architecture = SimpleNamespace(
                    n_quantizers=2,
                    codebook_size=8,
                    special_token_id=8,
                )
                self.config = TinyVuiConfig(architecture)
                self.codec = nn.Linear(1, 1)
                self.codec.config = {
                    "sample_rate": 22_050,
                }
                self.pattern_provider = DelayedPatternProvider(n_q=2)
                self.token_emb = nn.Embedding(32, 8)
                self.audio_embeddings = nn.ModuleList([nn.Embedding(16, 8), nn.Embedding(16, 8)], )
                self.decoder = TinyDecoder()
                self.audio_heads = nn.ModuleList([nn.Linear(8, 16), nn.Linear(8, 16)], )

            @property
            def device(self):
                return self.token_emb.weight.device

        model = TinyVui()
        adapter = VuiTrainingAdapter(
            _Wrapper(
                model,
                config={
                    "name_or_path": "test",
                },
            ),
            _training_spec("vui", TrainingFamily.CAUSAL_LM),
        )
        output = adapter(
            input_ids=torch.randint(0, 32, (2, 4)),
            text_attention_mask=torch.tensor([[True, True, True, True], [True, True, False, False]], ),
            audio_codes=torch.randint(0, 8, (2, 2, 5)),
            audio_code_lengths=torch.tensor([5, 3]),
        )
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(model.audio_heads[0].weight.grad)
        self.assertTrue(model.decoder.cache_cleared)
        self.assertFalse(any(parameter.requires_grad for parameter in model.codec.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            from voicehub.checkpointing import SafeTensorReader

            with SafeTensorReader(Path(directory) / "model.safetensors") as reader:
                self.assertIn("audio_heads.0.weight", reader)
                self.assertFalse(any(name.startswith("codec.") for name in reader.keys()))
                self.assertEqual(reader.metadata["component"], "model")
            with SafeTensorReader(Path(directory) / "codec.safetensors") as reader:
                self.assertIn("weight", reader)
                self.assertEqual(reader.metadata["component"], "codec")
            config = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["checkpoint_filename"], "model.safetensors")
            self.assertEqual(config["codec_filename"], "codec.safetensors")
            self.assertEqual(config["native_artifact_format"], "voicehub-vui")

    def test_zonos_training_graph_is_batch_safe_and_differentiable(self):
        from voicehub.models.zonos.source.zonos.backbone._torch import TorchZonosBackbone
        from voicehub.models.zonos.source.zonos.codebook_pattern import apply_delay_pattern
        from voicehub.models.zonos.source.zonos.config import BackboneConfig, PrefixConditionerConfig, ZonosConfig
        from voicehub.models.zonos.source.zonos.model import Zonos
        from voicehub.models.zonos.training import ZonosTrainingAdapter

        codes = torch.tensor([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ], )
        delayed = apply_delay_pattern(codes, 99)
        self.assertEqual(delayed[1, 0, 0].item(), 99)
        self.assertEqual(delayed[1, 0, 1].item(), 7)

        backbone_config = BackboneConfig(
            d_model=8,
            attn_mlp_d_intermediate=16,
            n_layer=1,
            attn_cfg={
                "num_heads": 2,
                "num_heads_kv": 1,
            },
        )

        class TinyZonos(nn.Module):
            teacher_forced_logits = Zonos.teacher_forced_logits

            def __init__(self):
                super().__init__()
                self.config = ZonosConfig(
                    backbone=backbone_config,
                    prefix_conditioner=PrefixConditionerConfig(
                        conditioners=[],
                        projection="none",
                    ),
                )
                self.eos_token_id = 1024
                self.masked_token_id = 1025
                self.embeddings = nn.ModuleList([nn.Embedding(1026, 8) for _ in range(9)], )
                self.heads = nn.ModuleList([nn.Linear(8, 1025, bias=False) for _ in range(9)], )
                self.backbone = TorchZonosBackbone(backbone_config)
                self.autoencoder = nn.Linear(1, 1)

            @property
            def device(self):
                return self.embeddings[0].weight.device

            def embed_codes(self, values):
                return sum(embedding(values[:, index]) for index, embedding in enumerate(self.embeddings))

            def apply_heads(self, hidden_states):
                return torch.stack(
                    [head(hidden_states) for head in self.heads],
                    dim=1,
                )

        model = TinyZonos()
        full_length_codes = torch.arange(36).reshape(9, 4) + 10
        short_codes = torch.full((9, 4), -100)
        short_codes[:, :2] = torch.arange(18).reshape(9, 2) + 100
        audio_codes = torch.stack([full_length_codes, short_codes])
        code_lengths = torch.tensor([4, 2])
        logits, targets = model.teacher_forced_logits(
            torch.randn(2, 3, 8),
            audio_codes,
            audio_code_lengths=code_lengths,
        )
        self.assertEqual(logits.shape[:3], targets.shape)
        for batch_index, length in enumerate(code_lengths.tolist()):
            for codebook_index in range(audio_codes.shape[1]):
                eos_position = length + codebook_index
                self.assertEqual(
                    targets[batch_index, codebook_index, eos_position].item(),
                    model.eos_token_id,
                )
                self.assertTrue(
                    torch.equal(
                        targets[
                            batch_index,
                            codebook_index,
                            codebook_index:codebook_index + length,
                        ],
                        audio_codes[
                            batch_index,
                            codebook_index,
                            :length,
                        ],
                    ))
                self.assertTrue(
                    torch.all(
                        targets[
                            batch_index,
                            codebook_index,
                            eos_position + 1:,
                        ] == model.masked_token_id, ))
            for cascade_step in range(audio_codes.shape[1]):
                delayed_position = length + cascade_step
                cascade_frame = targets[
                    batch_index,
                    :,
                    delayed_position,
                ]
                for codebook_index, label in enumerate(cascade_frame):
                    if codebook_index < cascade_step:
                        expected_label = model.masked_token_id
                    elif codebook_index == cascade_step:
                        expected_label = model.eos_token_id
                    else:
                        source_index = (length + cascade_step - codebook_index)
                        expected_label = (
                            audio_codes[
                                batch_index,
                                codebook_index,
                                source_index,
                            ].item() if source_index >= 0 else model.masked_token_id)
                    self.assertEqual(label.item(), expected_label)

        adapter = ZonosTrainingAdapter(
            _Wrapper(model),
            _training_spec("zonos", TrainingFamily.CAUSAL_LM),
        )
        output = adapter(
            prefix_conditioning=torch.randn(2, 3, 8),
            audio_codes=audio_codes,
            audio_code_lengths=code_lengths,
        )
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.metadata["supervised_tokens"], 72)
        self.assertIsNotNone(model.backbone.layers[0].mixer.in_proj.weight.grad)
        for head in model.heads:
            self.assertGreater(
                torch.count_nonzero(head.weight.grad[model.eos_token_id], ).item(),
                0,
            )
        self.assertFalse(any(parameter.requires_grad for parameter in model.autoencoder.parameters()), )

        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            saved_config = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"), )
            self.assertEqual(saved_config["backbone"]["d_model"], 8)
            self.assertTrue((Path(directory) / "model.safetensors").is_file())

    def test_vibevoice_combines_masked_ce_and_diffusion_gradients(self):
        from voicehub.models.vibevoice.training import VibeVoiceTrainingAdapter

        class TinyVibeVoice(nn.Module):

            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(32, 8)
                self.lm_head = nn.Linear(8, 32)
                runtime = nn.Module()
                runtime.acoustic_tokenizer = nn.Linear(1, 1)
                runtime.semantic_tokenizer = nn.Linear(1, 1)
                runtime.diffusion_head = nn.Linear(8, 1)
                self.model = runtime
                self.saved = False

            def forward(self, input_ids, **kwargs):
                del kwargs
                hidden = self.embedding(input_ids)
                logits = self.lm_head(hidden)
                diffusion_loss = self.model.diffusion_head(hidden[:, 2:3], ).square().mean()
                return SimpleNamespace(
                    logits=logits,
                    diffusion_loss=diffusion_loss,
                )

            def save_pretrained(self, destination, safe_serialization):
                self.saved = bool(safe_serialization)
                Path(destination).mkdir(parents=True, exist_ok=True)
                (Path(destination) / "model.safetensors").touch()

        class TinyProcessor:

            def save_pretrained(self, destination):
                (Path(destination) / "preprocessor_config.json").touch()

        model = TinyVibeVoice()
        wrapper = _Wrapper(
            model,
            config=SimpleNamespace(
                name_or_path="microsoft/VibeVoice-1.5B",
                training_ddpm_batch_mul=2,
                training_ce_loss_weight=0.75,
                training_diffusion_loss_weight=1.25,
            ),
            _processor=TinyProcessor(),
        )
        adapter = VibeVoiceTrainingAdapter(
            wrapper,
            _training_spec("vibevoice", TrainingFamily.COMPOSITE),
        )
        output = adapter(
            input_ids=torch.randint(0, 32, (1, 7)),
            attention_mask=torch.ones(1, 7, dtype=torch.long),
            speech_tensors=torch.randn(1, 3200),
            speech_masks=torch.tensor([[True, True]]),
            speeches_loss_input=torch.tensor([[True, True]]),
            speech_semantic_tensors=torch.randn(1, 2, 4),
            acoustic_input_mask=torch.tensor([[False, False, False, True, True, False, False]], ),
            acoustic_loss_mask=torch.tensor([[False, False, False, True, True, False, False]], ),
        )
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(model.lm_head.weight.grad)
        self.assertIsNotNone(model.model.diffusion_head.weight.grad)
        for tokenizer in (
                model.model.acoustic_tokenizer,
                model.model.semantic_tokenizer,
        ):
            self.assertFalse(tokenizer.training)
            self.assertFalse(any(p.requires_grad for p in tokenizer.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            self.assertTrue(model.saved)
            self.assertTrue((Path(directory) / "preprocessor_config.json").is_file())

    def test_vibevoice_realtime_checkpoint_fails_closed(self):
        from voicehub.models.vibevoice.modeling import VibeVoiceForTextToSpeech

        realtime = VibeVoiceForTextToSpeech(lazy_load=True)
        with self.assertRaisesRegex(ValueError, "non-streaming"):
            realtime._validate_training_runtime()

        supported = VibeVoiceForTextToSpeech(
            "microsoft/VibeVoice-1.5B",
            lazy_load=True,
        )
        supported._validate_training_runtime()


if __name__ == "__main__":
    unittest.main()
