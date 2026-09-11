from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from voicehub.models.qwen3tts.lora import (
    QWEN3_TTS_LORA_CONFIG_NAME,
    QWEN3_TTS_LORA_WEIGHTS_NAME,
    merged_qwen3_tts_state_dict,
    save_qwen3_tts_lora_adapter,
)
from voicehub.models.qwen3tts.modeling import Qwen3TTSConfig
from voicehub.models.qwen3tts.native.configuration import Qwen3TTSArchitectureConfig
from voicehub.models.qwen3tts.native.modeling import Qwen3TTSForConditionalGeneration
from voicehub.models.qwen3tts.native.registration import create_qwen3_tts_architecture_spec
from voicehub.optimization import LoRALinear
from voicehub.registry import get_model_spec
from voicehub.training.recipes import Qwen3TTSTrainingAdapter
from voicehub.training.specs import get_training_spec


def _tiny_base_config() -> Qwen3TTSArchitectureConfig:
    return Qwen3TTSArchitectureConfig.from_dict({
        "model_type": "qwen3_tts",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "tts_model_size": "0b6",
        "tts_model_type": "base",
        "im_start_token_id": 50,
        "im_end_token_id": 51,
        "tts_pad_token_id": 52,
        "tts_bos_token_id": 53,
        "tts_eos_token_id": 54,
        "speaker_encoder_config": {
            "mel_dim": 4,
            "enc_dim": 8,
            "enc_channels": [8, 8, 8, 16],
            "enc_kernel_sizes": [3, 3, 3, 1],
            "enc_dilations": [1, 2, 3, 1],
            "enc_attention_channels": 4,
            "enc_res2net_scale": 2,
            "enc_se_channels": 4,
            "sample_rate": 24_000,
        },
        "talker_config": {
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "rope_theta": 10_000,
            "num_code_groups": 4,
            "text_hidden_size": 12,
            "text_vocab_size": 64,
            "codec_eos_token_id": 30,
            "codec_think_id": 29,
            "codec_nothink_id": 28,
            "codec_think_bos_id": 27,
            "codec_think_eos_id": 26,
            "codec_pad_id": 25,
            "codec_bos_id": 24,
            "codec_language_id": {
                "english": 23,
            },
            "spk_id": {},
            "spk_is_dialect": {},
            "code_predictor_config": {
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "max_position_embeddings": 128,
                "rope_theta": 10_000,
                "num_code_groups": 4,
            },
        },
    })


class _TrainingWrapper:

    def __init__(self, native_model, config):
        self.config = config
        self.model = SimpleNamespace(
            model=native_model,
            processor=None,
        )

    def load_for_training(self):
        return self


def _adapter(
    *,
    rank: int | None,
    targets: tuple[str, ...] = ("q_proj", "down_proj"),
    seed: int = 7,
) -> tuple[Qwen3TTSTrainingAdapter, Qwen3TTSForConditionalGeneration]:
    native_model = Qwen3TTSForConditionalGeneration(_tiny_base_config())
    wrapper = _TrainingWrapper(
        native_model,
        Qwen3TTSConfig(
            name_or_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            training_speaker_id=20,
            training_lora_rank=rank,
            training_lora_alpha=4.0,
            training_lora_dropout=0.0,
            training_lora_target_modules=targets,
            training_lora_seed=seed,
        ),
    )
    adapter = Qwen3TTSTrainingAdapter(
        wrapper,
        get_training_spec("qwen3tts"),
    )
    adapter.setup()
    return adapter, native_model


class Qwen3TTSLoRATests(unittest.TestCase):

    def test_architecture_catalog_declares_native_lora_and_pinned_provenance(self):
        architecture = create_qwen3_tts_architecture_spec()

        self.assertIn(
            "native-lora-fine-tuning",
            architecture.capabilities.features,
        )
        self.assertTrue(architecture.metadata["lora_finetuning_ready"])
        self.assertFalse(architecture.metadata["upstream_lora_recipe_published"])
        self.assertIn(
            architecture.metadata["source"]["revision"],
            architecture.metadata["official_training_source"],
        )
        self.assertIn(
            architecture.metadata["source"]["revision"],
            architecture.metadata["official_training_documentation"],
        )
        self.assertIn(
            "lora-fine-tuning",
            get_model_spec("qwen3tts").capabilities,
        )

    def test_configuration_is_opt_in_and_validates_public_topology(self):
        config = Qwen3TTSConfig()

        self.assertIsNone(config.training_lora_rank)
        self.assertTrue(config.training_lora_export_adapter)
        self.assertIn("q_proj", config.training_lora_target_modules)
        self.assertIn("down_proj", config.training_lora_target_modules)
        with tempfile.TemporaryDirectory() as directory:
            configured = Qwen3TTSConfig(
                training_lora_rank=4,
                training_lora_target_modules=("q_proj", "v_proj"),
            )
            configured.save_pretrained(directory)
            reloaded = Qwen3TTSConfig.from_pretrained(directory)
        self.assertEqual(reloaded.training_lora_rank, 4)
        self.assertEqual(
            reloaded.training_lora_target_modules,
            ("q_proj", "v_proj"),
        )

        with self.assertRaisesRegex(ValueError, "positive integer"):
            Qwen3TTSConfig(training_lora_rank=0)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            Qwen3TTSConfig(
                training_lora_rank=2,
                training_lora_target_modules=("speaker_encoder", ),
            )
        with self.assertRaisesRegex(TypeError, "sequence"):
            Qwen3TTSConfig(
                training_lora_rank=2,
                training_lora_target_modules="q_proj",
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            Qwen3TTSConfig(training_lora_dropout=1.0)

    def test_full_finetuning_remains_default(self):
        adapter, model = _adapter(rank=None)

        self.assertIsNone(adapter._lora_injection)
        self.assertTrue(all(parameter.requires_grad for parameter in model.talker.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.speaker_encoder.parameters()))
        manifest = adapter.artifact_manifest()
        self.assertNotIn(
            "lora_adapter",
            manifest["checkpoint_semantics"],
        )
        self.assertFalse(any(isinstance(module, LoRALinear) for module in model.modules()))

    def test_lora_freezes_every_base_parameter_and_covers_both_decoders(self):
        adapter, model = _adapter(rank=2)
        injection = adapter._lora_injection
        self.assertIsNotNone(injection)
        assert injection is not None

        self.assertEqual(
            injection.module_names,
            (
                "talker.code_predictor.model.layers.0.mlp.down_proj",
                "talker.code_predictor.model.layers.0.self_attn.q_proj",
                "talker.code_predictor.model.layers.1.mlp.down_proj",
                "talker.code_predictor.model.layers.1.self_attn.q_proj",
                "talker.model.layers.0.mlp.down_proj",
                "talker.model.layers.0.self_attn.q_proj",
                "talker.model.layers.1.mlp.down_proj",
                "talker.model.layers.1.self_attn.q_proj",
            ),
        )
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(len(trainable), 2 * len(injection.module_names))
        self.assertTrue(all(name.endswith((".lora_a", ".lora_b")) for name in trainable))
        self.assertEqual(
            {id(parameter)
             for parameter in adapter.parameters()},
            {id(parameter)
             for parameter in trainable.values()},
        )
        self.assertFalse(model.talker.model.layers[0].self_attn.q_proj.base.weight.requires_grad)
        self.assertFalse(model.talker.model.text_embedding.weight.requires_grad)
        self.assertFalse(model.talker.codec_head.weight.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.speaker_encoder.parameters()))
        manifest = adapter.artifact_manifest()
        self.assertEqual(
            manifest["checkpoint_semantics"]["lora_adapter"],
            "strict-adapter-only-safetensors",
        )

    def test_lora_restore_recovers_graph_and_original_trainability(self):
        adapter, model = _adapter(rank=2)
        injection = adapter._lora_injection
        assert injection is not None

        injection.restore()

        self.assertFalse(any(isinstance(module, LoRALinear) for module in model.modules()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.talker.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.speaker_encoder.parameters()))

    def test_lora_rejects_an_incomplete_decoder_topology_atomically(self):
        model = Qwen3TTSForConditionalGeneration(_tiny_base_config())
        replacement = torch.nn.Identity()
        model.talker.code_predictor.model.layers[0].self_attn.q_proj = replacement
        trainability = {id(parameter): parameter.requires_grad for parameter in model.parameters()}

        from voicehub.models.qwen3tts.lora import inject_qwen3_tts_lora

        with self.assertRaisesRegex(TypeError, "must be torch.nn.Linear"):
            inject_qwen3_tts_lora(
                model,
                rank=2,
                alpha=4.0,
                dropout=0.0,
                target_modules=("q_proj", ),
                seed=7,
            )

        self.assertIs(
            model.talker.code_predictor.model.layers[0].self_attn.q_proj,
            replacement,
        )
        self.assertFalse(any(isinstance(module, LoRALinear) for module in model.modules()))
        self.assertEqual(
            {id(parameter): parameter.requires_grad
             for parameter in model.parameters()},
            trainability,
        )

    def test_both_sft_losses_backpropagate_only_into_adapters(self):
        adapter, model = _adapter(rank=2)
        injection = adapter._lora_injection
        assert injection is not None
        inputs = torch.randn(2, 6, 8)
        labels = torch.tensor([
            [-100, 1, 2, 3, 4, 5],
            [-100, 2, 3, 4, 5, 6],
        ])
        talker_output = model.talker(
            inputs_embeds=inputs,
            attention_mask=torch.ones(2, 6, dtype=torch.long),
            labels=labels,
        )
        codes = torch.randint(0, 32, (10, 4))
        _, predictor_loss = model.talker.forward_sub_talker_finetune(
            codes,
            talker_output.last_hidden_state[:, :-1].reshape(-1, 8),
        )
        loss = talker_output.loss + predictor_loss
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.talker.model.layers[0].self_attn.q_proj.lora_b.grad)
        self.assertIsNotNone(model.talker.code_predictor.model.layers[0].self_attn.q_proj.lora_b.grad)
        self.assertIsNone(model.talker.model.layers[0].self_attn.q_proj.base.weight.grad)
        self.assertIsNone(model.talker.codec_head.weight.grad)

    def test_clone_merge_is_clean_equivalent_and_does_not_mutate_live_base(self):
        adapter, model = _adapter(rank=2)
        injection = adapter._lora_injection
        assert injection is not None
        with torch.no_grad():
            for index, module_name in enumerate(injection.module_names, start=1):
                injection.modules[module_name].lora_b.fill_(index / 100)
        original_bases = {
            name: module.base.weight.detach().clone()
            for name, module in injection.modules.items()
        }
        inputs = torch.randn(1, 5, 8)
        expected = model.talker(
            inputs_embeds=inputs,
            attention_mask=torch.ones(1, 5, dtype=torch.long),
        ).logits

        merged = merged_qwen3_tts_state_dict(model, injection)

        self.assertFalse(any(".lora_" in name or ".base." in name for name in merged))
        for name, module in injection.modules.items():
            torch.testing.assert_close(module.base.weight, original_bases[name])
            self.assertFalse(module.merged)
        reloaded = Qwen3TTSForConditionalGeneration(_tiny_base_config())
        reloaded.load_state_dict(merged, strict=True)
        actual = reloaded.talker(
            inputs_embeds=inputs,
            attention_mask=torch.ones(1, 5, dtype=torch.long),
        ).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_adapter_only_checkpoint_round_trips_strictly(self):
        source_adapter, _ = _adapter(rank=2)
        injection = source_adapter._lora_injection
        assert injection is not None
        with torch.no_grad():
            for index, parameter in enumerate(injection.parameters(), start=1):
                parameter.fill_(index / 100)
        source_adapter._target_speaker_embedding = torch.arange(
            8,
            dtype=torch.float32,
        )

        with tempfile.TemporaryDirectory() as directory:
            adapter_path = Path(directory) / "lora_adapter"
            save_qwen3_tts_lora_adapter(
                injection,
                adapter_path,
                target_modules=("q_proj", "down_proj"),
                base_model=("Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
                target_speaker_embedding=(source_adapter._target_speaker_embedding),
                speaker_name="voicehub",
                speaker_id=20,
            )
            self.assertTrue((adapter_path / QWEN3_TTS_LORA_CONFIG_NAME).is_file())
            self.assertTrue((adapter_path / QWEN3_TTS_LORA_WEIGHTS_NAME).is_file())

            destination_adapter, _ = _adapter(rank=2)
            destination_adapter.load_lora_adapter(adapter_path)
            destination = destination_adapter._lora_injection
            assert destination is not None
            for name, tensor in injection.adapter_state_dict().items():
                torch.testing.assert_close(
                    destination.adapter_state_dict()[name],
                    tensor,
                )
            torch.testing.assert_close(
                destination_adapter._target_speaker_embedding,
                source_adapter._target_speaker_embedding,
            )

            incompatible, _ = _adapter(rank=1)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                incompatible.load_lora_adapter(adapter_path)

    def test_save_pretrained_writes_merged_runtime_and_adapter_supplement(self):
        adapter, model = _adapter(rank=2)
        injection = adapter._lora_injection
        assert injection is not None
        with torch.no_grad():
            for index, module_name in enumerate(injection.module_names, start=1):
                injection.modules[module_name].lora_b.fill_(index / 100)
        original_bases = {
            name: module.base.weight.detach().clone()
            for name, module in injection.modules.items()
        }
        adapter._target_speaker_embedding = torch.arange(
            8,
            dtype=torch.float32,
        )

        class Saveable:

            def save_pretrained(self, directory, **kwargs):
                self.directory = Path(directory)
                self.kwargs = kwargs
                self.directory.mkdir(parents=True, exist_ok=True)

        class RuntimeOwner:

            def save_pretrained(self, directory, *, model_state_override):
                self.directory = Path(directory)
                self.state = {name: tensor.detach().clone() for name, tensor in model_state_override.items()}
                self.saved_model_type = model.config.tts_model_type

        owner = RuntimeOwner()
        model._runtime_owner = owner
        model.speech_tokenizer = SimpleNamespace(
            model=Saveable(),
            feature_extractor=Saveable(),
        )
        adapter.model.model.processor = Saveable()

        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            adapter_path = Path(directory) / "lora_adapter"
            self.assertTrue((adapter_path / QWEN3_TTS_LORA_CONFIG_NAME).is_file())
            self.assertTrue((adapter_path / QWEN3_TTS_LORA_WEIGHTS_NAME).is_file())

        self.assertEqual(owner.saved_model_type, "custom_voice")
        self.assertEqual(model.config.tts_model_type, "base")
        self.assertFalse(any(".lora_" in name or ".base." in name for name in owner.state))
        self.assertNotIn("speaker_encoder.conv1.weight", owner.state)
        torch.testing.assert_close(
            owner.state["talker.model.codec_embedding.weight"][20],
            adapter._target_speaker_embedding,
        )
        for name, module in injection.modules.items():
            torch.testing.assert_close(module.base.weight, original_bases[name])
            self.assertFalse(module.merged)

    def test_portable_training_state_exactly_resumes_injected_topology(self):
        source_adapter, _ = _adapter(rank=2)
        source = source_adapter._lora_injection
        assert source is not None
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters(), start=1):
                parameter.fill_(index / 50)
        checkpoint = source_adapter.state_dict()

        destination_adapter, _ = _adapter(rank=2)
        destination_adapter.load_state_dict(checkpoint, strict=True)
        destination = destination_adapter._lora_injection
        assert destination is not None
        for name, tensor in source.adapter_state_dict().items():
            torch.testing.assert_close(
                destination.adapter_state_dict()[name],
                tensor,
            )


if __name__ == "__main__":
    unittest.main()
