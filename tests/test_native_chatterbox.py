from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from voicehub.models.chatterbox.checkpoint import (
    CHECKPOINT_REVISION,
    export_module_safetensors,
    inspect_t3_text_vocabulary_size,
    load_module_safetensors,
)
from voicehub.models.chatterbox.models.s3gen import S3Gen
from voicehub.models.chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec
from voicehub.models.chatterbox.models.s3gen.matcha.decoder import ConformerWrapper, Decoder
from voicehub.models.chatterbox.models.t3 import T3
from voicehub.models.chatterbox.models.t3.llama_configs import LLAMA_CONFIGS
from voicehub.models.chatterbox.models.t3.modules.cond_enc import T3Cond
from voicehub.models.chatterbox.models.tokenizers import EnTokenizer
from voicehub.models.chatterbox.models.voice_encoder import VoiceEncoder
from voicehub.models.chatterbox.native.registration import create_chatterbox_architecture_spec
from voicehub.models.chatterbox.training import ChatterboxTrainingAdapter, resize_t3_text_vocabulary
from voicehub.models.chatterbox.watermark import NativePerthWatermarker
from voicehub.training.contracts import TrainingContext
from voicehub.training.specs import get_training_spec

PACKAGE_ROOT = (Path(__file__).resolve().parents[1] / "voicehub" / "models" / "chatterbox")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TinyT3Config:
    start_text_token = 1
    stop_text_token = 2
    text_tokens_dict_size = 16
    max_text_tokens = 16
    start_speech_token = 10
    stop_speech_token = 11
    speech_tokens_dict_size = 12
    max_speech_tokens = 16
    llama_config_name = "VoiceHub_Chatterbox_Test"
    input_pos_emb = "learned"
    speech_cond_prompt_len = 2
    encoder_type = "voice_encoder"
    speaker_embed_size = 4
    use_perceiver_resampler = False
    emotion_adv = True

    @property
    def n_channels(self):
        return 16


_TINY_LLAMA = {
    "vocab_size": 8,
    "max_position_embeddings": 128,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 4,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "mlp_bias": False,
    "model_type": "llama",
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10_000.0,
    "use_cache": True,
}


class _TinyFlowEncoder(nn.Module):

    def __init__(self, width: int):
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.width = width

    def output_size(self):
        return self.width

    def forward(self, values, lengths):
        mask = (torch.arange(values.shape[1], device=values.device)[None, :] < lengths[:, None]).unsqueeze(1)
        return self.projection(values), mask


class _TinyFlowDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def compute_loss(self, target, mask, mu, speaker, *, cond):
        loss = (
            target.square().mean() + mu.square().mean() + speaker.square().mean() +
            cond.square().mean()) * self.scale
        return loss, mu


class _FakeSpeechTokenizer:

    def __call__(self, waveforms, max_len=None):
        length = 4 if max_len is None else min(4, int(max_len))
        tokens = torch.arange(1, length + 1, dtype=torch.long)
        tokens = tokens.repeat(len(waveforms), 1)
        return tokens, torch.full(
            (len(waveforms), ),
            length,
            dtype=torch.long,
        )


class _FakeVoiceEncoder:

    @staticmethod
    def embeds_from_wavs(waveforms, sample_rate):
        del sample_rate
        return torch.stack(
            [torch.tensor([waveform.mean(), waveform.std(), 0.25, 0.5]) for waveform in waveforms])


class _FakeTextTokenizer:

    @staticmethod
    def text_to_tokens(text):
        return torch.tensor([[3, 4, 5]], dtype=torch.long)


class _FakeSpeakerEncoder:

    @staticmethod
    def inference(waveforms):
        return torch.ones(len(waveforms), 192)


class NativeChatterboxTests(unittest.TestCase):

    def test_generation_limit_rejects_values_beyond_t3_capacity(self):
        from voicehub.models.chatterbox.tts import ChatterboxTTS

        runtime = ChatterboxTTS.__new__(ChatterboxTTS)
        runtime.t3 = SimpleNamespace(hp=SimpleNamespace(max_speech_tokens=16), )
        with self.assertRaisesRegex(ValueError, "checkpoint limit"):
            runtime.generate("hello", max_new_tokens=17)

    def test_native_architecture_declaration_is_complete_and_lazy(self):
        spec = create_chatterbox_architecture_spec()

        self.assertEqual(spec.architecture_id, "chatterbox")
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.capabilities.supports_checkpoint_format("safetensors"))
        self.assertEqual(
            spec.metadata["reference_tensor_count"],
            2_797,
        )
        self.assertEqual(
            spec.metadata["reference_parameter_count"],
            797_870_659,
        )
        self.assertFalse(spec.metadata["author_end_to_end_recipe_published"])

    def test_public_runtime_imports_only_torch_stdlib_and_voicehub(self):
        allowed = set(sys.stdlib_module_names) | {"torch", "voicehub"}
        violations = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            if "source" in path.relative_to(PACKAGE_ROOT).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                for name in names:
                    root = name.partition(".")[0]
                    if name.startswith("voicehub.models.chatterbox.source"):
                        violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}: {name}")
                    elif root and root not in allowed:
                        violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}: {name}")
        self.assertEqual(violations, [])

    def test_default_runtime_does_not_enter_legacy_s3tokenizer_package(self):
        script = """
import builtins
import sys
import torch

blocked = {"numpy", "onnx", "torchaudio", "tqdm"}
for module_name in tuple(sys.modules):
    if module_name.split(".", 1)[0] in blocked:
        del sys.modules[module_name]

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError("blocked import: " + name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from voicehub.models.chatterbox.models.s3tokenizer import S3Tokenizer
from voicehub.models.chatterbox.tts import ChatterboxTTS

assert S3Tokenizer.__module__.startswith(
    "voicehub.models.chatterbox.models.s3tokenizer"
)
assert ChatterboxTTS.__module__ == "voicehub.models.chatterbox.tts"
assert not any(
    name.startswith("voicehub.models.chatterbox.source.s3tokenizer")
    for name in sys.modules
)

import voicehub.models.chatterbox.source.s3tokenizer as legacy_s3tokenizer
assert "speech_tokenizer_v2_25hz" in legacy_s3tokenizer.available_models()
assert (
    "voicehub.models.chatterbox.source.s3tokenizer.utils"
    not in sys.modules
)
assert (
    "voicehub.models.chatterbox.source.s3tokenizer.model_v3"
    not in sys.modules
)
print("ok")
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_source_and_checkpoint_provenance_are_immutable(self):
        payload = json.loads((PACKAGE_ROOT / "source" / "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(
            payload["architecture"]["revision"],
            "eb90621fa748f341a5b768aed0c0c12fc561894b",
        )
        self.assertEqual(
            payload["checkpoint"]["revision"],
            CHECKPOINT_REVISION,
        )
        integration = payload["training_audit"]["integrated_fine_tuning_implementation"]
        self.assertEqual(
            integration["revision"],
            "fac31c46ec96b37283a363a1a96c2a0e56640e03",
        )

    def test_released_graphs_match_audited_checkpoint_inventories(self):
        with torch.device("meta"):
            modules = {
                "t3_cfg.safetensors": T3(),
                "s3gen.safetensors": S3Gen(),
                "ve.safetensors": VoiceEncoder(),
            }
        expected = {
            "t3_cfg.safetensors": (292, 532_405_248),
            "s3gen.safetensors": (2_489, 264_041_793),
            "ve.safetensors": (16, 1_423_618),
        }
        for name, module in modules.items():
            state = module.state_dict()
            self.assertEqual(
                (len(state), sum(value.numel() for value in state.values())),
                expected[name],
            )

    def test_t3_loss_is_shifted_masked_and_differentiable_twice(self):
        LLAMA_CONFIGS[_TinyT3Config.llama_config_name] = _TINY_LLAMA
        try:
            torch.manual_seed(3)
            model = T3(_TinyT3Config())
            self.assertIs(model.get_input_embeddings(), model.text_emb)
            model.gradient_checkpointing_enable({"use_reentrant": False})
            self.assertTrue(model.tfmr.gradient_checkpointing)
            text = torch.tensor(
                [[1, 3, 4, 2], [1, 5, 6, 2]],
                dtype=torch.long,
            )
            speech = torch.tensor(
                [[10, 3, 4, 5, 11], [10, 6, 7, 8, 11]],
                dtype=torch.long,
            )
            condition = T3Cond(
                speaker_emb=torch.randn(2, 4),
                cond_prompt_speech_tokens=torch.tensor([[3, 4], [6, 7]]),
                emotion_adv=0.5,
            )
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                text_loss, speech_loss = model.loss(
                    t3_cond=condition,
                    text_tokens=text,
                    text_token_lens=torch.tensor([4, 4]),
                    speech_tokens=speech,
                    speech_token_lens=torch.tensor([5, 5]),
                    prompt_lens=torch.tensor([1, 1]),
                )
                (text_loss + speech_loss).backward()
                self.assertIsNotNone(model.text_emb.weight.grad)
                self.assertIsNotNone(model.speech_head.weight.grad)
            self.assertIsNone(condition.cond_prompt_speech_emb)
            model.gradient_checkpointing_disable()
            model.eval()
            inference_condition = T3Cond(
                speaker_emb=torch.randn(1, 4),
                cond_prompt_speech_tokens=torch.tensor([[3, 4]]),
                emotion_adv=torch.tensor([[[0.5]]]),
            )
            generated = model.inference(
                t3_cond=inference_condition,
                text_tokens=torch.tensor([1, 3, 2]),
                max_new_tokens=2,
                do_sample=False,
                cfg_weight=0.5,
                stop_on_eos=False,
            )
            self.assertEqual(tuple(generated.shape), (1, 2))
        finally:
            LLAMA_CONFIGS.pop(_TinyT3Config.llama_config_name, None)

    def test_causal_loss_matches_manual_shift_and_prompt_mask(self):
        logits = torch.randn(2, 5, 13, requires_grad=True)
        tokens = torch.tensor(
            [[1, 2, 3, 4, 5], [1, 7, 8, 9, 0]],
            dtype=torch.long,
        )
        lengths = torch.tensor([5, 4])
        prompt_lens = torch.tensor([2, 1])
        actual = T3._causal_cross_entropy(
            logits,
            tokens,
            lengths,
            prompt_lens=prompt_lens,
            name="speech",
        )
        targets = tokens[:, 1:].clone()
        positions = torch.arange(4).unsqueeze(0)
        targets[(positions >= (lengths - 1).unsqueeze(1)) | (positions < prompt_lens.unsqueeze(1))] = -100
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 13),
            targets.reshape(-1),
            ignore_index=-100,
        )
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertIsNotNone(logits.grad)

    def test_community_vocabulary_expansion_uses_mean_initialization(self):
        t3 = SimpleNamespace(
            text_emb=nn.Embedding(4, 3),
            text_head=nn.Linear(3, 4, bias=False),
            hp=SimpleNamespace(
                start_text_token=1,
                stop_text_token=0,
                text_tokens_dict_size=4,
            ),
        )
        with torch.no_grad():
            t3.text_emb.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3))
            t3.text_head.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10)
        old_embedding = t3.text_emb.weight.detach().clone()
        old_head = t3.text_head.weight.detach().clone()
        resize_t3_text_vocabulary(t3, 7)
        self.assertTrue(torch.equal(t3.text_emb.weight[:4], old_embedding))
        self.assertTrue(torch.equal(t3.text_head.weight[:4], old_head))
        self.assertTrue(
            torch.equal(
                t3.text_emb.weight[4:],
                old_embedding.mean(dim=0, keepdim=True).expand(3, -1),
            ))
        self.assertTrue(
            torch.equal(
                t3.text_head.weight[4:],
                old_head.mean(dim=0, keepdim=True).expand(3, -1),
            ))
        self.assertEqual(t3.hp.text_tokens_dict_size, 7)

    def test_native_lora_policy_freezes_dense_t3_parameters(self):
        LLAMA_CONFIGS[_TinyT3Config.llama_config_name] = _TINY_LLAMA
        try:
            runtime = nn.Module()
            runtime.t3 = T3(_TinyT3Config())
            runtime.s3gen = nn.Module()
            runtime.s3gen.flow = nn.Linear(4, 4)
            runtime.ve = nn.Linear(4, 4)
            wrapper = SimpleNamespace(
                model=runtime,
                config=SimpleNamespace(
                    training_component="language_model",
                    training_text_vocab_size=None,
                    training_lora_rank=2,
                    training_lora_alpha=4.0,
                    training_lora_dropout=0.0,
                    training_lora_target_modules=("q_proj", ),
                    training_lora_modules_to_train=(
                        "text_emb",
                        "text_head",
                    ),
                    training_lora_seed=7,
                ),
                load_for_training=lambda: None,
            )
            adapter = ChatterboxTrainingAdapter(
                wrapper,
                get_training_spec("chatterbox"),
            ).setup()
            self.assertIsNotNone(adapter._lora_injection)
            trainable = {name for name, parameter in runtime.t3.named_parameters() if parameter.requires_grad}
            self.assertTrue(any(name.endswith("lora_a") for name in trainable))
            self.assertTrue(any(name.endswith("lora_b") for name in trainable))
            self.assertIn("text_emb.weight", trainable)
            self.assertIn("text_head.weight", trainable)
            self.assertNotIn("speech_emb.weight", trainable)
            adapter.train()
            self.assertTrue(runtime.t3.training)
            self.assertFalse(runtime.s3gen.training)
            self.assertFalse(runtime.ve.training)
            for module in adapter._lora_injection.modules.values():
                with torch.no_grad():
                    module.lora_b.fill_(0.01)
            adapter._lora_injection.merge()
            portable = {
                name.replace(".base.", "."): value
                for name, value in runtime.t3.state_dict().items()
                if not name.endswith((".lora_a", ".lora_b"))
            }
            dense = T3(_TinyT3Config())
            dense.load_state_dict(portable, strict=True)
            wrapped_q = adapter._lora_injection.modules["tfmr.layers.0.self_attn.q_proj"].base.weight
            self.assertTrue(torch.equal(
                dense.tfmr.layers[0].self_attn.q_proj.weight,
                wrapped_q,
            ))
            adapter._lora_injection.unmerge()
        finally:
            LLAMA_CONFIGS.pop(_TinyT3Config.llama_config_name, None)

    def test_community_precomputed_records_derive_lengths(self):
        wrapper = SimpleNamespace(
            model=None,
            config=SimpleNamespace(training_component="language_model"),
        )
        adapter = ChatterboxTrainingAdapter(
            wrapper,
            get_training_spec("chatterbox"),
        )
        dataset = adapter.create_dataset([{
            "text_tokens": torch.tensor([1, 3, 2]),
            "speech_tokens": torch.tensor([10, 4, 5, 11]),
            "speaker_emb": torch.ones(4),
            "prompt_tokens": torch.tensor([4, 5]),
        }])
        sample = dataset[0]
        self.assertEqual(sample["text_token_lens"], 3)
        self.assertEqual(sample["speech_token_lens"], 4)
        self.assertEqual(sample["prompt_lens"], 2)

        raw_dataset = adapter.create_dataset([
            {
                "audio": torch.ones(100),
                "sampling_rate": 16_000,
                "text": "first",
            },
            {
                "audio": torch.ones(80),
                "sampling_rate": 16_000,
                "text": "second",
            },
        ])
        collated = adapter.data_collator([raw_dataset[0], raw_dataset[1]])
        self.assertEqual(tuple(collated["audio"].shape), (2, 100))
        self.assertEqual(collated["audio_lengths"].tolist(), [100, 80])

    def test_raw_audio_preprocessing_builds_t3_and_flow_batches(self):
        hp = SimpleNamespace(
            max_text_tokens=16,
            max_speech_tokens=16,
            speech_cond_prompt_len=3,
            start_text_token=1,
            stop_text_token=2,
            start_speech_token=10,
            stop_speech_token=11,
        )
        t3 = SimpleNamespace(device=torch.device("cpu"), hp=hp)
        s3gen = SimpleNamespace(
            device=torch.device("cpu"),
            tokenizer=_FakeSpeechTokenizer(),
            mel_extractor=lambda waveform: torch.ones(
                waveform.shape[0],
                80,
                10,
            ),
            speaker_encoder=_FakeSpeakerEncoder(),
        )
        wrapper = SimpleNamespace(
            model=SimpleNamespace(
                t3=t3,
                s3gen=s3gen,
                ve=_FakeVoiceEncoder(),
                tokenizer=_FakeTextTokenizer(),
            ),
            config=SimpleNamespace(
                training_component="language_model",
                training_max_text_tokens=12,
                training_max_speech_tokens=12,
                training_prompt_duration=0.1,
                training_conditioning_dropout=0.0,
            ),
        )
        adapter = ChatterboxTrainingAdapter(
            wrapper,
            get_training_spec("chatterbox"),
        )
        audio = torch.randn(2, 3_200) * 0.01
        raw = {
            "audio": audio,
            "audio_lengths": torch.tensor([3_200, 2_800]),
            "sampling_rate": 16_000,
            "text": ["first", "second"],
        }
        context = TrainingContext(
            phase=get_training_spec("chatterbox").get_phase("language_model"),
            inputs=raw,
            is_training=False,
        )
        language_batch = adapter._prepare_raw_language_model_batch(
            raw,
            context,
        )
        self.assertEqual(tuple(language_batch["text_tokens"].shape), (2, 5))
        self.assertTrue(torch.equal(
            language_batch["speech_tokens"][:, 0],
            torch.tensor([10, 10]),
        ))
        self.assertTrue(torch.equal(
            language_batch["speech_tokens"][:, -1],
            torch.tensor([11, 11]),
        ))
        self.assertEqual(language_batch["prompt_lens"].tolist(), [3, 3])

        flow_batch = adapter._prepare_raw_flow_batch({
            "audio": audio,
            "audio_lengths": torch.tensor([3_200, 2_800]),
            "sampling_rate": 16_000,
        })
        self.assertEqual(tuple(flow_batch["speech_token"].shape), (2, 4))
        self.assertEqual(tuple(flow_batch["speech_feat"].shape), (2, 80, 8))
        self.assertEqual(flow_batch["speech_feat_len"].tolist(), [8, 8])
        self.assertEqual(tuple(flow_batch["embedding"].shape), (2, 192))

    def test_causal_flow_objective_backpropagates(self):
        flow = CausalMaskedDiffWithXvec(
            input_size=8,
            output_size=4,
            spk_embed_dim=3,
            vocab_size=10,
            encoder=_TinyFlowEncoder(8),
            decoder=_TinyFlowDecoder(),
        )
        losses = flow.compute_loss(
            {
                "speech_token": torch.tensor([[1, 2, 3], [4, 5, 0]]),
                "speech_token_len": torch.tensor([3, 2]),
                "speech_feat": torch.randn(2, 4, 3),
                "speech_feat_len": torch.tensor([3, 2]),
                "embedding": torch.randn(2, 3),
            },
            torch.device("cpu"),
        )
        losses["loss"].backward()
        self.assertIsNotNone(flow.input_embedding.weight.grad)
        self.assertIsNotNone(flow.spk_embed_affine_layer.weight.grad)
        self.assertIsNotNone(flow.decoder.scale.grad)

    def test_shared_conformer_remains_checkpoint_and_mask_compatible(self):
        block = ConformerWrapper(
            dim=8,
            dim_head=4,
            heads=2,
            ff_mult=1,
            conv_expansion_factor=2,
            conv_kernel_size=3,
        ).eval()
        state = block.state_dict()
        self.assertEqual((len(state), sum(x.numel() for x in state.values())), (34, 5_285))
        self.assertEqual(tuple(state["attn.fn.to_q.weight"].shape), (8, 8))
        self.assertEqual(
            tuple(state["conv.net.4.conv.weight"].shape),
            (16, 1, 3),
        )

        hidden = torch.randn(2, 7, 8)
        mask = torch.tensor([
            [True, True, True, True, True, True, True],
            [True, True, True, True, False, False, False],
        ])
        output = block(
            hidden_states=hidden,
            attention_mask=mask,
        )
        self.assertEqual(output.shape, hidden.shape)
        self.assertTrue(torch.isfinite(output).all())

        decoder = Decoder(
            in_channels=4,
            out_channels=2,
            channels=(8, ),
            dropout=0.0,
            attention_head_dim=4,
            n_blocks=1,
            num_mid_blocks=1,
            num_heads=2,
            down_block_type="conformer",
            mid_block_type="conformer",
            up_block_type="conformer",
        ).eval()
        decoder_mask = torch.tensor([
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, False, False, False],
        ])
        prediction = decoder(
            torch.randn(2, 2, 8),
            decoder_mask[:, None, :],
            torch.randn(2, 2, 8),
            torch.tensor([0.2, 0.8]),
        )
        self.assertEqual(tuple(prediction.shape), (2, 2, 8))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_native_tokenizer_executes_released_bpe_contract(self):
        vocabulary = {
            "[STOP]": 0,
            "[UNK]": 1,
            "[SPACE]": 2,
            "h": 3,
            "e": 4,
            "l": 5,
            "o": 6,
            "he": 7,
            "hel": 8,
            "hell": 9,
            "hello": 10,
            "[START]": 11,
        }
        added = [{
            "id": vocabulary[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
        } for token in ("[STOP]", "[UNK]", "[SPACE]", "[START]")]
        payload = {
            "added_tokens": added,
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": "[UNK]",
                "vocab": vocabulary,
                "merges": ["h e", "he l", "hel l", "hell o"],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            tokenizer = EnTokenizer(path)
            token_ids = tokenizer.text_to_tokens("hello hello")
            self.assertEqual(token_ids.tolist(), [[10, 2, 10]])
            self.assertEqual(tokenizer.decode(token_ids), "hello hello")

    def test_native_safetensors_round_trip_is_strict(self):
        source = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        target = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.safetensors"
            export_module_safetensors(
                source,
                checkpoint,
                component="test",
            )
            load_module_safetensors(target, checkpoint)
            for name, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[name]))
            incompatible = nn.Sequential(
                nn.Linear(3, 5),
                nn.Linear(5, 2),
            )
            before = {name: value.detach().clone() for name, value in incompatible.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                load_module_safetensors(incompatible, checkpoint)
            for name, value in incompatible.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))

    def test_expanded_t3_vocabulary_is_inspected_without_loading_payload(self):
        module = SimpleNamespace(
            state_dict=lambda: {
                "text_emb.weight": torch.zeros(709, 8),
                "text_head.weight": torch.zeros(709, 8),
            })
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "t3.safetensors"
            export_module_safetensors(
                module,
                checkpoint,
                component="t3",
            )
            self.assertEqual(
                inspect_t3_text_vocabulary_size(checkpoint),
                709,
            )

    def test_watermarker_uses_weights_only_and_preserves_public_rate(self):
        with patch.object(torch, "load", wraps=torch.load) as loader:
            watermarker = NativePerthWatermarker(device="cpu")
        self.assertTrue(loader.call_args.kwargs["weights_only"])
        waveform = torch.randn(5_000) * 0.01
        output = watermarker.apply_watermark(
            waveform,
            sample_rate=24_000,
        )
        self.assertEqual(output.ndim, 1)
        self.assertGreater(output.numel(), 0)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
