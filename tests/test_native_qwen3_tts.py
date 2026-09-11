from __future__ import annotations

import ast
import base64
import inspect
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import torch

from voicehub.models.qwen3tts.native import modeling as qwen3_tts_modeling
from voicehub.models.qwen3tts.native.checkpoint import (
    export_qwen3_tts_decoder,
    export_qwen3_tts_model,
    load_qwen3_tts_decoder_checkpoint,
    load_qwen3_tts_model_checkpoint,
)
from voicehub.models.qwen3tts.native.codec import Qwen3TTSSpeechDecoder
from voicehub.models.qwen3tts.native.configuration import (
    Qwen3TTSArchitectureConfig,
    Qwen3TTSDecoderConfig,
    Qwen3TTSTokenizerConfig,
)
from voicehub.models.qwen3tts.native.modeling import Qwen3TTSForConditionalGeneration
from voicehub.models.qwen3tts.native.registration import create_qwen3_tts_architecture_spec
from voicehub.models.qwen3tts.native.runtime import _load_reference_audio, load_qwen3_tts_runtime
from voicehub.models.qwen3tts.native.tokenization import EXPECTED_TTS_TOKEN_IDS
from voicehub.optimization.codecs import discover_codec_compile_targets
from voicehub.tokenization import encode_gpt2_token


def _tiny_architecture(*, role: str = "custom_voice") -> Qwen3TTSArchitectureConfig:
    speaker_ids = {"voicehub": 20} if role == "custom_voice" else {}
    return Qwen3TTSArchitectureConfig.from_dict({
        "model_type": "qwen3_tts",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
        "tts_model_size": "0b6",
        "tts_model_type": role,
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
            "spk_id": speaker_ids,
            "spk_is_dialect": {
                name: False
                for name in speaker_ids
            },
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


def _tiny_decoder() -> Qwen3TTSDecoderConfig:
    return Qwen3TTSDecoderConfig.from_dict({
        "latent_dim": 8,
        "codebook_dim": 8,
        "codebook_size": 16,
        "decoder_dim": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "max_position_embeddings": 128,
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_quantizers": 4,
        "num_semantic_quantizers": 1,
        "sliding_window": 8,
        "upsample_rates": [2, 2],
        "upsampling_ratios": [2],
        "vector_quantization_hidden_dimension": 8,
    })


def _eager_attention_reference(
    attention,
    hidden_states,
    *,
    cosine,
    sine,
    attention_bias,
):
    batch, time, _ = hidden_states.shape
    query = attention.q_norm(
        attention.q_proj(hidden_states).view(
            batch,
            time,
            attention.num_heads,
            attention.head_dim,
        )).transpose(1, 2)
    key = attention.k_norm(
        attention.k_proj(hidden_states).view(
            batch,
            time,
            attention.num_kv_heads,
            attention.head_dim,
        )).transpose(1, 2)
    value = attention.v_proj(hidden_states).view(
        batch,
        time,
        attention.num_kv_heads,
        attention.head_dim,
    ).transpose(1, 2)
    query, key = qwen3_tts_modeling.apply_rotary_embedding(
        query,
        key,
        cosine,
        sine,
    )
    key = qwen3_tts_modeling._expand_kv(key, attention.groups)
    value = qwen3_tts_modeling._expand_kv(value, attention.groups)
    weights = torch.matmul(query, key.transpose(-1, -2)) * attention.scale
    weights = torch.softmax(
        weights.float() + attention_bias,
        dim=-1,
    ).to(dtype=query.dtype)
    weights = torch.nn.functional.dropout(
        weights,
        p=attention.dropout,
        training=attention.training,
    )
    output = torch.matmul(weights, value)
    output = output.transpose(1, 2).reshape(
        batch,
        time,
        attention.num_heads * attention.head_dim,
    )
    return attention.o_proj(output)


def _portable_architecture() -> Qwen3TTSArchitectureConfig:
    values = _tiny_architecture().to_dict()
    values.update({
        "im_start_token_id": EXPECTED_TTS_TOKEN_IDS["<|im_start|>"],
        "im_end_token_id": EXPECTED_TTS_TOKEN_IDS["<|im_end|>"],
        "tts_pad_token_id": EXPECTED_TTS_TOKEN_IDS["<tts_pad>"],
        "tts_bos_token_id": EXPECTED_TTS_TOKEN_IDS["<tts_text_bos>"],
        "tts_eos_token_id": EXPECTED_TTS_TOKEN_IDS["<tts_text_eod>"],
    })
    values["talker_config"]["text_vocab_size"] = 151_936
    return Qwen3TTSArchitectureConfig.from_dict(values)


def _write_tokenizer_assets(directory: Path) -> None:
    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    vocabulary[encode_gpt2_token(b"aa")] = 256
    (directory / "vocab.json").write_text(
        json.dumps(vocabulary),
        encoding="utf-8",
    )
    (directory / "merges.txt").write_text(
        "#version: 0.2\na a\n",
        encoding="utf-8",
    )
    added_tokens = {
        str(token_id): {
            "content": spelling,
            "lstrip": False,
            "rstrip": False,
            "special": True,
        }
        for spelling, token_id in EXPECTED_TTS_TOKEN_IDS.items()
    }
    (directory / "tokenizer_config.json").write_text(
        json.dumps({
            "added_tokens_decoder": added_tokens,
            "add_prefix_space": False,
            "errors": "replace",
        }),
        encoding="utf-8",
    )


class NativeQwen3TTSTests(unittest.TestCase):

    def test_source_metadata_is_pinned_and_apache_licensed(self):
        root = Path(__file__).parents[1]
        metadata = json.loads(
            (root / 'voicehub/models/qwen3tts/native/SOURCE.json').read_text(encoding="utf-8"))
        self.assertEqual(metadata["license"], "Apache-2.0")
        self.assertEqual(len(metadata["revision"]), 40)
        self.assertIn("Mimi-derived", metadata["limitations"][0])

    def test_native_modules_do_not_import_upstream_runtimes(self):
        root = (Path(__file__).parents[1] / 'voicehub/models/qwen3tts/native')
        forbidden = {
            "diffusers",
            "huggingface_hub",
            "librosa",
            "numpy",
            "onnxruntime",
            "safetensors",
            "soundfile",
            "transformers",
        }
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".", 1)[0])
            self.assertFalse(
                imported & forbidden,
                f"{path.name} imports {sorted(imported & forbidden)!r}",
            )

    def test_official_base_namespaces_have_audited_counts(self):
        root = Path(__file__).parents[1]
        official = json.loads((root / "voicehub" / "models" / "qwen3tts" / "source" /
                               "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(official["license"], "Apache-2.0")

        # The tiny test does not download multi-GB weights. Meta construction
        # still proves the graph's complete persistent namespace.
        config = Qwen3TTSArchitectureConfig.from_dict({
            'architectures': ["Qwen3TTSForConditionalGeneration"],
            "model_type": "qwen3_tts",
            "tokenizer_type": "qwen3_tts_tokenizer_12hz",
            "tts_model_size": "0b6",
            "tts_model_type": "base",
            "speaker_encoder_config": {
                "enc_dim": 1024,
                "sample_rate": 24_000,
            },
            "talker_config": {
                "vocab_size": 3072,
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rope_theta": 1_000_000,
                "num_code_groups": 16,
                "text_hidden_size": 2048,
                "text_vocab_size": 151_936,
                "codec_eos_token_id": 2150,
                "codec_think_id": 2154,
                "codec_nothink_id": 2155,
                "codec_think_bos_id": 2156,
                "codec_think_eos_id": 2157,
                "codec_pad_id": 2148,
                "codec_bos_id": 2149,
                "codec_language_id": {
                    "english": 2050,
                },
                "spk_id": {},
                "spk_is_dialect": {},
                "code_predictor_config": {
                    "vocab_size": 2048,
                    "hidden_size": 1024,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 5,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "rope_theta": 1_000_000,
                    "num_code_groups": 16,
                },
            },
        })
        model = Qwen3TTSForConditionalGeneration(config, initialize=False)
        self.assertEqual(len(model.state_dict()), 478)
        self.assertEqual(
            sum(value.numel() for value in model.state_dict().values()),
            914_643_008,
        )

        decoder = Qwen3TTSSpeechDecoder(
            Qwen3TTSDecoderConfig(),
            initialize=False,
        )
        self.assertEqual(len(decoder.state_dict()), 271)
        self.assertEqual(
            sum(value.numel() for value in decoder.state_dict().values()),
            114_323_137,
        )

    def test_exact_sft_losses_are_differentiable(self):
        torch.manual_seed(7)
        model = Qwen3TTSForConditionalGeneration(_tiny_architecture())
        embeddings = torch.randn(2, 7, 8, requires_grad=True)
        labels = torch.tensor([
            [-100, -100, 1, 2, 3, 4, 5],
            [-100, -100, 2, 3, 4, 5, 6],
        ])
        output = model.talker(
            inputs_embeds=embeddings,
            attention_mask=torch.ones(2, 7, dtype=torch.long),
            labels=labels,
            output_hidden_states=True,
        )
        self.assertIsNotNone(output.loss)
        assert output.loss is not None
        hidden = output.hidden_states[0][-1][:, :-1].reshape(-1, 8)
        codes = torch.randint(0, 16, (hidden.shape[0], 4))
        _, sub_loss = model.talker.forward_sub_talker_finetune(
            codes,
            hidden,
        )
        loss = output.loss + 0.3 * sub_loss
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.talker.model.layers[0].self_attn.q_proj.weight.grad)
        self.assertIsNotNone(model.talker.code_predictor.lm_head[0].weight.grad)

    def test_sdpa_matches_eager_attention_forward_and_backward(self):
        torch.manual_seed(41)
        attention = (
            Qwen3TTSForConditionalGeneration(_tiny_architecture()).talker.model.layers[0].self_attn.train())
        hidden_states = torch.randn(2, 5, 8, requires_grad=True)
        cosine = torch.ones(2, 5, attention.head_dim)
        sine = torch.zeros_like(cosine)
        attention_bias = qwen3_tts_modeling._causal_bias(
            torch.tensor([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
            ]),
            batch=2,
            time=5,
            sliding_window=None,
            device=hidden_states.device,
        )

        actual = attention(
            hidden_states,
            cosine=cosine,
            sine=sine,
            attention_bias=attention_bias,
        )
        actual.square().mean().backward()
        actual_input_gradient = hidden_states.grad.detach().clone()
        actual_parameter_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in attention.named_parameters() if parameter.grad is not None
        }

        attention.zero_grad(set_to_none=True)
        reference_hidden_states = hidden_states.detach().clone().requires_grad_(True)
        expected = _eager_attention_reference(
            attention,
            reference_hidden_states,
            cosine=cosine,
            sine=sine,
            attention_bias=attention_bias,
        )
        expected.square().mean().backward()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            actual_input_gradient,
            reference_hidden_states.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(
            set(actual_parameter_gradients),
            {name
             for name, parameter in attention.named_parameters() if parameter.grad is not None},
        )
        for name, parameter in attention.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(
                    actual_parameter_gradients[name],
                    parameter.grad,
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_sdpa_routes_gqa_and_dropout_with_legacy_fallback(self):
        attention = (Qwen3TTSForConditionalGeneration(_tiny_architecture()).talker.model.layers[0].self_attn)
        attention.dropout = 0.25
        hidden_states = torch.randn(1, 3, 8)
        cosine = torch.ones(1, 3, attention.head_dim)
        sine = torch.zeros_like(cosine)
        attention_bias = qwen3_tts_modeling._causal_bias(
            None,
            batch=1,
            time=3,
            sliding_window=None,
            device=hidden_states.device,
        )
        calls = []

        def capture_sdpa(query, key, value, **kwargs):
            calls.append((query.shape, key.shape, value.shape, dict(kwargs)))
            return torch.zeros_like(query)

        attention.train()
        with (
                mock.patch.object(qwen3_tts_modeling, "_SDPA_SUPPORTS_GQA", True),
                mock.patch.object(
                    qwen3_tts_modeling.functional,
                    "scaled_dot_product_attention",
                    side_effect=capture_sdpa,
                ),
        ):
            attention(
                hidden_states,
                cosine=cosine,
                sine=sine,
                attention_bias=attention_bias,
            )

        attention.eval()
        with (
                mock.patch.object(qwen3_tts_modeling, "_SDPA_SUPPORTS_GQA", False),
                mock.patch.object(
                    qwen3_tts_modeling.functional,
                    "scaled_dot_product_attention",
                    side_effect=capture_sdpa,
                ),
        ):
            attention(
                hidden_states,
                cosine=cosine,
                sine=sine,
                attention_bias=attention_bias,
            )

        native, fallback = calls
        self.assertEqual(native[1][1], attention.num_kv_heads)
        self.assertTrue(native[3]["enable_gqa"])
        self.assertEqual(native[3]["dropout_p"], 0.25)
        self.assertEqual(native[3]["scale"], attention.scale)
        self.assertEqual(fallback[1][1], attention.num_heads)
        self.assertNotIn("enable_gqa", fallback[3])
        self.assertEqual(fallback[3]["dropout_p"], 0.0)
        self.assertEqual(fallback[3]["scale"], attention.scale)

    def test_talker_exposes_last_hidden_state_without_history(self):
        torch.manual_seed(43)
        talker = Qwen3TTSForConditionalGeneration(_tiny_architecture()).talker.eval()
        embeddings = torch.randn(2, 5, 8)
        attention_mask = torch.ones(2, 5, dtype=torch.long)

        compact = talker(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        detailed = talker(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        self.assertIsNone(compact.hidden_states)
        self.assertEqual(compact.last_hidden_state.shape, (2, 5, 8))
        self.assertIsNotNone(detailed.hidden_states)
        assert detailed.hidden_states is not None
        self.assertEqual(
            len(detailed.hidden_states[0]),
            talker.config.num_hidden_layers + 1,
        )
        torch.testing.assert_close(
            compact.last_hidden_state,
            detailed.hidden_states[0][-1],
        )
        torch.testing.assert_close(
            compact.last_hidden_state,
            detailed.last_hidden_state,
        )
        torch.testing.assert_close(compact.logits, detailed.logits)

    def test_generation_uses_checkpoint_text_pad_token(self):
        torch.manual_seed(9)
        model = Qwen3TTSForConditionalGeneration(_tiny_architecture()).eval()
        codes = model.talker.generate_codes(
            prompt_embeds=torch.randn(1, 3, 8),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            trailing_text_hidden=torch.empty(1, 0, 8),
            max_new_tokens=1,
            do_sample=False,
            subtalker_dosample=False,
        )
        self.assertEqual(codes.shape, (1, 4))

    def test_generation_suppresses_eos_for_two_initial_frames(self):

        class EosHead(torch.nn.Module):

            def __init__(self, vocabulary_size: int, eos_token_id: int):
                super().__init__()
                self.vocabulary_size = vocabulary_size
                self.eos_token_id = eos_token_id

            def forward(self, hidden_states):
                logits = torch.zeros(
                    hidden_states.shape[0],
                    self.vocabulary_size,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                logits[:, self.eos_token_id] = 10
                return logits

        torch.manual_seed(8)
        model = Qwen3TTSForConditionalGeneration(_tiny_architecture()).eval()
        model.talker.codec_head = EosHead(
            model.talker.config.vocab_size,
            model.talker.config.codec_eos_token_id,
        )
        codes = model.talker.generate_codes(
            prompt_embeds=torch.randn(1, 3, 8),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            trailing_text_hidden=torch.randn(1, 5, 8),
            max_new_tokens=5,
            do_sample=False,
            subtalker_dosample=False,
        )
        self.assertEqual(codes.shape, (2, 4))
        self.assertFalse(bool((codes[:, 0] == model.talker.config.codec_eos_token_id).any()))

    def test_talker_kv_cache_matches_full_causal_forward(self):
        torch.manual_seed(10)
        backbone = Qwen3TTSForConditionalGeneration(_tiny_architecture()).eval().talker.model
        prefix = torch.randn(2, 4, 8)
        continuation = torch.randn(2, 3, 8)
        full_mask = torch.ones(2, 7, dtype=torch.long)
        with torch.no_grad():
            expected, _ = backbone(
                torch.cat((prefix, continuation), dim=1),
                attention_mask=full_mask,
            )
            _, cache = backbone.forward_with_cache(
                prefix,
                attention_mask=full_mask[:, :4],
            )
            actual, cache = backbone.forward_with_cache(
                continuation,
                attention_mask=full_mask,
                past_key_values=cache,
            )
        torch.testing.assert_close(actual, expected[:, 4:])
        self.assertEqual(cache[0][0].shape[2], 7)

    def test_cached_subtalker_generation_matches_full_recomputation(self):
        torch.manual_seed(12)
        talker = Qwen3TTSForConditionalGeneration(_tiny_architecture()).eval().talker
        hidden = torch.randn(2, 8)
        first_code = torch.tensor([2, 3])
        with torch.no_grad():
            actual = talker._residual_codes(
                hidden,
                first_code,
                do_sample=False,
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                generator=None,
            )
            expected_codes = [first_code]
            embeddings = [
                hidden.unsqueeze(1),
                talker.get_input_embeddings()(first_code[:, None]),
            ]
            for index, (table, head) in enumerate(zip(
                    talker.code_predictor.get_input_embeddings(),
                    talker.code_predictor.lm_head,
            )):
                projected = talker.code_predictor.small_to_mtp_projection(torch.cat(embeddings, dim=1))
                states, _ = talker.code_predictor.model(projected)
                code = head(states[:, -1]).argmax(dim=-1)
                expected_codes.append(code)
                if index + 1 < len(talker.code_predictor.lm_head):
                    embeddings.append(table(code[:, None]))
        torch.testing.assert_close(
            actual,
            torch.stack(expected_codes, dim=-1),
        )

    def test_reference_audio_data_url_uses_native_wave_decoder(self):
        payload = io.BytesIO()
        with wave.open(payload, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(24_000)
            stream.writeframes(b"\x00\x00" * 512)
        encoded = base64.b64encode(payload.getvalue()).decode("ascii")
        audio = _load_reference_audio(f"data:audio/wav;base64,{encoded}")
        self.assertEqual(audio.sampling_rate, 24_000)
        self.assertEqual(audio.waveform.shape, (512, ))
        raw_audio = _load_reference_audio(encoded)
        self.assertEqual(raw_audio.waveform.shape, (512, ))

    def test_model_safetensors_roundtrip_is_exact(self):
        torch.manual_seed(11)
        config = _tiny_architecture()
        source = Qwen3TTSForConditionalGeneration(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = export_qwen3_tts_model(
                source,
                Path(directory) / "model.safetensors",
            )
            target = Qwen3TTSForConditionalGeneration(
                config,
                initialize=False,
            )
            load_qwen3_tts_model_checkpoint(
                target,
                path,
                device="cpu",
                dtype=torch.float32,
            )
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name])

    def test_speech_decoder_roundtrip_and_gradients(self):
        torch.manual_seed(13)
        config = _tiny_decoder()
        source = Qwen3TTSSpeechDecoder(config)
        codes = torch.randint(0, config.codebook_size, (1, 4, 3))
        waveform = source(codes)
        self.assertEqual(waveform.shape, (1, 1, 24))
        waveform.square().mean().backward()
        self.assertIsNotNone(source.decoder[0].conv.weight.grad)

        with tempfile.TemporaryDirectory() as directory:
            path = export_qwen3_tts_decoder(
                source,
                Path(directory) / "model.safetensors",
            )
            target = Qwen3TTSSpeechDecoder(config, initialize=False)
            load_qwen3_tts_decoder_checkpoint(
                target,
                path,
                device="cpu",
                dtype=torch.float32,
                verify_official=False,
            )
            with torch.no_grad():
                actual = target(codes)
        torch.testing.assert_close(waveform.detach(), actual)

    def test_speech_decoder_optimizer_targets_capture_safe_inner_boundary(self):
        torch.manual_seed(14)
        config = _tiny_decoder()
        decoder = Qwen3TTSSpeechDecoder(config).eval()
        codes = torch.randint(
            0,
            config.codebook_size,
            (1, config.num_quantizers, 3),
        )
        target, = discover_codec_compile_targets(
            decoder,
            mode="inference",
        )
        self.assertEqual(target.label, "codec.decode.qwen3_tts.decode_codes")
        self.assertEqual(target.attribute, "_decode_codes_unchecked")
        self.assertEqual(target.component, "decode")
        source = inspect.getsource(getattr(type(decoder), target.attribute))
        for value_sync in (
                "bool(",
                ".item(",
                ".any(",
                ".tolist(",
                ".numpy(",
        ):
            self.assertNotIn(value_sync, source)

        with mock.patch.object(
                decoder,
                "_decode_codes_unchecked",
                wraps=decoder._decode_codes_unchecked,
        ) as inner:
            expected = decoder(codes)
        inner.assert_called_once_with(codes)

        with mock.patch.object(
                decoder,
                "_validate_codes",
                side_effect=AssertionError("validation entered capture target"),
        ):
            actual = getattr(target.owner, target.attribute)(codes)
        torch.testing.assert_close(actual, expected)

        invalid = codes.clone()
        invalid[..., 0] = config.codebook_size
        with mock.patch.object(
                decoder,
                "_decode_codes_unchecked",
                wraps=decoder._decode_codes_unchecked,
        ) as inner:
            with self.assertRaisesRegex(ValueError, "outside"):
                decoder(invalid)
        inner.assert_not_called()

    def test_portable_runtime_export_reloads_without_upstream_runtime(self):
        torch.manual_seed(17)
        architecture = _portable_architecture()
        decoder_config = _tiny_decoder()
        tokenizer_config = Qwen3TTSTokenizerConfig(
            decoder_config=decoder_config,
            encoder_valid_num_quantizers=decoder_config.num_quantizers,
            decode_upsample_rate=decoder_config.total_upsample,
            encode_downsample_rate=decoder_config.total_upsample,
        )
        tokenizer_config.validate()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "source"
            speech = source / "speech_tokenizer"
            speech.mkdir(parents=True)
            (source / "config.json").write_text(
                json.dumps(architecture.to_dict()),
                encoding="utf-8",
            )
            (speech / "config.json").write_text(
                json.dumps(tokenizer_config.to_dict()),
                encoding="utf-8",
            )
            _write_tokenizer_assets(source)
            export_qwen3_tts_model(
                Qwen3TTSForConditionalGeneration(architecture),
                source / "model.safetensors",
            )
            export_qwen3_tts_decoder(
                Qwen3TTSSpeechDecoder(decoder_config),
                speech / "model.safetensors",
            )

            runtime = load_qwen3_tts_runtime(
                source,
                device="cpu",
                compute_dtype="float32",
            )
            exported = runtime.save_pretrained(workspace / "exported")
            reloaded = load_qwen3_tts_runtime(
                exported,
                device="cpu",
                compute_dtype="float32",
            )

        self.assertEqual(reloaded.config.tts_model_type, "custom_voice")
        self.assertEqual(
            tuple(reloaded.model.state_dict()),
            tuple(runtime.model.state_dict()),
        )
        for name, expected in runtime.model.state_dict().items():
            torch.testing.assert_close(
                reloaded.model.state_dict()[name],
                expected,
            )

    def test_architecture_spec_is_honest_about_native_scope(self):
        spec = create_qwen3_tts_architecture_spec()
        self.assertTrue(spec.capabilities.training)
        self.assertIn("sdpa", spec.capabilities.optimization_passes)
        self.assertIn("voice-clone-xvector", spec.capabilities.features)
        self.assertIn(
            "Checkpoint-exact native Mimi-derived encoder",
            spec.metadata["icl_reference_encoder"],
        )
        self.assertIn(
            "native-speech-tokenizer-encoder",
            spec.capabilities.features,
        )


if __name__ == "__main__":
    unittest.main()
