from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from voicehub.architectures.irodoritts.checkpoint import (
    irodori_header_fingerprint,
    load_irodori_safetensors,
    native_irodori_tensor_shapes,
    save_irodori_safetensors,
)
from voicehub.architectures.irodoritts.codec import load_digest_gated_codec_payload
from voicehub.architectures.irodoritts.configuration import IrodoriModelConfig
from voicehub.architectures.irodoritts.metadata import IRODORI_CHECKPOINTS, IRODORI_CODEC_CHECKPOINT
from voicehub.architectures.irodoritts.modeling import TextToLatentRFDiT
from voicehub.architectures.irodoritts.runtime import InferenceRuntime, SamplingRequest
from voicehub.architectures.irodoritts.tokenization import IrodoriTokenizer
from voicehub.architectures.irodoritts.training import IrodoriBatchProcessor, irodori_training_step
from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.irodoritts.inference import IrodoriTTSForTextToSpeech
from voicehub.registry import get_model_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(**overrides) -> IrodoriModelConfig:
    values = {
        "adaln_rank": 8,
        "duration_attention_heads": 2,
        "duration_hidden_dim": 16,
        "duration_layers": 1,
        "latent_dim": 4,
        "mlp_ratio": 2.0,
        "model_dim": 16,
        "num_heads": 2,
        "num_layers": 2,
        "speaker_dim": 16,
        "speaker_heads": 2,
        "speaker_layers": 1,
        "speaker_mlp_ratio": 2.0,
        "text_dim": 16,
        "text_heads": 2,
        "text_layers": 1,
        "text_mlp_ratio": 2.0,
        "text_vocab_size": 266,
        "timestep_embed_dim": 8,
        "variant": "custom",
    }
    values.update(overrides)
    return IrodoriModelConfig(**values)


def _tokenizer_payload() -> dict:
    vocabulary = [
        ["<unk>", 0.0],
        ["<s>", 0.0],
        ["</s>", 0.0],
        ["<MASK|LLM-jp>", 0.0],
        ["<PAD|LLM-jp>", 0.0],
        ["<CLS|LLM-jp>", 0.0],
        ["<SEP|LLM-jp>", 0.0],
        ["<EOD|LLM-jp>", 0.0],
    ]
    vocabulary.extend([f"<0x{value:02X}>", -10.0] for value in range(256))
    vocabulary.extend((["\u2581", -1.0], ["known", -0.1]))
    return {
        "normalizer": {
            "type":
            "Sequence",
            "normalizers": [
                {
                    "type": "Replace",
                    "pattern": {
                        "Regex": "(?<!\\n)^",
                    },
                    "content": "\u2581",
                },
                {
                    "type": "Replace",
                    "pattern": {
                        "Regex": " ",
                    },
                    "content": "\u2581",
                },
            ],
        },
        "pre_tokenizer": None,
        "decoder": {
            "type":
            "Sequence",
            "decoders": [
                {
                    "type": "ByteFallback",
                },
                {
                    "type": "Replace",
                    "pattern": {
                        "Regex": "\u2581",
                    },
                    "content": " ",
                },
                {
                    "type": "Fuse",
                },
                {
                    "type": "Replace",
                    "pattern": {
                        "Regex": "(?<!\\n)^ ",
                    },
                    "content": "",
                },
            ],
        },
        "model": {
            "type": "Unigram",
            "unk_id": 0,
            "byte_fallback": True,
            "vocab": vocabulary,
        },
    }


def _write_tokenizer(directory: Path) -> IrodoriTokenizer:
    tokenizer_path = directory / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(_tokenizer_payload(), ensure_ascii=False),
        encoding="utf-8",
    )
    config_path = directory / "tokenizer_config.json"
    config_path.write_text(
        json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast",
        }),
        encoding="utf-8",
    )
    return IrodoriTokenizer.from_files(
        tokenizer_path,
        tokenizer_config=config_path,
        expected_vocabulary_size=266,
    )


class _FakeCodec:
    sample_rate = 48_000

    def __init__(self, latent_dim: int) -> None:
        self.latent_dim = latent_dim
        self.calls = 0

    def encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        value = waveform.float().mean()
        return value.expand(1, 3, self.latent_dim).clone()


class NativeIrodoriTests(unittest.TestCase):

    def test_no_reference_synthesis_matches_fixed_padding_and_omits_speaker_guidance(self):
        torch.manual_seed(12)
        config = _tiny_config(use_duration_predictor=False)
        model = TextToLatentRFDiT(config).eval()
        nn.init.normal_(model.out_proj.weight, std=0.02)
        codec = SimpleNamespace(
            device=torch.device("cpu"),
            sample_rate=48_000,
            hop_length=1920,
            decode_latent=lambda latent: latent[:, :, 0].repeat_interleave(1920, dim=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = InferenceRuntime(
                model=model,
                model_cfg=config,
                tokenizer=_write_tokenizer(Path(directory)),
                codec=codec,
                model_device="cpu",
            )
            ids, mask = runtime._batch_tokens("known", batch_size=2, max_length=8)
            self.assertEqual(ids.tolist(), [[1, 264, 265, 4, 4, 4, 4, 4]] * 2)
            self.assertEqual(mask.tolist(), [[True] * 3 + [False] * 5] * 2)
            empty_ids, empty_mask = runtime._batch_tokens("", batch_size=1, max_length=8, allow_empty=True)
            self.assertEqual(empty_ids.tolist(), [[1] + [4] * 7])
            self.assertFalse(empty_mask.any())
            batches = []
            handle = model.blocks[0].register_forward_pre_hook(
                lambda module, args, kwargs: batches.append(kwargs["x"].shape[0]), with_kwargs=True)
            try:
                for mode, scale in (("independent", None), ("joint", None), ("independent", 4.0)):
                    with self.subTest(mode=mode, scale=scale):
                        batches.clear()
                        result = runtime.synthesize(
                            SamplingRequest(
                                text="known",
                                no_ref=True,
                                seconds=0.08,
                                min_seconds=0.01,
                                max_text_len=8,
                                num_steps=3,
                                seed=12,
                                cfg_guidance_mode=mode,
                                cfg_scale=scale,
                                speaker_kv_scale=2.0,
                                trim_tail=False))
                        self.assertTrue(torch.isfinite(result.audio).all())
                        self.assertEqual(max(batches), 1 if mode == "joint" else 2)
            finally:
                handle.remove()

    def test_public_namespaces_stay_lazy_and_registry_points_to_native_graph(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import voicehub.architectures.irodoritts; "
                    "import voicehub.models.irodoritts; "
                    "print('torch' in sys.modules)"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False")
        spec = get_model_spec("irodori-tts")
        self.assertEqual(spec.architecture, "irodoritts-rf-dit")
        self.assertIn("voicehub-native", spec.capabilities)
        self.assertIsNone(spec.install_extra)

    def test_all_published_meta_inventories_match_audited_facts(self):
        for variant, facts in IRODORI_CHECKPOINTS.items():
            with self.subTest(variant=variant):
                shapes = native_irodori_tensor_shapes(IrodoriModelConfig.for_variant(variant))
                self.assertEqual(len(shapes), facts["tensors"])
                self.assertEqual(
                    sum(math.prod(shape) for shape in shapes.values()),
                    facts["parameters"],
                )
                self.assertEqual(
                    irodori_header_fingerprint({
                        name: ("F32", shape)
                        for name, shape in shapes.items()
                    }),
                    facts["header_fingerprint"],
                )

    def test_tokenizer_uses_utf8_byte_fallback_and_rejects_nonfinite_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = _write_tokenizer(root)
            marker_id = tokenizer.vocabulary["\u2581"]
            encoded = tokenizer.encode("🦊")
            self.assertEqual(
                encoded,
                (
                    tokenizer.bos_token_id,
                    marker_id,
                    *(tokenizer.vocabulary[f"<0x{value:02X}>"] for value in "🦊".encode()),
                ),
            )
            self.assertEqual(tokenizer.decode(encoded), "🦊")

            malformed = _tokenizer_payload()
            malformed["model"]["vocab"][-1][1] = float("nan")
            path = root / "malformed.json"
            path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid record"):
                IrodoriTokenizer.from_files(
                    path,
                    expected_vocabulary_size=266,
                )

    def test_joint_flow_duration_backward_supports_gradient_checkpointing(self):
        torch.manual_seed(7)
        model = TextToLatentRFDiT(_tiny_config()).train()
        model.set_gradient_checkpointing(True)
        nn.init.normal_(model.out_proj.weight, std=0.02)
        batch = {
            "target_latent": torch.randn(2, 5, 4),
            "latent_mask": torch.tensor([[True, True, True, True, True], [True, True, True, False, False]]),
            "text_input_ids": torch.randint(0, 266, (2, 4)),
            "text_mask": torch.tensor([[True, True, True, True], [True, True, False, False]]),
            "ref_latent": torch.randn(2, 3, 4),
            "ref_mask": torch.tensor([[True, True, True], [True, True, False]]),
            "caption_input_ids": None,
            "caption_mask": None,
            "duration_features": torch.randn(2, 14),
            "duration_target": torch.log1p(torch.tensor([5.0, 3.0])),
            "duration_has_speaker": torch.tensor([True, True]),
            "duration_has_caption": None,
        }
        output = irodori_training_step(
            model,
            batch,
            objective="joint",
            text_condition_dropout=0.0,
            speaker_condition_dropout=0.0,
        )
        self.assertTrue(torch.isfinite(output["loss"]).item())
        self.assertTrue(torch.isfinite(output["flow_loss"]).item())
        self.assertTrue(torch.isfinite(output["duration_loss"]).item())
        output["loss"].backward()
        self.assertGreater(model.in_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            model.duration_predictor.token_out_proj.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_checkpoint_roundtrip_and_malformed_exports_fail_closed(self):
        config = _tiny_config()
        model = TextToLatentRFDiT(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_irodori_safetensors(
                model,
                config,
                root / "model.safetensors",
            )
            loaded, loaded_config = load_irodori_safetensors(checkpoint)
            self.assertEqual(loaded_config.to_dict(), config.to_dict())
            self.assertFalse(any(tensor.is_meta for tensor in loaded.state_dict().values()))

            blob = root / "checkpoint-blob"
            checkpoint.rename(blob)
            checkpoint.symlink_to(blob.name)
            linked, linked_config = load_irodori_safetensors(checkpoint)
            self.assertEqual(linked_config.to_dict(), config.to_dict())
            self.assertFalse(any(tensor.is_meta for tensor in linked.state_dict().values()))

            incompatible_config = _tiny_config(dropout=0.2)
            with self.assertRaisesRegex(
                    CheckpointCompatibilityError,
                    "does not match",
            ):
                save_irodori_safetensors(
                    model,
                    incompatible_config,
                    root / "misrepresented.safetensors",
                )

            missing_state = dict(model.state_dict())
            missing_state.pop(next(iter(missing_state)))
            malformed = save_safetensors(
                missing_state,
                root / "missing.safetensors",
                metadata={
                    "config_json": json.dumps(config.to_dict()),
                },
            )
            with self.assertRaisesRegex(
                    CheckpointCompatibilityError,
                    "missing=",
            ):
                load_irodori_safetensors(malformed)

            invalid_variant = config.to_dict()
            invalid_variant["variant"] = "v1"
            v1_checkpoint = save_safetensors(
                model.state_dict(),
                root / "v1.safetensors",
                metadata={
                    "config_json": json.dumps(invalid_variant),
                },
            )
            with self.assertRaisesRegex(
                    CheckpointCompatibilityError,
                    "Unsupported Irodori variant",
            ):
                load_irodori_safetensors(v1_checkpoint)

            legacy_config = config.to_dict()
            legacy_config.pop("variant")
            legacy_config["version"] = 1
            legacy_checkpoint = save_safetensors(
                model.state_dict(),
                root / "legacy-v1.safetensors",
                metadata={
                    "config_json": json.dumps(legacy_config),
                },
            )
            with self.assertRaisesRegex(
                    CheckpointCompatibilityError,
                    "v1 checkpoints",
            ):
                load_irodori_safetensors(legacy_checkpoint)

            with torch.no_grad():
                model.out_proj.bias[0] = float("nan")
            destination = root / "nonfinite.safetensors"
            with self.assertRaisesRegex(
                    CheckpointCompatibilityError,
                    "non-finite",
            ):
                save_irodori_safetensors(model, config, destination)
            self.assertFalse(destination.exists())

    def test_codec_archive_is_digest_gated_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            path.write_bytes(b"audited-codec-fixture")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            payload = {
                "state_dict": {},
                "metadata": {
                    "kwargs": dict(IRODORI_CODEC_CHECKPOINT["constructor"]),
                },
            }
            with patch(
                    "voicehub.architectures.irodoritts.codec.torch.load",
                    return_value=payload,
            ) as loader:
                with self.assertRaisesRegex(ValueError, "Refusing to unpickle"):
                    load_digest_gated_codec_payload(
                        path,
                        expected_sha256="0" * 64,
                    )
                loader.assert_not_called()
                self.assertIs(
                    load_digest_gated_codec_payload(
                        path,
                        expected_sha256=digest,
                    ),
                    payload,
                )
                self.assertTrue(loader.call_args.kwargs["weights_only"])

                payload["metadata"]["kwargs"]["unexpected"] = True
                with self.assertRaisesRegex(ValueError, "constructor metadata"):
                    load_digest_gated_codec_payload(
                        path,
                        expected_sha256=digest,
                    )

    def test_batch_processor_accepts_raw_file_and_batched_preencoded_inputs(self):
        config = _tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = _write_tokenizer(root)
            codec = _FakeCodec(config.latent_dim)
            processor = IrodoriBatchProcessor(
                config=config,
                tokenizer=tokenizer,
                codec=codec,
                device="cpu",
            )

            batched = processor({
                "text": ["known", "🦊"],
                "target_latent": torch.randn(2, 4, config.latent_dim),
                "reference_latent": torch.randn(2, 3, config.latent_dim),
            })
            self.assertEqual(batched["target_latent"].shape, (2, 4, 4))
            self.assertEqual(batched["ref_latent"].shape, (2, 3, 4))
            self.assertEqual(batched["duration_target"].shape, (2, ))
            self.assertEqual(
                batched["duration_has_speaker"].tolist(),
                [True, True],
            )

            latent_path = save_safetensors(
                {
                    "latent": torch.randn(5, config.latent_dim),
                },
                root / "latent.safetensors",
            )
            encoded = processor({
                "text": "known",
                "target_latent": latent_path,
            })
            self.assertEqual(encoded["target_latent"].shape, (1, 5, 4))
            torch.testing.assert_close(
                encoded["ref_latent"],
                encoded["target_latent"],
            )
            self.assertEqual(
                encoded["duration_has_speaker"].tolist(),
                [False],
            )

            batched_latent_path = save_safetensors(
                {
                    "latent": torch.randn(2, 6, config.latent_dim),
                },
                root / "batched-latent.safetensors",
            )
            encoded_batch = processor({
                "text": ["known", "known"],
                "target_latent": batched_latent_path,
            })
            self.assertEqual(
                encoded_batch["target_latent"].shape,
                (2, 6, 4),
            )

            raw = processor({
                "text": ["known", "known"],
                "waveform": [torch.ones(2_000), torch.full((2_000, ), 0.5)],
                "sample_rate": [48_000, 48_000],
            })
            self.assertEqual(raw["target_latent"].shape, (2, 3, 4))
            torch.testing.assert_close(raw["ref_latent"], raw["target_latent"])
            self.assertEqual(
                raw["duration_has_speaker"].tolist(),
                [False, False],
            )
            self.assertEqual(codec.calls, 2)

    def test_training_device_sync_supports_codec_with_read_only_device(self):

        class ReadOnlyCodec:

            def __init__(self) -> None:
                self.model = nn.Linear(2, 2)

            @property
            def device(self) -> torch.device:
                return next(self.model.parameters()).device

        runtime = SimpleNamespace(
            model=nn.Linear(2, 2),
            model_device=torch.device("cpu"),
            codec=ReadOnlyCodec(),
        )
        wrapper = IrodoriTTSForTextToSpeech(device="cpu")
        wrapper.model = runtime
        wrapper._set_training_device("cpu")
        self.assertEqual(runtime.model_device, torch.device("cpu"))
        self.assertEqual(runtime.codec_device, torch.device("cpu"))

    def test_native_export_tokenizer_is_preferred_for_fresh_runtime(self):
        captured = {}

        class RuntimeFactory:

            @staticmethod
            def from_key(key):
                captured["key"] = key
                return SimpleNamespace(
                    codec=SimpleNamespace(sample_rate=48_000),
                    synthesize=lambda request: request,
                )

        runtime_module = SimpleNamespace(
            RuntimeKey=lambda **values: SimpleNamespace(**values),
            InferenceRuntime=RuntimeFactory,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.safetensors"
            checkpoint.touch()
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            codec = root / "weights.pth"
            codec.touch()
            wrapper = IrodoriTTSForTextToSpeech(
                model_path=str(checkpoint),
                device="cpu",
            )
            with patch(
                    "voicehub.models.irodoritts.inference.resolve_pretrained_file",
                    return_value=codec,
            ) as resolver:
                _, sample_rate = wrapper._build_runtime(runtime_module)
            self.assertIsNone(captured["key"].checkpoint_model_id)
            self.assertIsNone(captured["key"].checkpoint_revision)

            published_wrapper = IrodoriTTSForTextToSpeech(device="cpu")
            with (
                    patch.object(
                        published_wrapper,
                        "_resolve_checkpoint",
                        return_value=checkpoint,
                    ),
                    patch(
                        "voicehub.models.irodoritts.inference.resolve_pretrained_file",
                        return_value=codec,
                    ),
            ):
                published_wrapper._build_runtime(runtime_module)
            published_key = captured["key"]

        self.assertEqual(sample_rate, 48_000)
        self.assertEqual(
            published_key.checkpoint_model_id,
            IRODORI_CHECKPOINTS["v3"]["model_id"],
        )
        self.assertEqual(
            published_key.checkpoint_revision,
            IRODORI_CHECKPOINTS["v3"]["revision"],
        )
        self.assertEqual(
            Path(published_key.tokenizer_directory).resolve(),
            root.resolve(),
        )
        resolver.assert_called_once_with(
            wrapper.config.codec_name_or_path,
            "weights.pth",
            revision=wrapper.config.codec_revision,
        )


if __name__ == "__main__":
    unittest.main()
