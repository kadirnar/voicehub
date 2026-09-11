import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from voicehub.models.causal_lm.native.configuration import Qwen3Config
from voicehub.models.mosstts.native.checkpoint import inspect_mosstts_checkpoint
from voicehub.models.mosstts.native.codec import MossAudioCodecConfig, NativeMossAudioCodec
from voicehub.models.mosstts.native.codec_checkpoint import (
    export_moss_audio_tokenizer_checkpoint,
    load_moss_audio_tokenizer_checkpoint,
)
from voicehub.models.mosstts.native.codec_configuration import MossAudioTokenizerConfig
from voicehub.models.mosstts.native.codec_modeling import MossAudioTokenizerModel as MossAudioTokenizerV2Model
from voicehub.models.mosstts.native.codec_modeling_v1 import MossAudioTokenizerModel as MossAudioTokenizerV1Model
from voicehub.models.mosstts.native.configuration import MossGPT2Config, MossTTSConfig
from voicehub.models.mosstts.native.metadata import MOSS_CODEC_CHECKPOINTS
from voicehub.models.mosstts.native.modeling import build_mosstts_model
from voicehub.models.mosstts.native.processing import MossTTSProcessor
from voicehub.models.mosstts.native.runtime import MossTTSRuntime
from voicehub.models.mosstts.native.tokenization import MossTextTokenizer
from voicehub.models.mosstts.native.training import MossTTSDataset
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import get_training_spec


def _header_fingerprint(state: dict[str, torch.Tensor]) -> str:
    labels = {
        torch.bfloat16: "BF16",
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
    }
    rows = [
        f"{name}|{labels[value.dtype]}|" + "x".join(str(dimension) for dimension in value.shape)
        for name, value in state.items()
    ]
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


def _tiny_codec(
    version: int,
    *,
    num_quantizers: int = 2,
) -> NativeMossAudioCodec:
    sample_rate = 24_000 if version == 1 else 48_000
    downsample_rate = 1_920 if version == 1 else 3_840
    channels = 1 if version == 1 else 2
    architecture_config = MossAudioTokenizerConfig(
        version=str(version),
        sampling_rate=sample_rate,
        downsample_rate=downsample_rate,
        number_channels=channels,
        enable_channel_interleave=version == 2,
        compute_dtype="fp32",
        encoder_kwargs=[{
            "module_type": "PatchedPretransform",
            "patch_size": downsample_rate,
        }],
        decoder_kwargs=[{
            "module_type": "PatchedPretransform",
            "patch_size": downsample_rate,
        }],
        quantizer_type="rlfq",
        quantizer_kwargs={
            "input_dim": downsample_rate,
            "rvq_dim": 4,
            "output_dim": downsample_rate,
            "num_quantizers": num_quantizers,
            "codebook_size": 8,
            "codebook_dim": 2,
            "quantizer_type": "rlfq",
        },
    )
    codec_config = MossAudioCodecConfig(
        version=version,
        sample_rate=sample_rate,
        downsample_rate=downsample_rate,
        channels=channels,
        code_dimension=downsample_rate,
        rvq_dimension=4,
        output_dimension=downsample_rate,
        num_quantizers=num_quantizers,
        codebook_size=8,
        codebook_dimension=2,
        channel_interleave=version == 2,
    )
    model_class = (MossAudioTokenizerV1Model if version == 1 else MossAudioTokenizerV2Model)
    return NativeMossAudioCodec(
        model_class(architecture_config),
        codec_config,
        architecture_config=architecture_config,
    )


def _tiny_qwen(vocabulary_size: int = 64) -> Qwen3Config:
    return Qwen3Config(
        model_type="qwen3",
        vocab_size=vocabulary_size,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _tiny_tts_config(variant: str) -> MossTTSConfig:
    common = {
        "language_config": _tiny_qwen(),
        "audio_vocab_size": 8,
        "audio_pad_token_id": 8,
        "pad_token_id": 0,
        "im_start_token_id": 1,
        "im_end_token_id": 2,
        "audio_start_token_id": 3,
        "audio_end_token_id": 4,
        "audio_user_slot_token_id": 5,
        "audio_assistant_slot_token_id": 6,
        "initializer_range": 0.02,
        "codec_repository": "native-test-codec",
    }
    if variant == "delay":
        return MossTTSConfig(
            variant="delay",
            n_vq=32,
            audio_codebook_sizes=(8, ) * 32,
            sample_rate=24_000,
            audio_assistant_delay_slot_token_id=7,
            **common,
        )
    if variant == "local":
        return MossTTSConfig(
            variant="local",
            n_vq=32,
            audio_codebook_sizes=(8, ) * 32,
            sample_rate=24_000,
            audio_assistant_delay_slot_token_id=7,
            local_config=_tiny_qwen(),
            additional_mlp_ffn_hidden_size=16,
            **common,
        )
    if variant == "local_v1_5":
        return MossTTSConfig(
            variant="local_v1_5",
            n_vq=12,
            audio_codebook_sizes=(8, ) * 12,
            sample_rate=48_000,
            local_config=MossGPT2Config(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=1,
                max_position_embeddings=128,
            ),
            local_text_head_mode="binary",
            **common,
        )
    if variant == "realtime":
        common.update({
            "audio_vocab_size": 1_027,
            "audio_pad_token_id": 1_024,
        })
        return MossTTSConfig(
            variant="realtime",
            n_vq=16,
            audio_codebook_sizes=(1_027, ) * 16,
            sample_rate=24_000,
            local_config=_tiny_qwen(1_027),
            reference_audio_pad_token_id=8,
            text_pad_token_id=9,
            **common,
        )
    raise AssertionError(variant)


class _TinyTokenizer(MossTextTokenizer):
    _SPECIAL_TOKENS = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "<|audio_start|>": 3,
        "<|audio_end|>": 4,
        "<|audio_user_slot|>": 5,
        "<|audio_assistant_gen_slot|>": 6,
        "<|audio_assistant_delay_slot|>": 7,
        "<|audio_pad|>": 8,
        "<|text_pad|>": 9,
    }

    def __init__(self):
        pass

    @property
    def pad_token_id(self) -> int:
        return 0

    def encode_ids(self, text: str) -> list[int]:
        output: list[int] = []
        remaining = text
        while remaining:
            match = next(
                ((spelling, token_id)
                 for spelling, token_id in self._SPECIAL_TOKENS.items() if remaining.startswith(spelling)),
                None,
            )
            if match is None:
                output.append(20 + ord(remaining[0]) % 20)
                remaining = remaining[1:]
            else:
                output.append(match[1])
                remaining = remaining[len(match[0]):]
        return output


class _TinySemanticModel(nn.Module):

    def __init__(self, config: MossTTSConfig):
        super().__init__()
        self.config = config
        self.anchor = nn.Parameter(torch.zeros(()))


class NativeMossCodecTests(unittest.TestCase):

    def test_official_codec_graph_inventories_match_pinned_headers(self):
        cases = (
            (
                "OpenMOSS-Team/MOSS-Audio-Tokenizer",
                MossAudioTokenizerConfig(
                    version="1",
                    sampling_rate=24_000,
                    downsample_rate=1_920,
                    number_channels=1,
                    enable_channel_interleave=False,
                ),
                MossAudioTokenizerV1Model,
            ),
            (
                "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
                MossAudioTokenizerConfig(),
                MossAudioTokenizerV2Model,
            ),
        )
        for repository, config, model_class in cases:
            with self.subTest(repository=repository):
                with torch.device("meta"):
                    model = model_class(config)
                state = model.state_dict()
                facts = MOSS_CODEC_CHECKPOINTS[repository]
                self.assertEqual(len(state), facts["tensors"])
                self.assertEqual(
                    sum(value.numel() for value in state.values()),
                    facts["parameters"],
                )
                self.assertEqual(
                    sum(value.numel() * value.element_size() for value in state.values()),
                    facts["tensor_bytes"],
                )
                self.assertEqual(
                    _header_fingerprint(state),
                    facts["header_fingerprint"],
                )

    def test_v1_and_v2_encode_and_decode_raw_waveforms(self):
        for version in (1, 2):
            with self.subTest(version=version):
                codec = _tiny_codec(version)
                frames = (codec.config.downsample_rate if version == 1 else codec.config.downsample_rate // 2)
                waveform = torch.randn(
                    2,
                    codec.config.channels,
                    frames,
                )

                encoded = codec.encode(waveform)
                decoded = codec.decode(
                    encoded.audio_codes,
                    encoded.audio_code_lengths,
                )

                self.assertEqual(
                    tuple(encoded.audio_codes.shape),
                    (2, 1, 2),
                )
                self.assertEqual(
                    tuple(decoded.waveform.shape),
                    (2, codec.config.channels, frames),
                )
                self.assertEqual(decoded.sample_rate, codec.config.sample_rate)

    def test_tiny_codec_safetensors_round_trip_is_exact(self):
        codec = _tiny_codec(2)
        with tempfile.TemporaryDirectory() as directory:
            path = export_moss_audio_tokenizer_checkpoint(
                codec.model,
                Path(directory) / "model.safetensors",
            )
            report = inspect_mosstts_checkpoint(path)
            with torch.device("meta"):
                reloaded = MossAudioTokenizerV2Model(codec.architecture_config, )
            load_moss_audio_tokenizer_checkpoint(
                reloaded,
                path,
                device="cpu",
            )

        self.assertEqual(report.tensor_count, len(codec.model.state_dict()))
        for name, expected in codec.model.state_dict().items():
            self.assertTrue(
                torch.equal(
                    reloaded.state_dict()[name],
                    expected,
                ),
                name,
            )


class NativeMossTrainingTests(unittest.TestCase):

    def test_shared_training_profile_uses_the_native_adapter(self):
        from voicehub.models.mosstts.modeling import MossTTSForTextToSpeech
        from voicehub.models.mosstts.native.training import NativeMossTTSTrainingAdapter

        spec = get_training_spec("mosstts")
        model = MossTTSForTextToSpeech(device="cpu")
        adapter = AutoTrainingAdapter.from_model(model)

        self.assertIsInstance(adapter, NativeMossTTSTrainingAdapter)
        self.assertFalse(adapter.is_ready)
        self.assertIs(spec.support, TrainingSupport.NATIVE)
        self.assertEqual(spec.module_paths, ("model", ))
        self.assertEqual(spec.component_paths, ("model", ))
        self.assertEqual(spec.default_phase, "semantic_language_model")
        self.assertEqual(
            spec.get_phase().required_inputs,
            ("input_ids", "attention_mask", "labels"),
        )
        self.assertEqual(
            spec.get_phase().frozen_component_paths,
            ("training_backend.codec", ),
        )
        self.assertIn(
            "voicehub.models.mosstts.native.training:"
            "NativeMossTTSTrainingAdapter",
            spec.source_entrypoints,
        )

    def test_all_semantic_variants_have_full_gradient_training(self):
        for variant in ("delay", "local", "local_v1_5", "realtime"):
            with self.subTest(variant=variant):
                config = _tiny_tts_config(variant)
                model = build_mosstts_model(config)
                input_ids = torch.zeros(
                    2,
                    4,
                    config.channels,
                    dtype=torch.long,
                )
                input_ids[..., 0] = torch.randint(0, 10, (2, 4))
                audio_high = 100 if variant == "realtime" else 7
                input_ids[..., 1:] = torch.randint(
                    0,
                    audio_high,
                    (2, 4, config.n_vq),
                )

                output = model(
                    input_ids,
                    labels=input_ids.clone(),
                )
                self.assertIsNotNone(output.loss)
                output.loss.backward()

                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_realtime_has_buffered_high_level_generation(self):
        config = _tiny_tts_config("realtime")
        model = build_mosstts_model(config)
        for parameter in model.parameters():
            parameter.data.zero_()
        runtime = SimpleNamespace(
            model=model,
            processor=MossTTSProcessor(config, _TinyTokenizer()),
            device=torch.device("cpu"),
            codec=SimpleNamespace(config=SimpleNamespace(codebook_size=1_024), ),
        )

        generated = MossTTSRuntime.generate_codes(
            runtime,
            "hello",
            max_new_tokens=3,
            audio_temperature=0.0,
        )

        self.assertEqual(len(generated), 1)
        self.assertEqual(tuple(generated[0].audio_codes.shape), (3, 16))
        self.assertTrue(generated[0].audio_codes.eq(0).all())

    def test_raw_audio_dataset_encodes_with_frozen_native_codec(self):
        config = _tiny_tts_config("delay")
        tokenizer = _TinyTokenizer()
        runtime = MossTTSRuntime(
            model=_TinySemanticModel(config),
            tokenizer=tokenizer,
            processor=MossTTSProcessor(config, tokenizer),
            codec=_tiny_codec(1, num_quantizers=32),
        )
        dataset = MossTTSDataset(
            [{
                "text": "hello",
                "audio": torch.randn(1_920),
                "sampling_rate": 24_000,
            }],
            processor=runtime.processor,
            runtime=runtime,
        )

        batch = dataset.collate_fn([dataset[0]])

        self.assertEqual(batch["input_ids"].shape[0], 1)
        self.assertEqual(batch["input_ids"].shape[2], 33)
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)
        self.assertTrue(bool(batch["labels"].ne(-100).any()))
        self.assertFalse(any(parameter.requires_grad for parameter in runtime.codec.parameters()))

    def test_raw_tensor_training_requires_sampling_rate(self):
        config = _tiny_tts_config("delay")
        tokenizer = _TinyTokenizer()
        runtime = MossTTSRuntime(
            model=_TinySemanticModel(config),
            tokenizer=tokenizer,
            processor=MossTTSProcessor(config, tokenizer),
            codec=_tiny_codec(1, num_quantizers=32),
        )

        with self.assertRaisesRegex(ValueError, "sampling_rate"):
            runtime.prepare_training_batch([{
                "text": "hello",
                "audio": torch.randn(1_920),
            }])


if __name__ == "__main__":
    unittest.main()
