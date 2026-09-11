from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional at import time
    torch = None

from voicehub.models.asr_nemotron import NemotronASRConfig, NemotronForSpeechRecognition
from voicehub.models.asr_nemotron.native.artifacts import resolve_nemotron_asr_artifacts
from voicehub.models.asr_nemotron.native.configuration import NemotronASRArchitectureConfig, NemotronEncoderConfig
from voicehub.models.asr_nemotron.native.metadata import (
    NEMOTRON_ASR_HEADER_FINGERPRINT,
    NEMOTRON_ASR_PARAMETER_COUNT,
    NEMOTRON_ASR_REVISION,
    NEMOTRON_ASR_TENSOR_COUNT,
)
from voicehub.models.asr_nemotron.native.tokenization import NemotronASRTokenizer
from voicehub.models.asr_nemotron.training_asr_nemotron import NativeNemotronASRTrainingAdapter
from voicehub.processing.waveform import save_pcm_wave
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.specs import get_training_spec


def _tiny_config(
    *,
    vocab_size: int = 16,
    blank_token_id: int = 15,
    num_mel_bins: int = 8,
) -> NemotronASRArchitectureConfig:
    return NemotronASRArchitectureConfig(
        vocab_size=vocab_size,
        decoder_hidden_size=8,
        num_decoder_layers=1,
        hidden_act="relu",
        max_symbols_per_step=3,
        encoder_config=NemotronEncoderConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            conv_kernel_size=3,
            subsampling_factor=2,
            subsampling_conv_channels=4,
            num_mel_bins=num_mel_bins,
            subsampling_conv_kernel_size=3,
            subsampling_conv_stride=2,
            dropout=0.0,
            dropout_positions=0.0,
            layerdrop=0.0,
            activation_dropout=0.0,
            attention_dropout=0.0,
            max_position_embeddings=128,
            sliding_window=8,
            default_num_lookahead_tokens=0,
            supported_num_lookahead_tokens=(0, ),
        ),
        pad_token_id=0,
        blank_token_id=blank_token_id,
        num_prompts=4,
        prompt_intermediate_size=8,
        default_prompt_id=0,
    )


def _write_tokenizer(path: Path) -> Path:
    document = {
        "version":
        "1.0",
        "truncation":
        None,
        "padding":
        None,
        "added_tokens": [
            {
                "id": 0,
                "content": "<unk>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 7,
                "content": "<en-US>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 13087,
                "content": "<pad>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 13088,
                "content": "<blank>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": {
            "type":
            "Sequence",
            "normalizers": [
                {
                    "type": "Precompiled",
                    "precompiled_charsmap": "audited-test-map",
                },
                {
                    "type": "Strip",
                    "strip_left": False,
                    "strip_right": True,
                },
                {
                    "type": "Replace",
                    "pattern": {
                        "Regex": " {2,}"
                    },
                    "content": "▁",
                },
            ],
        },
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": True,
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{
                "Sequence": {
                    "id": "A",
                    "type_id": 0
                }
            }],
            "pair": [
                {
                    "Sequence": {
                        "id": "A",
                        "type_id": 0
                    }
                },
                {
                    "Sequence": {
                        "id": "B",
                        "type_id": 1
                    }
                },
            ],
            "special_tokens": {},
        },
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": {
                "<unk>": 0,
                "▁": 1,
                "h": 2,
                "i": 3,
                "▁h": 4,
                "▁hi": 5,
            },
            "merges": [["▁", "h"], ["▁h", "i"]],
        },
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


@unittest.skipUnless(torch is not None, "PyTorch is required")
class NemotronNativeArchitectureTests(unittest.TestCase):

    def test_provider_package_discovery_does_not_import_torch(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_nemotron as provider; "
            "assert 'torch' not in sys.modules; "
            "assert 'NemotronForSpeechRecognition' in provider.__all__; "
            "assert 'NativeNemotronASRTrainingAdapter' in provider.__all__")
        subprocess.run(
            [sys.executable, "-c", command],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_shared_training_registry_selects_native_full_model_adapter(self):
        wrapper = NemotronForSpeechRecognition(
            lazy_load=True,
            device="cpu",
        )
        adapter = AutoTrainingAdapter.from_model(wrapper)
        spec = get_training_spec("asr_nemotron")

        self.assertIsInstance(adapter, NativeNemotronASRTrainingAdapter)
        self.assertEqual(spec.module_paths, ("model", ))
        self.assertEqual(
            spec.component_paths,
            (
                "model.encoder",
                "model.encoder_projector",
                "model.prompt_projector",
                "model.decoder",
                "model.joint",
            ),
        )
        self.assertTrue(all(entrypoint.startswith("voicehub.") for entrypoint in spec.source_entrypoints))

    def test_official_meta_inventory_matches_audited_checkpoint(self):
        from voicehub.models.asr_nemotron.native.checkpoint import (
            native_nemotron_asr_tensor_shapes,
            nemotron_asr_header_fingerprint,
        )

        shapes = native_nemotron_asr_tensor_shapes(NemotronASRArchitectureConfig(), )
        self.assertEqual(len(shapes), NEMOTRON_ASR_TENSOR_COUNT)
        self.assertEqual(
            sum(
                tensor.numel()
                for tensor in (torch.empty(shape, device="meta") for shape in shapes.values())),
            NEMOTRON_ASR_PARAMETER_COUNT,
        )
        self.assertEqual(
            nemotron_asr_header_fingerprint({
                name: ("F32", shape)
                for name, shape in shapes.items()
            }),
            NEMOTRON_ASR_HEADER_FINGERPRINT,
        )

    def test_native_rnnt_forward_and_backward_are_finite(self):
        from voicehub.models.asr_nemotron.native.modeling import Nemotron3_5ASRForRNNT

        model = Nemotron3_5ASRForRNNT(_tiny_config())
        model.gradient_checkpointing_enable()
        model.train()
        output = model(
            input_features=torch.randn(1, 17, 8),
            attention_mask=torch.ones(1, 17, dtype=torch.long),
            prompt_ids=torch.tensor([0]),
            decoder_input_ids=torch.tensor([[15, 2, 3, 4]]),
            labels=torch.tensor([[2, 3, 4]]),
            label_lengths=torch.tensor([3]),
            num_lookahead_tokens=0,
        )
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_checkpoint_streaming_assignment_rebuilds_meta_buffer(self):
        from voicehub.checkpointing import SafeTensorReader, save_safetensors
        from voicehub.models.asr_nemotron.native.checkpoint import NemotronASRCheckpointAdapter
        from voicehub.models.asr_nemotron.native.modeling import Nemotron3_5ASRForRNNT

        config = _tiny_config()
        original = Nemotron3_5ASRForRNNT(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_safetensors(
                original.state_dict(),
                Path(directory) / "model.safetensors",
            )
            with torch.device("meta"):
                restored = Nemotron3_5ASRForRNNT(
                    config,
                    initialize=False,
                )
            with SafeTensorReader(checkpoint) as reader:
                report = NemotronASRCheckpointAdapter().load_assign_streaming(
                    restored,
                    reader,
                    config,
                    device="cpu",
                )
        self.assertTrue(report.is_compatible)
        self.assertFalse(any(tensor.device.type == "meta" for tensor in restored.state_dict().values()))
        self.assertEqual(
            restored.encoder.encode_positions.inv_freq.device.type,
            "cpu",
        )

    def test_checkpoint_preflight_rejects_invalid_tensor_kinds_atomically(self):
        from voicehub.checkpointing.errors import CheckpointCompatibilityError
        from voicehub.models.asr_nemotron.native.checkpoint import NemotronASRCheckpointAdapter
        from voicehub.models.asr_nemotron.native.modeling import Nemotron3_5ASRForRNNT

        config = _tiny_config()
        original = Nemotron3_5ASRForRNNT(config)
        state = dict(original.state_dict())
        name = sorted(state)[-1]
        value = state[name]
        variants = {
            "integral": torch.zeros_like(
                value,
                dtype=torch.int64,
            ),
            "complex": torch.complex(
                value,
                torch.zeros_like(value),
            ),
            "quantized": torch.quantize_per_tensor(
                value,
                scale=0.1,
                zero_point=0,
                dtype=torch.qint8,
            ),
            "sparse": value.to_sparse(),
            "meta": torch.empty_like(
                value,
                device="meta",
            ),
        }
        for label, invalid in variants.items():
            with self.subTest(tensor_kind=label):
                source = {
                    **state,
                    name: invalid,
                }
                with torch.device("meta"):
                    target = Nemotron3_5ASRForRNNT(
                        config,
                        initialize=False,
                    )
                with self.assertRaisesRegex(
                        CheckpointCompatibilityError,
                        "dtypes/layouts",
                ):
                    NemotronASRCheckpointAdapter().load_assign_streaming(
                        target,
                        source,
                        config,
                        device="cpu",
                    )
                self.assertTrue(all(tensor.device.type == "meta" for tensor in target.state_dict().values()))

    def test_custom_safetensors_dtype_preflight_is_atomic(self):
        from voicehub.checkpointing import SafeTensorReader, save_safetensors
        from voicehub.checkpointing.errors import CheckpointCompatibilityError
        from voicehub.models.asr_nemotron.native.checkpoint import NemotronASRCheckpointAdapter
        from voicehub.models.asr_nemotron.native.modeling import Nemotron3_5ASRForRNNT

        config = _tiny_config()
        original = Nemotron3_5ASRForRNNT(config)
        state = dict(original.state_dict())
        name = sorted(state)[-1]
        state[name] = torch.zeros_like(
            state[name],
            dtype=torch.int64,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_safetensors(
                state,
                Path(directory) / "invalid.safetensors",
            )
            with torch.device("meta"):
                target = Nemotron3_5ASRForRNNT(
                    config,
                    initialize=False,
                )
            with (
                    SafeTensorReader(checkpoint) as reader,
                    self.assertRaisesRegex(
                        CheckpointCompatibilityError,
                        "dtypes/layouts",
                    ),
            ):
                NemotronASRCheckpointAdapter().load_assign_streaming(
                    target,
                    reader,
                    config,
                    device="cpu",
                )
            self.assertTrue(all(tensor.device.type == "meta" for tensor in target.state_dict().values()))


class NemotronTokenizerAndProviderTests(unittest.TestCase):

    def test_tokenizer_normalizes_published_whitespace_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_tokenizer(Path(directory) / "tokenizer.json")
            tokenizer = NemotronASRTokenizer.from_tokenizer_json(path)
            self.assertEqual(
                tokenizer.encode("  hi \n").input_ids,
                (5, ),
            )
            self.assertEqual(tokenizer.decode((5, )), "hi")
            with self.assertRaisesRegex(ValueError, "reserved token"):
                tokenizer.encode("<en-US>hi")

    def test_provider_configuration_rejects_external_runtime_options(self):
        config = NemotronASRConfig(
            target_language=" de-DE ",
            num_lookahead_tokens=3,
        )
        self.assertEqual(config.target_language, "de-DE")
        self.assertEqual(config.architecture_family, "rnnt")
        self.assertEqual(
            NemotronForSpeechRecognition(device="cpu").config.model_type,
            "asr_nemotron",
        )
        with self.assertRaisesRegex(ValueError, "never executes"):
            NemotronASRConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "0, 3, 6, 13"):
            NemotronASRConfig(num_lookahead_tokens=1)

    def test_provider_rejects_invalid_request_language_before_processing(self):
        provider = NemotronForSpeechRecognition(device="cpu")
        provider.model = object()
        provider.native_config = object()
        provider.nemotron_processor = object()
        with self.assertRaisesRegex(TypeError, "string or None"):
            provider._transcribe([], language=3)
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            provider._transcribe([], language=" \t")

    def test_wrapper_trims_file_audio_before_training_resample(self):

        class CapturingProcessor:

            sampling_rate = 16_000

            def __call__(self, waveforms, **kwargs):
                self.waveforms = tuple(waveforms)
                self.kwargs = kwargs
                return {
                    "input_features": torch.zeros(1, 2, 8),
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                    "prompt_ids": torch.zeros(1, dtype=torch.long),
                    "labels": torch.ones(1, 1, dtype=torch.long),
                    "label_lengths": torch.ones(1, dtype=torch.long),
                    "decoder_input_ids": torch.ones(1, 2, dtype=torch.long),
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = save_pcm_wave(
                root / "padded.wav",
                torch.cat((torch.zeros(200), torch.ones(200))),
                8_000,
            )
            processor = CapturingProcessor()
            wrapper = NemotronForSpeechRecognition(device="cpu")
            wrapper.model = object()
            wrapper.nemotron_processor = processor

            wrapper.prepare_training_inputs(
                {
                    "audio": str(path),
                    "audio_lengths": 200,
                    "sampling_rate": 8_000,
                    "language": "en-US",
                    "text": "hi",
                },
                phase="speech_recognition",
            )

        self.assertEqual(processor.waveforms[0].numel(), 400)
        torch.testing.assert_close(
            processor.waveforms[0],
            torch.zeros(400),
        )
        self.assertEqual(processor.kwargs["sampling_rate"], 16_000)

    def test_configuration_fails_closed_on_adjacent_transducer_graphs(self):
        values = NemotronASRArchitectureConfig().to_dict()
        values["durations"] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "RNN-T"):
            NemotronASRArchitectureConfig.from_dict(values)
        values = NemotronASRArchitectureConfig().to_dict()
        values['architectures'] = ["ParakeetForTDT"]
        with self.assertRaisesRegex(ValueError, "Nemotron3_5AsrForRNNT"):
            NemotronASRArchitectureConfig.from_dict(values)

    def test_local_artifact_resolution_rejects_weight_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in (
                    "config.json",
                    "processor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                    "model.safetensors.index.json",
            ):
                (root / filename).write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                resolve_nemotron_asr_artifacts(root)

    def test_provenance_is_immutable(self):
        self.assertEqual(
            NEMOTRON_ASR_REVISION,
            "f3d333391852ba876df169dcc9ba902d25b6ab0b",
        )

    def test_generation_configuration_is_bound_to_native_graph(self):
        from voicehub.models.asr_nemotron.native.runtime import validate_nemotron_asr_generation_config

        config = _tiny_config()
        valid = {
            "blank_token_id": config.blank_token_id,
            "max_symbols_per_step": config.max_symbols_per_step,
            "num_lookahead_tokens": 0,
            "supported_num_lookahead_tokens": [0],
        }
        self.assertEqual(
            validate_nemotron_asr_generation_config(
                valid,
                config,
            ),
            valid,
        )
        invalid_values = {
            "blank_token_id": {
                **valid,
                "blank_token_id": 0,
            },
            "max_symbols_per_step": {
                **valid,
                "max_symbols_per_step": 99,
            },
            "num_lookahead_tokens": {
                **valid,
                "num_lookahead_tokens": 3,
            },
            "supported_num_lookahead_tokens": {
                **valid,
                "supported_num_lookahead_tokens": [0, 3],
            },
        }
        for name, invalid in invalid_values.items():
            with (
                    self.subTest(setting=name),
                    self.assertRaisesRegex(ValueError, name),
            ):
                validate_nemotron_asr_generation_config(
                    invalid,
                    config,
                )

    def test_registration_does_not_claim_unverified_optimizations(self):
        from voicehub.models.asr_nemotron.native.registration import create_nemotron_asr_architecture_spec

        spec = create_nemotron_asr_architecture_spec()
        self.assertEqual(
            spec.capabilities.optimization_passes,
            (),
        )
        self.assertTrue(spec.capabilities.has_feature("gradient-checkpointing"), )

    def test_training_adapter_does_not_create_partial_export_directory(self):
        from voicehub.models.asr_nemotron.training_asr_nemotron import NativeNemotronASRTrainingAdapter

        adapter = object.__new__(NativeNemotronASRTrainingAdapter)
        adapter.model = SimpleNamespace(
            export_native_pretrained=Mock(side_effect=ValueError("invalid runtime state"), ), )
        adapter.setup = lambda: adapter
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "rejected"
            with self.assertRaisesRegex(ValueError, "invalid runtime state"):
                adapter.save_pretrained(destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
