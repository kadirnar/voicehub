import ast
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.asr_parakeet_tdt import ParakeetTDTASRConfig, ParakeetTDTForSpeechRecognition
from voicehub.models.asr_parakeet_tdt.native.artifacts import ParakeetTDTArtifacts, resolve_parakeet_tdt_artifacts
from voicehub.models.asr_parakeet_tdt.native.checkpoint import (
    ParakeetTDTCheckpointAdapter,
    native_parakeet_tdt_tensor_shapes,
    parakeet_tdt_header_fingerprint,
)
from voicehub.models.asr_parakeet_tdt.native.configuration import ParakeetEncoderConfig, ParakeetTDTConfig
from voicehub.models.asr_parakeet_tdt.native.decoding import decode_tdt_sequence
from voicehub.models.asr_parakeet_tdt.native.loss import tdt_loss
from voicehub.models.asr_parakeet_tdt.native.metadata import PARAKEET_TDT_CHECKPOINTS, PARAKEET_TRANSFORMERS_REVISION
from voicehub.models.asr_parakeet_tdt.native.modeling import ParakeetEncoderOutput, ParakeetForTDT
from voicehub.models.asr_parakeet_tdt.native.processing import ParakeetFeatureExtractor, ParakeetProcessor
from voicehub.models.asr_parakeet_tdt.native.runtime import (
    ParakeetTDTRuntime,
    load_parakeet_tdt_runtime,
    save_parakeet_tdt_runtime,
)
from voicehub.models.asr_parakeet_tdt.native.tokenization import ParakeetTokenizer, ParakeetTokenizerAssets
from voicehub.models.asr_parakeet_tdt.training_asr_parakeet_tdt import NativeParakeetTDTTrainingAdapter
from voicehub.processing.waveform import save_pcm_wave
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.specs import get_training_spec

TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


class _IdentityNormalizer:

    @staticmethod
    def normalize(text):
        return text


def _tiny_config():
    encoder = ParakeetEncoderConfig(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        hidden_act="silu",
        attention_bias=False,
        convolution_bias=False,
        conv_kernel_size=3,
        subsampling_factor=2,
        subsampling_conv_channels=4,
        num_mel_bins=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=100,
        scale_input=False,
    )
    return ParakeetTDTConfig(
        vocab_size=9,
        decoder_hidden_size=4,
        num_decoder_layers=1,
        max_symbols_per_step=4,
        encoder_config=encoder,
        pad_token_id=2,
        eos_token_id=3,
        blank_token_id=8,
        durations=(0, 1, 2),
    )


def _tiny_tokenizer():
    vocabulary = {
        "<unk>": 0,
        "▁": 1,
        "<pad>": 2,
        "<|endoftext|>": 3,
        "h": 4,
        "i": 5,
        "x": 6,
        "!": 7,
    }
    id_to_token = {value: key for key, value in vocabulary.items()}
    id_to_token[8] = "<blank>"
    assets = ParakeetTokenizerAssets(
        vocabulary=MappingProxyType(vocabulary),
        id_to_token=MappingProxyType(id_to_token),
        merge_ranks=MappingProxyType({}),
        added_tokens=MappingProxyType({
            "<unk>": 0,
            "<pad>": 2,
            "<|endoftext|>": 3,
            "<blank>": 8,
        }),
        special_ids=frozenset({0, 2, 3, 8}),
        unk_token_id=0,
        pad_token_id=2,
        eos_token_id=3,
        blank_token_id=8,
        normalizer=_IdentityNormalizer(),
        replacement="▁",
        original_document=MappingProxyType({"version": "1.0"}),
        original_config=MappingProxyType({
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "eos_token": "<|endoftext|>",
        }),
    )
    return ParakeetTokenizer(assets)


def _tiny_processor():
    return ParakeetProcessor(
        ParakeetFeatureExtractor(
            feature_size=8,
            sampling_rate=16_000,
            hop_length=160,
            n_fft=512,
            win_length=400,
        ),
        _tiny_tokenizer(),
        subsampling_factor=2,
    )


def _tiny_runtime(root):
    config = _tiny_config()
    model = ParakeetForTDT(config)
    processor = _tiny_processor()
    artifacts = ParakeetTDTArtifacts(
        source=str(root),
        revision=None,
        config=root / "config.json",
        processor_config=root / "processor_config.json",
        tokenizer=root / "tokenizer.json",
        tokenizer_config=root / "tokenizer_config.json",
        checkpoint=root / "model.safetensors",
        generation_config=root / "generation_config.json",
    )
    return ParakeetTDTRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config={
            "decoder_start_token_id": 8,
            "eos_token_id": 3,
            "pad_token_id": 2,
            "suppress_tokens": [9, 10, 11],
        },
    )


class NativeParakeetTDTTests(unittest.TestCase):

    def test_package_discovery_does_not_import_torch(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_parakeet_tdt as provider; "
            "import voicehub.models.asr_parakeet_tdt.native as architecture; "
            "assert 'torch' not in sys.modules; "
            "assert 'ParakeetTDTForSpeechRecognition' in provider.__all__; "
            "assert 'NativeParakeetTDTTrainingAdapter' in provider.__all__; "
            "assert 'ParakeetForTDT' in architecture.__all__")
        subprocess.run(
            [sys.executable, "-c", command],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_shared_training_registry_selects_the_native_adapter_and_graph(self):
        wrapper = ParakeetTDTForSpeechRecognition(
            lazy_load=True,
            device="cpu",
        )
        adapter = AutoTrainingAdapter.from_model(wrapper)
        spec = get_training_spec("asr_parakeet_tdt")

        self.assertIsInstance(adapter, NativeParakeetTDTTrainingAdapter)
        self.assertEqual(spec.module_paths, ("model", ))
        self.assertEqual(
            spec.component_paths,
            (
                "model.encoder",
                "model.encoder_projector",
                "model.decoder",
                "model.joint",
            ),
        )
        self.assertTrue(all(entrypoint.startswith("voicehub.") for entrypoint in spec.source_entrypoints))

    def test_published_graph_inventory_and_provenance_are_exact(self):
        expected = PARAKEET_TDT_CHECKPOINTS["nvidia/parakeet-tdt-0.6b-v3"]
        self.assertEqual(
            PARAKEET_TRANSFORMERS_REVISION,
            "af71155683b4d34dd92d8f037392fa6bf334035e",
        )
        config = ParakeetTDTConfig()
        shapes = native_parakeet_tdt_tensor_shapes(config)
        self.assertEqual(len(shapes), 723)
        self.assertEqual(sum(tensor.numel() for tensor in _meta_state(config).values()), 627_057_310)
        inventory = {
            name: (
                "I64" if tensor.dtype == torch.int64 else "F32",
                tuple(tensor.shape),
            )
            for name, tensor in _meta_state(config).items()
        }
        self.assertEqual(
            parakeet_tdt_header_fingerprint(inventory),
            expected["header_fingerprint"],
        )
        self.assertEqual(
            sum(name.startswith("encoder.") for name in shapes),
            708,
        )
        self.assertEqual(
            sum(name.startswith("decoder.") for name in shapes),
            11,
        )

    def test_configuration_rejects_non_tdt_and_incoherent_durations(self):
        with self.assertRaisesRegex(ValueError, "rejects CTC/RNNT"):
            ParakeetTDTConfig.from_dict({"model_type": "parakeet_ctc"})
        with self.assertRaisesRegex(ValueError, "begin with zero"):
            ParakeetTDTConfig(durations=(1, 2))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            ParakeetTDTConfig(durations=(0, 2, 1))
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            ParakeetTDTASRConfig(use_safetensors=False)

    def test_artifacts_pin_known_remote_and_require_one_snapshot(self):
        revision = PARAKEET_TDT_CHECKPOINTS["nvidia/parakeet-tdt-0.6b-v3"]["revision"]
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / revision
            snapshot.mkdir()
            for name in (
                    "config.json",
                    "processor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                    "generation_config.json",
            ):
                (snapshot / name).write_text("{}", encoding="utf-8")
            calls = []

            def resolve(repo_id, filename, **kwargs):
                calls.append((repo_id, filename, kwargs["revision"]))
                if filename == "model.safetensors.index.json":
                    raise FileNotFoundError(filename)
                return snapshot / filename

            with (
                    patch(
                        "voicehub.models.asr_parakeet_tdt.native.artifacts."
                        "resolve_pretrained_file",
                        side_effect=resolve,
                    ),
                    patch(
                        "voicehub.models.asr_parakeet_tdt.native.artifacts."
                        "get_cached_hugging_face_commit",
                        return_value=revision,
                    ),
            ):
                artifacts = resolve_parakeet_tdt_artifacts("nvidia/parakeet-tdt-0.6b-v3")
            self.assertEqual(artifacts.revision, revision)
            self.assertTrue(all(call[2] == revision for call in calls))

            other = Path(temporary) / "other"
            other.mkdir()
            (other / "tokenizer.json").write_text("{}", encoding="utf-8")

            def incoherent(repo_id, filename, **kwargs):
                del repo_id, kwargs
                if filename == "model.safetensors.index.json":
                    raise FileNotFoundError(filename)
                if filename == "tokenizer.json":
                    return other / filename
                return snapshot / filename

            with (
                    patch(
                        "voicehub.models.asr_parakeet_tdt.native.artifacts."
                        "resolve_pretrained_file",
                        side_effect=incoherent,
                    ),
                    patch(
                        "voicehub.models.asr_parakeet_tdt.native.artifacts."
                        "get_cached_hugging_face_commit",
                        return_value=revision,
                    ),
                    self.assertRaisesRegex(RuntimeError, "one immutable snapshot"),
            ):
                resolve_parakeet_tdt_artifacts("nvidia/parakeet-tdt-0.6b-v3")

    def test_artifacts_reject_unproven_mutable_revision_and_ambiguous_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            config = snapshot / "config.json"
            config.write_text("{}", encoding="utf-8")
            with (
                    patch(
                        "voicehub.models.asr_parakeet_tdt.native.artifacts."
                        "resolve_pretrained_file",
                        return_value=config,
                    ),
                    patch(
                        "voicehub.models.asr_parakeet_tdt.native.artifacts."
                        "get_cached_hugging_face_commit",
                        return_value=None,
                    ),
                    self.assertRaisesRegex(RuntimeError, "immutable"),
            ):
                resolve_parakeet_tdt_artifacts(
                    "unverified/parakeet",
                    revision="main",
                )

            ambiguous = root / "ambiguous"
            ambiguous.mkdir()
            (ambiguous / "model.safetensors").touch()
            (ambiguous / "model.safetensors.index.json").touch()
            with self.assertRaisesRegex(ValueError, "both single-file"):
                resolve_parakeet_tdt_artifacts(ambiguous)

    def test_frontend_masks_padding_and_normalizes_valid_frames(self):
        torch.manual_seed(10)
        extractor = ParakeetFeatureExtractor(feature_size=8)
        output = extractor(
            (torch.randn(3_200), torch.randn(4_377)),
            sampling_rate=16_000,
        )
        self.assertEqual(output["input_features"].shape, (2, 28, 8))
        self.assertEqual(
            output["attention_mask"].sum(-1).tolist(),
            [20, 27],
        )
        first = output["input_features"][0, :20]
        torch.testing.assert_close(
            first.mean(dim=0),
            torch.zeros(8),
            atol=2e-6,
            rtol=0,
        )
        self.assertTrue(torch.all(output["input_features"][0, 20:] == 0))

    def test_tdt_loss_matches_reference_values_and_backpropagates(self):
        torch.manual_seed(3)
        token_logits = torch.randn(2, 5, 4, 7, requires_grad=True)
        duration_logits = torch.randn(2, 5, 4, 3, requires_grad=True)
        targets = torch.tensor([[1, 2, 3], [3, 2, 0]])
        logit_lengths = torch.tensor([5, 4])
        target_lengths = torch.tensor([3, 2])
        losses = tdt_loss(
            token_logits,
            duration_logits,
            targets,
            logit_lengths,
            target_lengths,
            6,
            (0, 1, 2),
            sigma=0.05,
            reduction="none",
        )
        torch.testing.assert_close(
            losses,
            torch.tensor([10.9064, 7.4465]),
            atol=5e-5,
            rtol=5e-5,
        )
        losses.mean().backward()
        self.assertTrue(torch.isfinite(token_logits.grad).all())
        self.assertTrue(torch.isfinite(duration_logits.grad).all())
        with self.assertRaisesRegex(ValueError, "outside"):
            tdt_loss(
                token_logits.detach(),
                duration_logits.detach(),
                torch.tensor([[1, 7, 3], [3, 2, 0]]),
                logit_lengths,
                target_lengths,
                6,
                (0, 1, 2),
            )
        with self.assertRaisesRegex(ValueError, "blank"):
            tdt_loss(
                token_logits.detach(),
                duration_logits.detach(),
                torch.tensor([[1, 6, 3], [3, 2, 0]]),
                logit_lengths,
                target_lengths,
                6,
                (0, 1, 2),
            )

    def test_full_tiny_graph_loss_backward_and_gradient_checkpointing(self):
        torch.manual_seed(7)
        model = ParakeetForTDT(_tiny_config())
        model.gradient_checkpointing_enable()
        model.train()
        output = model(
            torch.randn(2, 31, 8),
            attention_mask=torch.ones(2, 31, dtype=torch.long),
            decoder_input_ids=torch.tensor([[8, 4, 5], [8, 5, 4]]),
            labels=torch.tensor([[4, 5], [5, 4]]),
        )
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.encoder.layers[0].self_attn.q_proj.weight.grad)
        self.assertIsNotNone(model.decoder.embedding.weight.grad)
        self.assertIsNotNone(model.joint.head.weight.grad)

    def test_model_rejects_incoherent_decoder_targets_and_internal_padding(self):
        model = ParakeetForTDT(_tiny_config()).eval()
        common = {
            "input_features": torch.randn(1, 31, 8),
            "attention_mask": torch.ones(1, 31, dtype=torch.long),
        }
        with self.assertRaisesRegex(ValueError, r"\[blank\] \+ labels"):
            model(
                **common,
                decoder_input_ids=torch.tensor([[8, 5, 4]]),
                labels=torch.tensor([[4, 5]]),
            )
        with self.assertRaisesRegex(ValueError, "contiguous right padding"):
            model(
                **common,
                decoder_input_ids=torch.tensor([[8, 4, 2, 5]]),
                labels=torch.tensor([[4, 2, 5]]),
            )

    def test_checkpoint_rejects_integral_or_complex_replacement_weights(self):
        config = _tiny_config()
        for dtype in (torch.int64, torch.complex64):
            with self.subTest(dtype=dtype):
                model = ParakeetForTDT(config)
                state = dict(model.state_dict())
                name = "joint.head.weight"
                state[name] = torch.zeros(
                    state[name].shape,
                    dtype=dtype,
                )
                with self.assertRaisesRegex(
                        CheckpointCompatibilityError,
                        "incompatible tensor dtypes",
                ):
                    ParakeetTDTCheckpointAdapter().load_assign_streaming(
                        model,
                        state,
                        config,
                        device="cpu",
                    )

    def test_greedy_decoding_uses_duration_values_and_forces_blank_progress(self):
        model = ParakeetForTDT(_tiny_config())

        def audio_features(input_features, attention_mask=None):
            del input_features, attention_mask
            hidden = torch.zeros(1, 3, 4)
            return ParakeetEncoderOutput(
                last_hidden_state=torch.zeros(1, 3, 8),
                pooler_output=hidden,
                attention_mask=torch.ones(1, 3, dtype=torch.int),
            )

        class Decoder(nn.Module):

            def forward(self, input_ids, cache=None):
                del cache
                return torch.zeros(input_ids.shape[0], 1, 4)

        class Joint(nn.Module):

            def __init__(self):
                super().__init__()
                self.step = 0

            def forward(self, **kwargs):
                batch = kwargs["encoder_hidden_states"].shape[0]
                output = torch.full((batch, 1, 12), -100.0)
                tokens = (4, 8, 5)
                duration_indices = (0, 0, 2)
                output[:, :, tokens[self.step]] = 100.0
                output[:, :, 9 + duration_indices[self.step]] = 100.0
                self.step += 1
                return output

        model.get_audio_features = audio_features
        model.decoder = Decoder()
        model.joint = Joint()
        generated = model.generate(
            torch.zeros(1, 2, 8),
            torch.ones(1, 2, dtype=torch.long),
        )
        self.assertEqual(generated.sequences.tolist(), [[8, 4, 8, 5]])
        self.assertEqual(generated.durations.tolist(), [[0, 0, 1, 2]])

    def test_timestamp_decoder_preserves_repeats_and_zeroes_punctuation(self):
        decoded = decode_tdt_sequence(
            _tiny_tokenizer(),
            torch.tensor([8, 1, 4, 5, 5, 7]),
            torch.tensor([0, 1, 0, 1, 2, 1]),
            frame_seconds=0.08,
        )
        self.assertEqual(decoded.text, "hii!")
        repeated = [value for value in decoded.tokens if value.text == "i"]
        self.assertEqual(len(repeated), 2)
        punctuation = decoded.tokens[-1]
        self.assertEqual(punctuation.text, "!")
        self.assertEqual(punctuation.start, punctuation.end)
        self.assertEqual(decoded.words[0].text, "hii!")

    def test_processor_builds_blank_prefixed_training_inputs(self):
        processor = _tiny_processor()
        prepared = processor(
            (torch.zeros(1_600), torch.zeros(2_000)),
            text=("hi", "h"),
            sampling_rate=16_000,
        )
        self.assertEqual(prepared["labels"].tolist(), [[1, 4, 5], [1, 4, 2]])
        self.assertEqual(
            prepared["decoder_input_ids"].tolist(),
            [[8, 1, 4, 5], [8, 1, 4, 2]],
        )
        self.assertEqual(prepared["input_features"].shape[0], 2)

    def test_wrapper_trims_file_audio_before_training_resample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_root = root / "runtime"
            runtime_root.mkdir()
            runtime = _tiny_runtime(runtime_root)
            wrapper = ParakeetTDTForSpeechRecognition(
                ParakeetTDTASRConfig(name_or_path=runtime_root),
                device="cpu",
            )
            wrapper.runtime = runtime
            wrapper.artifacts = runtime.artifacts
            wrapper.native_config = runtime.config
            wrapper.parakeet_processor = runtime.processor
            wrapper.model = runtime.model
            path = save_pcm_wave(
                root / "padded.wav",
                torch.cat((torch.zeros(200), torch.ones(200))),
                8_000,
            )
            common = {
                "sampling_rate": 8_000,
                "text": "hi",
            }

            file_batch = wrapper.prepare_training_inputs(
                {
                    **common,
                    "audio": str(path),
                    "audio_lengths": 200,
                },
                phase="speech_recognition",
            )
            tensor_batch = wrapper.prepare_training_inputs(
                {
                    **common,
                    "audio": torch.zeros(200),
                },
                phase="speech_recognition",
            )

            torch.testing.assert_close(
                file_batch["input_features"],
                tensor_batch["input_features"],
            )
            torch.testing.assert_close(
                file_batch["attention_mask"],
                tensor_batch["attention_mask"],
            )

    def test_export_validates_before_creating_destination_and_reloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _tiny_runtime(root / "source")
            invalid_state = dict(runtime.model.state_dict())
            invalid_state.pop("joint.head.bias")
            rejected = root / "rejected"
            with self.assertRaisesRegex(ValueError, "exact model namespace"):
                save_parakeet_tdt_runtime(
                    runtime,
                    rejected,
                    state_dict=invalid_state,
                )
            self.assertFalse(rejected.exists())

            invalid_variants = {
                "shape":
                torch.zeros(runtime.model.joint.head.bias.shape[0] + 1),
                "dtype":
                runtime.model.joint.head.bias.double(),
                "complex":
                runtime.model.joint.head.bias.to(torch.complex64),
                "sparse":
                runtime.model.joint.head.bias.to_sparse(),
                "quantized":
                torch.quantize_per_tensor(
                    runtime.model.joint.head.bias,
                    scale=0.1,
                    zero_point=0,
                    dtype=torch.qint8,
                ),
            }
            for label, invalid in invalid_variants.items():
                with self.subTest(validation=label):
                    invalid_state = dict(runtime.model.state_dict())
                    invalid_state["joint.head.bias"] = invalid
                    invalid_destination = root / f"rejected-{label}"
                    with self.assertRaises((TypeError, ValueError)):
                        save_parakeet_tdt_runtime(
                            runtime,
                            invalid_destination,
                            state_dict=invalid_state,
                        )
                    self.assertFalse(invalid_destination.exists())

            destination = root / "portable"
            save_parakeet_tdt_runtime(
                runtime,
                destination,
                maximum_shard_bytes=1_024,
            )
            self.assertTrue((destination / "model.safetensors.index.json").is_file())
            with patch.object(
                    ParakeetProcessor,
                    "from_files",
                    return_value=runtime.processor,
            ):
                restored = load_parakeet_tdt_runtime(
                    destination,
                    device="cpu",
                )
            for name, value in runtime.model.state_dict().items():
                torch.testing.assert_close(
                    restored.model.state_dict()[name],
                    value,
                )

    def test_training_adapter_does_not_create_directory_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "failed-training-export"
            adapter = object.__new__(NativeParakeetTDTTrainingAdapter)
            adapter.model = SimpleNamespace(
                export_native_pretrained=Mock(side_effect=ValueError("invalid runtime state")))
            adapter.setup = lambda: adapter
            with self.assertRaisesRegex(ValueError, "invalid runtime state"):
                adapter.save_pretrained(destination)
            self.assertFalse(destination.exists())

    def test_public_wrapper_rejects_unimplemented_decoding_modes_before_load(self):
        wrapper = ParakeetTDTForSpeechRecognition(
            lazy_load=True,
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "auto-detects"):
            wrapper._validate_request(
                language="en",
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=None,
                max_new_tokens=None,
                hotwords=None,
            )
        with self.assertRaisesRegex(ValueError, "greedy"):
            wrapper._validate_request(
                language=None,
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=4,
                max_new_tokens=None,
                hotwords=None,
            )

    def test_active_runtime_has_no_external_model_library_imports(self):
        root = Path(__file__).resolve().parents[1]
        files = tuple((root / 'voicehub/models/asr_parakeet_tdt/native').glob("*.py")) + tuple(
            (root / "voicehub" / "models" / "asr_parakeet_tdt").glob("*.py"))
        forbidden = {
            "huggingface_hub",
            "librosa",
            "nemo",
            "numpy",
            "safetensors",
            "tokenizers",
            "transformers",
        }
        imported = set()
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".", 1)[0])
        self.assertFalse(imported & forbidden)

    @unittest.skipUnless(
        TRANSFORMERS_AVAILABLE,
        "Transformers is used only as an optional parity oracle",
    )
    def test_tiny_graph_matches_pinned_transformers_equations(self):
        from transformers import ParakeetForTDT as ReferenceModel
        from transformers import ParakeetTDTConfig as ReferenceConfig

        config = _tiny_config()
        reference_config = ReferenceConfig(**config.to_dict())
        torch.manual_seed(8)
        reference = ReferenceModel(reference_config).eval()
        native = ParakeetForTDT(config).eval()
        self.assertEqual(set(reference.state_dict()), set(native.state_dict()))
        native.load_state_dict(reference.state_dict(), strict=True)
        features = torch.randn(2, 31, 8)
        mask = torch.zeros(2, 31, dtype=torch.long)
        mask[0, :] = 1
        mask[1, :25] = 1
        decoder_ids = torch.tensor([[8, 4, 5], [8, 5, 4]])
        with torch.no_grad():
            expected = reference(
                features,
                attention_mask=mask,
                decoder_input_ids=decoder_ids,
            )
            actual = native(
                features,
                attention_mask=mask,
                decoder_input_ids=decoder_ids,
            )
        torch.testing.assert_close(
            actual.last_hidden_state,
            expected.last_hidden_state,
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            actual.logits,
            expected.logits,
            atol=1e-6,
            rtol=1e-6,
        )


def _meta_state(config):
    with torch.device("meta"):
        return ParakeetForTDT(config, initialize=False).state_dict()


if __name__ == "__main__":
    unittest.main()
