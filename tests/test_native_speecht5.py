from __future__ import annotations

import ast
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.speecht5.checkpoint import (
    inspect_safetensors_checkpoint,
    load_speecht5_checkpoint,
    save_speecht5_checkpoint,
    state_dict_inventory,
    tensor_inventory_fingerprint,
)
from voicehub.models.speecht5.metadata import (
    SPEECHT5_HIFIGAN_REVISION,
    SPEECHT5_HIFIGAN_STATE_VALUES,
    SPEECHT5_HIFIGAN_TENSOR_COUNT,
    SPEECHT5_HIFIGAN_TENSOR_FINGERPRINT,
    SPEECHT5_REVISION,
    SPEECHT5_STATE_VALUES,
    SPEECHT5_TENSOR_COUNT,
    SPEECHT5_TENSOR_FINGERPRINT,
    TRANSFORMERS_REFERENCE_REVISION,
)
from voicehub.models.speecht5.modeling import SpeechT5Config, SpeechT5ForTextToSpeech
from voicehub.models.speecht5.native_configuration import NativeSpeechT5Config, NativeSpeechT5HifiGanConfig
from voicehub.models.speecht5.native_modeling import SpeechT5ForTextToSpeechModel, SpeechT5HifiGan
from voicehub.models.speecht5.processing import (
    SpeechT5FeatureConfig,
    SpeechT5FeatureExtractor,
    SpeechT5Processor,
    SpeechT5Tokenizer,
)
from voicehub.models.speecht5.training import NativeSpeechT5TrainingAdapter
from voicehub.training.specs import get_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_AVAILABLE = (
    importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("numpy") is not None)


def _varint(value: int) -> bytes:
    if value < 0:
        value &= (1 << 64) - 1
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        encoded.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(encoded)


def _length_field(number: int, value: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(value)) + value


def _varint_field(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _piece(text: str, score: float, piece_type: int) -> bytes:
    return (
        _length_field(1, text.encode("utf-8")) + _varint(2 << 3 | 5) + struct.pack("<f", score) +
        _varint_field(3, piece_type))


def _write_test_sentencepiece(path: Path) -> None:
    pieces = (
        ("<s>", 0.0, 3),
        ("<pad>", 0.0, 3),
        ("</s>", 0.0, 3),
        ("<unk>", 0.0, 2),
        ("\u2581", -0.1, 1),
        ("a", -0.1, 1),
        ("b", -0.1, 1),
        ("c", -0.1, 1),
    )
    trainer = (_varint_field(40, 3) + _varint_field(41, 0) + _varint_field(42, 2) + _varint_field(43, 1))
    normalizer = (
        _length_field(1, b"identity") + _varint_field(3, 1) + _varint_field(4, 1) + _varint_field(5, 1))
    payload = b"".join(
        _length_field(1, _piece(text, score, piece_type)) for text, score, piece_type in pieces)
    path.write_bytes(payload + _length_field(2, trainer) + _length_field(3, normalizer))


def _tiny_config() -> NativeSpeechT5Config:
    return NativeSpeechT5Config(
        vocab_size=8,
        hidden_size=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        encoder_layerdrop=0.0,
        decoder_layers=1,
        decoder_ffn_dim=32,
        decoder_attention_heads=2,
        decoder_layerdrop=0.0,
        positional_dropout=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        speech_decoder_prenet_layers=1,
        speech_decoder_prenet_units=8,
        speech_decoder_prenet_dropout=0.5,
        speaker_embedding_dim=4,
        speech_decoder_postnet_layers=2,
        speech_decoder_postnet_units=8,
        speech_decoder_postnet_kernel=3,
        speech_decoder_postnet_dropout=0.0,
        max_speech_positions=64,
        max_text_positions=64,
        encoder_max_relative_position=8,
        use_guided_attention_loss=True,
        guided_attention_loss_num_heads=1,
    )


def _tiny_vocoder_config() -> NativeSpeechT5HifiGanConfig:
    return NativeSpeechT5HifiGanConfig(
        upsample_initial_channel=16,
        upsample_rates=(2, 2),
        upsample_kernel_sizes=(4, 4),
        resblock_kernel_sizes=(3, ),
        resblock_dilation_sizes=((1, 3), ),
    )


def _processor(root: Path) -> SpeechT5Processor:
    tokenizer_path = root / "spm_char.model"
    _write_test_sentencepiece(tokenizer_path)
    return SpeechT5Processor(
        SpeechT5Tokenizer(tokenizer_path, model_max_length=64),
        SpeechT5FeatureExtractor(SpeechT5FeatureConfig(reduction_factor=2)),
    )


def _wrapper(root: Path) -> SpeechT5ForTextToSpeech:
    model_config = _tiny_config()
    vocoder_config = _tiny_vocoder_config()
    wrapper = SpeechT5ForTextToSpeech(
        SpeechT5Config(
            native_model_config=model_config.to_dict(),
            native_vocoder_config=vocoder_config.to_dict(),
            verify_official_integrity=False,
        ),
        device="cpu",
    )
    wrapper._torch = torch
    wrapper.native_config = model_config
    wrapper.native_vocoder_config = vocoder_config
    wrapper.model = SpeechT5ForTextToSpeechModel(model_config)
    wrapper.vocoder = SpeechT5HifiGan(vocoder_config)
    wrapper.vocoder.requires_grad_(False)
    wrapper.transformers_processor = _processor(root)
    wrapper.processor = wrapper.transformers_processor
    return wrapper


class NativeSpeechT5DependencyTests(unittest.TestCase):

    def test_runtime_has_no_external_architecture_or_processing_imports(self):
        root = PROJECT_ROOT / "voicehub" / "models" / "speecht5"
        forbidden = {
            "huggingface_hub",
            "librosa",
            "numpy",
            "safetensors",
            "sentencepiece",
            "tokenizers",
            "torchaudio",
            "transformers",
        }
        violations = []
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    names = []
                for name in names:
                    if name.split(".", 1)[0] in forbidden:
                        violations.append((path.name, name))
        self.assertEqual(violations, [])

    def test_public_package_import_is_dependency_lazy(self):
        script = (
            "import sys; import voicehub.models.speecht5; "
            "print('torch' in sys.modules, 'transformers' in sys.modules, "
            "'numpy' in sys.modules, 'safetensors' in sys.modules, "
            "'sentencepiece' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False False False False False")

    def test_configuration_rejects_external_runtime_injection(self):
        with self.assertRaisesRegex(ValueError, "never executes remote"):
            SpeechT5Config(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "atomically"):
            SpeechT5Config(config_name_or_path="other")
        with self.assertRaisesRegex(ValueError, "provider-runtime"):
            SpeechT5Config(model_kwargs={"device_map": "auto"})
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            SpeechT5Config(processor_kwargs={"token": "secret"})

    def test_provenance_pins_source_and_both_checkpoints(self):
        source = json.loads(
            (PROJECT_ROOT / "voicehub" / "models" / "speecht5" / "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(
            source["source"]["revision"],
            TRANSFORMERS_REFERENCE_REVISION,
        )
        self.assertEqual(
            source["checkpoints"]["text_to_spectrogram"]["revision"],
            SPEECHT5_REVISION,
        )
        self.assertEqual(
            source["checkpoints"]["vocoder"]["revision"],
            SPEECHT5_HIFIGAN_REVISION,
        )
        self.assertEqual(
            source["verification"]["runtime_dependencies"],
            "PyTorch, the Python standard library, and VoiceHub only.",
        )


class NativeSpeechT5InventoryTests(unittest.TestCase):

    def test_full_graph_matches_the_published_checkpoint_inventory(self):
        with torch.device("meta"):
            model = SpeechT5ForTextToSpeechModel(NativeSpeechT5Config())
            vocoder = SpeechT5HifiGan(NativeSpeechT5HifiGanConfig())
        model_inventory = state_dict_inventory(model.state_dict())
        vocoder_inventory = state_dict_inventory(vocoder.state_dict())

        self.assertEqual(len(model_inventory), SPEECHT5_TENSOR_COUNT)
        self.assertEqual(
            sum(tensor.numel() for tensor in model.state_dict().values()),
            SPEECHT5_STATE_VALUES,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(model_inventory),
            SPEECHT5_TENSOR_FINGERPRINT,
        )
        self.assertEqual(len(vocoder_inventory), SPEECHT5_HIFIGAN_TENSOR_COUNT)
        self.assertEqual(
            sum(tensor.numel() for tensor in vocoder.state_dict().values()),
            SPEECHT5_HIFIGAN_STATE_VALUES,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(vocoder_inventory),
            SPEECHT5_HIFIGAN_TENSOR_FINGERPRINT,
        )
        self.assertEqual(
            tuple(model.state_dict()["speecht5.encoder.prenet.encode_positions.pe"].shape),
            (1, 600, 768),
        )
        self.assertEqual(
            tuple(model.state_dict()["speecht5.decoder.prenet.encode_positions.pe"].shape),
            (1, 1876, 768),
        )


class NativeSpeechT5ProcessorTests(unittest.TestCase):

    def test_tokenizer_adds_eos_and_batches_with_the_published_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = _processor(Path(directory))
            encoded = processor(
                text=["ab", "a"],
                padding=True,
                return_tensors="pt",
            )
        self.assertEqual(
            encoded["input_ids"].tolist(),
            [[4, 5, 6, 2], [4, 5, 2, 1]],
        )
        self.assertEqual(
            encoded["attention_mask"].tolist(),
            [[1, 1, 1, 1], [1, 1, 1, 0]],
        )

    def test_raw_targets_are_resampled_padded_and_completely_masked(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = _processor(Path(directory))
            batch = processor(
                text=["ab", "a"],
                audio_target=[
                    torch.sin(torch.linspace(0, 20, 1_600)),
                    torch.sin(torch.linspace(0, 15, 1_200)),
                ],
                sampling_rate=16_000,
                padding=True,
                return_tensors="pt",
            )
        self.assertEqual(batch["labels"].shape[0], 2)
        self.assertEqual(batch["labels"].shape[-1], 80)
        self.assertEqual(batch["labels"].shape[1] % 2, 0)
        mask = batch["decoder_attention_mask"].bool()
        self.assertTrue(torch.isfinite(batch["labels"][mask]).all())
        self.assertTrue((batch["labels"][~mask] == -100.0).all())

    @unittest.skipUnless(
        TRANSFORMERS_AVAILABLE,
        "Transformers is an optional audit oracle",
    )
    def test_log_mel_frontend_matches_the_pinned_reference_within_float32_roundoff(self):
        from transformers import SpeechT5FeatureExtractor as ReferenceExtractor

        waveform = torch.linspace(-0.2, 0.3, 16_000).sin()
        actual = SpeechT5FeatureExtractor().extract_mel(
            waveform,
            sampling_rate=16_000,
        )
        reference = torch.from_numpy(ReferenceExtractor()._extract_mel_features(waveform.numpy()))
        # Windows and Unix numerical backends can round a handful of log-mel
        # values differently by one float32 ULP at this output magnitude.
        # Keep relative tolerance disabled so the absolute roundoff boundary
        # remains explicit and still catches meaningful frontend drift.
        roundoff = 4 * torch.finfo(torch.float32).eps
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=roundoff)


@unittest.skipUnless(
    TRANSFORMERS_AVAILABLE,
    "Transformers is an optional audit oracle",
)
class NativeSpeechT5SourceParityTests(unittest.TestCase):

    def test_tiny_acoustic_graph_and_cached_generation_match_reference(self):
        from transformers import SpeechT5Config as ReferenceConfig
        from transformers import SpeechT5ForTextToSpeech as ReferenceModel

        config = _tiny_config()
        reference = ReferenceModel(ReferenceConfig(**config.to_dict())).eval()
        native = SpeechT5ForTextToSpeechModel(config).eval()
        native_state = native.state_dict()
        reference_state = reference.state_dict()
        self.assertEqual(
            set(native_state) - set(reference_state),
            {
                "speecht5.decoder.prenet.encode_positions.pe",
                "speecht5.encoder.prenet.encode_positions.pe",
            },
        )
        for name, tensor in reference_state.items():
            self.assertEqual(tuple(native_state[name].shape), tuple(tensor.shape))
            with torch.no_grad():
                native_state[name].copy_(tensor)

        input_ids = torch.tensor([[4, 5, 6, 2]])
        attention_mask = torch.ones_like(input_ids)
        speaker = torch.randn(1, config.speaker_embedding_dim)
        torch.manual_seed(31)
        expected, expected_lengths = reference.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speaker_embeddings=speaker,
            threshold=1.0,
            minlenratio=1.0,
            maxlenratio=1.0,
            return_output_lengths=True,
        )
        torch.manual_seed(31)
        actual, actual_lengths = native.generate(
            input_ids,
            attention_mask,
            speaker_embeddings=speaker,
            threshold=1.0,
            minlenratio=1.0,
            maxlenratio=1.0,
            return_output_lengths=True,
        )
        self.assertEqual(actual_lengths, expected_lengths)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_tiny_hifigan_is_value_exact(self):
        from transformers import SpeechT5HifiGan as ReferenceModel
        from transformers import SpeechT5HifiGanConfig as ReferenceConfig

        config = _tiny_vocoder_config()
        reference = ReferenceModel(ReferenceConfig(**config.to_dict())).eval()
        native = SpeechT5HifiGan(config).eval()
        native.load_state_dict(reference.state_dict())
        spectrogram = torch.randn(2, 7, 80)
        with torch.no_grad():
            expected = reference(spectrogram)
            actual = native(spectrogram)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class NativeSpeechT5CheckpointTests(unittest.TestCase):

    def test_safetensors_export_is_deterministic_and_strictly_reloadable(self):
        config = _tiny_config()
        source = SpeechT5ForTextToSpeechModel(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = save_speecht5_checkpoint(source, root / "first.safetensors")
            second = save_speecht5_checkpoint(source, root / "second.safetensors")
            report = inspect_safetensors_checkpoint(first)
            restored = SpeechT5ForTextToSpeechModel(config)
            load_speecht5_checkpoint(restored, first)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(report.tensor_count, len(source.state_dict()))
            for name, expected in source.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], expected)

            incomplete = dict(source.state_dict())
            incomplete.pop("speech_decoder_postnet.feat_out.bias")
            broken = save_safetensors(
                incomplete,
                root / "broken.safetensors",
            )
            untouched = restored.speech_decoder_postnet.feat_out.bias.detach().clone()
            with self.assertRaises(CheckpointCompatibilityError):
                load_speecht5_checkpoint(restored, broken)
            torch.testing.assert_close(
                restored.speech_decoder_postnet.feat_out.bias,
                untouched,
            )

    def test_restricted_pytorch_archive_is_supported_without_pickle_fallback(self):
        config = _tiny_vocoder_config()
        source = SpeechT5HifiGan(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pytorch_model.bin"
            torch.save(source.state_dict(), checkpoint)
            restored = SpeechT5HifiGan(config)
            report = load_speecht5_checkpoint(
                restored,
                checkpoint,
                vocoder=True,
            )
        self.assertEqual(report.tensor_count, len(source.state_dict()))
        torch.testing.assert_close(
            restored.conv_pre.weight,
            source.conv_pre.weight,
        )


class NativeSpeechT5TrainingAndRuntimeTests(unittest.TestCase):

    def test_shared_registry_and_trainer_select_the_native_runtime(self):
        from voicehub.registry import get_model_spec
        from voicehub.runtime import get_architecture_spec
        from voicehub.training.recipes import BUILTIN_MODEL_ADAPTERS
        from voicehub.training.specs import get_training_spec

        model_spec = get_model_spec("speecht5")
        architecture = get_architecture_spec("speecht5")
        training = get_training_spec("speecht5")

        self.assertTrue(model_spec.is_voicehub_native)
        self.assertIs(model_spec.native_architecture, architecture)
        self.assertTrue(architecture.capabilities.training)
        self.assertIn(
            "raw-audio-fine-tuning",
            model_spec.capabilities,
        )
        self.assertTrue(training.native_training)
        self.assertEqual(training.support.value, "native")
        self.assertEqual(training.default_phase, "spectrogram")
        self.assertEqual(
            training.adapter_factory,
            ("voicehub.models.speecht5.training:"
             "NativeSpeechT5TrainingAdapter"),
        )
        self.assertEqual(
            BUILTIN_MODEL_ADAPTERS["speecht5"].__name__,
            "NativeSpeechT5TrainingAdapter",
        )

    def test_native_training_adapter_accepts_raw_audio_and_backpropagates(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = _wrapper(Path(directory))
            adapter = NativeSpeechT5TrainingAdapter(
                wrapper,
                get_training_spec("speecht5"),
            )
            output = adapter(
                text=["ab", "a"],
                audio=[
                    {
                        "array": torch.sin(torch.linspace(0, 12, 800)),
                        "sampling_rate": 8_000,
                    },
                    {
                        "array": torch.sin(torch.linspace(0, 20, 1_600)),
                        "sampling_rate": 16_000,
                    },
                ],
                speaker_embeddings=torch.randn(1, 4),
            )
            manifest = adapter.artifact_manifest()
            output.loss.backward()

        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.logits.shape[0], 2)
        self.assertIsNotNone(wrapper.model.speecht5.encoder.prenet.embed_tokens.weight.grad)
        self.assertTrue(torch.isfinite(wrapper.model.speecht5.encoder.prenet.embed_tokens.weight.grad).all())
        self.assertTrue(all(parameter.grad is None for parameter in wrapper.vocoder.parameters()))
        self.assertTrue(manifest["raw_data_fine_tuning"])
        self.assertEqual(manifest["frozen_components"], ["vocoder"])

    def test_native_bundle_round_trip_preserves_seeded_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_root = root / "build"
            build_root.mkdir()
            export = root / "export"
            wrapper = _wrapper(build_root)
            wrapper._prepare_for_inference()
            wrapper.save_pretrained(export)
            restored = SpeechT5ForTextToSpeech(
                export,
                device="cpu",
                lazy_load=False,
            )
            first = wrapper._generate(
                "ab",
                speaker_embeddings=torch.zeros(4),
                threshold=0.0,
                maxlenratio=1.0,
                seed=19,
            )
            second = restored._generate(
                "ab",
                speaker_embeddings=torch.zeros(4),
                threshold=0.0,
                maxlenratio=1.0,
                seed=19,
            )

        torch.testing.assert_close(first.audio, second.audio, rtol=0.0, atol=0.0)
        self.assertEqual(second.metadata["backend"], "voicehub-native")
        self.assertEqual(second.metadata["seed"], 19)
        self.assertEqual(restored.native_config, _tiny_config())
        self.assertFalse(any(parameter.requires_grad for parameter in restored.vocoder.parameters()))


if __name__ == "__main__":
    unittest.main()
