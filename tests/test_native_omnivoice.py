from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional

from voicehub.models.omnivoice.configuration import OmniVoiceConfig
from voicehub.models.omnivoice.modeling import OmniVoiceForTextToSpeech
from voicehub.models.omnivoice.native.checkpoint import (
    export_omnivoice_checkpoint,
    load_omnivoice_checkpoint,
    tensor_inventory_fingerprint,
)
from voicehub.models.omnivoice.native.codec import HiggsAudioV2Tokenizer
from voicehub.models.omnivoice.native.configuration import HiggsAudioV2Config, OmniVoiceArchitectureConfig
from voicehub.models.omnivoice.native.generation import OmniVoiceGenerationConfig, OmniVoiceGenerator, OmniVoicePrompt
from voicehub.models.omnivoice.native.metadata import (
    HIGGS_AUDIO_V2_HEADER_FINGERPRINT,
    HIGGS_AUDIO_V2_PARAMETER_COUNT,
    HIGGS_AUDIO_V2_TENSOR_COUNT,
    OMNIVOICE_MODEL_HEADER_FINGERPRINT,
    OMNIVOICE_MODEL_PARAMETER_COUNT,
    OMNIVOICE_MODEL_TENSOR_COUNT,
    OMNIVOICE_UPSTREAM_REVISION,
)
from voicehub.models.omnivoice.native.modeling import OmniVoiceModel
from voicehub.models.omnivoice.native.processing import (
    DENOISE,
    END_OF_TEXT,
    IM_END,
    INSTRUCT_END,
    INSTRUCT_START,
    LANG_END,
    LANG_START,
    TEXT_END,
    TEXT_START,
    OmniVoiceMaskingConfig,
    OmniVoicePaddingCollator,
    OmniVoiceSampleProcessor,
    OmniVoiceTokenizer,
)
from voicehub.models.omnivoice.native.runtime import OmniVoiceRuntime
from voicehub.models.omnivoice.training_omnivoice import OmniVoiceTrainingAdapter
from voicehub.tokenization import ByteBPETokenizer
from voicehub.training.contracts import TrainingPhaseSpec, TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_tokenizer() -> OmniVoiceTokenizer:
    regular = {bytes([value]): value + 16 for value in range(256)}
    special = {
        END_OF_TEXT: 0,
        IM_END: 1,
        DENOISE: 2,
        LANG_START: 3,
        LANG_END: 4,
        INSTRUCT_START: 5,
        INSTRUCT_END: 6,
        TEXT_START: 7,
        TEXT_END: 8,
    }
    tokenizer = ByteBPETokenizer(
        regular,
        special_tokens=special,
        pad_token_id=0,
        use_regex=False,
    )
    return OmniVoiceTokenizer(
        tokenizer,
        validate_published_ids=False,
    )


def _tiny_codec() -> HiggsAudioV2Tokenizer:
    codec = object.__new__(HiggsAudioV2Tokenizer)
    torch.nn.Module.__init__(codec)
    codec.config = SimpleNamespace(
        frame_rate=25,
        hop_length=960,
        num_quantizers=2,
        sample_rate=24_000,
    )
    codec.semantic_model = torch.nn.Identity()
    codec.register_parameter(
        "_test_parameter",
        torch.nn.Parameter(torch.zeros(())),
    )
    return codec


def _training_spec() -> ModelTrainingSpec:
    phase = TrainingPhaseSpec(
        name="masked_audio",
        component_paths=("model", ),
        optimizer_names=("model", ),
        loss_keys=("loss", ),
        prediction_keys=("logits", ),
        required_inputs=("input_ids", "audio_mask", "labels"),
    )
    return ModelTrainingSpec(
        model_type="omnivoice",
        family=TrainingFamily.COMPOSITE,
        module_paths=("model", ),
        component_paths=("model", ),
        loss_keys=("loss", ),
        prediction_keys=("logits", ),
        source_entrypoints=("voicehub.models.omnivoice.native.modeling:OmniVoiceModel.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(phase, ),
        default_phase=phase.name,
    )


class NativeOmniVoiceDependencyTests(unittest.TestCase):

    def test_native_files_do_not_import_external_model_runtimes(self):
        roots = (
            PROJECT_ROOT / 'voicehub/models/omnivoice/native',
            PROJECT_ROOT / 'voicehub/models/omnivoice',
        )
        forbidden = {
            "accelerate",
            "einops",
            "huggingface_hub",
            "librosa",
            "numpy",
            "safetensors",
            "tokenizers",
            "torchaudio",
            "transformers",
        }
        violations = []
        for root in roots:
            for path in root.glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
                        names = [node.module]
                    else:
                        names = []
                    violations.extend(
                        (path.name, name) for name in names if name.split(".", 1)[0] in forbidden)
        self.assertEqual(violations, [])

    def test_public_provider_import_remains_lazy(self):
        command = (
            "import sys; import voicehub.models.omnivoice; "
            "print('torch' in sys.modules, 'transformers' in sys.modules, "
            "'huggingface_hub' in sys.modules, 'safetensors' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False False False False")

    def test_provenance_pins_complete_training_boundary(self):
        source = json.loads(
            (PROJECT_ROOT / 'voicehub/models/omnivoice/native/SOURCE.json').read_text(encoding="utf-8"))
        self.assertEqual(source["source"]["revision"], OMNIVOICE_UPSTREAM_REVISION)
        self.assertTrue(source["training"]["full_finetuning"])
        self.assertEqual(source["training"]["audio_codebooks"], 8)
        self.assertTrue((PROJECT_ROOT / 'voicehub/models/omnivoice/native/THIRD_PARTY_LICENSE').is_file())


class NativeOmniVoiceInventoryTests(unittest.TestCase):

    def test_official_model_matches_audited_safetensors_inventory(self):
        with torch.device("meta"):
            model = OmniVoiceModel(
                OmniVoiceArchitectureConfig(),
                initialize=False,
            )
        state = model.state_dict()
        dtypes = {torch.float32: "F32", torch.int64: "I64"}
        inventory = {name: (dtypes[value.dtype], tuple(value.shape)) for name, value in state.items()}
        self.assertEqual(len(state), OMNIVOICE_MODEL_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            OMNIVOICE_MODEL_PARAMETER_COUNT,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(inventory),
            OMNIVOICE_MODEL_HEADER_FINGERPRINT,
        )

    def test_official_higgs_codec_matches_audited_inventory(self):
        with torch.device("meta"):
            codec = HiggsAudioV2Tokenizer(
                HiggsAudioV2Config(),
                initialize=False,
            )
        state = codec.state_dict()
        inventory = {name: ("F32", tuple(value.shape)) for name, value in state.items()}
        self.assertEqual(len(state), HIGGS_AUDIO_V2_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            HIGGS_AUDIO_V2_PARAMETER_COUNT,
        )
        self.assertEqual(
            tensor_inventory_fingerprint(inventory),
            HIGGS_AUDIO_V2_HEADER_FINGERPRINT,
        )


class NativeOmniVoiceModelTests(unittest.TestCase):

    def test_attention_is_bidirectional(self):
        torch.manual_seed(7)
        config = OmniVoiceArchitectureConfig.tiny(vocab_size=320)
        model = OmniVoiceModel(config)
        input_ids = torch.randint(0, 16, (1, 2, 5))
        audio_mask = torch.ones(1, 5, dtype=torch.bool)
        output = model(
            input_ids,
            audio_mask,
            output_attentions=True,
        )
        self.assertIsNotNone(output.attentions)
        attention = output.attentions[0]
        self.assertGreater(float(attention[0, 0, 0, -1].detach()), 0.0)

    def test_weighted_codebook_objective_matches_reference_math(self):
        torch.manual_seed(11)
        config = OmniVoiceArchitectureConfig.tiny(vocab_size=320)
        model = OmniVoiceModel(config)
        input_ids = torch.randint(0, 16, (2, 2, 4))
        audio_mask = torch.ones(2, 4, dtype=torch.bool)
        labels = torch.tensor([
            [[1, 2, -100, 3], [4, -100, 5, 6]],
            [[2, -100, 3, 4], [-100, 7, 8, 9]],
        ])
        output = model(input_ids, audio_mask, labels=labels)
        per_token = functional.cross_entropy(
            output.logits.permute(0, 3, 1, 2),
            labels,
            reduction="none",
            ignore_index=-100,
        )
        valid = (labels != -100).float()
        codebook = (per_token * valid).sum((0, 2)) / valid.sum((0, 2))
        expected = (codebook * torch.tensor([2 / 3, 1 / 3])).sum()
        torch.testing.assert_close(output.codebook_losses, codebook)
        torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        self.assertIsNotNone(model.audio_heads.weight.grad)

    def test_tiny_safetensors_roundtrip_is_strict(self):
        torch.manual_seed(13)
        config = OmniVoiceArchitectureConfig.tiny(vocab_size=320)
        source = OmniVoiceModel(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            export_omnivoice_checkpoint(source, path)
            target = OmniVoiceModel(config, initialize=False)
            load_omnivoice_checkpoint(target, path, device="cpu")
            for name, expected in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], expected)

    def test_iterative_generation_reveals_every_mask(self):
        config = OmniVoiceArchitectureConfig.tiny(vocab_size=320)
        model = OmniVoiceModel(config)
        codec = _tiny_codec()
        generator = OmniVoiceGenerator(model, _tiny_tokenizer(), codec)
        tokens = generator.generate_tokens(
            "a",
            duration=0.08,
            config=OmniVoiceGenerationConfig(
                num_steps=2,
                guidance_scale=0.0,
                position_temperature=0.0,
            ),
        )
        self.assertEqual(tuple(tokens.shape), (2, 2))
        self.assertFalse((tokens == config.audio_mask_id).any())


class NativeOmniVoiceProcessingTests(unittest.TestCase):

    def test_preprocessed_record_masks_only_audio_and_collates(self):
        config = OmniVoiceArchitectureConfig.tiny(vocab_size=320)
        processor = OmniVoiceSampleProcessor(
            _tiny_tokenizer(),
            config,
            masking=OmniVoiceMaskingConfig(
                prompt_ratio_range=(0.0, 0.0),
                mask_ratio_range=(1.0, 1.0),
                drop_cond_ratio=0.0,
                language_ratio=1.0,
                use_pinyin_ratio=0.0,
                instruct_ratio=1.0,
                only_instruct_ratio=0.0,
            ),
        )
        sample = processor({
            "audio_tokens": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "instruct": "happy",
            "language_id": "en",
            "text": "a",
        })
        self.assertTrue(
            torch.equal(
                sample["input_ids"][:, sample["audio_mask"]],
                torch.full((2, 3), config.audio_mask_id),
            ))
        self.assertTrue((sample["labels"][:, ~sample["audio_mask"]] == -100).all())
        batch = OmniVoicePaddingCollator(0)([sample, sample])
        self.assertEqual(tuple(batch["input_ids"].shape[:2]), (2, 2))
        self.assertEqual(batch["attention_mask"].dtype, torch.bool)

    def test_raw_audio_requires_the_native_codec(self):
        processor = OmniVoiceSampleProcessor(
            _tiny_tokenizer(),
            OmniVoiceArchitectureConfig.tiny(vocab_size=320),
        )
        with self.assertRaisesRegex(ValueError, "Higgs Audio V2"):
            processor({
                "sampling_rate": 24_000,
                "text": "a",
                "waveform": torch.zeros(2_400),
            })

    def test_prompt_artifact_is_pickle_free(self):
        prompt = OmniVoicePrompt(
            torch.tensor([[1, 2], [3, 4]]),
            "reference.",
            0.05,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt"
            prompt.save(path)
            restored = OmniVoicePrompt.load(path)
            torch.testing.assert_close(restored.audio_tokens, prompt.audio_tokens)
            self.assertEqual(restored.reference_text, prompt.reference_text)
            self.assertFalse(any(path.glob("*.pt")))


class NativeOmniVoiceProviderConfigurationTests(unittest.TestCase):

    def test_configuration_fails_closed_on_external_code_and_pickle(self):
        with self.assertRaisesRegex(ValueError, "never executes"):
            OmniVoiceConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            OmniVoiceConfig(use_safetensors=False)

    def test_training_and_generation_controls_are_validated(self):
        config = OmniVoiceConfig(
            training_masking_config={"mask_ratio_range": [0.2, 0.8]},
            training_packing_tokens=4_096,
            generation_config={"num_steps": 8},
        )
        self.assertEqual(config.training_packing_tokens, 4_096)
        self.assertEqual(config.generation_config["num_steps"], 8)
        hub_codec = OmniVoiceConfig(codec_source="eustlb/higgs-audio-v2-tokenizer")
        self.assertEqual(
            hub_codec.codec_source,
            "eustlb/higgs-audio-v2-tokenizer",
        )
        with self.assertRaisesRegex(ValueError, "masking option"):
            OmniVoiceConfig(training_masking_config={"mystery": 1})

    def test_full_finetuning_runs_and_keeps_higgs_frozen(self):
        architecture_config = OmniVoiceArchitectureConfig.tiny(vocab_size=320)
        native_model = OmniVoiceModel(architecture_config)
        codec = _tiny_codec()
        runtime = OmniVoiceRuntime(
            native_model,
            _tiny_tokenizer(),
            codec,
        )
        wrapper = OmniVoiceForTextToSpeech(
            OmniVoiceConfig(
                training_masking_config={
                    "drop_cond_ratio": 0.0,
                    "instruct_ratio": 0.0,
                    "language_ratio": 0.0,
                    "mask_ratio_range": [1.0, 1.0],
                    "only_instruct_ratio": 0.0,
                    "prompt_ratio_range": [0.0, 0.0],
                    "use_pinyin_ratio": 0.0,
                }),
            device="cpu",
            lazy_load=True,
        )
        wrapper._runtime = runtime
        adapter = OmniVoiceTrainingAdapter(wrapper, _training_spec()).setup()
        self.assertIs(adapter.primary_model, native_model)
        self.assertFalse(any(parameter.requires_grad for parameter in codec.parameters()))
        batch = wrapper.prepare_training_inputs(
            {"records": [{
                "audio_tokens": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "text": "a",
            }]},
            phase="masked_audio",
        )
        output = native_model(**batch)
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        self.assertIsNotNone(native_model.audio_heads.weight.grad)


if __name__ == "__main__":
    unittest.main()
