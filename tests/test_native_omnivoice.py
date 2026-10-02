from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional

from voicehub.architectures.omnivoice.artifacts import resolve_omnivoice_artifacts
from voicehub.architectures.omnivoice.checkpoint import (
    export_omnivoice_checkpoint,
    inspect_omnivoice_checkpoint,
    load_omnivoice_checkpoint,
    tensor_inventory_fingerprint,
)
from voicehub.architectures.omnivoice.codec import HiggsAudioV2Tokenizer
from voicehub.architectures.omnivoice.configuration import HiggsAudioV2Config, OmniVoiceArchitectureConfig
from voicehub.architectures.omnivoice.generation import (
    OmniVoiceGenerationConfig,
    OmniVoiceGenerator,
    OmniVoicePrompt,
    _chunk_text,
)
from voicehub.architectures.omnivoice.languages import resolve_language
from voicehub.architectures.omnivoice.metadata import (
    HIGGS_AUDIO_V2_HEADER_FINGERPRINT,
    HIGGS_AUDIO_V2_PARAMETER_COUNT,
    HIGGS_AUDIO_V2_TENSOR_COUNT,
    OMNIVOICE_MODEL_HEADER_FINGERPRINT,
    OMNIVOICE_MODEL_PARAMETER_COUNT,
    OMNIVOICE_MODEL_TENSOR_COUNT,
    OMNIVOICE_UPSTREAM_REVISION,
)
from voicehub.architectures.omnivoice.modeling import OmniVoiceModel
from voicehub.architectures.omnivoice.processing import (
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
from voicehub.architectures.omnivoice.runtime import OmniVoiceRuntime
from voicehub.architectures.omnivoice.silence import remove_silence
from voicehub.architectures.omnivoice.voice_design import resolve_instruction
from voicehub.models.omnivoice_native.configuration_omnivoice import OmniVoiceConfig
from voicehub.models.omnivoice_native.modeling_omnivoice import OmniVoiceForTextToSpeech
from voicehub.models.omnivoice_native.training_omnivoice import OmniVoiceTrainingAdapter
from voicehub.processing.waveform import resample_waveform_hann
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
        source_entrypoints=("voicehub.architectures.omnivoice.modeling:OmniVoiceModel.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(phase, ),
        default_phase=phase.name,
    )


class NativeOmniVoiceDependencyTests(unittest.TestCase):

    def test_native_files_do_not_import_external_model_runtimes(self):
        roots = (
            PROJECT_ROOT / "voicehub" / "architectures" / "omnivoice",
            PROJECT_ROOT / "voicehub" / "models" / "omnivoice_native",
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
            "import sys; import voicehub.models.omnivoice_native; "
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
        source = json.loads((PROJECT_ROOT / "voicehub" / "architectures" / "omnivoice" /
                             "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(source["source"]["revision"], OMNIVOICE_UPSTREAM_REVISION)
        self.assertTrue(source["training"]["full_finetuning"])
        self.assertEqual(source["training"]["audio_codebooks"], 8)
        self.assertTrue(
            (PROJECT_ROOT / "voicehub" / "architectures" / "omnivoice" / "THIRD_PARTY_LICENSE").is_file())


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


def _tone(milliseconds: int, amplitude: float, *, sample_rate: int = 24_000) -> torch.Tensor:
    count = sample_rate * milliseconds // 1000
    phase = torch.arange(count, dtype=torch.float64) * (2 * torch.pi * 220 / sample_rate)
    return (amplitude * phase.sin()).float()


def _silence_fixture() -> torch.Tensor:
    return torch.cat([
        torch.zeros(7_200),
        _tone(400, 0.3),
        torch.zeros(16_800),
        _tone(300, 0.2),
        torch.zeros(6_000 + 13),
    ])


class _RecordingCodec(torch.nn.Module):
    """Minimal codec stand-in that records what the prompt encoder sends."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(frame_rate=25, hop_length=960, num_quantizers=2, sample_rate=24_000)
        self.sample_rate = 24_000
        self.frame_rate = 25
        self.register_buffer("_device_anchor", torch.zeros(()))
        self.inputs = []

    @property
    def device(self):
        return self._device_anchor.device

    def encode(self, values):
        self.inputs.append(values.detach().clone())
        frames = values.shape[-1] // self.config.hop_length
        return SimpleNamespace(audio_codes=torch.zeros((1, 2, frames), dtype=torch.long))


def _recording_generator() -> tuple[OmniVoiceGenerator, _RecordingCodec]:
    generator = object.__new__(OmniVoiceGenerator)
    codec = _RecordingCodec()
    generator.model = OmniVoiceModel(OmniVoiceArchitectureConfig.tiny(vocab_size=320))
    generator.text_tokenizer = _tiny_tokenizer()
    generator.audio_tokenizer = codec
    return generator, codec


class NativeOmniVoiceUpstreamParityTests(unittest.TestCase):
    """Regressions found by the k2-fsa/OmniVoice 468e927 parity audit."""

    def test_meta_loaded_checkpoint_materializes_rotary_buffers(self):
        torch.manual_seed(5)
        config = OmniVoiceArchitectureConfig.tiny(vocab_size=320)
        source = OmniVoiceModel(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            export_omnivoice_checkpoint(source, path)
            with torch.device("meta"):
                target = OmniVoiceModel(config, initialize=False)
            load_omnivoice_checkpoint(target, path, device="cpu")
        self.assertFalse(any(buffer.is_meta for buffer in target.buffers()))
        input_ids = torch.randint(0, 16, (1, 2, 5))
        audio_mask = torch.ones(1, 5, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(
                target.eval()(input_ids, audio_mask).logits,
                source(input_ids, audio_mask).logits,
                rtol=0.0,
                atol=0.0,
            )

    def test_hugging_face_snapshot_symlinks_resolve_to_named_files(self):
        torch.manual_seed(6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = root / "blobs"
            snapshot = root / "snapshots" / "abc"
            (snapshot / "audio_tokenizer").mkdir(parents=True)
            blobs.mkdir()
            files = {
                "model.safetensors": None,
                "config.json": b"{}",
                "tokenizer.json": b"{}",
                "audio_tokenizer/model.safetensors": b"x",
                "audio_tokenizer/config.json": b"{}",
            }
            for index, (name, payload) in enumerate(files.items()):
                blob = blobs / f"{index:064x}"
                if payload is None:
                    export_omnivoice_checkpoint(
                        OmniVoiceModel(OmniVoiceArchitectureConfig.tiny(vocab_size=320)),
                        blob.with_suffix(".safetensors"),
                    ).rename(blob)
                else:
                    blob.write_bytes(payload)
                (snapshot / name).symlink_to(blob)
            artifacts = resolve_omnivoice_artifacts(snapshot, verify_integrity=False)
            self.assertEqual(artifacts.model_checkpoint, (snapshot / "model.safetensors").absolute())
            self.assertEqual(
                artifacts.codec_checkpoint,
                (snapshot / "audio_tokenizer" / "model.safetensors").absolute(),
            )
            report = inspect_omnivoice_checkpoint(artifacts.model_checkpoint)
            self.assertGreater(report.tensor_count, 0)

    def test_silence_removal_matches_upstream_pydub_reference(self):
        # Expected values were produced by omnivoice.utils.audio.remove_silence
        # (pydub 0.25.1) at upstream revision 468e927 on the same signal.
        waveform = _silence_fixture()
        self.assertEqual(waveform.numel(), 46_813)
        for (middle, leading, trailing), length in (
            ((200, 100, 200), 33_600),
            ((500, 100, 100), 38_424),
            ((0, 100, 300), 42_024),
        ):
            result = remove_silence(
                waveform,
                24_000,
                middle_ms=middle,
                leading_ms=leading,
                trailing_ms=trailing,
            )
            self.assertEqual(result.numel(), length)
            nonzero = result.nonzero().flatten()
            self.assertEqual(int(nonzero[0]), 2_401)
            self.assertEqual(float(result[2_401]), 0.017242431640625)
            self.assertEqual(float(result.abs().max()), 9_830 / 32_768)
            self.assertTrue(torch.equal(result * 32_768, (result * 32_768).round()))

    def test_prompt_is_resampled_trimmed_and_punctuated_like_upstream(self):
        generator, codec = _recording_generator()
        quiet = _silence_fixture() * 0.1
        prompt = generator.create_prompt(
            quiet,
            sampling_rate=24_000,
            reference_text=" Hello there, ",
            preprocess_prompt=True,
        )
        rms = float(quiet.square().mean().sqrt())
        expected = remove_silence(
            quiet * 0.1 / rms,
            24_000,
            middle_ms=200,
            leading_ms=100,
            trailing_ms=200,
        )
        expected = expected[:expected.numel() - expected.numel() % 960]
        self.assertTrue(torch.equal(codec.inputs[-1][0, 0], expected))
        # "," is upstream end punctuation, so no period is appended.
        self.assertEqual(prompt.reference_text, "Hello there,")
        self.assertEqual(prompt.reference_rms, rms)

        raw = generator.create_prompt(
            quiet,
            sampling_rate=24_000,
            reference_text=" Hello there ",
            preprocess_prompt=False,
        )
        self.assertEqual(raw.reference_text, " Hello there ")
        self.assertEqual(codec.inputs[-1].shape[-1], quiet.numel() - quiet.numel() % 960)

    def test_output_postprocessing_follows_upstream_order(self):
        generator, _ = _recording_generator()
        raw = _silence_fixture()
        config = OmniVoiceGenerationConfig(postprocess_output=False)
        # Prompt-free output is peak-normalized even without silence removal.
        output = generator._postprocess(raw.clone(), prompt=None, config=config)
        self.assertEqual(output.numel(), raw.numel() + 2 * 2_400)
        self.assertAlmostEqual(float(output.abs().max()), 0.5, places=6)
        prompt = OmniVoicePrompt(torch.zeros((2, 1), dtype=torch.long), "a.", 0.05)
        trimmed = generator._postprocess(raw.clone(), prompt=prompt, config=OmniVoiceGenerationConfig())
        expected = remove_silence(raw, 24_000, middle_ms=500, leading_ms=100, trailing_ms=100) * 0.05 / 0.1
        self.assertEqual(trimmed.numel(), expected.numel() + 2 * 2_400)
        self.assertTrue(torch.equal(trimmed[2_400 + 2_400:-2_400 - 2_400], expected[2_400:-2_400]))

    def test_prompt_resamples_with_the_torchaudio_default_kernel(self):
        generator, codec = _recording_generator()
        waveform = _silence_fixture()[::3][:15_000] * 0.5
        generator.create_prompt(
            waveform,
            sampling_rate=16_000,
            reference_text="a.",
            preprocess_prompt=False,
        )
        expected = resample_waveform_hann(waveform, 16_000, 24_000)
        rms = float(expected.square().mean().sqrt())
        self.assertLess(rms, 0.1)
        expected = expected * 0.1 / rms
        expected = expected[:expected.numel() - expected.numel() % 960]
        self.assertTrue(torch.equal(codec.inputs[-1][0, 0], expected))

    def test_sdpa_attention_matches_the_explicit_float32_path(self):
        torch.manual_seed(9)
        model = OmniVoiceModel(OmniVoiceArchitectureConfig.tiny(vocab_size=320)).eval()
        input_ids = torch.randint(0, 16, (2, 2, 6))
        audio_mask = torch.ones(2, 6, dtype=torch.bool)
        attention_mask = torch.zeros(2, 1, 6, 6, dtype=torch.bool)
        attention_mask[0, :, :, :] = True
        attention_mask[1, :, :4, :4] = True
        attention_mask[1, :, 4:, 4:] = torch.eye(2, dtype=torch.bool)
        with torch.no_grad():
            fused = model(input_ids, audio_mask, attention_mask=attention_mask).logits
            explicit = model(
                input_ids,
                audio_mask,
                attention_mask=attention_mask,
                output_attentions=True,
            ).logits
        torch.testing.assert_close(fused, explicit, rtol=1e-5, atol=1e-5)

    def test_language_names_resolve_like_upstream(self):
        self.assertEqual(resolve_language("English"), "en")
        self.assertEqual(resolve_language("chinese"), "zh")
        self.assertEqual(resolve_language("en"), "en")
        self.assertIsNone(resolve_language("None"))
        # ``assertWarns`` walks ``sys.modules``; record warnings directly.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(resolve_language("Klingon"))
        self.assertTrue(any("Klingon" in str(item.message) for item in caught))

    def test_voice_design_instructions_are_normalized_like_upstream(self):
        # Expected strings come from upstream ``_resolve_instruct``.
        self.assertEqual(resolve_instruction("Female, British Accent"), "female, british accent")
        self.assertEqual(resolve_instruction("male, low pitch", use_chinese=True), "男，低音调")
        self.assertEqual(resolve_instruction("男，河南话"), "男，河南话")
        self.assertEqual(resolve_instruction("female，elderly"), "female, elderly")
        with self.assertRaisesRegex(ValueError, "did you mean 'female'"):
            resolve_instruction("femal")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            resolve_instruction("male, female")

    def test_style_prompt_receives_resolved_language_and_instruction(self):
        generator, _ = _recording_generator()
        captured = {}

        def prepare(text, **options):
            captured.update(options)
            raise RuntimeError("stop")

        generator._prepare_inputs = prepare
        with self.assertRaisesRegex(RuntimeError, "stop"):
            generator.generate(
                "你好。",
                language="Chinese",
                instruction="Male, Low Pitch",
                duration=0.08,
            )
        self.assertEqual(captured["language"], "zh")
        self.assertEqual(captured["instruction"], "男，低音调")

    def test_long_text_chunking_matches_upstream_punctuation_splitter(self):
        # Expected chunks come from upstream ``chunk_text_punctuation``: long
        # sentences are never split mid-sentence and "Fr."/"Rd." are
        # abbreviations.
        text = (
            "Mr. Smith went to Fr. Brown's house on Main Rd. today. It was a very long and "
            "winding sentence without any stops at all whatsoever. Ok. Hi")
        self.assertEqual(
            _chunk_text(text, maximum_characters=20),
            [
                "Mr. Smith went to Fr. Brown's house on Main Rd. today.",
                "It was a very long and winding sentence without any stops at all whatsoever.",
                "Ok. Hi",
            ],
        )
        self.assertEqual(_chunk_text("A. B. This is it.", maximum_characters=5), ["A. B.", "This is it."])


if __name__ == "__main__":
    unittest.main()
