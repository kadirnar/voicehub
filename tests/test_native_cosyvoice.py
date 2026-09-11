from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.checkpointing.errors import CheckpointCompatibilityError, CheckpointIntegrityError
from voicehub.models.cosyvoice.configuration import CosyVoiceConfig
from voicehub.models.cosyvoice.modeling import CosyVoiceForTextToSpeech
from voicehub.models.cosyvoice.native.checkpoint import (
    convert_audited_cosyvoice_legacy_checkpoint,
    inspect_cosyvoice_checkpoint,
    tensor_inventory_fingerprint,
    validate_cosyvoice_checkpoint,
)
from voicehub.models.cosyvoice.native.configuration import CosyVoiceArchitectureConfig
from voicehub.models.cosyvoice.native.metadata import (
    COSYVOICE3_LEGACY_FILES,
    COSYVOICE3_MODEL_REVISION,
    COSYVOICE_SOURCE_REVISION,
)
from voicehub.models.cosyvoice.native.modeling import CosyVoiceNativeModel
from voicehub.models.cosyvoice.native.registration import create_cosyvoice_architecture_spec
from voicehub.models.cosyvoice.native.runtime import CosyVoiceNativeRuntime, load_cosyvoice_runtime
from voicehub.models.cosyvoice.native.tokenization import (
    END_OF_PROMPT,
    END_OF_TEXT,
    IM_END,
    IM_START,
    CosyVoiceTextTokenizer,
)
from voicehub.models.cosyvoice.training_cosyvoice import CosyVoiceTrainingAdapter
from voicehub.tokenization import encode_gpt2_token
from voicehub.training.contracts import TrainingContext, TrainingPhaseSpec
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_tokenizer(directory: Path) -> CosyVoiceTextTokenizer:
    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    special_tokens = (END_OF_TEXT, IM_START, IM_END, END_OF_PROMPT)
    for token_id, spelling in enumerate(special_tokens, 256):
        vocabulary[spelling] = token_id
    vocabulary[encode_gpt2_token(b"hi")] = 260
    (directory / "vocab.json").write_text(
        json.dumps(vocabulary),
        encoding="utf-8",
    )
    (directory / "merges.txt").write_text(
        "#version: 0.2\nh i\n",
        encoding="utf-8",
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({
            "add_prefix_space": False,
            "errors": "replace",
            "added_tokens_decoder": {
                str(token_id): {
                    "content": spelling,
                    "special": True,
                }
                for token_id, spelling in enumerate(special_tokens, 256)
            },
        }),
        encoding="utf-8",
    )
    return CosyVoiceTextTokenizer.from_files(
        directory / "vocab.json",
        directory / "merges.txt",
        directory / "tokenizer_config.json",
        validate_published_ids=False,
    )


def _tiny_runtime(directory: Path) -> CosyVoiceNativeRuntime:
    torch.manual_seed(11)
    return CosyVoiceNativeRuntime(
        CosyVoiceNativeModel(CosyVoiceArchitectureConfig.tiny()),
        _write_tiny_tokenizer(directory),
    )


def _inventory(module) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {name: ("F32", tuple(value.shape)) for name, value in module.state_dict().items()}


class NativeCosyVoicePolicyTests(unittest.TestCase):

    def test_native_slice_has_no_external_model_runtime_imports(self):
        forbidden = {
            "diffusers",
            "einops",
            "huggingface_hub",
            "hyperpyyaml",
            "modelscope",
            "numpy",
            "safetensors",
            "scipy",
            "tokenizers",
            "torchaudio",
            "transformers",
            "x_transformers",
        }
        roots = (
            PROJECT_ROOT / 'voicehub/models/cosyvoice/native',
            PROJECT_ROOT / 'voicehub/models/cosyvoice',
        )
        violations = []
        for root in roots:
            for path in (root.rglob("*.py") if root.name == "native" else root.glob("*.py")):
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

    def test_public_packages_are_torch_lazy(self):
        script = (
            "import sys; "
            "import voicehub.models.cosyvoice.native; "
            "import voicehub.models.cosyvoice; "
            "print('torch' in sys.modules, 'transformers' in sys.modules, "
            "'numpy' in sys.modules, 'safetensors' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False False False False")

    def test_provenance_and_capability_boundary_are_immutable(self):
        source = json.loads(
            (PROJECT_ROOT / 'voicehub/models/cosyvoice/native/SOURCE.json').read_text(encoding="utf-8"))
        self.assertEqual(source["source"]["revision"], COSYVOICE_SOURCE_REVISION)
        self.assertEqual(
            source["checkpoint"]["revision"],
            COSYVOICE3_MODEL_REVISION,
        )
        self.assertEqual(source["source"]["license"], "Apache-2.0")
        self.assertIn(
            "CosyVoice 1 and 2",
            source["implementation"]["executable_checkpoint_compatibility"],
        )
        spec = create_cosyvoice_architecture_spec()
        self.assertTrue(spec.capabilities.training)
        self.assertFalse(spec.capabilities.streaming)
        self.assertEqual(spec.capabilities.checkpoint_formats, ("safetensors", ))
        self.assertEqual(
            spec.metadata["executable_checkpoint_compatibility"],
            "cosyvoice3-only",
        )
        self.assertNotIn(
            "cosyvoice-1-2-3-family",
            spec.capabilities.features,
        )

    def test_configuration_rejects_unimplemented_generations_and_unsafe_loading(self):
        for generation in (1, 2):
            with self.subTest(generation=generation):
                with self.assertRaisesRegex(ValueError, "CosyVoice 3 only"):
                    CosyVoiceArchitectureConfig(generation=generation)
        with self.assertRaisesRegex(ValueError, "never executes repository code"):
            CosyVoiceConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "requires Safetensors"):
            CosyVoiceConfig(use_safetensors=False)
        config = CosyVoiceArchitectureConfig.tiny()
        self.assertEqual(
            CosyVoiceArchitectureConfig.from_dict(config.to_dict()),
            config,
        )


class NativeCosyVoiceInventoryTests(unittest.TestCase):

    def test_complete_graph_matches_all_three_audited_official_inventories(self):
        with torch.device("meta"):
            model = CosyVoiceNativeModel(
                CosyVoiceArchitectureConfig(),
                initialize=False,
            )
        for component, module in (
            ("llm", model.llm),
            ("flow", model.flow),
            ("hift", model.hift),
        ):
            with self.subTest(component=component):
                expected = COSYVOICE3_LEGACY_FILES[component]
                inventory = _inventory(module)
                self.assertEqual(len(inventory), expected["tensor_count"])
                self.assertEqual(
                    sum(value.numel() for value in module.state_dict().values()),
                    expected["parameter_count"],
                )
                self.assertEqual(
                    tensor_inventory_fingerprint(inventory),
                    expected["header_fingerprint"],
                )

    def test_legacy_converter_rejects_every_unaudited_pickle_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blob = root / "checkpoint-blob"
            blob.write_bytes(b"not an audited checkpoint")
            source = root / "llm.pt"
            source.symlink_to(blob.name)
            model = CosyVoiceNativeModel(CosyVoiceArchitectureConfig.tiny()).llm
            with self.assertRaisesRegex(
                    CheckpointIntegrityError,
                    "file size differs",
            ):
                convert_audited_cosyvoice_legacy_checkpoint(
                    model,
                    source,
                    Path(temporary) / "llm.safetensors",
                    component="llm",
                )


class NativeCosyVoiceObjectiveTests(unittest.TestCase):

    def test_language_and_flow_objectives_are_differentiable(self):
        model = CosyVoiceNativeModel(CosyVoiceArchitectureConfig.tiny())
        language = model(
            component="llm",
            text_tokens=torch.tensor([[2, 3, 4]]),
            text_lengths=torch.tensor([3]),
            instruction_tokens=torch.tensor([[5, 6]]),
            instruction_lengths=torch.tensor([2]),
            speech_tokens=torch.tensor([[7, 8, 9]]),
            speech_lengths=torch.tensor([3]),
        )
        self.assertEqual(tuple(language.logits.shape), (1, 10, 40))
        self.assertTrue((language.labels[0, :6] == -100).all())
        self.assertEqual(
            language.labels[0, 6:].tolist(),
            [7, 8, 9, model.config.language.eos_token_id],
        )
        language.loss.backward()
        self.assertIsNotNone(model.llm.llm_decoder.weight.grad)

        model.zero_grad(set_to_none=True)
        flow = model(
            component="flow",
            speech_tokens=torch.tensor([[2, 3, 4]]),
            speech_lengths=torch.tensor([3]),
            speech_features=torch.randn(1, 6, 8),
            feature_lengths=torch.tensor([6]),
            speaker_embeddings=torch.randn(1, 4),
            generator=torch.Generator().manual_seed(7),
        )
        self.assertEqual(tuple(flow.path.shape), (1, 8, 6))
        self.assertTrue(torch.isfinite(flow.loss))
        flow.loss.backward()
        self.assertIsNotNone(model.flow.input_embedding.weight.grad)

    def test_hift_generator_and_discriminator_have_separate_real_objectives(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _tiny_runtime(Path(temporary))
            features = torch.randn(1, 8, 8)
            waveform, pitch = runtime.model.hift(features)
            real_waveform = torch.randn_like(waveform)
            target_pitch = torch.randn_like(pitch)

            runtime.prepare_for_training("hifigan_generator")
            generator = runtime.model(
                component="hifigan_generator",
                speech_features=features,
                waveform=real_waveform,
                pitch=target_pitch,
            )
            self.assertEqual(
                set(generator.losses),
                {
                    "adversarial_loss",
                    "feature_matching_loss",
                    "pitch_loss",
                    "spectral_reconstruction_loss",
                },
            )
            generator.loss.backward()
            self.assertTrue(any(parameter.grad is not None for parameter in runtime.model.hift.parameters()))
            self.assertTrue(
                all(parameter.grad is None for parameter in runtime.model.hifigan.discriminator.parameters()))

            runtime.model.zero_grad(set_to_none=True)
            runtime.prepare_for_training("hifigan_discriminator")
            discriminator = runtime.model(
                component="hifigan_discriminator",
                speech_features=features,
                waveform=real_waveform,
                pitch=target_pitch,
            )
            self.assertEqual(set(discriminator.losses), {"discriminator_loss"})
            discriminator.loss.backward()
            self.assertTrue(
                any(
                    parameter.grad is not None
                    for parameter in runtime.model.hifigan.discriminator.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in runtime.model.hift.parameters()))


class NativeCosyVoiceRuntimeTests(unittest.TestCase):

    def test_strict_safetensors_roundtrip_reload_and_public_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _tiny_runtime(root)
            export = runtime.save_pretrained(root / "export")
            reports = {
                component:
                inspect_cosyvoice_checkpoint(
                    export / f"{component}.safetensors",
                    component=component,
                )
                for component in ("llm", "flow", "hift")
            }
            self.assertEqual(
                reports["llm"].tensor_count,
                len(runtime.model.llm.state_dict()),
            )
            with self.assertRaises(CheckpointCompatibilityError):
                validate_cosyvoice_checkpoint(
                    runtime.model.llm,
                    export / "flow.safetensors",
                    component="llm",
                )

            reloaded = load_cosyvoice_runtime(export)
            self.assertEqual(reloaded.model.config, runtime.model.config)
            self.assertFalse(
                any(value.device.type == "meta" for value in reloaded.model.state_dict().values()))
            self.assertFalse(any(value.device.type == "meta" for _, value in reloaded.model.named_buffers()))
            for original, restored in zip(runtime.model.state_dict().values(),
                                          reloaded.model.state_dict().values()):
                self.assertTrue(torch.equal(original, restored))
            qwen = reloaded.model.llm.llm.model
            self.assertIs(
                qwen.model.embed_tokens.weight,
                qwen.lm_head.weight,
            )

            wrapper = CosyVoiceForTextToSpeech(
                CosyVoiceConfig(name_or_path=str(export)),
                device="cpu",
            )
            wrapper._runtime = reloaded
            output = wrapper.forward(
                "hi",
                speaker_embedding=torch.ones(4),
                min_new_tokens=2,
                max_new_tokens=2,
                flow_steps=2,
                top_k=5,
                top_p=0.8,
                temperature=1.0,
                seed=19,
            )
            self.assertEqual(output.sample_rate, 24_000)
            self.assertGreater(output.audio.numel(), 0)
            self.assertEqual(output.metadata["backend"], "voicehub-native")
            with self.assertRaisesRegex(ValueError, "shape \\[4\\]"):
                wrapper._validate_generation_inputs({
                    "speaker_embedding": torch.ones(192),
                })

    def test_language_adapter_uses_raw_text_and_frozen_preencoded_codec_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _tiny_runtime(Path(temporary))
            wrapper = CosyVoiceForTextToSpeech(
                CosyVoiceConfig(
                    name_or_path="unused",
                    training_component="llm",
                ),
                device="cpu",
            )
            wrapper._runtime = runtime
            phase = TrainingPhaseSpec(
                name="llm",
                component_paths=("model.llm", ),
                optimizer_names=("llm", ),
            )
            spec = ModelTrainingSpec(
                model_type="cosyvoice",
                family=TrainingFamily.COMPOSITE,
                module_paths=("model", ),
                component_paths=("model.llm", ),
                native_training=True,
                phases=(phase, ),
                default_phase="llm",
            )
            adapter = CosyVoiceTrainingAdapter(wrapper, spec)
            context = TrainingContext(
                phase=phase,
                inputs={
                    "records": [{
                        "text": "hi",
                        "instruction": "Speak clearly.",
                        "speech_tokens": [1, 2, 3],
                    }],
                },
            )
            output = adapter.execute_training_phase(context)
            self.assertEqual(output.training_phase, "llm")
            self.assertTrue(output.metadata["speech_tokenizer_frozen"])
            output.loss.backward()
            self.assertTrue(any(parameter.grad is not None for parameter in runtime.model.llm.parameters()))
            self.assertTrue(all(not parameter.requires_grad for parameter in runtime.model.flow.parameters()))
            self.assertTrue(all(not parameter.requires_grad for parameter in runtime.model.hift.parameters()))
            with self.assertRaisesRegex(ValueError, "pre-encoded"):
                runtime.prepare_language_batch([{"text": "hi"}])


if __name__ == "__main__":
    unittest.main()
