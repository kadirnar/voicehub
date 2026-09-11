from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from voicehub.models.asr_granite_speech import (
    GraniteSpeechASRConfig,
    GraniteSpeechForSpeechRecognition,
    NativeGraniteSpeechTrainingAdapter,
)
from voicehub.models.asr_granite_speech.native.artifacts import GraniteSpeechArtifacts, resolve_granite_speech_artifacts
from voicehub.models.asr_granite_speech.native.checkpoint import (
    granite_speech_header_fingerprint,
    math_product,
    native_granite_speech_tensor_shapes,
)
from voicehub.models.asr_granite_speech.native.configuration import (
    GraniteSpeechArchitectureConfig,
    GraniteSpeechEncoderConfig,
    GraniteSpeechProjectorConfig,
)
from voicehub.models.asr_granite_speech.native.frontend import GraniteSpeechFeatureExtractor
from voicehub.models.asr_granite_speech.native.modeling import GraniteSpeechForConditionalGeneration
from voicehub.models.asr_granite_speech.native.processing import GraniteSpeechProcessor
from voicehub.models.asr_granite_speech.native.runtime import (
    GraniteSpeechRuntime,
    load_granite_speech_runtime,
    save_granite_speech_runtime,
)
from voicehub.models.asr_granite_speech.native.tokenization import (
    AUDIO_TOKEN,
    DEFAULT_AUDIO_TOKEN_ID,
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_PAD_TOKEN_ID,
    GraniteSpeechTokenizer,
)
from voicehub.models.causal_lm.native.configuration import GraniteConfig
from voicehub.processing.waveform import save_pcm_wave
from voicehub.tasks import SpeechTask
from voicehub.tokenization.assets import encode_gpt2_token
from voicehub.training import ModelTrainingSpec, TrainingFamily, TrainingSupport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCHAUDIO_AVAILABLE = (importlib.util.find_spec("torchaudio") is not None)
TRANSFORMERS_AVAILABLE = (importlib.util.find_spec("transformers") is not None)


def _tiny_config() -> GraniteSpeechArchitectureConfig:
    return GraniteSpeechArchitectureConfig(
        text_config=GraniteConfig(
            vocab_size=100_353,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=256,
            initializer_range=0.02,
            pad_token_id=DEFAULT_PAD_TOKEN_ID,
            bos_token_id=DEFAULT_EOS_TOKEN_ID,
            eos_token_id=DEFAULT_EOS_TOKEN_ID,
            tie_word_embeddings=False,
            embedding_multiplier=1.0,
            logits_scaling=1.0,
            residual_multiplier=1.0,
        ),
        encoder_config=GraniteSpeechEncoderConfig(
            input_dim=160,
            num_layers=2,
            hidden_dim=8,
            feedforward_mult=2,
            num_heads=2,
            dim_head=4,
            output_dim=4,
            context_size=4,
            max_pos_emb=8,
            dropout=0.0,
            conv_kernel_size=3,
            conv_expansion_factor=2,
        ),
        projector_config=GraniteSpeechProjectorConfig(
            hidden_size=8,
            encoder_hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            cross_attention_frequency=1,
        ),
        audio_token_index=DEFAULT_AUDIO_TOKEN_ID,
        initializer_range=0.02,
        has_lora_adapter=False,
        downsample_rate=1,
        window_size=3,
        tie_word_embeddings=False,
    )


def _official_config() -> GraniteSpeechArchitectureConfig:
    return GraniteSpeechArchitectureConfig.from_dict({
        "model_type": "granite_speech",
        "audio_token_index": 100_352,
        "downsample_rate": 5,
        "has_lora_adapter": False,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "window_size": 15,
        "encoder_config": {
            "model_type": "granite_speech_encoder",
            "input_dim": 160,
            "num_layers": 16,
            "hidden_dim": 1_024,
            "feedforward_mult": 4,
            "num_heads": 8,
            "dim_head": 128,
            "output_dim": 348,
            "context_size": 200,
            "max_pos_emb": 512,
            "dropout": 0.1,
            "conv_kernel_size": 15,
            "conv_expansion_factor": 2,
        },
        "projector_config": {
            "model_type": "blip_2_qformer",
            "hidden_size": 1_024,
            "encoder_hidden_size": 1_024,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "intermediate_size": 4_096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "layer_norm_eps": 1e-12,
            "cross_attention_frequency": 1,
            "use_qformer_text_input": False,
        },
        "text_config": {
            "model_type": "granite",
            "vocab_size": 100_353,
            "hidden_size": 2_048,
            "intermediate_size": 4_096,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_position_embeddings": 4_096,
            "initializer_range": 0.1,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10_000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "mlp_bias": False,
            "embedding_multiplier": 12.0,
            "logits_scaling": 8.0,
            "residual_multiplier": 0.22,
            "attention_multiplier": 0.0078125,
            "pad_token_id": 100_256,
            "bos_token_id": 100_257,
            "eos_token_id": 100_257,
            "tie_word_embeddings": False,
        },
    })


def _write_tokenizer_assets(directory: Path, ) -> GraniteSpeechTokenizer:
    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    document = {
        "version":
        "1.0",
        "added_tokens": [
            {
                "id": DEFAULT_PAD_TOKEN_ID,
                "content": "<|pad|>",
                "special": True,
                "lstrip": False,
                "rstrip": False,
            },
            {
                "id": DEFAULT_EOS_TOKEN_ID,
                "content": "<|end_of_text|>",
                "special": True,
                "lstrip": False,
                "rstrip": False,
            },
            {
                "id": DEFAULT_AUDIO_TOKEN_ID,
                "content": AUDIO_TOKEN,
                "special": True,
                "lstrip": False,
                "rstrip": False,
            },
        ],
        "normalizer":
        None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "vocab": vocabulary,
            "merges": [],
            "unk_token": None,
        },
    }
    (directory / "tokenizer.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({
            "add_bos_token": False,
            "add_prefix_space": False,
        }),
        encoding="utf-8",
    )
    (directory / "special_tokens_map.json").write_text(
        json.dumps({
            "bos_token": "<|end_of_text|>",
            "eos_token": "<|end_of_text|>",
            "pad_token": "<|pad|>",
        }),
        encoding="utf-8",
    )
    (directory / "added_tokens.json").write_text(
        json.dumps({AUDIO_TOKEN: DEFAULT_AUDIO_TOKEN_ID}),
        encoding="utf-8",
    )
    (directory / "chat_template.jinja").write_text(
        "USER: {{ message['content'] }}\n ASSISTANT:",
        encoding="utf-8",
    )
    return GraniteSpeechTokenizer.from_files(
        directory / "tokenizer.json",
        tokenizer_config=directory / "tokenizer_config.json",
        special_tokens_map=directory / "special_tokens_map.json",
        added_tokens=directory / "added_tokens.json",
        chat_template=directory / "chat_template.jinja",
    )


def _tiny_runtime(directory: Path) -> GraniteSpeechRuntime:
    directory.mkdir(parents=True, exist_ok=True)
    config = _tiny_config()
    tokenizer = _write_tokenizer_assets(directory)
    processor = GraniteSpeechProcessor(config, tokenizer)
    model = GraniteSpeechForConditionalGeneration(config)
    artifacts = GraniteSpeechArtifacts(
        source=str(directory),
        revision=None,
        config=directory / "config.json",
        checkpoint=directory / "model.safetensors",
        preprocessor_config=directory / "preprocessor_config.json",
        processor_config=directory / "processor_config.json",
        tokenizer=directory / "tokenizer.json",
        tokenizer_config=directory / "tokenizer_config.json",
        special_tokens_map=directory / "special_tokens_map.json",
        added_tokens=directory / "added_tokens.json",
        chat_template=directory / "chat_template.jinja",
    )
    return GraniteSpeechRuntime(
        model=model,
        processor=processor,
        config=config,
        artifacts=artifacts,
        generation_config={
            "do_sample": False,
            "eos_token_id": DEFAULT_EOS_TOKEN_ID,
            "pad_token_id": DEFAULT_PAD_TOKEN_ID,
        },
    )


class GraniteSpeechConfigurationTests(unittest.TestCase):

    def test_package_import_is_dependency_free(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_granite_speech; "
            "print('transformers' in sys.modules, 'torch' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False False")

    def test_public_config_fails_closed_for_delegated_or_ambiguous_options(self):
        config = GraniteSpeechASRConfig()
        self.assertEqual(config.sample_rate, 16_000)
        self.assertIn(AUDIO_TOKEN, config.transcription_prompt)
        with self.assertRaisesRegex(ValueError, "audio"):
            GraniteSpeechASRConfig(transcription_prompt="transcribe", )
        with self.assertRaisesRegex(ValueError, "prompt-conditioned"):
            GraniteSpeechASRConfig(training_language="English")
        with self.assertRaisesRegex(ValueError, "never executes"):
            GraniteSpeechASRConfig(trust_remote_code=True)
        with self.assertRaisesRegex(ValueError, "Safetensors"):
            GraniteSpeechASRConfig(use_safetensors=False)

    def test_official_checkpoint_namespace_is_exact(self):
        shapes = native_granite_speech_tensor_shapes(_official_config(), )
        self.assertEqual(len(shapes), 954)
        self.assertEqual(
            sum(name.startswith("encoder.") for name in shapes),
            534,
        )
        self.assertEqual(
            sum(name.startswith("projector.") for name in shapes),
            57,
        )
        self.assertEqual(
            sum(name.startswith("language_model.") for name in shapes),
            363,
        )
        self.assertEqual(
            shapes["language_model.model.embed_tokens.weight"],
            (100_353, 2_048),
        )
        self.assertEqual(
            shapes["projector.query"],
            (1, 3, 1_024),
        )
        inventory = {
            name: (
                "I64" if name.endswith("num_batches_tracked") else "BF16",
                shape,
            )
            for name, shape in shapes.items()
        }
        self.assertEqual(
            sum(dtype == "I64" for dtype, _ in inventory.values()),
            16,
        )
        self.assertEqual(
            sum(math_product(shape) * (8 if dtype == "I64" else 2) for dtype, shape in inventory.values()),
            4_626_414_392,
        )
        self.assertEqual(
            granite_speech_header_fingerprint(inventory),
            "8889064efd770b05c39cc62dba5fac842e006530649b12bcd1c3d90c7a474001",
        )

    def test_plus_and_embedded_adapter_graphs_fail_closed(self):
        values = _official_config().to_dict()
        values["model_type"] = "granite_speech_plus"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            GraniteSpeechArchitectureConfig.from_dict(values)
        values["model_type"] = "granite_speech"
        values["has_lora_adapter"] = True
        with self.assertRaisesRegex(ValueError, "embedded PEFT"):
            GraniteSpeechArchitectureConfig.from_dict(values)


class GraniteSpeechNativeRuntimeTests(unittest.TestCase):

    def test_raw_training_masks_prompt_and_backpropagates_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _tiny_runtime(Path(temporary))
            prepared = runtime.processor.prepare_training_batch(
                (
                    torch.linspace(-0.1, 0.1, 1_600),
                    torch.zeros(2_400),
                ),
                ("hello", "world"),
                sampling_rates=(16_000, 16_000),
            )
            labels = prepared["labels"]
            self.assertTrue(torch.all(labels[prepared["input_ids"] == DEFAULT_AUDIO_TOKEN_ID] == -100))
            self.assertTrue(torch.equal(
                labels[labels.ne(-100)],
                prepared["input_ids"][labels.ne(-100)],
            ))
            output = runtime.model(**prepared)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(runtime.model.encoder.input_linear.weight.grad, )
            self.assertIsNotNone(runtime.model.projector.linear.weight.grad, )
            self.assertIsNotNone(runtime.model.language_model.model.layers[0].self_attn.q_proj.weight.grad, )

    def test_wrapper_trims_file_audio_before_training_resample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _tiny_runtime(root / "runtime")
            wrapper = GraniteSpeechForSpeechRecognition(
                GraniteSpeechASRConfig(name_or_path=root / "runtime"),
                device="cpu",
            )
            wrapper.runtime = runtime
            wrapper.artifacts = runtime.artifacts
            wrapper.native_config = runtime.config
            wrapper.granite_processor = runtime.processor
            wrapper.model = runtime.model
            path = save_pcm_wave(
                root / "padded.wav",
                torch.cat((torch.zeros(800), torch.ones(800))),
                8_000,
            )
            common = {
                "sampling_rate": 8_000,
                "text": "hello",
            }

            file_batch = wrapper.prepare_training_inputs(
                {
                    **common,
                    "audio": str(path),
                    "audio_lengths": 800,
                },
                phase="speech_recognition",
            )
            tensor_batch = wrapper.prepare_training_inputs(
                {
                    **common,
                    "audio": torch.zeros(800),
                },
                phase="speech_recognition",
            )

            torch.testing.assert_close(
                file_batch["input_features"],
                tensor_batch["input_features"],
            )
            torch.testing.assert_close(
                file_batch["input_features_mask"],
                tensor_batch["input_features_mask"],
            )

    def test_processor_expands_audio_tokens_and_pads_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _tiny_runtime(Path(temporary))
            prepared = runtime.processor.prepare_inference_batch(
                (
                    torch.zeros(1_600),
                    torch.zeros(4_800),
                ),
                sampling_rates=(16_000, 16_000),
                prompts=("transcribe", "transcribe"),
            )
            counts = (prepared["input_ids"] == DEFAULT_AUDIO_TOKEN_ID).sum(dim=-1)
            torch.testing.assert_close(
                counts,
                prepared["input_features_mask"].sum(dim=-1),
            )
            self.assertEqual(
                tuple(prepared["attention_mask"].shape),
                tuple(prepared["input_ids"].shape),
            )
            self.assertFalse(bool(prepared["attention_mask"][0, 0]), )

    def test_safe_export_reloads_a_fresh_native_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _tiny_runtime(root / "source")
            output = save_granite_speech_runtime(
                runtime,
                root / "export",
            )
            reloaded = load_granite_speech_runtime(
                output,
                device="cpu",
                compute_dtype="float32",
                local_files_only=True,
            )
            self.assertEqual(
                set(runtime.model.state_dict()),
                set(reloaded.model.state_dict()),
            )
            self.assertFalse(reloaded.model.training)
            self.assertEqual(
                reloaded.processor.tokenizer.audio_token_id,
                DEFAULT_AUDIO_TOKEN_ID,
            )

    def test_safe_export_rejects_incomplete_or_malformed_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _tiny_runtime(root / "source")
            state = dict(runtime.model.state_dict())
            removed_name = next(iter(state))
            incomplete = dict(state)
            incomplete.pop(removed_name)
            with self.assertRaisesRegex(ValueError, "missing"):
                save_granite_speech_runtime(
                    runtime,
                    root / "incomplete",
                    state_dict=incomplete,
                )
            self.assertFalse((root / "incomplete").exists())

            malformed = dict(state)
            malformed[removed_name] = torch.zeros(1)
            with self.assertRaisesRegex(ValueError, "shape_mismatches"):
                save_granite_speech_runtime(
                    runtime,
                    root / "malformed",
                    state_dict=malformed,
                )
            self.assertFalse((root / "malformed").exists())

            tracker_name = next(name for name in state if name.endswith("num_batches_tracked"))
            invalid_dtype = dict(state)
            invalid_dtype[tracker_name] = state[tracker_name].float()
            with self.assertRaisesRegex(ValueError, "invalid_tensors"):
                save_granite_speech_runtime(
                    runtime,
                    root / "invalid-dtype",
                    state_dict=invalid_dtype,
                )
            self.assertFalse((root / "invalid-dtype").exists())

    def test_artifact_resolver_rejects_backslash_shard_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({
                    "weight_map": {
                        "encoder.weight": "..\\outside.safetensors",
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                resolve_granite_speech_artifacts(root)

    def test_lora_export_merges_into_a_fresh_adapter_free_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = _tiny_runtime(root / "source")
            wrapper = GraniteSpeechForSpeechRecognition(
                model_path=root / "source",
                device="cpu",
            )
            wrapper.runtime = runtime
            wrapper.artifacts = runtime.artifacts
            wrapper.native_config = runtime.config
            wrapper.granite_processor = runtime.processor
            wrapper.training_processor = runtime.processor
            wrapper.model = runtime.model

            injection = wrapper.enable_lora(
                rank=2,
                alpha=2.0,
                target_modules=("language_model.model.layers.0.self_attn.q_proj", ),
            )
            module = injection.modules["language_model.model.layers.0.self_attn.q_proj"]
            with torch.no_grad():
                module.lora_a.fill_(0.25)
                module.lora_b.fill_(0.5)
            original = module.base.weight.detach().clone()
            expected = original + module.adapter_delta()

            wrapper.export_native_pretrained(root / "export")

            self.assertFalse(module.merged)
            torch.testing.assert_close(
                module.base.weight,
                original,
                rtol=0.0,
                atol=0.0,
            )
            reloaded = load_granite_speech_runtime(
                root / "export",
                device="cpu",
                compute_dtype="float32",
                local_files_only=True,
            )
            reloaded_weight = (reloaded.model.language_model.model.layers[0].self_attn.q_proj.weight)
            torch.testing.assert_close(
                reloaded_weight,
                expected,
                rtol=0.0,
                atol=0.0,
            )
            self.assertFalse(
                any("lora" in name or ".base." in name for name in reloaded.model.state_dict()), )

    @unittest.skipUnless(
        TORCHAUDIO_AVAILABLE,
        "torchaudio is used only as an audit reference",
    )
    def test_frontend_is_numerically_equal_to_torchaudio(self):
        import torchaudio

        torch.manual_seed(9)
        audio = torch.randn(2, 16_000) * 0.1
        native = GraniteSpeechFeatureExtractor().extract(
            audio,
            sampling_rates=(16_000, 16_000),
        )["input_features"]
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
        )(audio)
        reference = mel.transpose(-1, -2).clamp_min_(1e-10).log10_()
        maximum = reference.amax(
            dim=(-2, -1),
            keepdim=True,
        )
        reference = torch.maximum(
            reference,
            maximum - 8.0,
        ).div_(4.0).add_(1.0)
        if reference.shape[1] % 2:
            reference = reference[:, :-1]
        reference = reference.reshape(2, -1, 160)
        float32_epsilon = torch.finfo(torch.float32).eps
        torch.testing.assert_close(
            native,
            reference,
            rtol=0.0,
            atol=float32_epsilon,
        )

    @unittest.skipUnless(
        TRANSFORMERS_AVAILABLE,
        "Transformers is used only as an audit reference",
    )
    def test_encoder_and_projector_match_the_reference_tiny_graph(self):
        from transformers import Blip2QFormerConfig
        from transformers import GraniteConfig as ReferenceGraniteConfig
        from transformers import GraniteSpeechConfig as ReferenceSpeechConfig
        from transformers import GraniteSpeechEncoderConfig as ReferenceEncoderConfig
        from transformers.models.granite_speech import modeling_granite_speech

        from voicehub.models.asr_granite_speech.native.modeling import (
            GraniteSpeechCTCEncoder,
            GraniteSpeechEncoderProjector,
        )

        ReferenceEncoder = modeling_granite_speech.GraniteSpeechCTCEncoder
        ReferenceProjector = (modeling_granite_speech.GraniteSpeechEncoderProjector)
        config = _tiny_config()
        reference_encoder_config = ReferenceEncoderConfig(
            **{
                key: value
                for key, value in config.encoder_config.to_dict().items()
                if key not in {"extra_config", "model_type"}
            }, )
        reference_projector_config = Blip2QFormerConfig(
            **{
                key: value
                for key, value in config.projector_config.to_dict().items()
                if key not in {"extra_config", "model_type"}
            }, )
        reference_text_config = ReferenceGraniteConfig(
            **{
                key: value
                for key, value in config.text_config.to_dict().items() if key not in {
                    'architectures',
                    "extra_config",
                    "model_type",
                }
            }, )
        reference_config = ReferenceSpeechConfig(
            text_config=reference_text_config,
            encoder_config=reference_encoder_config,
            projector_config=reference_projector_config,
            audio_token_index=config.audio_token_index,
            has_lora_adapter=False,
            downsample_rate=config.downsample_rate,
            window_size=config.window_size,
            tie_word_embeddings=False,
        )
        reference_encoder = ReferenceEncoder(reference_encoder_config, ).eval()
        native_encoder = GraniteSpeechCTCEncoder(config.encoder_config, ).eval()
        native_encoder.load_state_dict(
            reference_encoder.state_dict(),
            strict=True,
        )
        features = torch.randn(2, 7, 160)
        torch.testing.assert_close(
            native_encoder(features),
            reference_encoder(features).last_hidden_state,
            rtol=0.0,
            atol=0.0,
        )

        reference_projector = ReferenceProjector(reference_config, ).eval()
        native_projector = GraniteSpeechEncoderProjector(config, ).eval()
        native_projector.load_state_dict(
            reference_projector.state_dict(),
            strict=True,
        )
        hidden = torch.randn(2, 7, 8)
        torch.testing.assert_close(
            native_projector(hidden),
            reference_projector(hidden),
            rtol=1e-5,
            atol=1e-6,
        )


class _FakeTokenizer:

    def decode(self, token_ids, *, skip_special_tokens):
        self.call = (tuple(token_ids), skip_special_tokens)
        return "hello Granite"


class _FakeProcessor:
    sample_rate = 16_000

    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def prepare_inference_batch(self, *args, **kwargs):
        self.call = (args, kwargs)
        return {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "input_features": torch.zeros(1, 3, 160),
            "input_features_mask": torch.ones(
                1,
                3,
                dtype=torch.bool,
            ),
            "audio_lengths": torch.tensor([16_000]),
        }


class _FakeModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def generate(self, input_ids, **kwargs):
        self.call = (input_ids, kwargs)
        return SimpleNamespace(
            sequences=torch.tensor([[10, 11, 12, 90, 91]]),
            generated_lengths=torch.tensor([2]),
        )


def _granite_training_spec() -> ModelTrainingSpec:
    return ModelTrainingSpec(
        model_type="asr_granite_speech",
        family=TrainingFamily.SPEECH_SEQ2SEQ,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
    )


class _TinyGraniteTrainingGraph(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.encoder = torch.nn.Linear(4, 4)
        self.projector = torch.nn.Linear(4, 4)
        self.language_model = torch.nn.Linear(4, 4)

    def forward(self, input_ids, **kwargs):
        del input_ids, kwargs
        return SimpleNamespace(loss=(self.anchor.square() + self.projector.weight.square().mean()), )


class _TinyGraniteLoRAInjection:

    def __init__(self, model, config):
        self.config = config
        self.module_names = (
            "language_model.model.layers.0.self_attn.q_proj",
            "language_model.model.layers.0.self_attn.v_proj",
        )
        self._parameters = (
            model.language_model.lora_a,
            model.language_model.lora_b,
        )

    def parameters(self):
        return iter(self._parameters)


class _TinyGraniteTrainingWrapper:
    architecture_family = "speech-seq2seq"

    def __init__(self, prepared=None):
        self.config = SimpleNamespace(name_or_path="tiny-native-granite-speech", )
        self.model = _TinyGraniteTrainingGraph()
        self.runtime = SimpleNamespace(model=self.model)
        self._lora_injection = None
        self.load_count = 0
        self.enable_lora_count = 0
        self.enable_lora_target_modules = ()
        self.prepare_calls = []
        self.export_calls = []
        self.prepared = prepared or {
            "input_ids": torch.tensor([[10, 11, 12]]),
            "attention_mask": torch.ones(
                1,
                3,
                dtype=torch.bool,
            ),
            "labels": torch.tensor([[-100, 20, 21]]),
            "input_features": torch.zeros(1, 4, 160),
            "input_features_mask": torch.ones(
                1,
                4,
                dtype=torch.bool,
            ),
            "audio_lengths": torch.tensor([640]),
        }

    def load_for_training(self):
        self.load_count += 1
        self.model.train()

    def enable_lora(
            self,
            *,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            target_modules=(
                "*.q_proj",
                "*.k_proj",
                "*.v_proj",
                "*.o_proj",
            ),
            freeze_base=True,
            seed=0,
    ):
        self.enable_lora_count += 1
        self.enable_lora_target_modules = tuple(target_modules)
        if self._lora_injection is not None:
            raise RuntimeError("LoRA is already enabled.")
        if freeze_base:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self.model.language_model.register_parameter(
            "lora_a",
            torch.nn.Parameter(torch.zeros(rank, 4)),
        )
        self.model.language_model.register_parameter(
            "lora_b",
            torch.nn.Parameter(torch.zeros(4, rank)),
        )
        config = SimpleNamespace(
            alpha=alpha,
            dropout=dropout,
            freeze_base=freeze_base,
            rank=rank,
            seed=seed,
            target_modules=target_modules,
        )
        self._lora_injection = _TinyGraniteLoRAInjection(
            self.model,
            config,
        )
        return self._lora_injection

    def prepare_training_inputs(self, inputs, *, phase):
        self.prepare_calls.append((dict(inputs), phase))
        return dict(self.prepared)

    def export_native_pretrained(self, directory):
        destination = Path(directory)
        self.export_calls.append(destination)
        (destination / "native-export.marker").write_text(
            "native-granite-speech-v1",
            encoding="utf-8",
        )
        return destination


class GraniteSpeechTrainingAdapterTests(unittest.TestCase):

    def test_setup_targets_the_exact_native_wrapper_model(self):
        wrapper = _TinyGraniteTrainingWrapper()
        adapter = NativeGraniteSpeechTrainingAdapter(
            wrapper,
            _granite_training_spec(),
        ).setup()

        self.assertEqual(wrapper.load_count, 1)
        self.assertEqual(wrapper.enable_lora_count, 1)
        self.assertEqual(
            wrapper.enable_lora_target_modules,
            (
                "language_model.model.layers.*.self_attn.q_proj",
                "language_model.model.layers.*.self_attn.v_proj",
            ),
        )
        self.assertIs(adapter.primary_model, wrapper.model)
        self.assertEqual(adapter.primary_path, "model")
        self.assertIs(wrapper.runtime.model, wrapper.model)
        trainable = {name for name, parameter in wrapper.model.named_parameters() if parameter.requires_grad}
        self.assertEqual(
            trainable,
            {
                "language_model.lora_a",
                "language_model.lora_b",
                "projector.bias",
                "projector.weight",
            },
        )
        self.assertFalse(wrapper.model.encoder.weight.requires_grad)
        self.assertFalse(wrapper.model.encoder.bias.requires_grad)
        self.assertFalse(
            any(name.startswith(("encoder.", "projector."))
                for name in wrapper._lora_injection.module_names), )

        mismatched = _TinyGraniteTrainingWrapper()
        mismatched.runtime.model = _TinyGraniteTrainingGraph()
        with self.assertRaisesRegex(
                ValueError,
                "different model graphs",
        ):
            NativeGraniteSpeechTrainingAdapter(
                mismatched,
                _granite_training_spec(),
            ).setup()

    def test_full_model_fine_tuning_requires_an_explicit_opt_in(self):
        wrapper = _TinyGraniteTrainingWrapper()
        adapter = NativeGraniteSpeechTrainingAdapter(
            wrapper,
            _granite_training_spec(),
        )
        adapter.configure_trainable_scope("full-model").setup()

        self.assertEqual(wrapper.enable_lora_count, 0)
        self.assertTrue(all(parameter.requires_grad for parameter in wrapper.model.parameters()), )
        grouped_names = {name for _, parameters in adapter.named_parameter_groups() for name, _ in parameters}
        self.assertEqual(
            grouped_names,
            {name
             for name, _ in wrapper.model.named_parameters()},
        )
        self.assertIn("encoder.weight", grouped_names)
        self.assertIn("encoder.bias", grouped_names)
        self.assertEqual(
            adapter.artifact_manifest()["trainable_scope"],
            "full-model",
        )

    def test_input_preparation_filters_to_the_native_forward_contract(self):
        wrapper = _TinyGraniteTrainingWrapper()
        wrapper.prepared["debug_metadata"] = "not-a-forward-argument"
        adapter = NativeGraniteSpeechTrainingAdapter(
            wrapper,
            _granite_training_spec(),
        )
        context = adapter.create_training_context({
            "audio": torch.zeros(640),
            "text": "hello",
        })

        prepared = adapter.prepare_training_inputs(
            context.inputs,
            context,
        )

        self.assertEqual(
            set(prepared),
            {
                "attention_mask",
                "audio_lengths",
                "input_features",
                "input_features_mask",
                "input_ids",
                "labels",
                "use_cache",
            },
        )
        self.assertFalse(prepared["use_cache"])
        self.assertEqual(wrapper.prepare_calls[0][1], "default")

        wrapper.prepared["use_cache"] = True
        with self.assertRaisesRegex(ValueError, "use_cache=True"):
            adapter.prepare_training_inputs(context.inputs, context)

        wrapper.prepared.pop("use_cache")
        wrapper.prepared.pop("input_features_mask")
        with self.assertRaisesRegex(
                ValueError,
                "input_features_mask",
        ):
            adapter.prepare_training_inputs(context.inputs, context)

    def test_export_manifest_describes_the_portable_native_artifact(self):
        wrapper = _TinyGraniteTrainingWrapper()
        adapter = NativeGraniteSpeechTrainingAdapter(
            wrapper,
            _granite_training_spec(),
            lora_options={
                "alpha": 8.0,
                "dropout": 0.05,
                "rank": 4,
                "seed": 17,
            },
        ).setup()

        manifest = adapter.artifact_manifest()
        self.assertEqual(
            manifest["checkpoint_format"],
            "native-granite-speech-v1",
        )
        self.assertEqual(
            manifest["native_model_path"],
            "model",
        )
        self.assertEqual(
            manifest["native_objective"],
            ("audio-conditioned-completion-only-"
             "causal-language-modeling"),
        )
        self.assertEqual(
            manifest["label_policy"],
            "transcript-completion-only",
        )
        self.assertEqual(manifest["lora"]["rank"], 4)
        self.assertEqual(
            manifest["lora"]["module_names"],
            [
                "language_model.model.layers.0.self_attn.q_proj",
                "language_model.model.layers.0.self_attn.v_proj",
            ],
        )
        self.assertEqual(
            manifest["trainable_scope"],
            "projector-and-native-lora",
        )
        self.assertEqual(
            manifest["source_recommended_training"],
            {
                "bf16": True,
                "learning_rate": 3e-5,
                "warmup_ratio": 0.2,
            },
        )
        self.assertIn(
            manifest["source_notebook"]["revision"],
            manifest["source_notebook"]["url"],
        )
        self.assertEqual(
            manifest["trainable_parameter_count"],
            52,
        )
        self.assertEqual(
            manifest["trainable_tensor_count"],
            4,
        )
        self.assertEqual(
            manifest["checkpoint_semantics"]["save_pretrained"],
            adapter.native_export_semantics,
        )
        self.assertEqual(
            adapter.recipe_resume_configuration()["model_path"],
            "model",
        )

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "export"
            adapter.save_pretrained(destination)
            self.assertEqual(wrapper.export_calls, [destination])
            self.assertEqual(
                (destination / "native-export.marker").read_text(encoding="utf-8", ),
                "native-granite-speech-v1",
            )


class GraniteSpeechWrapperTests(unittest.TestCase):

    def test_inference_is_native_and_decodes_only_completion_tokens(self):
        model = GraniteSpeechForSpeechRecognition(device="cpu", )
        processor = _FakeProcessor()
        model.model = _FakeModel()
        model.granite_processor = processor
        model.runtime = SimpleNamespace(
            generation_config={
                "do_sample": False,
                "eos_token_id": DEFAULT_EOS_TOKEN_ID,
                "pad_token_id": DEFAULT_PAD_TOKEN_ID,
            }, )
        model.native_config = SimpleNamespace(
            text_config=SimpleNamespace(
                eos_token_id=DEFAULT_EOS_TOKEN_ID,
                pad_token_id=DEFAULT_PAD_TOKEN_ID,
                max_position_embeddings=128,
            ), )
        model.artifacts = SimpleNamespace(revision="pinned")
        output = model._transcribe(
            torch.zeros(16_000),
            sampling_rate=16_000,
            prompt="transcribe with punctuation",
            hotwords=("VoiceHub", "Granite"),
            max_new_tokens=8,
        )
        self.assertEqual(
            processor.tokenizer.call,
            ((90, 91), True),
        )
        self.assertEqual(output.text, "hello Granite")
        self.assertEqual(
            output.metadata["backend"],
            "voicehub-native",
        )
        self.assertEqual(
            processor.call[1]["hotwords"],
            ("VoiceHub", "Granite"),
        )

    def test_translation_builds_the_published_prompt_and_normalizes_language(self):
        model = GraniteSpeechForSpeechRecognition(device="cpu", )
        processor = _FakeProcessor()
        model.model = _FakeModel()
        model.granite_processor = processor
        model.runtime = SimpleNamespace(
            generation_config={
                "do_sample": False,
                "eos_token_id": DEFAULT_EOS_TOKEN_ID,
                "pad_token_id": DEFAULT_PAD_TOKEN_ID,
            }, )
        model.native_config = SimpleNamespace(
            text_config=SimpleNamespace(
                eos_token_id=DEFAULT_EOS_TOKEN_ID,
                pad_token_id=DEFAULT_PAD_TOKEN_ID,
                max_position_embeddings=128,
            ), )
        model.artifacts = SimpleNamespace(revision="pinned")

        output = model._transcribe(
            torch.zeros(16_000),
            sampling_rate=16_000,
            task="translate",
            language="fra",
            max_new_tokens=8,
        )

        self.assertEqual(output.language, "fr")
        self.assertEqual(output.metadata["task"], "translate")
        self.assertEqual(
            processor.call[1]["prompts"],
            ("<|audio|>translate the speech to French with proper "
             "punctuation and capitalization."),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            model._validate_request(
                language="tr",
                task="translate",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=None,
            )

    def test_inference_rejects_unimplemented_language_and_timestamp_modes(self):
        model = GraniteSpeechForSpeechRecognition(device="cpu", )
        with self.assertRaisesRegex(ValueError, "language-ID"):
            model._validate_request(
                language="fr",
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=None,
            )
        with self.assertRaisesRegex(ValueError, "timestamps"):
            model._validate_request(
                language=None,
                task="transcribe",
                return_timestamps=True,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=None,
            )


if __name__ == "__main__":
    unittest.main()
