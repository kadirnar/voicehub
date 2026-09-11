from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.checkpointing import CheckpointCompatibilityError, ShardedSafeTensorReader, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.asr_vibevoice import VibeVoiceASRConfig as ProviderConfig
from voicehub.models.asr_vibevoice import VibeVoiceForSpeechRecognition
from voicehub.models.asr_vibevoice.training_asr_vibevoice import NativeVibeVoiceASRTrainingAdapter
from voicehub.models.causal_lm.native import Qwen2Config
from voicehub.models.vibevoice import VibeVoiceForTextToSpeech
from voicehub.models.vibevoice.native.checkpoint import VibeVoiceCheckpointAdapter, build_vibevoice_model
from voicehub.models.vibevoice.native.configuration import (
    VibeVoiceASRConfig,
    VibeVoiceASRTokenizerConfig,
    VibeVoiceDiffusionConfig,
    VibeVoiceLegacyTokenizerConfig,
    VibeVoiceTTSConfig,
)
from voicehub.models.vibevoice.native.diffusion import VibeVoiceDiffusionHead, VibeVoiceDPMSolver
from voicehub.models.vibevoice.native.metadata import (
    VIBEVOICE_ASR_REPOSITORY,
    VIBEVOICE_CHECKPOINTS,
    VIBEVOICE_REALTIME_REPOSITORY,
    VIBEVOICE_TTS_REPOSITORY,
)
from voicehub.models.vibevoice.native.modeling import (
    VibeVoiceASRForConditionalGeneration,
    VibeVoiceForConditionalGeneration,
    VibeVoiceRealtimeForConditionalGeneration,
)
from voicehub.models.vibevoice.native.registration import (
    create_vibevoice_asr_architecture_spec,
    create_vibevoice_tts_architecture_spec,
)
from voicehub.models.vibevoice.native.runtime import load_vibevoice_runtime, save_vibevoice_runtime
from voicehub.models.vibevoice.native.tokenization import VIBEVOICE_TOKEN_IDS
from voicehub.optimization.diffusion_sampling import DiffusionSamplingConfig, DiffusionSamplingMixin
from voicehub.tokenization.assets import encode_gpt2_token
from voicehub.training import AutoTrainingAdapter, get_training_spec


def _text_config(*, layers: int = 1) -> Qwen2Config:
    return Qwen2Config(
        vocab_size=151_936,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1_024,
        max_window_layers=layers,
        bos_token_id=None,
        eos_token_id=VIBEVOICE_TOKEN_IDS["<|endoftext|>"],
        pad_token_id=VIBEVOICE_TOKEN_IDS["<|endoftext|>"],
        tie_word_embeddings=False,
    )


def _asr_config() -> VibeVoiceASRConfig:
    acoustic = VibeVoiceASRTokenizerConfig(
        hidden_size=4,
        num_filters=2,
        depths=(1, 1, 1, 1, 1, 1, 1),
    )
    semantic = VibeVoiceASRTokenizerConfig(
        hidden_size=6,
        num_filters=2,
        depths=(1, 1, 1, 1, 1, 1, 1),
    )
    return VibeVoiceASRConfig(
        acoustic_tokenizer_encoder_config=acoustic,
        semantic_tokenizer_encoder_config=semantic,
        text_config=_text_config(),
        acoustic_tokenizer_chunk_size=3_200,
    )


class VibeVoiceASRTrainingAdapterTests(unittest.TestCase):

    def test_structured_segments_are_raw_evaluation_targets(self):
        adapter = NativeVibeVoiceASRTrainingAdapter(
            object(),
            get_training_spec("asr_vibevoice"),
        )
        segments = [[{
            "start": 0.0,
            "end": 0.5,
            "speaker": 0,
            "text": "hello",
        }]]

        self.assertEqual(
            adapter.evaluation_label_values(
                {
                    "audio": ["clip.wav"],
                    "segments": segments
                },
                get_training_spec("asr_vibevoice").get_phase(),
            ),
            (segments, ),
        )


def _codec_config(*, latent_size: int) -> VibeVoiceLegacyTokenizerConfig:
    return VibeVoiceLegacyTokenizerConfig(
        vae_dim=latent_size,
        encoder_n_filters=2,
        decoder_n_filters=2,
        encoder_depths=(1, 1, 1, 1, 1, 1, 1),
    )


def _diffusion_config() -> VibeVoiceDiffusionConfig:
    return VibeVoiceDiffusionConfig(
        hidden_size=8,
        head_layers=1,
        head_ffn_ratio=2.0,
        latent_size=2,
        ddpm_num_steps=10,
        ddpm_num_inference_steps=2,
        ddpm_batch_mul=1,
    )


def _tts_config() -> VibeVoiceTTSConfig:
    return VibeVoiceTTSConfig(
        acoustic_tokenizer_config=_codec_config(latent_size=2),
        semantic_tokenizer_config=_codec_config(latent_size=3),
        decoder_config=_text_config(),
        diffusion_head_config=_diffusion_config(),
        acoustic_vae_dim=2,
        semantic_vae_dim=3,
    )


def _realtime_config() -> VibeVoiceTTSConfig:
    return VibeVoiceTTSConfig(
        acoustic_tokenizer_config=_codec_config(latent_size=2),
        semantic_tokenizer_config=None,
        decoder_config=_text_config(layers=3),
        diffusion_head_config=_diffusion_config(),
        acoustic_vae_dim=2,
        semantic_vae_dim=None,
        tts_backbone_num_hidden_layers=2,
        model_type="vibevoice_streaming",
    )


def _write_tokenizer(root: Path) -> None:
    vocabulary = {encode_gpt2_token(bytes((value, ))): value for value in range(256)}
    added_tokens = [{
        "id": token_id,
        "content": spelling,
        "special": True,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "single_word": False,
    } for spelling, token_id in VIBEVOICE_TOKEN_IDS.items()]
    write_json_file(
        root / "tokenizer.json",
        {
            "version": "1.0",
            "added_tokens": added_tokens,
            "normalizer": None,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            },
            "model": {
                "type": "BPE",
                "vocab": vocabulary,
                "merges": [],
                "unk_token": None,
            },
        },
    )
    write_json_file(
        root / "tokenizer_config.json",
        {
            "errors": "replace",
            "pad_token": "<|endoftext|>",
        },
    )


def _write_asr_artifact(root: Path) -> VibeVoiceASRForConditionalGeneration:
    root.mkdir(parents=True)
    config = _asr_config()
    model = VibeVoiceASRForConditionalGeneration(config)
    write_json_file(root / "config.json", config.to_dict())
    _write_tokenizer(root)
    write_json_file(
        root / "processor_config.json",
        {
            "processor_class": "VibeVoiceAsrProcessor",
            "feature_extractor": {
                "sampling_rate": 24_000,
                "normalize_audio": True,
                "target_dB_FS": -25,
                "eps": 1e-6,
            },
        },
    )
    write_json_file(
        root / "generation_config.json",
        {
            "do_sample": False,
            "eos_token_id": VIBEVOICE_TOKEN_IDS["<|endoftext|>"],
            "pad_token_id": VIBEVOICE_TOKEN_IDS["<|image_pad|>"],
            "use_cache": True,
            "max_new_tokens": 32_768,
        },
    )
    (root / "chat_template.jinja").write_text(
        "audited by the native prompt renderer",
        encoding="utf-8",
    )
    save_safetensors(
        model.state_dict(),
        root / "model.safetensors",
    )
    return model


class NativeVibeVoiceTests(unittest.TestCase):

    def test_provider_discovery_does_not_import_torch(self):
        command = (
            "import sys; "
            "import voicehub.models.vibevoice.native as architectures; "
            "import voicehub.models.asr_vibevoice as asr; "
            "import voicehub.models.vibevoice as tts; "
            "assert 'torch' not in sys.modules; "
            "assert 'VibeVoiceRuntime' in architectures.__all__; "
            "assert 'VibeVoiceForSpeechRecognition' in asr.__all__; "
            "assert 'VibeVoiceForTextToSpeech' in tts.__all__")
        subprocess.run(
            [sys.executable, "-c", command],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_source_and_checkpoint_inventories_are_pinned(self):
        source = (Path(__file__).resolve().parents[1] / 'voicehub/models/vibevoice/native/source/SOURCE.json')
        metadata = json.loads(source.read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["implementation_sources"][0]["revision"],
            "94da20d98b2fa7688e9cbfaf7692ddb4954f7600",
        )
        expected = {
            VIBEVOICE_ASR_REPOSITORY: (901, 8_330_325_888, 16_660_651_776),
            VIBEVOICE_TTS_REPOSITORY: (1_204, 2_704_021_987, 5_408_043_974),
            VIBEVOICE_REALTIME_REPOSITORY: (605, 1_017_626_724, 2_035_253_448),
        }
        for repository, facts in expected.items():
            checkpoint = VIBEVOICE_CHECKPOINTS[repository]
            self.assertEqual(
                (
                    checkpoint["tensors"],
                    checkpoint["parameters"],
                    checkpoint["tensor_bytes"],
                ),
                facts,
            )
            self.assertEqual(len(checkpoint["header_fingerprint"]), 64)

    def test_native_modules_do_not_import_external_model_runtimes(self):
        root = Path(__file__).resolve().parents[1]
        files = [
            *(root / 'voicehub/models/vibevoice/native').glob("*.py"),
            *(root / "voicehub" / "models" / "asr_vibevoice").glob("*.py"),
            root / 'voicehub/models/vibevoice/modeling.py',
            root / "voicehub" / "models" / "vibevoice" / "training.py",
        ]
        forbidden = {
            "diffusers",
            "huggingface_hub",
            "librosa",
            "numpy",
            "safetensors",
            "soundfile",
            "tokenizers",
            "transformers",
        }
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".", 1)[0])
            self.assertFalse(
                imported & forbidden,
                f"{path.name} imports {sorted(imported & forbidden)!r}",
            )

    def test_asr_raw_training_and_portable_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_asr_artifact(root / "source")
            runtime = load_vibevoice_runtime(
                root / "source",
                device="cpu",
                compute_dtype="float32",
                for_training=True,
            )
            wrapper = VibeVoiceForSpeechRecognition(
                ProviderConfig(name_or_path=str(root / "source")),
                device="cpu",
            )
            wrapper.runtime = runtime
            wrapper.artifacts = runtime.artifacts
            wrapper.native_config = runtime.config
            wrapper.vibevoice_processor = runtime.processor
            wrapper.training_processor = runtime.processor
            wrapper.transformers_processor = runtime.processor
            wrapper.model = runtime.model
            runtime.model.gradient_checkpointing_enable()
            adapter = AutoTrainingAdapter.from_model(wrapper)

            prepared = wrapper.prepare_training_inputs(
                {
                    "audio": torch.zeros(3_200),
                    "sampling_rate": 24_000,
                    "segments": [{
                        "start": 0.0,
                        "end": 0.13,
                        "speaker": 0,
                        "text": "hello",
                    }],
                },
                phase="main",
            )
            active = prepared["labels"][0].ne(-100)
            completion = runtime.processor.tokenizer.decode(
                prepared["labels"][0, active],
                skip_special_tokens=False,
            )
            self.assertTrue(completion.startswith("<|im_start|>assistant\n"))
            self.assertIn('"Content":"hello"', completion)
            self.assertTrue(completion.endswith("<|im_end|>\n"))
            self.assertIsInstance(
                adapter,
                NativeVibeVoiceASRTrainingAdapter,
            )
            self.assertTrue(
                torch.all(
                    prepared["labels"][prepared["input_ids"].eq(runtime.config.audio_token_id)].eq(-100)))
            batched = wrapper.prepare_training_inputs(
                {
                    "audio":
                    torch.zeros(2, 3_200),
                    "audio_lengths":
                    torch.tensor([3_200, 1_600]),
                    "sampling_rate":
                    torch.tensor([24_000, 24_000]),
                    "segments": [
                        [{
                            "start": 0.0,
                            "end": 0.1,
                            "speaker": 0,
                            "text": "first",
                        }],
                        [{
                            "start": 0.0,
                            "end": 0.05,
                            "speaker": 1,
                            "text": "second",
                        }],
                    ],
                },
                phase="main",
            )
            self.assertEqual(
                batched["padding_mask"].sum(dim=1).tolist(),
                [3_200, 1_600],
            )

            output = runtime.model(
                **prepared,
                use_cache=False,
                generator=torch.Generator().manual_seed(7),
            )
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(runtime.model.model.multi_modal_projector.acoustic_linear_1.weight.grad)
            self.assertIsNotNone(runtime.model.model.language_model.layers[0].self_attn.q_proj.weight.grad)
            for encoder in (
                    runtime.model.model.acoustic_tokenizer_encoder,
                    runtime.model.model.semantic_tokenizer_encoder,
            ):
                self.assertFalse(encoder.training)
                self.assertFalse(any(parameter.requires_grad for parameter in encoder.parameters()))

            destination = save_vibevoice_runtime(
                runtime,
                root / "export",
            )
            restored = load_vibevoice_runtime(
                destination,
                device="cpu",
                compute_dtype="float32",
                for_training=True,
            )
            self.assertEqual(
                set(restored.model.state_dict()),
                set(runtime.model.state_dict()),
            )
            torch.testing.assert_close(
                restored.model.lm_head.weight,
                runtime.model.lm_head.weight,
            )

    def test_architecture_specs_report_only_verified_public_contracts(self):
        asr = create_vibevoice_asr_architecture_spec()
        tts = create_vibevoice_tts_architecture_spec()

        self.assertTrue(asr.capabilities.training)
        self.assertFalse(asr.capabilities.streaming)
        self.assertEqual(
            asr.metadata["reference_checkpoint_revision"],
            VIBEVOICE_CHECKPOINTS[VIBEVOICE_ASR_REPOSITORY]["revision"],
        )
        self.assertTrue(tts.capabilities.training)
        self.assertFalse(tts.capabilities.streaming)
        self.assertIn(
            "diffusion-sampling",
            tts.capabilities.optimization_passes,
        )
        self.assertIn(
            "diffusion-sampling-prediction-cache",
            tts.capabilities.features,
        )
        self.assertIn(
            "high-level-realtime-generation-fails-closed",
            tts.capabilities.features,
        )

    def test_tts_objectives_train_only_author_trainable_components(self):
        config = _tts_config()
        model = VibeVoiceForConditionalGeneration(config)
        model.train()
        model.gradient_checkpointing_enable()
        for codec in (
                model.model.acoustic_tokenizer,
                model.model.semantic_tokenizer,
        ):
            codec.eval()
            for parameter in codec.parameters():
                parameter.requires_grad_(False)

        input_ids = torch.randint(0, 256, (1, 6))
        output = model(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            speech_tensors=torch.randn(1, 3_200),
            speech_masks=torch.tensor([[True]]),
            speeches_loss_input=torch.tensor([[True]]),
            speech_semantic_tensors=torch.randn(1, 1, 3),
            acoustic_input_mask=torch.tensor([[False, True, False, False, False, False]]),
            acoustic_loss_mask=torch.tensor([[False, False, False, True, False, False]]),
            labels=input_ids,
            generator=torch.Generator().manual_seed(11),
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(torch.isfinite(output.diffusion_loss))
        output.loss.backward()
        self.assertIsNotNone(model.model.language_model.layers[0].self_attn.q_proj.weight.grad)
        self.assertIsNotNone(model.model.prediction_head.final_layer.linear.weight.grad)
        for codec in (
                model.model.acoustic_tokenizer,
                model.model.semantic_tokenizer,
        ):
            self.assertFalse(any(parameter.grad is not None for parameter in codec.parameters()))

    def test_native_dpm_solver_matches_audited_upstream_regression(self):
        solver = VibeVoiceDPMSolver(_diffusion_config())
        solver.set_timesteps(2)
        self.assertEqual(solver.timesteps.tolist(), [9, 4])
        torch.testing.assert_close(
            solver.sigmas,
            torch.tensor([203.7340545654297, 1.0123896598815918, 0.0]),
        )
        sample = torch.tensor([
            [0.8032760620117188, 0.17483338713645935],
            [0.08897809684276581, -0.6137180328369141],
        ])
        predictions = (
            torch.tensor([
                [-1.605276346206665, 0.23248571157455444],
                [2.239870071411133, 0.8472937941551208],
            ]),
            torch.tensor([
                [1.2006442546844482, -0.4015503227710724],
                [-1.4260196685791016, 0.903931736946106],
            ]),
        )
        for timestep, prediction in zip(solver.timesteps, predictions):
            sample = solver.step(
                prediction,
                timestep,
                sample,
            ).prev_sample
        torch.testing.assert_close(
            sample,
            torch.tensor([
                [0.3381619453430176, 0.25927478075027466],
                [-0.0413975715637207, -1.367765188217163],
            ]),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_realtime_diffusion_sampling_rebuilds_dpm_history_and_narrows_cfg(self, ):
        model = VibeVoiceRealtimeForConditionalGeneration(_realtime_config()).eval()
        head = model.model.prediction_head
        self.assertIsInstance(head, DiffusionSamplingMixin)
        head.enable_diffusion_sampling(
            DiffusionSamplingConfig(
                target_steps=2,
                guidance="limited_interval",
                guidance_start=0.0,
                guidance_end=0.0,
            ))
        batch_sizes: list[int] = []
        hook = head.register_forward_hook(
            lambda _module, arguments, _output: batch_sizes.append(arguments[0].shape[0]))
        condition = torch.randn(1, 8)
        negative_condition = torch.randn(1, 8)
        try:
            first = model.sample_speech_latents(
                condition,
                negative_condition,
                inference_steps=4,
                generator=torch.Generator().manual_seed(17),
            )
            self.assertEqual(batch_sizes, [2, 1])
            self.assertEqual(
                model.model.noise_scheduler.timesteps.tolist(),
                [9, 4],
            )
            self.assertEqual(model.model.noise_scheduler._step_index, 2)
            stats = head.diffusion_sampling_stats()
            self.assertEqual(stats["native_steps"], 4)
            self.assertEqual(stats["prepared_steps"], 2)
            self.assertEqual(stats["model_calls"], 2)
            self.assertEqual(stats["guidance_calls"], 1)
            self.assertEqual(stats["guidance_skips"], 1)

            batch_sizes.clear()
            second = model.sample_speech_latents(
                condition,
                negative_condition,
                inference_steps=4,
                generator=torch.Generator().manual_seed(17),
            )
        finally:
            hook.remove()
        torch.testing.assert_close(first, second)
        self.assertEqual(batch_sizes, [2, 1])
        self.assertEqual(model.model.noise_scheduler._step_index, 2)

    def test_realtime_prediction_cache_preserves_every_dpm_step(self):
        model = VibeVoiceRealtimeForConditionalGeneration(_realtime_config()).eval()
        head = model.model.prediction_head
        head.enable_diffusion_sampling(
            DiffusionSamplingConfig(
                prediction_cache="fora",
                cache_interval=2,
                cache_warmup_steps=0,
            ))
        batch_sizes: list[int] = []
        hook = head.register_forward_hook(
            lambda _module, arguments, _output: batch_sizes.append(arguments[0].shape[0]))
        try:
            sampled = model.sample_speech_latents(
                torch.randn(1, 8),
                torch.randn(1, 8),
                guidance_scale=1.0,
                inference_steps=4,
                generator=torch.Generator().manual_seed(23),
            )
        finally:
            hook.remove()
        self.assertEqual(sampled.shape, (1, 2))
        self.assertEqual(batch_sizes, [1, 1])
        self.assertEqual(model.model.noise_scheduler._step_index, 4)
        stats = head.diffusion_sampling_stats()
        self.assertEqual(stats["model_calls"], 2)
        self.assertEqual(stats["predicted_calls"], 2)

    def test_realtime_nonuniform_schedule_rebuilds_dpm_sigma_grid(self):
        model = VibeVoiceRealtimeForConditionalGeneration(_realtime_config()).eval()
        model.model.prediction_head.enable_diffusion_sampling(
            DiffusionSamplingConfig(
                target_steps=2,
                schedule="quadratic",
            ))
        model.sample_speech_latents(
            torch.randn(1, 8),
            torch.randn(1, 8),
            guidance_scale=1.0,
            inference_steps=4,
            generator=torch.Generator().manual_seed(29),
        )
        solver = model.model.noise_scheduler
        self.assertEqual(solver.timesteps.tolist(), [9, 6])
        torch.testing.assert_close(
            solver.sigmas[:-1],
            solver.training_sigmas[torch.tensor([9, 6])],
        )
        self.assertEqual(solver._step_index, 2)

    def test_vibevoice_rejects_direct_velocity_stork_solver(self):
        head = VibeVoiceDiffusionHead(_diffusion_config())
        with self.assertRaisesRegex(
                ValueError,
                "stork2",
        ):
            head.enable_diffusion_sampling(DiffusionSamplingConfig(solver="stork2"))

    def test_realtime_graph_exposes_stages_but_rejects_training_forward(self):
        model = VibeVoiceRealtimeForConditionalGeneration(_realtime_config())
        with self.assertRaisesRegex(RuntimeError, "staged inference"):
            model()
        lower = model.forward_lm(
            torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
        )
        self.assertEqual(lower.last_hidden_state.shape, (1, 2, 8))
        with self.assertRaisesRegex(RuntimeError, "unified forward"):
            model.model()

    def test_tts_wrapper_fails_before_loading_unverified_high_level_loop(self):
        wrapper = VibeVoiceForTextToSpeech(
            lazy_load=True,
            device="cpu",
        )
        with self.assertRaisesRegex(
                RuntimeError,
                "waveform parity",
        ):
            wrapper("Speaker 0: hello")
        self.assertIsNone(wrapper.model)

    def test_sharded_index_is_reconciled_before_meta_assignment(self):
        config = _asr_config()
        source = VibeVoiceASRForConditionalGeneration(config)
        state = source.state_dict()
        names = sorted(state)
        midpoint = len(names) // 2
        groups = (names[:midpoint], names[midpoint:])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight_map: dict[str, str] = {}
            for index, group in enumerate(groups, start=1):
                filename = f"model-{index:05d}-of-00002.safetensors"
                save_safetensors(
                    {name: state[name]
                     for name in group},
                    root / filename,
                )
                weight_map.update({name: filename for name in group})
            index_path = root / "model.safetensors.index.json"
            write_json_file(
                index_path,
                {
                    "metadata": {},
                    "weight_map": weight_map,
                },
            )
            with torch.device("meta"):
                restored = build_vibevoice_model(
                    config,
                    initialize=False,
                )
            with ShardedSafeTensorReader(index_path) as reader:
                VibeVoiceCheckpointAdapter().load_assign_streaming(
                    restored,
                    reader,
                    config,
                    device="cpu",
                    dtype=torch.float32,
                )
            self.assertFalse(any(value.device.type == "meta" for value in restored.state_dict().values()))

            first = names[0]
            original_shard = weight_map[first]
            other_shard = next(
                filename for filename in set(weight_map.values()) if filename != original_shard)
            weight_map[first] = other_shard
            write_json_file(
                index_path,
                {
                    "metadata": {},
                    "weight_map": weight_map,
                },
            )
            with torch.device("meta"):
                rejected = build_vibevoice_model(
                    config,
                    initialize=False,
                )
            with (
                    ShardedSafeTensorReader(index_path) as reader,
                    self.assertRaises(CheckpointCompatibilityError),
            ):
                VibeVoiceCheckpointAdapter().load_assign_streaming(
                    rejected,
                    reader,
                    config,
                    device="cpu",
                    dtype=torch.float32,
                )
            self.assertTrue(all(value.device.type == "meta" for value in rejected.state_dict().values()))


if __name__ == "__main__":
    unittest.main()
