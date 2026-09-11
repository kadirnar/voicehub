from __future__ import annotations

import ast
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

from voicehub.checkpointing import save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.zonos2.modeling import Zonos2Config, Zonos2ForTextToSpeech
from voicehub.models.zonos2.native.artifacts import resolve_zonos2_artifacts
from voicehub.models.zonos2.native.checkpoint import export_zonos2_checkpoint, load_zonos2_checkpoint
from voicehub.models.zonos2.native.configuration import Zonos2ArchitectureConfig
from voicehub.models.zonos2.native.metadata import (
    ZONOS2_OFFICIAL_CHECKPOINT,
    ZONOS2_OFFICIAL_CHECKPOINT_REVISION,
    ZONOS2_PARAMETER_COUNT,
    ZONOS2_SAFE_CONVERSION,
    ZONOS2_SAFE_CONVERSION_FILENAME,
    ZONOS2_SAFE_CONVERSION_REVISION,
    ZONOS2_SOURCE_REVISION,
    ZONOS2_TENSOR_COUNT,
)
from voicehub.models.zonos2.native.modeling import Zonos2ForCausalLM
from voicehub.models.zonos2.native.objective import zonos2_causal_cross_entropy
from voicehub.models.zonos2.native.prompting import (
    build_zonos2_prompt,
    delay_audio_completion,
    prepare_zonos2_training_batch,
    shear,
    shear_up,
    text_to_byte_ids,
)
from voicehub.models.zonos2.native.registration import create_zonos2_architecture_spec
from voicehub.models.zonos2.native.runtime import (
    NativeZonos2Runtime,
    Zonos2Generation,
    normalize_zonos2_text,
    speaking_rate_bucket_from_speed,
)
from voicehub.models.zonos2.native.sampling import Zonos2SamplingOptions, sample_zonos2_codes
from voicehub.models.zonos2.native.speaker import zonos2_speaker_mel
from voicehub.models.zonos2.training import Zonos2TrainingAdapter
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import get_training_spec


def tiny_config(*, speaker_enabled: bool = False):
    return Zonos2ArchitectureConfig(
        n_layers=4,
        dim=32,
        head_dim=8,
        n_kv_heads=2,
        ffn_dim_multiplier=1.0,
        multiple_of=8,
        max_seqlen=128,
        n_codebooks=3,
        codebook_size=16,
        eoa_id=16,
        audio_pad_id=17,
        text_vocab=448,
        speaker_enabled=speaker_enabled,
        speaker_embedding_dim=12,
        speaker_lda_dim=8 if speaker_enabled else None,
        speaker_background_token_enabled=False,
        accurate_mode_token_enabled=False,
        speaking_rate_num_buckets=0,
        speaking_rate_buckets=(),
        quality_num_buckets=0,
        quality_features=(),
        quality_buckets={},
        quality_dropout={},
        moe_n_experts=2,
        moe_router_topk=1,
        special_topk_layers={2: 2},
        moe_router_dim=8,
        moe_start_from_layer=1,
        moe_end_from_layer=1,
    )


class NativeZonos2ConfigTests(unittest.TestCase):

    def test_published_inventory_is_exact(self):
        config_path = (
            Path(__file__).parents[1] / "voicehub" / "models" / "zonos2" / "source" / "SOURCE.json")
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["revision"], ZONOS2_SOURCE_REVISION)
        self.assertFalse(metadata["reference_checkpoint"]["safetensors_published"])

        config = Zonos2ArchitectureConfig()
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(config.intermediate_size, 3_072)
        with torch.device("meta"):
            model = Zonos2ForCausalLM(config)
        state = model.state_dict()
        self.assertEqual(len(state), ZONOS2_TENSOR_COUNT)
        self.assertEqual(
            sum(value.numel() for value in state.values()),
            ZONOS2_PARAMETER_COUNT,
        )
        self.assertEqual(
            state["layers.26.feed_forward.experts.w13"].shape,
            (16, 6_144, 2_048),
        )

    def test_config_round_trip_preserves_unknown_values(self):
        values = tiny_config().to_dict()
        values["future_conditioner"] = {"version": 2}
        restored = Zonos2ArchitectureConfig.from_dict(values)
        self.assertEqual(
            restored.to_dict()["future_conditioner"],
            {"version": 2},
        )

    def test_architecture_registration_is_honest_about_training_boundary(self):
        spec = create_zonos2_architecture_spec()
        self.assertTrue(spec.capabilities.training)
        self.assertEqual(spec.capabilities.checkpoint_formats, ("safetensors", ))
        self.assertFalse(spec.metadata["author_verified_training_recipe"])
        self.assertTrue(spec.metadata["full_model_gradient_ready"])


class NativeZonos2ModelTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(13)
        self.config = tiny_config()
        self.model = Zonos2ForCausalLM(self.config)
        self.input_ids = torch.zeros(
            (2, 9, self.config.frame_width),
            dtype=torch.long,
        )
        self.input_ids[..., :-1] = torch.randint(
            0,
            self.config.codebook_size,
            self.input_ids[..., :-1].shape,
        )
        self.input_ids[..., -1] = 2

    def test_full_model_objective_backpropagates_through_dense_and_moe(self):
        labels = self.input_ids[..., :-1].clone()
        labels[:, :3] = -100
        output = self.model(self.input_ids, labels=labels)
        self.assertEqual(
            output.logits.shape,
            (2, 9, 3, self.config.audio_vocab_size),
        )
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(self.model.layers[0].attention.wq.weight.grad)
        self.assertIsNotNone(self.model.layers[1].feed_forward.experts.w13.grad)
        self.assertIsNotNone(self.model.layers[2].feed_forward.router.down_proj.weight.grad)
        self.assertIsNotNone(self.model.multi_output.weight.grad)

    def test_cached_chunks_match_full_causal_forward(self):
        self.model.eval()
        expected = self.model(self.input_ids[:1]).logits
        cache = self.model.create_kv_cache(
            batch_size=1,
            max_length=self.input_ids.shape[1],
        )
        first = self.model(
            self.input_ids[:1, :5],
            kv_cache=cache,
        ).logits
        second = self.model(
            self.input_ids[:1, 5:],
            kv_cache=cache,
        ).logits
        actual = torch.cat((first, second), dim=1)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)

    def test_speaker_projection_is_part_of_the_gradient_graph(self):
        config = tiny_config(speaker_enabled=True)
        model = Zonos2ForCausalLM(config)
        speaker = torch.randn(2, config.speaker_embedding_dim)
        output = model(
            self.input_ids,
            speaker_embedding=speaker,
            speaker_position=0,
        )
        output.logits.square().mean().backward()
        self.assertIsNotNone(model.speaker_lda_projection.weight.grad)
        self.assertIsNotNone(model.speaker_projection.weight.grad)

    def test_objective_matches_manual_masked_cross_entropy(self):
        logits = torch.randn(1, 5, 3, 18, requires_grad=True)
        labels = torch.tensor([[
            [1, 2, 3],
            [4, 17, 6],
            [7, 8, 9],
            [-100, 10, 11],
            [16, 12, 13],
        ]])
        output = zonos2_causal_cross_entropy(
            logits,
            labels,
            audio_pad_id=17,
        )
        shifted_labels = labels[:, 1:]
        valid = (shifted_labels != -100) & (shifted_labels != 17)
        manual_labels = shifted_labels.masked_fill(~valid, 0)
        manual = F.cross_entropy(
            logits[:, :-1].reshape(-1, 18),
            manual_labels.reshape(-1),
            reduction="none",
        ).view_as(manual_labels)
        manual = (manual * valid).sum() / valid.sum()
        torch.testing.assert_close(output.loss, manual)
        self.assertEqual(int(output.token_count), int(valid.sum()))

    def test_objective_handles_a_fully_masked_batch_without_host_sync(self):
        logits = torch.randn(1, 4, 3, 18, requires_grad=True)
        labels = torch.full((1, 4, 3), -100)
        output = zonos2_causal_cross_entropy(
            logits,
            labels,
            audio_pad_id=17,
        )
        self.assertEqual(float(output.loss.detach()), 0.0)
        self.assertEqual(int(output.token_count), 0)
        output.loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(int(torch.count_nonzero(logits.grad)), 0)


class NativeZonos2PromptTests(unittest.TestCase):

    def test_byte_tokenizer_is_utf8_and_source_compatible(self):
        self.assertEqual(text_to_byte_ids("A"), [2, 257, 3])
        self.assertEqual(
            text_to_byte_ids("é"),
            [2, 195 + 192, 169 + 192, 3],
        )

    def test_shear_inverse_and_training_tail(self):
        codes = torch.tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ])
        sheared = shear(codes, 17)
        torch.testing.assert_close(
            shear_up(sheared, 17)[:2],
            codes[:2],
        )
        completion = delay_audio_completion(
            codes,
            pad_id=17,
            eoa_id=16,
        )
        torch.testing.assert_close(
            shear_up(completion, 17)[:codes.shape[0]],
            codes,
        )
        for codebook in range(3):
            self.assertEqual(
                int(completion[codes.shape[0] + codebook, codebook]),
                16,
            )

    def test_training_batch_masks_prompt_and_padding(self):
        config = tiny_config()
        batch = prepare_zonos2_training_batch(
            config,
            ["short", "a longer line"],
            [
                torch.randint(0, 16, (5, 3)),
                torch.randint(0, 16, (8, 3)),
            ],
            prepend_silence=False,
        )
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["labels"].shape[-1], 3)
        self.assertTrue((batch["labels"][~batch["loss_mask"]] == -100).all())
        output = Zonos2ForCausalLM(config)(**batch)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertGreater(int(output.token_count), 0)

    def test_prompt_conditioning_and_speed_bucket(self):
        config = Zonos2ArchitectureConfig()
        prompt, position = build_zonos2_prompt(
            config,
            "Hello",
            include_speaker_slot=True,
            speaking_rate_bucket=4,
            quality_buckets=(None, None, None, None, None, 3),
            prepend_silence=False,
        )
        self.assertEqual(position, 0)
        self.assertEqual(prompt.shape[-1], 10)
        self.assertEqual(speaking_rate_bucket_from_speed(config, 1.0), 4)
        self.assertEqual(normalize_zonos2_text("  Ｈi  \n there "), "Hi there")


class NativeZonos2CheckpointTests(unittest.TestCase):

    def test_strict_safetensors_round_trip(self):
        config = tiny_config()
        source = Zonos2ForCausalLM(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            export_zonos2_checkpoint(source, checkpoint)
            with torch.device("meta"):
                restored = Zonos2ForCausalLM(config)
            report = load_zonos2_checkpoint(
                restored,
                checkpoint,
                device="cpu",
            )
            self.assertEqual(report.tensor_count, len(source.state_dict()))
            for name, value in source.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value)

    def test_loader_rejects_incomplete_namespace_before_assignment(self):
        config = tiny_config()
        source = Zonos2ForCausalLM(config)
        state = dict(source.state_dict())
        state.pop(next(iter(state)))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_safetensors(state, checkpoint)
            target = Zonos2ForCausalLM(config)
            first = next(target.parameters()).detach().clone()
            with self.assertRaises(CheckpointCompatibilityError):
                load_zonos2_checkpoint(target, checkpoint, device="cpu")
            torch.testing.assert_close(next(target.parameters()), first)

    def test_loader_rejects_integer_model_weights_before_assignment(self):
        config = tiny_config()
        source = Zonos2ForCausalLM(config)
        state = dict(source.state_dict())
        first_name = next(iter(state))
        state[first_name] = state[first_name].to(dtype=torch.int16)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_safetensors(state, checkpoint)
            target = Zonos2ForCausalLM(config)
            first = next(target.parameters()).detach().clone()
            with self.assertRaises(CheckpointCompatibilityError):
                load_zonos2_checkpoint(target, checkpoint, device="cpu")
            torch.testing.assert_close(next(target.parameters()), first)

    def test_local_artifacts_never_accept_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(tiny_config().to_dict()),
                encoding="utf-8",
            )
            (root / "model.pth").write_bytes(b"not a safe checkpoint")
            with self.assertRaises(FileNotFoundError):
                resolve_zonos2_artifacts(root)

    def test_official_alias_uses_pinned_safe_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "params.json"
            checkpoint = root / "zonos2-bf16.safetensors"
            config.touch()
            checkpoint.touch()
            calls = []

            def resolve(source, filename, **kwargs):
                calls.append((source, filename, kwargs["revision"]))
                return config if filename == "params.json" else checkpoint

            with patch(
                    "voicehub.models.zonos2.native.artifacts."
                    "resolve_pretrained_file",
                    side_effect=resolve,
            ):
                artifacts = resolve_zonos2_artifacts(ZONOS2_OFFICIAL_CHECKPOINT)
            self.assertTrue(artifacts.safe_conversion)
            self.assertIn(
                (
                    ZONOS2_SAFE_CONVERSION,
                    ZONOS2_SAFE_CONVERSION_FILENAME,
                    ZONOS2_SAFE_CONVERSION_REVISION,
                ),
                calls,
            )
            self.assertIn(
                (
                    ZONOS2_OFFICIAL_CHECKPOINT,
                    "params.json",
                    ZONOS2_OFFICIAL_CHECKPOINT_REVISION,
                ),
                calls,
            )


class NativeZonos2TrainingAdapterTests(unittest.TestCase):

    @staticmethod
    def _adapter():
        config = tiny_config()
        native = Zonos2ForCausalLM(config)
        wrapper = SimpleNamespace(
            model=native,
            config=SimpleNamespace(to_dict=lambda: {}),
        )
        return (
            Zonos2TrainingAdapter(
                wrapper,
                get_training_spec("zonos2"),
            ),
            config,
            native,
        )

    def test_training_profile_accepts_raw_text_and_audio(self):
        spec = get_training_spec("zonos2")

        self.assertIs(spec.support, TrainingSupport.NATIVE)

    def test_integrated_adapter_executes_native_objective(self):
        adapter, config, native = self._adapter()
        batch = prepare_zonos2_training_batch(
            config,
            ["fine tune"],
            [torch.randint(0, config.codebook_size, (5, config.n_codebooks))],
            prepend_silence=False,
        )
        output = adapter(**batch)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(
            output.training_phase,
            "reconstructed_codec_language_model",
        )
        self.assertFalse(output.metadata["objective_author_verified"])
        output.loss.backward()
        self.assertIsNotNone(native.multi_output.weight.grad)

    def test_adapter_export_reloads_in_a_fresh_native_runtime(self):
        adapter, _, native = self._adapter()
        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            restored = NativeZonos2Runtime.from_pretrained(
                directory,
                device="cpu",
                dtype=torch.float32,
            )
        self.assertEqual(
            restored.config.to_dict(),
            native.config.to_dict(),
        )
        for name, value in native.state_dict().items():
            torch.testing.assert_close(
                restored.model.state_dict()[name],
                value,
            )


class NativeZonos2RuntimeTests(unittest.TestCase):

    def test_training_collator_records_variable_audio_lengths(self):
        adapter = Zonos2TrainingAdapter(
            SimpleNamespace(),
            get_training_spec("zonos2"),
        )
        batch = adapter.data_collator([
            {
                "text": "short",
                "audio_codes": torch.ones(3, 3, dtype=torch.long),
            },
            {
                "text": "long",
                "audio_codes": torch.ones(5, 3, dtype=torch.long),
            },
        ])
        self.assertEqual(batch["audio_code_lengths"].tolist(), [3, 5])
        self.assertEqual(batch["audio_codes"].shape, (2, 5, 3))
        self.assertTrue((batch["audio_codes"][0, 3:] == 1_025).all())

    def test_training_collator_records_nested_waveform_lengths(self):
        adapter = Zonos2TrainingAdapter(
            SimpleNamespace(),
            get_training_spec("zonos2"),
        )
        batch = adapter.data_collator([
            {
                "text": "short",
                "audio": {
                    "array": torch.zeros(4),
                    "sampling_rate": 16_000,
                },
            },
            {
                "text": "long",
                "audio": {
                    "array": torch.zeros(6),
                    "sampling_rate": 24_000,
                },
            },
        ])
        self.assertEqual(
            batch["audio"]["audio_lengths"].tolist(),
            [4, 6],
        )
        self.assertEqual(batch["audio"]["array"].shape, (2, 6))

    def test_public_training_preparation_trims_collated_code_padding(self):
        runtime = SimpleNamespace(prepare_training_batch=Mock(return_value={"prepared": True}), )
        wrapper = Zonos2ForTextToSpeech(device="cpu")
        wrapper._runtime = runtime
        result = wrapper.prepare_training_inputs(
            {
                "texts": ["short", "long"],
                "audio_codes": torch.zeros(2, 5, 3, dtype=torch.long),
                "audio_code_lengths": torch.tensor([3, 5]),
            },
            phase=None,
        )
        self.assertTrue(result["prepared"])
        codes = runtime.prepare_training_batch.call_args.args[1]
        self.assertEqual([item.shape for item in codes], [(3, 3), (5, 3)])

    def test_public_training_preparation_trims_raw_audio_and_keeps_rates(self):
        encoded = []

        def encode_audio(audio, *, sampling_rate):
            encoded.append((audio.clone(), sampling_rate))
            return torch.zeros(2, 3, dtype=torch.long)

        runtime = SimpleNamespace(
            encode_audio=Mock(side_effect=encode_audio),
            prepare_training_batch=Mock(return_value={"prepared": True}),
        )
        wrapper = Zonos2ForTextToSpeech(device="cpu")
        wrapper._runtime = runtime
        result = wrapper.prepare_training_inputs(
            {
                "texts": ["first", "second"],
                "audio_values": torch.zeros(2, 8),
                "audio_lengths": torch.tensor([5, 8]),
                "sampling_rate": torch.tensor([16_000, 24_000]),
            },
            phase=None,
        )
        self.assertTrue(result["prepared"])
        self.assertEqual(
            [(item.shape[-1], rate) for item, rate in encoded],
            [(5, 16_000), (8, 24_000)],
        )

    def test_public_training_preparation_consumes_nested_audio_lengths(self):
        encoded = []

        def encode_audio(audio, *, sampling_rate):
            encoded.append((audio, sampling_rate))
            return torch.zeros(2, 3, dtype=torch.long)

        runtime = SimpleNamespace(
            encode_audio=Mock(side_effect=encode_audio),
            prepare_training_batch=Mock(return_value={"prepared": True}),
        )
        wrapper = Zonos2ForTextToSpeech(device="cpu")
        wrapper._runtime = runtime
        wrapper.prepare_training_inputs(
            {
                "texts": ["first", "second"],
                "audio": {
                    "array": torch.zeros(2, 8),
                    "sampling_rate": torch.tensor([16_000, 24_000]),
                    "audio_lengths": torch.tensor([5, 8]),
                },
            },
            phase=None,
        )
        self.assertEqual(
            [(
                item["array"].shape[-1],
                int(item["sampling_rate"]),
                rate,
            ) for item, rate in encoded],
            [
                (5, 16_000, None),
                (8, 24_000, None),
            ],
        )

    def test_public_training_preparation_rejects_duplicate_length_sources(self):
        wrapper = Zonos2ForTextToSpeech(device="cpu")
        wrapper._runtime = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "either beside.*or inside"):
            wrapper.prepare_training_inputs(
                {
                    "texts": ["one"],
                    "audio": {
                        "array": torch.zeros(1, 8),
                        "audio_lengths": torch.tensor([5]),
                    },
                    "audio_lengths": torch.tensor([5]),
                },
                phase=None,
            )

    def test_sampling_is_request_local_and_repeatable(self):
        logits = torch.randn(1, 3, 18)
        options = Zonos2SamplingOptions(
            temperature=0.8,
            top_k=8,
            top_p=0.9,
            min_p=0.05,
        )
        first_generator = torch.Generator().manual_seed(37)
        second_generator = torch.Generator().manual_seed(37)
        first = sample_zonos2_codes(
            logits,
            generated=[],
            options=options,
            generator=first_generator,
        )
        second = sample_zonos2_codes(
            logits,
            generated=[],
            options=options,
            generator=second_generator,
        )
        torch.testing.assert_close(first, second)

    def test_speaker_frontend_is_torch_only(self):
        waveform = torch.randn(4_800)
        features = zonos2_speaker_mel(waveform)
        self.assertEqual(features.ndim, 3)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[-1], 128)
        self.assertTrue(torch.isfinite(features).all())

    def test_public_wrapper_scopes_seed_for_native_runtime(self):
        wrapper = Zonos2ForTextToSpeech(device="cpu")
        wrapper.device = "cpu"
        wrapper.artifacts = SimpleNamespace(safe_conversion=False)
        wrapper._runtime = SimpleNamespace(
            generate=lambda *args, **kwargs: Zonos2Generation(
                audio=torch.tensor([0.1, -0.1]),
                audio_codes=torch.zeros(10, 3, dtype=torch.long),
                sample_rate=44_100,
                eos_frame=1,
                speaker_embedding=None,
            ))
        with patch(
                "voicehub.models.zonos2.modeling.seeded_inference",
                return_value=nullcontext(41),
        ) as seeded:
            output = wrapper._generate("hello")
        seeded.assert_called_once_with(
            None,
            device="cpu",
            model_type="zonos2",
        )
        self.assertEqual(output.metadata["seed"], 41)
        self.assertEqual(output.sample_rate, 44_100)

    def test_native_files_do_not_import_external_model_libraries(self):
        root = (Path(__file__).parents[1] / 'voicehub/models/zonos2/native')
        forbidden = {
            "transformers",
            "safetensors",
            "huggingface_hub",
            "torchaudio",
            "librosa",
            "numpy",
            "flash_attn",
            "triton",
            "zonos2",
        }
        violations = []
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [item.name.split(".", 1)[0] for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.split(".", 1)[0]]
                else:
                    continue
                for name in names:
                    if name in forbidden:
                        violations.append((path.name, name))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
