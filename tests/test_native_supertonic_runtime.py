from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

import torch

from voicehub.checkpointing import (
    ONNXAttribute,
    ONNXGraph,
    ONNXModel,
    ONNXNode,
    ONNXTensor,
    ONNXValueInfo,
    save_safetensors,
)
from voicehub.models.supertonic.configuration import SUPERTONIC_SAMPLE_RATE, SupertonicConfig
from voicehub.models.supertonic.modeling import SupertonicForTextToSpeech
from voicehub.models.supertonic.native.checkpoint import load_supertonic_native_weights, save_supertonic_native_weights
from voicehub.models.supertonic.native.configuration import SupertonicArchitectureConfig
from voicehub.models.supertonic.native.frontend import SupertonicUnicodeProcessor
from voicehub.models.supertonic.native.runtime import NativeSupertonicRuntime
from voicehub.models.supertonic.training import SupertonicTrainingAdapter
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import get_training_spec


def _value(name: str, shape: tuple[int | str | None, ...]) -> ONNXValueInfo:
    return ONNXValueInfo(
        name=name,
        element_type=1,
        shape=shape,
    )


def _tensor(
        name: str,
        values: tuple[float, ...],
        shape: tuple[int, ...] = (),
) -> ONNXTensor:
    return ONNXTensor(
        name=name,
        data_type=1,
        dimensions=shape,
        raw_data=b"",
        float_data=values,
    )


def _axes(name: str, values: tuple[int, ...]) -> ONNXTensor:
    return ONNXTensor(
        name=name,
        data_type=7,
        dimensions=(len(values), ),
        raw_data=b"",
        int64_data=values,
    )


def _keep_dimensions(value: bool) -> MappingProxyType:
    return MappingProxyType({
        "keepdims": ONNXAttribute(
            name="keepdims",
            attribute_type=2,
            value=int(value),
        ),
    })


def _model(
    *,
    inputs: tuple[str, ...],
    output: str,
    nodes: tuple[ONNXNode, ...],
    initializers: dict[str, ONNXTensor],
) -> ONNXModel:
    return ONNXModel(
        ir_version=9,
        producer_name="voicehub-test",
        producer_version="1",
        domain="",
        model_version=1,
        opsets=(("", 19), ),
        metadata=MappingProxyType({}),
        graph=ONNXGraph(
            name="supertonic-test",
            inputs=tuple(_value(name, ()) for name in inputs),
            outputs=(_value(output, ()), ),
            nodes=nodes,
            initializers=MappingProxyType(initializers),
        ),
    )


def _runtime() -> NativeSupertonicRuntime:
    duration = _model(
        inputs=("text_ids", "style_dp", "text_mask"),
        output="duration",
        nodes=(
            ONNXNode(
                op_type="ReduceMean",
                domain="",
                inputs=("style_dp", "duration_axes"),
                outputs=("duration_mean", ),
                attributes=_keep_dimensions(False),
            ),
            ONNXNode(
                op_type="Add",
                domain="",
                inputs=("duration_mean", "duration_bias"),
                outputs=("duration", ),
            ),
        ),
        initializers={
            "duration_axes": _axes("duration_axes", (1, 2)),
            "duration_bias": _tensor("duration_bias", (0.5, )),
        },
    )
    text_encoder = _model(
        inputs=("text_ids", "style_ttl", "text_mask"),
        output="text_emb",
        nodes=(
            ONNXNode(
                op_type="Add",
                domain="",
                inputs=("style_ttl", "text_bias"),
                outputs=("text_emb", ),
            ), ),
        initializers={
            "text_bias": _tensor("text_bias", (0.1, )),
        },
    )
    vector = _model(
        inputs=(
            "noisy_latent",
            "text_emb",
            "style_ttl",
            "latent_mask",
            "text_mask",
            "current_step",
            "total_step",
        ),
        output="denoised_latent",
        nodes=(
            ONNXNode(
                op_type="ReduceMean",
                domain="",
                inputs=("text_emb", "text_axes"),
                outputs=("text_condition", ),
                attributes=_keep_dimensions(True),
            ),
            ONNXNode(
                op_type="Add",
                domain="",
                inputs=("noisy_latent", "text_condition"),
                outputs=("conditioned", ),
            ),
            ONNXNode(
                op_type="Mul",
                domain="",
                inputs=("conditioned", "vector_scale"),
                outputs=("denoised_latent", ),
            ),
        ),
        initializers={
            "text_axes": _axes("text_axes", (1, 2)),
            "vector_scale": _tensor("vector_scale", (0.5, )),
        },
    )
    vocoder = _model(
        inputs=("latent", ),
        output="wav_tts",
        nodes=(
            ONNXNode(
                op_type="Mul",
                domain="",
                inputs=("latent", "vocoder_scale"),
                outputs=("scaled", ),
            ),
            ONNXNode(
                op_type="ReduceMean",
                domain="",
                inputs=("scaled", "channel_axis"),
                outputs=("wav_tts", ),
                attributes=_keep_dimensions(False),
            ),
        ),
        initializers={
            "channel_axis": _axes("channel_axis", (1, )),
            "vocoder_scale": _tensor("vocoder_scale", (1.0, )),
        },
    )
    processor = SupertonicUnicodeProcessor(tuple(range(128)))
    return NativeSupertonicRuntime(
        architecture=SupertonicArchitectureConfig(
            sample_rate=100,
            base_chunk_size=2,
            latent_dimension=2,
            text_to_latent_compression=3,
        ),
        processor=processor,
        duration_predictor=duration,
        text_encoder=text_encoder,
        vector_estimator=vector,
        vocoder=vocoder,
    )


class NativeSupertonicRuntimeTests(unittest.TestCase):

    def test_frontend_normalizes_unicode_without_external_packages(self):
        processor = SupertonicUnicodeProcessor(tuple(range(128)))

        normalized = processor.normalize_text(
            "Hello – world 😀",
            "en",
        )
        ids, mask = processor.encode(
            ["Hello"],
            ["en"],
        )

        self.assertEqual(normalized, "<en>Hello - world.</en>")
        self.assertEqual(ids.shape[0], 1)
        self.assertEqual(mask.shape, (1, 1, ids.shape[1]))
        self.assertTrue(bool(mask.all()))

    def test_config_keeps_fixed_rate_and_requires_explicit_training_opt_in(self):
        config = SupertonicConfig(
            sample_rate=8_000,
            language="TR",
        )

        self.assertEqual(config.sample_rate, SUPERTONIC_SAMPLE_RATE)
        self.assertEqual(config.language, "tr")
        self.assertFalse(config.enable_preprocessed_training)
        with self.assertRaisesRegex(
                ValueError,
                "Unsupported Supertonic language",
        ):
            SupertonicConfig(language="xx")

    def test_latent_mask_preserves_the_released_sample_truncation(self):
        runtime = _runtime()

        mask, maximum = runtime._latent_mask(torch.tensor([0.061]))

        self.assertEqual(maximum, 1)
        torch.testing.assert_close(mask, torch.ones(1, 1, 1))

    def test_published_graph_objective_backpropagates_through_all_components(self, ):
        runtime = _runtime()
        text_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        text_mask = torch.ones(1, 1, 3)
        style_ttl = torch.zeros(1, 50, 256)
        style_dp = torch.zeros(1, 8, 16)
        target_latent = torch.randn(
            1,
            runtime.architecture.latent_channels,
            4,
        )

        output = runtime.fine_tuning_loss(
            text_ids=text_ids,
            text_mask=text_mask,
            style_ttl=style_ttl,
            style_dp=style_dp,
            target_duration=torch.tensor([1.0]),
            target_latent=target_latent,
            source_noise=torch.zeros_like(target_latent),
            current_step=torch.tensor([0.0]),
            total_steps=2,
            target_audio=torch.zeros(1, 4),
        )
        output.loss.backward()

        self.assertEqual(
            set(output.losses),
            {
                "duration_loss",
                "flow_step_loss",
                "vocoder_l1_loss",
            },
        )
        for graph, name in (
            (runtime.duration_predictor, "duration_bias"),
            (runtime.text_encoder, "text_bias"),
            (runtime.vector_estimator, "vector_scale"),
            (runtime.vocoder, "vocoder_scale"),
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(graph.initializer_tensor(name).grad, )

    def test_native_weight_export_strictly_round_trips_original_names(self):
        runtime = _runtime()
        expected = {
            role: {
                name: value.clone()
                for name, value in getattr(
                    runtime,
                    role,
                ).onnx_state_dict().items()
            }
            for role in (
                "duration_predictor",
                "text_encoder",
                "vector_estimator",
                "vocoder",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = save_supertonic_native_weights(runtime, directory)
            with torch.no_grad():
                for parameter in runtime.parameters():
                    parameter.add_(7.0)

            load_supertonic_native_weights(runtime, paths)

        for role, state in expected.items():
            actual = getattr(runtime, role).onnx_state_dict()
            self.assertEqual(tuple(actual), tuple(state))
            for name in state:
                torch.testing.assert_close(actual[name], state[name])

    def test_native_overlay_cannot_replace_structural_initializers(self):
        runtime = _runtime()
        with tempfile.TemporaryDirectory() as directory:
            paths = save_supertonic_native_weights(runtime, directory)
            state = runtime.duration_predictor.onnx_state_dict()
            state["duration_axes"] = state["duration_axes"] + 1
            save_safetensors(
                state,
                paths["duration_predictor"],
            )

            with self.assertRaisesRegex(
                    ValueError,
                    "structural initializer",
            ):
                load_supertonic_native_weights(runtime, paths)

    def test_training_profile_is_honestly_preprocessed(self):
        spec = get_training_spec("supertonic")

        self.assertIs(spec.support, TrainingSupport.PREPROCESSED)
        self.assertTrue(spec.native_training)
        self.assertEqual(spec.default_phase, "published_graph")
        self.assertEqual(
            spec.get_phase().forward_method,
            "fine_tuning_loss",
        )

    def test_public_training_adapter_executes_the_native_objective(self):
        wrapper = SupertonicForTextToSpeech(
            SupertonicConfig(enable_preprocessed_training=True),
            device="cpu",
        )
        wrapper.model = _runtime()
        adapter = wrapper.get_training_adapter()

        output = adapter(
            text_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
            text_mask=torch.ones(1, 1, 3),
            style_ttl=torch.zeros(1, 50, 256),
            style_dp=torch.zeros(1, 8, 16),
            target_duration=torch.tensor([1.0]),
        )

        self.assertIsInstance(adapter, SupertonicTrainingAdapter)
        self.assertTrue(output.loss.requires_grad)
        self.assertEqual(
            output.metadata["training_phase"],
            "published_graph",
        )


if __name__ == "__main__":
    unittest.main()
