from __future__ import annotations

import unittest

from voicehub.optimization import (
    DIFFUSION_FAMILY_FEATURE,
    DIFFUSION_KIND_FEATURE_PREFIX,
    DIFFUSION_OPERATION_FEATURE_PREFIX,
    DIFFUSION_SAMPLING_FEATURE_PREFIX,
    DiffusionArchitectureKind,
    DiffusionOperation,
    diffusion_kind_feature,
    diffusion_operation_feature,
    get_diffusion_model_optimization_support,
    get_tts_optimization_support,
    list_diffusion_model_optimization_support,
)
from voicehub.registry import list_model_specs
from voicehub.runtime import get_architecture_spec
from voicehub.tasks import SpeechTask

_EXPECTED = {
    "chatterbox": {
        "architecture":
        "chatterbox",
        "kind":
        DiffusionArchitectureKind.CONDITIONAL_FLOW_MATCHING,
        "operations": (
            DiffusionOperation.DENOISER,
            DiffusionOperation.CLASSIFIER_FREE_GUIDANCE,
            DiffusionOperation.EULER_SOLVER,
        ),
        "passes": (
            "compile",
            "sdpa",
            "diffusion-cache",
            "diffusion-sampling",
        ),
        "sampling": (
            "schedule",
            "prediction-cache",
        ),
    },
    "cosyvoice": {
        "architecture":
        "cosyvoice-native",
        "kind":
        DiffusionArchitectureKind.CONDITIONAL_FLOW_MATCHING,
        "operations": (
            DiffusionOperation.DENOISER,
            DiffusionOperation.CLASSIFIER_FREE_GUIDANCE,
            DiffusionOperation.EULER_SOLVER,
        ),
        "passes": (
            "compile",
            "custom-kernels",
            "diffusion-cache",
            "diffusion-sampling",
        ),
        "sampling": (
            "schedule",
            "guidance",
            "prediction-cache",
            "stork2",
        ),
    },
    "echo": {
        "architecture":
        "echo-dit",
        "kind":
        DiffusionArchitectureKind.RECTIFIED_FLOW,
        "operations": (
            DiffusionOperation.DENOISER,
            DiffusionOperation.CLASSIFIER_FREE_GUIDANCE,
            DiffusionOperation.EULER_SOLVER,
        ),
        "passes": (
            "compile",
            "custom-kernels",
            "diffusion-cache",
            "diffusion-sampling",
        ),
        "sampling": (
            "schedule",
            "guidance",
            "prediction-cache",
        ),
    },
    "f5tts": {
        "architecture":
        "f5tts",
        "kind":
        DiffusionArchitectureKind.CONDITIONAL_FLOW_MATCHING,
        "operations": (
            DiffusionOperation.DENOISER,
            DiffusionOperation.CLASSIFIER_FREE_GUIDANCE,
            DiffusionOperation.EULER_SOLVER,
            DiffusionOperation.MIDPOINT_SOLVER,
        ),
        "passes": (
            "compile",
            "attention-backend",
            "custom-kernels",
            "diffusion-cache",
            "diffusion-sampling",
        ),
        "sampling": (
            "schedule",
            "guidance",
            "prediction-cache",
            "stork2",
        ),
    },
    "irodoritts": {
        "architecture":
        "irodoritts-rf-dit",
        "kind":
        DiffusionArchitectureKind.RECTIFIED_FLOW,
        "operations": (
            DiffusionOperation.DENOISER,
            DiffusionOperation.CLASSIFIER_FREE_GUIDANCE,
            DiffusionOperation.EULER_SOLVER,
        ),
        "passes": (
            "compile",
            "sdpa",
            "custom-kernels",
            "diffusion-cache",
            "diffusion-sampling",
        ),
        "sampling": (
            "schedule",
            "guidance",
            "prediction-cache",
        ),
    },
    "styletts2": {
        "architecture":
        "styletts2",
        "kind":
        DiffusionArchitectureKind.STYLE_DIFFUSION,
        "operations": (
            DiffusionOperation.DENOISER,
            DiffusionOperation.CLASSIFIER_FREE_GUIDANCE,
            DiffusionOperation.ADPM2_SOLVER,
        ),
        "passes": (
            "compile",
            "diffusion-cache",
            "diffusion-sampling",
        ),
        "sampling": ("schedule", ),
    },
    "supertonic": {
        "architecture": "supertonic",
        "kind": DiffusionArchitectureKind.FLOW_MATCHING,
        "operations": (
            DiffusionOperation.DENOISER,
            DiffusionOperation.ITERATIVE_ESTIMATOR,
        ),
        "passes": (
            "compile",
            "diffusion-cache",
            "diffusion-sampling",
        ),
        "sampling": ("discrete-step-count", ),
    },
    "vibevoice": {
        "architecture":
        "vibevoice-tts",
        "kind":
        DiffusionArchitectureKind.DENOISING_DIFFUSION,
        "operations": (
            DiffusionOperation.DENOISER,
            DiffusionOperation.CLASSIFIER_FREE_GUIDANCE,
            DiffusionOperation.DPM_SOLVER_PLUS_PLUS,
        ),
        "passes": (
            "compile",
            "sdpa",
            "custom-kernels",
            "diffusion-cache",
            "diffusion-sampling",
        ),
        "sampling": (
            "schedule",
            "guidance",
            "prediction-cache",
        ),
    },
    "voxcpm": {
        "architecture":
        "voxcpm2",
        "kind":
        DiffusionArchitectureKind.CONDITIONAL_FLOW_MATCHING,
        "operations": (
            DiffusionOperation.DENOISER,
            DiffusionOperation.CLASSIFIER_FREE_GUIDANCE,
            DiffusionOperation.EULER_SOLVER,
        ),
        "passes": (
            "compile",
            "sdpa",
            "diffusion-cache",
            "diffusion-sampling",
        ),
        "sampling": (
            "schedule",
            "guidance",
            "prediction-cache",
        ),
    },
}


class DiffusionFamilyInventoryTests(unittest.TestCase):

    def test_inventory_is_trait_driven_and_exact(self):
        support = list_diffusion_model_optimization_support()

        self.assertEqual(
            [item.model_type for item in support],
            sorted(_EXPECTED),
        )
        for item in support:
            expected = _EXPECTED[item.model_type]
            with self.subTest(model_type=item.model_type):
                self.assertEqual(item.architecture, expected["architecture"])
                self.assertIs(item.kind, expected["kind"])
                self.assertEqual(item.operations, expected["operations"])
                self.assertEqual(
                    item.optimization_passes,
                    expected["passes"],
                )
                self.assertEqual(
                    item.sampling_techniques,
                    expected.get("sampling", ()),
                )
                self.assertTrue(item.training)
                self.assertTrue(item.distributed_training)
                self.assertTrue(item.compile_supported)
                self.assertTrue(item.supports_optimization_pass("compile"))
                self.assertEqual(
                    item.diffusion_cache_supported,
                    "diffusion-cache" in expected["passes"],
                )
                self.assertEqual(item.diffusion_cache_methods, ("dbcache", "first_block"))
                self.assertEqual(item.diffusion_cache_predictors, ("reuse", "taylor"))
                self.assertEqual(item.diffusion_cache_step_policies, ("dynamic", "static"))
                self.assertEqual(
                    item.diffusion_sampling_supported,
                    "diffusion-sampling" in expected["passes"],
                )

    def test_normalized_features_and_metadata_are_kept_in_lockstep(self):
        for model_type, expected in _EXPECTED.items():
            architecture = get_architecture_spec(expected["architecture"])
            capabilities = architecture.capabilities
            metadata = architecture.metadata
            with self.subTest(model_type=model_type):
                self.assertTrue(capabilities.has_feature(DIFFUSION_FAMILY_FEATURE), )
                self.assertEqual(
                    tuple(
                        item for item in capabilities.features
                        if item.startswith(DIFFUSION_KIND_FEATURE_PREFIX)),
                    (diffusion_kind_feature(expected["kind"]), ),
                )
                self.assertEqual(
                    tuple(
                        item for item in capabilities.features
                        if item.startswith(DIFFUSION_OPERATION_FEATURE_PREFIX)),
                    tuple(diffusion_operation_feature(item) for item in expected["operations"]),
                )
                self.assertEqual(
                    tuple(
                        item for item in capabilities.features
                        if item.startswith(DIFFUSION_SAMPLING_FEATURE_PREFIX)),
                    tuple(
                        f"{DIFFUSION_SAMPLING_FEATURE_PREFIX}{item}"
                        for item in expected.get("sampling", ())),
                )
                self.assertEqual(
                    metadata["diffusion_architecture_kind"],
                    expected["kind"].value,
                )
                self.assertEqual(
                    metadata["diffusion_operations"],
                    tuple(item.value for item in expected["operations"]),
                )
                self.assertEqual(
                    metadata.get("diffusion_sampling_capabilities", ()),
                    expected.get("sampling", ()),
                )

    def test_pass_inventory_matches_the_universal_tts_resolver(self):
        selectable_attention = []
        custom_kernels = []
        native_sdpa = []
        for item in list_diffusion_model_optimization_support():
            universal = get_tts_optimization_support(item.model_type)
            with self.subTest(model_type=item.model_type):
                self.assertTrue(universal.compile)
                self.assertEqual(
                    universal.optimization_kinds,
                    item.optimization_passes,
                )
            if item.supports_optimization_pass("attention_backend"):
                selectable_attention.append(item.model_type)
            if item.supports_optimization_pass("custom_kernels"):
                custom_kernels.append(item.model_type)
            if item.supports_optimization_pass("sdpa"):
                native_sdpa.append(item.model_type)

        self.assertEqual(selectable_attention, ["f5tts"])
        self.assertEqual(
            custom_kernels,
            ["cosyvoice", "echo", "f5tts", "irodoritts", "vibevoice"],
        )
        self.assertEqual(
            native_sdpa,
            ["chatterbox", "irodoritts", "vibevoice", "voxcpm"],
        )

    def test_vendored_or_non_diffusion_graphs_are_not_misclassified(self):
        excluded = (
            "gptsovits",
            "kokoro",
            "omnivoice",
            "qwen3tts",
            "vits",
        )
        for model_type in excluded:
            with (
                    self.subTest(model_type=model_type),
                    self.assertRaisesRegex(
                        ValueError,
                        "not a registered diffusion-family",
                    ),
            ):
                get_diffusion_model_optimization_support(model_type)

        marked = {
            model.model_type
            for model in list_model_specs(task=SpeechTask.TEXT_TO_SPEECH) if (
                model.architecture is not None and
                get_architecture_spec(model.architecture).capabilities.has_feature(DIFFUSION_FAMILY_FEATURE))
        }
        self.assertEqual(marked, set(_EXPECTED))

    def test_model_aliases_resolve_to_canonical_support(self):
        aliases = {
            "cosy-voice": "cosyvoice",
            "f5-tts": "f5tts",
            "irodori": "irodoritts",
            "style-tts2": "styletts2",
            "supertonic3": "supertonic",
            "vibe-voice": "vibevoice",
            "vox-cpm": "voxcpm",
        }
        for alias, canonical in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(
                    get_diffusion_model_optimization_support(alias).model_type,
                    canonical,
                )

    def test_support_manifest_is_primitive_and_explicit(self):
        manifest = get_diffusion_model_optimization_support("f5tts").to_dict()
        self.assertEqual(manifest["model_type"], "f5tts")
        self.assertEqual(
            manifest["kind"],
            "conditional-flow-matching",
        )
        self.assertEqual(
            manifest["operations"],
            [
                "denoiser",
                "classifier-free-guidance",
                "euler-solver",
                "midpoint-solver",
            ],
        )
        self.assertTrue(manifest["compile_supported"])
        self.assertTrue(manifest["diffusion_cache_supported"])
        self.assertTrue(manifest["diffusion_sampling_supported"])
        self.assertEqual(
            manifest["sampling_techniques"],
            [
                "schedule",
                "guidance",
                "prediction-cache",
                "stork2",
            ],
        )
        self.assertEqual(
            manifest["optimization_passes"],
            [
                "compile",
                "attention-backend",
                "custom-kernels",
                "diffusion-cache",
                "diffusion-sampling",
            ],
        )


if __name__ == "__main__":
    unittest.main()
