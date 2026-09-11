import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voicehub.diffusion_serving import (
    DiffusionServingBackend,
    DiffusionServingCompatibilityError,
    VLLMOmniDiffusionPlugin,
    bridge_vllm_omni_tts_config,
    detect_vllm_omni_features,
    get_diffusion_serving_capability,
    list_diffusion_serving_capabilities,
    resolve_diffusion_tts_backend,
)
from voicehub.llm_serving import LLMBackend, LLMBackendConfig, LLMBackendTransport
from voicehub.registry import ModelSpec, register_model_spec, unregister_model_spec
from voicehub.runtime import (
    ArchitectureCapabilities,
    ArchitectureSpec,
    register_architecture_spec,
    unregister_architecture_spec,
)


class DiffusionServingCapabilityTests(unittest.TestCase):

    def test_backend_names_keep_sglang_visual_and_omni_runtimes_separate(self):
        self.assertIs(
            DiffusionServingBackend.coerce("sglang-diffusion"),
            DiffusionServingBackend.SGLANG_DIFFUSION,
        )
        self.assertIs(
            DiffusionServingBackend.coerce("sglang-omni"),
            DiffusionServingBackend.SGLANG_OMNI,
        )
        self.assertIs(
            DiffusionServingBackend.coerce("vllm"),
            DiffusionServingBackend.VLLM_OMNI,
        )

    def test_capabilities_are_truthful_about_modalities_and_tts(self):
        native = get_diffusion_serving_capability("native")
        self.assertEqual(
            native.verified_tts_models,
            (
                "chatterbox",
                "cosyvoice",
                "echo",
                "f5tts",
                "irodoritts",
                "styletts2",
                "supertonic",
                "voxcpm",
            ),
        )

        vllm = get_diffusion_serving_capability("vllm-omni")
        self.assertTrue(vllm.supports_tts_diffusion)
        self.assertEqual(vllm.verified_tts_models, ("cosyvoice", "voxcpm"))
        self.assertTrue(vllm.supports_custom_plugins)

        visual_sglang = get_diffusion_serving_capability("sglang-diffusion")
        self.assertTrue(visual_sglang.supports_visual_diffusion)
        self.assertFalse(visual_sglang.supports_tts)
        self.assertFalse(visual_sglang.supports_tts_diffusion)

        omni_sglang = get_diffusion_serving_capability("sglang-omni")
        self.assertTrue(omni_sglang.supports_tts)
        self.assertFalse(omni_sglang.supports_visual_diffusion)
        self.assertFalse(omni_sglang.supports_tts_diffusion)

    def test_registered_architecture_feature_extends_native_support(self):
        architecture_id = "test-diffusion-serving-architecture"
        model_type = "test-diffusion-serving-model"
        architecture = ArchitectureSpec(
            architecture_id=architecture_id,
            model_builder="tests.test_diffusion_serving:TestModel",
            capabilities=ArchitectureCapabilities(
                tasks=("text-to-speech", ),
                features=(
                    "diffusion-family",
                    "diffusion-serving-native",
                ),
            ),
        )
        model = ModelSpec(
            model_type=model_type,
            module="tests.test_diffusion_serving",
            class_name="TestModel",
            default_model_path="",
            capabilities=("voicehub-native", ),
            architecture=architecture_id,
        )

        try:
            register_architecture_spec(architecture)
            register_model_spec(model)

            capability = get_diffusion_serving_capability("native")
            self.assertIn(model_type, capability.verified_tts_models)
            plan = resolve_diffusion_tts_backend(model_type, "native")
            self.assertTrue(plan.verified)
        finally:
            unregister_model_spec(model_type, missing_ok=True)
            unregister_architecture_spec(architecture_id, missing_ok=True)

    def test_capability_filters_do_not_import_optional_engines(self):
        visual = list_diffusion_serving_capabilities(supports_visual_diffusion=True)
        self.assertEqual(
            {item.backend
             for item in visual},
            {
                DiffusionServingBackend.VLLM_OMNI,
                DiffusionServingBackend.SGLANG_DIFFUSION,
            },
        )
        tts = list_diffusion_serving_capabilities(supports_tts=True)
        self.assertIn(
            DiffusionServingBackend.NATIVE,
            {item.backend
             for item in tts},
        )

    def test_verified_native_and_vllm_pairs_resolve(self):
        native = resolve_diffusion_tts_backend("f5-tts", "native")
        self.assertEqual(native.model_type, "f5tts")
        self.assertTrue(native.verified)

        for model_type in ("cosyvoice", "vox-cpm"):
            with self.subTest(model_type=model_type):
                plan = resolve_diffusion_tts_backend(model_type, "vllm-omni")
                self.assertTrue(plan.verified)
                self.assertTrue(plan.uses_existing_llm_speech_bridge)

    def test_native_serving_does_not_claim_vibevoice_high_level_inference(self):
        with self.assertRaisesRegex(
                DiffusionServingCompatibilityError,
                "no diffusion-TTS support",
        ):
            resolve_diffusion_tts_backend("vibevoice", "native")

    def test_sglang_diffusion_and_sglang_omni_fail_closed_for_diffusion_tts(self):
        with self.assertRaisesRegex(
                DiffusionServingCompatibilityError,
                "supports_tts=False",
        ):
            resolve_diffusion_tts_backend("cosyvoice", "sglang-diffusion")
        with self.assertRaisesRegex(
                DiffusionServingCompatibilityError,
                "separate LLM-TTS runtime",
        ):
            resolve_diffusion_tts_backend("cosyvoice", "sglang-omni")

    def test_unverified_vllm_model_requires_complete_explicit_plugin(self):
        with self.assertRaisesRegex(
                DiffusionServingCompatibilityError,
                "no verified complete TTS diffusion adapter",
        ):
            resolve_diffusion_tts_backend("f5tts", "vllm-omni")

        denoiser_only = VLLMOmniDiffusionPlugin(
            model_type="f5tts",
            model_arch="F5TTSPipeline",
            module_name="voicehub_vllm_plugins.f5tts",
            class_name="F5TTSPipeline",
        )
        with self.assertRaisesRegex(
                DiffusionServingCompatibilityError,
                "does not declare a complete TTS pipeline",
        ):
            resolve_diffusion_tts_backend(
                "f5tts",
                "vllm-omni",
                plugin=denoiser_only,
            )

        complete = VLLMOmniDiffusionPlugin(
            model_type="f5-tts",
            model_arch="F5TTSPipeline",
            module_name="voicehub_vllm_plugins.f5tts",
            class_name="F5TTSPipeline",
            complete_tts_pipeline=True,
            post_process_func_name="get_audio_post_process",
        )
        plan = resolve_diffusion_tts_backend(
            "f5tts",
            "vllm-omni",
            plugin=complete,
        )
        self.assertTrue(plan.experimental)
        self.assertFalse(plan.verified)
        self.assertFalse(plan.uses_existing_llm_speech_bridge)

    def test_package_import_does_not_import_optional_engines(self):
        script = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "vllm_omni" or name.startswith("vllm_omni."):
        raise AssertionError("vLLM-Omni imported eagerly")
    if name == "sglang" or name.startswith("sglang."):
        raise AssertionError("SGLang imported eagerly")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import voicehub.diffusion_serving
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class VLLMOmniPluginTests(unittest.TestCase):

    def test_plugin_validates_public_registration_contract(self):
        with self.assertRaisesRegex(ValueError, "module_name"):
            VLLMOmniDiffusionPlugin(
                model_type="f5tts",
                model_arch="F5TTSPipeline",
                module_name="not-a-module",
                class_name="F5TTSPipeline",
            )
        with self.assertRaisesRegex(ValueError, "post_process_func_name"):
            VLLMOmniDiffusionPlugin(
                model_type="f5tts",
                model_arch="F5TTSPipeline",
                module_name="plugins.f5tts",
                class_name="F5TTSPipeline",
                complete_tts_pipeline=True,
            )

    @patch("voicehub.diffusion_serving.vllm_omni.import_module")
    def test_plugin_registration_is_lazy_and_uses_upstream_function(self, import_module_mock):
        register = Mock()
        import_module_mock.return_value = SimpleNamespace(register_diffusion_model=register)
        plugin = VLLMOmniDiffusionPlugin(
            model_type="f5tts",
            model_arch="F5TTSPipeline",
            module_name="plugins.f5tts",
            class_name="F5TTSPipeline",
            pre_process_func_name="get_pre_process",
            post_process_func_name="get_post_process",
        )

        plugin.register()

        import_module_mock.assert_called_once_with("vllm_omni.diffusion.registry")
        register.assert_called_once_with(
            model_arch="F5TTSPipeline",
            module_name="plugins.f5tts",
            class_name="F5TTSPipeline",
            pre_process_func_name="get_pre_process",
            post_process_func_name="get_post_process",
            action_post_process_func_name=None,
            ir_op_priority_func_name=None,
        )

    @patch("voicehub.diffusion_serving.vllm_omni.import_module")
    @patch("voicehub.diffusion_serving.vllm_omni.util.find_spec")
    @patch("voicehub.diffusion_serving.vllm_omni.metadata.version")
    def test_feature_detection_reports_installed_version_and_registry_api(
            self, version_mock, find_spec_mock, import_module_mock):
        version_mock.return_value = "0.24.0"
        find_spec_mock.return_value = object()
        import_module_mock.return_value = SimpleNamespace(register_diffusion_model=lambda **_: None)

        status = detect_vllm_omni_features()

        self.assertTrue(status.installed)
        self.assertEqual(status.version, "0.24.0")
        self.assertTrue(status.supports_out_of_tree_diffusion_plugins)
        import_module_mock.assert_called_once_with("vllm_omni.diffusion.registry")

    @patch("voicehub.diffusion_serving.vllm_omni.import_module")
    @patch("voicehub.diffusion_serving.vllm_omni.util.find_spec")
    @patch("voicehub.diffusion_serving.vllm_omni.metadata.version")
    def test_metadata_only_detection_does_not_import_registry(
            self, version_mock, find_spec_mock, import_module_mock):
        version_mock.return_value = "0.24.0"
        find_spec_mock.return_value = object()

        status = detect_vllm_omni_features(probe_registry=False)

        self.assertTrue(status.installed)
        self.assertIsNone(status.register_diffusion_model)
        import_module_mock.assert_not_called()


class DiffusionServingBridgeTests(unittest.TestCase):

    def test_verified_vllm_pair_reuses_llm_speech_configuration(self):
        config = LLMBackendConfig(
            backend="vllm",
            endpoint="https://voice.example",
            transport="auto",
            model="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        )

        plan, bridged = bridge_vllm_omni_tts_config("cosyvoice", config)

        self.assertTrue(plan.uses_existing_llm_speech_bridge)
        self.assertIs(bridged.backend, LLMBackend.VLLM)
        self.assertIs(bridged.transport, LLMBackendTransport.SPEECH)
        self.assertEqual(bridged.endpoint, config.endpoint)
        self.assertEqual(bridged.model, config.model)

    def test_bridge_rejects_sglang_and_token_transport(self):
        with self.assertRaisesRegex(
                DiffusionServingCompatibilityError,
                "backend='vllm'",
        ):
            bridge_vllm_omni_tts_config(
                "cosyvoice",
                LLMBackendConfig(
                    backend="sglang",
                    endpoint="https://voice.example",
                ),
            )
        with self.assertRaisesRegex(
                DiffusionServingCompatibilityError,
                "speech transport",
        ):
            bridge_vllm_omni_tts_config(
                "cosyvoice",
                LLMBackendConfig(
                    backend="vllm",
                    endpoint="https://voice.example",
                    transport="tokens",
                ),
            )

    def test_bridge_does_not_claim_unverified_vllm_models(self):
        with self.assertRaisesRegex(
                DiffusionServingCompatibilityError,
                "Verified VoiceHub models: cosyvoice, voxcpm",
        ):
            bridge_vllm_omni_tts_config(
                "f5tts",
                LLMBackendConfig(
                    backend="vllm",
                    endpoint="https://voice.example",
                ),
            )


if __name__ == "__main__":
    unittest.main()
