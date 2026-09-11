import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from voicehub.models.omnivoice.modeling import OmniVoiceConfig, OmniVoiceForTextToSpeech

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class OmniVoiceTrainingRuntimeTests(unittest.TestCase):

    @staticmethod
    def _runtime():
        import torch

        from tests.test_native_omnivoice import _tiny_codec, _tiny_tokenizer
        from voicehub.models.omnivoice.native.configuration import OmniVoiceArchitectureConfig
        from voicehub.models.omnivoice.native.modeling import OmniVoiceModel
        from voicehub.models.omnivoice.native.runtime import OmniVoiceRuntime

        torch.manual_seed(17)
        model = OmniVoiceModel(OmniVoiceArchitectureConfig.tiny(vocab_size=320))
        runtime = OmniVoiceRuntime(model, _tiny_tokenizer(), _tiny_codec())
        return runtime

    def test_runtime_mode_switches_preserve_the_trainable_graph(self):
        runtime = self._runtime()
        wrapper = OmniVoiceForTextToSpeech(
            OmniVoiceConfig(name_or_path="test/omnivoice"),
            device="cpu",
        )
        wrapper._runtime = runtime

        wrapper.load_for_training()
        optimizer_owned_model = wrapper.model
        optimizer_owned_model.audio_heads.weight.data.fill_(0.125)

        self.assertTrue(wrapper._training_ready)
        self.assertTrue(optimizer_owned_model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in runtime.audio_tokenizer.parameters()))

        wrapper.load()

        self.assertTrue(wrapper._inference_ready)
        self.assertFalse(wrapper._training_ready)
        self.assertIs(wrapper.model, optimizer_owned_model)
        self.assertFalse(optimizer_owned_model.training)
        self.assertEqual(
            float(optimizer_owned_model.audio_heads.weight[0, 0].detach()),
            0.125,
        )

        wrapper.load_for_training()

        self.assertTrue(wrapper._training_ready)
        self.assertIs(wrapper.model, optimizer_owned_model)
        self.assertTrue(optimizer_owned_model.training)
        self.assertEqual(
            float(optimizer_owned_model.audio_heads.weight[0, 0].detach()),
            0.125,
        )

    def test_shared_training_registry_selects_native_adapter(self):
        from voicehub.models.omnivoice.training import OmniVoiceTrainingAdapter
        from voicehub.training.specs import get_training_spec

        runtime = self._runtime()
        wrapper = OmniVoiceForTextToSpeech(device="cpu")
        wrapper._runtime = runtime

        adapter = wrapper.get_training_adapter()
        self.assertIsInstance(adapter, OmniVoiceTrainingAdapter)
        self.assertEqual(adapter.spec, get_training_spec("omnivoice"))
        self.assertEqual(adapter.spec.default_phase, "masked_audio")
        self.assertEqual(
            adapter.spec.source_entrypoints,
            ("voicehub.models.omnivoice.native.modeling:"
             "OmniVoiceModel.forward", ),
        )
        adapter.setup()
        self.assertIs(adapter.primary_model, runtime.model)

    def test_native_export_delegates_to_complete_runtime(self):
        runtime = self._runtime()
        wrapper = OmniVoiceForTextToSpeech(device="cpu")
        wrapper._runtime = runtime
        wrapper.model = runtime.model

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "omnivoice"
            with patch.object(
                    runtime,
                    "save_pretrained",
                    return_value=destination,
            ) as save:
                result = wrapper.export_native_pretrained(destination)

        self.assertEqual(result, destination)
        save.assert_called_once_with(destination)


if __name__ == "__main__":
    unittest.main()
