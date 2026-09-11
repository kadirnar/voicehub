import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voicehub.models.fishtts.modeling import FishTTSConfig, FishTTSForTextToSpeech
from voicehub.models.fishtts.native.configuration import FishS2Config
from voicehub.models.fishtts.native.modeling import FishS2ForConditionalGeneration

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class FishTTSTrainingRuntimeTests(unittest.TestCase):

    @staticmethod
    def _wrapper():
        import torch

        semantic_model = FishS2ForConditionalGeneration(
            FishS2Config.tiny(
                vocab_size=32,
                codebook_size=4,
                num_codebooks=2,
                hidden_size=8,
                num_hidden_layers=1,
                num_fast_layers=1,
            ))
        wrapper = FishTTSForTextToSpeech(
            FishTTSConfig(
                name_or_path="test/fish",
                torch_dtype="float32",
            ),
            device="cpu",
        )
        wrapper.model = semantic_model
        wrapper.native_config = semantic_model.config
        wrapper._torch = torch
        wrapper._loaded_for_training = True
        wrapper._training_ready = True
        semantic_model.train()
        return wrapper

    @staticmethod
    def _native_attachment(wrapper):
        import torch

        codec = torch.nn.Linear(1, 1, bias=False).eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)

        def prepare_for_inference():
            wrapper.model.eval()
            codec.eval()

        def prepare_for_training():
            wrapper.model.clear_caches()
            wrapper.model.train()
            codec.eval()

        runtime = SimpleNamespace(
            infer=Mock(return_value=torch.tensor([0.25, -0.25])),
            prepare_for_inference=Mock(side_effect=prepare_for_inference),
            prepare_for_training=Mock(side_effect=prepare_for_training),
        )
        wrapper.codec = codec
        wrapper._codec = codec
        wrapper._runtime = runtime
        return runtime, codec

    def test_training_generation_training_preserves_optimizer_owned_model(self):
        import torch

        wrapper = self._wrapper()
        semantic_model = wrapper.model
        optimizer = torch.optim.SGD(semantic_model.parameters(), lr=0.1)
        optimizer_parameter = optimizer.param_groups[0]["params"][0]
        attachment = {}

        def attach(*, torch, dtype):
            del torch, dtype
            runtime, codec = self._native_attachment(wrapper)
            attachment.update(runtime=runtime, codec=codec)

        with patch.object(
                wrapper,
                "_attach_codec_runtime",
                side_effect=attach,
        ) as attach_runtime:
            output = wrapper.generate("hello")

            self.assertIs(wrapper.model, semantic_model)
            self.assertIs(optimizer_parameter, next(wrapper.model.parameters()))
            self.assertFalse(wrapper._loaded_for_training)
            self.assertFalse(wrapper.model.training)
            self.assertIs(wrapper._runtime, attachment["runtime"])
            self.assertIs(wrapper._codec, attachment["codec"])
            self.assertTrue(all(not parameter.requires_grad for parameter in wrapper._codec.parameters()))
            torch.testing.assert_close(
                output.audio,
                torch.tensor([0.25, -0.25]),
            )

            wrapper.load_for_training()

            self.assertIs(wrapper.model, semantic_model)
            self.assertIs(optimizer_parameter, next(wrapper.model.parameters()))
            self.assertTrue(wrapper._loaded_for_training)
            self.assertTrue(wrapper.model.training)
            self.assertIs(wrapper._runtime, attachment["runtime"])
            self.assertIs(wrapper._codec, attachment["codec"])
            self.assertEqual(wrapper.model.max_batch_size, -1)
            self.assertEqual(wrapper.model.max_sequence_length, -1)

            wrapper.load()

        self.assertIs(wrapper.model, semantic_model)
        self.assertIs(optimizer_parameter, next(wrapper.model.parameters()))
        self.assertFalse(wrapper._loaded_for_training)
        attach_runtime.assert_called_once()
        self.assertEqual(
            attachment["runtime"].prepare_for_inference.call_count,
            2,
        )
        attachment["runtime"].prepare_for_training.assert_called_once()

    def test_failed_safe_codec_attachment_is_retryable(self):
        wrapper = self._wrapper()
        semantic_model = wrapper.model
        codec_error = RuntimeError("safe codec unavailable")

        with (
                patch.object(
                    wrapper,
                    "_attach_codec_runtime",
                    side_effect=codec_error,
                ),
                self.assertRaisesRegex(RuntimeError, "safe codec unavailable"),
        ):
            wrapper.load()

        self.assertIs(wrapper.model, semantic_model)
        self.assertTrue(wrapper._loaded_for_training)
        self.assertTrue(wrapper.model.training)
        self.assertIsNone(wrapper._runtime)
        self.assertIsNone(wrapper._codec)
        self.assertFalse(wrapper._inference_ready)

        with patch.object(
                wrapper,
                "_attach_codec_runtime",
                side_effect=lambda **unused: self._native_attachment(wrapper),
        ):
            wrapper.load()

        self.assertFalse(wrapper.model.training)
        self.assertTrue(wrapper._inference_ready)
        self.assertIsNotNone(wrapper._runtime)
        self.assertIsNotNone(wrapper._codec)


if __name__ == "__main__":
    unittest.main()
