from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from voicehub.models.bark.native.modeling import BarkModel
from voicehub.models.chatterbox.tts import ChatterboxTTS
from voicehub.models.csm.native.modeling import CSMModel
from voicehub.models.fishtts.native.modeling import FishS2ForConditionalGeneration
from voicehub.models.irodoritts.native.runtime import InferenceRuntime
from voicehub.models.kokoro.model import KModel
from voicehub.models.melotts.native.runtime import MeloTTSRuntime
from voicehub.models.neutts.native.modeling import NeuTTSRuntime
from voicehub.models.outetts.native.runtime import OuteTTSRuntime
from voicehub.models.speecht5.native_modeling import SpeechT5ForTextToSpeechModel
from voicehub.models.styletts2.native.runtime import StyleTTS2Runtime
from voicehub.models.supertonic.native.runtime import NativeSupertonicRuntime
from voicehub.models.voxcpm.native.modeling import VoxCPM2Model
from voicehub.models.vui.model import Vui
from voicehub.models.xtts.native.modeling import XTTS2Model
from voicehub.optimization.protocols import OptimizationCompileTarget


class _Generator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.infer_calls = 0

    def forward(self, value):
        return value * self.weight

    def infer(self, *args, **kwargs):
        del args, kwargs
        self.infer_calls += 1
        return (torch.ones(1, 1, 8) * self.weight, )


class _Codec(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def decode(self, value):
        return value * self.weight

    def decode_codes(self, value):
        return value * self.weight

    def decode_latent(self, value):
        return value * self.weight

    def decode_code(self, value):
        return value * self.weight

    def from_indices(self, value):
        return value * self.weight


class CompileTargetHookTests(unittest.TestCase):

    def _assert_targets(
        self,
        runtime,
        mode: str,
        expected: tuple[str, ...],
    ) -> tuple[OptimizationCompileTarget, ...]:
        targets = runtime.optimization_compile_targets(mode)
        self.assertIsInstance(targets, tuple)
        self.assertTrue(all(isinstance(target, OptimizationCompileTarget) for target in targets))
        self.assertEqual(
            tuple(target.label for target in targets),
            expected,
        )
        identities = tuple((id(target.owner), target.attribute) for target in targets)
        self.assertEqual(len(identities), len(set(identities)))
        return targets

    @staticmethod
    def _chatterbox() -> ChatterboxTTS:
        runtime = object.__new__(ChatterboxTTS)
        runtime.t3 = SimpleNamespace(
            inference=lambda **kwargs: kwargs,
            forward=lambda *args, **kwargs: (args, kwargs),
        )
        runtime.s3gen = SimpleNamespace(
            inference=lambda **kwargs: kwargs,
            flow_inference=lambda **kwargs: kwargs,
            hift_inference=lambda *args, **kwargs: (args, kwargs),
            flow=SimpleNamespace(forward=lambda *args, **kwargs: (args, kwargs),
                                 ),
        )
        return runtime

    @staticmethod
    def _melotts() -> MeloTTSRuntime:
        runtime = object.__new__(MeloTTSRuntime)
        runtime.model = _Generator()
        return runtime

    @staticmethod
    def _outetts() -> OuteTTSRuntime:
        runtime = object.__new__(OuteTTSRuntime)
        nn.Module.__init__(runtime)
        runtime.language_model = _Generator()
        runtime.codec = _Codec()
        return runtime

    @staticmethod
    def _styletts2() -> StyleTTS2Runtime:
        runtime = object.__new__(StyleTTS2Runtime)
        runtime.sampler = _Generator()
        runtime.model = SimpleNamespace(decoder=_Generator())
        return runtime

    @staticmethod
    def _irodori() -> InferenceRuntime:
        runtime = object.__new__(InferenceRuntime)
        runtime.model = SimpleNamespace(
            forward=lambda *args, **kwargs: (args, kwargs),
            forward_with_encoded_conditions=lambda *args, **kwargs: (
                args,
                kwargs,
            ),
        )
        runtime.codec = _Codec()
        return runtime

    @staticmethod
    def _voxcpm2() -> VoxCPM2Model:
        runtime = object.__new__(VoxCPM2Model)
        nn.Module.__init__(runtime)
        return runtime

    @staticmethod
    def _xtts2() -> XTTS2Model:
        runtime = object.__new__(XTTS2Model)
        nn.Module.__init__(runtime)
        runtime.gpt = _Generator()
        runtime.hifigan_decoder = _Generator()
        return runtime

    @staticmethod
    def _supertonic() -> NativeSupertonicRuntime:
        runtime = object.__new__(NativeSupertonicRuntime)
        nn.Module.__init__(runtime)
        runtime.duration_predictor = _Generator()
        runtime.text_encoder = _Generator()
        runtime.vector_estimator = _Generator()
        runtime.vocoder = _Generator()
        return runtime

    @staticmethod
    def _local_runtime_hooks():
        fish = object.__new__(FishS2ForConditionalGeneration)
        nn.Module.__init__(fish)

        csm = object.__new__(CSMModel)
        nn.Module.__init__(csm)
        csm.backbone = _Generator()
        csm.decoder = _Generator()

        neutts = object.__new__(NeuTTSRuntime)
        nn.Module.__init__(neutts)
        neutts.backbone = _Generator()
        neutts.codec = _Codec()

        bark = object.__new__(BarkModel)
        nn.Module.__init__(bark)
        bark.semantic = _Generator()
        bark.coarse_acoustics = _Generator()
        bark.fine_acoustics = _Generator()

        speecht5 = object.__new__(SpeechT5ForTextToSpeechModel)
        nn.Module.__init__(speecht5)
        speecht5.speecht5 = SimpleNamespace(
            encoder=_Generator(),
            decoder=SimpleNamespace(
                prenet=_Generator(),
                wrapped_decoder=_Generator(),
            ),
        )
        speecht5.speech_decoder_postnet = SimpleNamespace(postnet=lambda value: value, )

        vui = object.__new__(Vui)
        nn.Module.__init__(vui)
        vui.decoder = _Generator()
        vui.codec = _Codec()

        kokoro = object.__new__(KModel)
        nn.Module.__init__(kokoro)
        return fish, csm, neutts, bark, speecht5, vui, kokoro

    def test_inference_targets_match_real_execution_boundaries(self):
        cases = (
            (
                self._chatterbox(),
                (
                    "t3.inference",
                    "s3gen.flow_inference",
                    "s3gen.hift_inference",
                ),
            ),
            (
                self._melotts(),
                ("model.infer", ),
            ),
            (
                self._outetts(),
                ("language_model.forward", "codec.decode_codes"),
            ),
            (
                self._styletts2(),
                ("sampler.forward", "decoder.forward"),
            ),
            (
                self._irodori(),
                (
                    "model.forward_with_encoded_conditions",
                    "codec.decode_latent",
                ),
            ),
            (
                self._voxcpm2(),
                ("generate_features", ),
            ),
            (
                self._xtts2(),
                ("gpt.forward", "hifigan_decoder.forward"),
            ),
            (
                self._supertonic(),
                (
                    "duration_predictor.forward",
                    "text_encoder.forward",
                    "vector_estimator.forward",
                    "vocoder.forward",
                ),
            ),
        )
        for runtime, expected in cases:
            with self.subTest(runtime=type(runtime).__name__):
                self._assert_targets(runtime, "inference", expected)

    def test_training_targets_only_name_training_boundaries(self):
        cases = (
            (self._chatterbox(), ()),
            (self._melotts(), ("model.forward", )),
            (self._outetts(), ("language_model.forward", )),
            (self._styletts2(), ()),
            (self._irodori(), ("model.forward", )),
            (self._voxcpm2(), ("forward", )),
            (self._xtts2(), ("gpt.forward", )),
            (self._supertonic(), ("fine_tuning_loss", )),
        )
        for runtime, expected in cases:
            with self.subTest(runtime=type(runtime).__name__):
                self._assert_targets(runtime, "training", expected)

    def test_melotts_public_generate_hits_declared_inference_target(self):
        runtime = self._melotts()
        runtime.device = torch.device("cpu")
        runtime.dtype = torch.float32
        runtime.frontend = SimpleNamespace(
            prepare=lambda **kwargs: SimpleNamespace(
                input_ids=torch.ones(1, 2, dtype=torch.long),
                input_lengths=torch.tensor([2]),
                speaker_ids=torch.tensor([0]),
                tone_ids=torch.zeros(1, 2, dtype=torch.long),
                language_ids=torch.zeros(1, 2, dtype=torch.long),
                bert_features=torch.zeros(1, 1, 2),
                ja_bert_features=torch.zeros(1, 1, 2),
            ), )

        waveform = runtime.generate(
            input_ids=[1, 2],
            tone_ids=[0, 0],
            language_ids=[0, 0],
            bert_features=[[0.0, 0.0]],
            ja_bert_features=[[0.0, 0.0]],
        )

        self.assertEqual(runtime.model.infer_calls, 1)
        self.assertEqual(tuple(waveform.shape), (8, ))

    def test_invalid_modes_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported optimization mode"):
            self._xtts2().optimization_compile_targets("export")

    def test_additional_native_runtime_targets_are_mode_specific(self):
        fish, csm, neutts, bark, speecht5, vui, kokoro = (self._local_runtime_hooks())
        cases = (
            (
                fish,
                (
                    "semantic.forward_generate",
                    "semantic.forward_generate_fast",
                ),
                ("semantic.forward", ),
            ),
            (
                csm,
                ("backbone.forward", "depth_decoder.forward"),
                ("backbone.forward", "depth_decoder.forward"),
            ),
            (
                neutts,
                (),
                ("backbone.forward", ),
            ),
            (
                bark,
                (
                    "semantic.forward",
                    "coarse.forward",
                    "fine.forward",
                ),
                (
                    "semantic.forward",
                    "coarse.forward",
                    "fine.forward",
                ),
            ),
            (
                speecht5,
                (
                    "encoder.forward",
                    "decoder.prenet.forward",
                    "decoder.forward",
                    "postnet.forward",
                ),
                ("acoustic_model.forward", ),
            ),
            (
                vui,
                (),
                ("decoder.forward", ),
            ),
            (
                kokoro,
                ("decoder.forward_with_tokens", ),
                ("decoder.forward_preprocessed", ),
            ),
        )
        for runtime, inference, training in cases:
            with self.subTest(runtime=type(runtime).__name__):
                self._assert_targets(runtime, "inference", inference)
                self._assert_targets(runtime, "training", training)

    def test_neutts_exposes_only_the_training_compile_region(self):
        _, _, neutts, *_ = self._local_runtime_hooks()

        inference = self._assert_targets(
            neutts,
            "inference",
            (),
        )
        training = self._assert_targets(
            neutts,
            "training",
            ("backbone.forward", ),
        )

        self.assertEqual(inference, ())
        self.assertIs(training[0].owner, neutts.backbone)
        self.assertTrue(callable(neutts.codec.decode_code))

    def test_vui_exposes_only_the_training_compile_region(self):
        *_, vui, _ = self._local_runtime_hooks()

        inference = self._assert_targets(
            vui,
            "inference",
            (),
        )
        training = self._assert_targets(
            vui,
            "training",
            ("decoder.forward", ),
        )

        self.assertEqual(inference, ())
        self.assertIs(training[0].owner, vui.decoder)
        self.assertTrue(callable(vui.codec.from_indices))


if __name__ == "__main__":
    unittest.main()
