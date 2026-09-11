from __future__ import annotations

import unittest

from torch import nn

from voicehub.models.conversationtts.source.conversationtts.models.model_new import Model as ConversationTTSModel
from voicehub.models.cosyvoice.native.configuration import CosyVoiceArchitectureConfig
from voicehub.models.cosyvoice.native.modeling import CosyVoiceNativeModel
from voicehub.models.gptsovits.native.runtime import GPTSoVITSRuntime
from voicehub.models.mosstts.native.modeling import (
    MossDelayModel,
    MossLocalV15Model,
    MossOldLocalModel,
    MossRealtimeModel,
)
from voicehub.models.parlertts.native.modeling import ParlerTTSForConditionalGeneration
from voicehub.models.vibevoice.native.modeling import (
    VibeVoiceForConditionalGeneration,
    VibeVoiceRealtimeForConditionalGeneration,
)
from voicehub.models.zonos.native.modeling import ZonosForCausalLM
from voicehub.optimization import OptimizationCompileTarget


def _bare_module(model_class):
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    return model


class TTSCompileTargetProtocolTests(unittest.TestCase):

    def assert_targets(self, model, mode, expected):
        targets = model.optimization_compile_targets(mode)
        self.assertTrue(all(isinstance(target, OptimizationCompileTarget) for target in targets), )
        self.assertEqual(
            tuple((target.label, target.attribute) for target in targets),
            expected,
        )
        self.assertTrue(all(target.owner is model for target in targets))

    def test_conversationtts_targets_generation_and_training(self):
        model = _bare_module(ConversationTTSModel)
        self.assert_targets(
            model,
            "inference",
            (("conversationtts.generate_frame", "generate_frame"), ),
        )
        self.assert_targets(
            model,
            "training",
            (("conversationtts.forward", "forward"), ),
        )

    def test_cosyvoice_targets_repeated_inference_stages(self):
        model = CosyVoiceNativeModel(CosyVoiceArchitectureConfig.tiny())
        inference = model.optimization_compile_targets("inference")
        self.assertEqual(
            tuple((target.label, target.attribute) for target in inference),
            (
                ("cosyvoice.llm.transformer.forward", "forward"),
                ("cosyvoice.flow.estimator.forward", "forward"),
                ("cosyvoice.hift.forward", "forward"),
            ),
        )
        self.assertIs(inference[0].owner, model.llm.llm.model.model)
        self.assertIs(inference[1].owner, model.flow.decoder.estimator)
        self.assertIs(inference[2].owner, model.hift)
        self.assert_targets(
            model,
            "training",
            (("cosyvoice.forward", "forward"), ),
        )

    def test_gptsovits_targets_only_its_inference_graph(self):
        model = _bare_module(GPTSoVITSRuntime)
        self.assert_targets(
            model,
            "inference",
            ((
                "gptsovits.synthesize_prepared",
                "synthesize_prepared",
            ), ),
        )
        self.assert_targets(model, "training", ())

    def test_parlertts_targets_generation_and_training(self):
        model = _bare_module(ParlerTTSForConditionalGeneration)
        self.assert_targets(
            model,
            "inference",
            (("parlertts.generate", "generate"), ),
        )
        self.assert_targets(
            model,
            "training",
            (("parlertts.forward", "forward"), ),
        )

    def test_all_mosstts_variants_share_the_mode_contract(self):
        for model_class in (
                MossDelayModel,
                MossOldLocalModel,
                MossLocalV15Model,
                MossRealtimeModel,
        ):
            with self.subTest(model_class=model_class.__name__):
                model = _bare_module(model_class)
                self.assert_targets(
                    model,
                    "inference",
                    (("mosstts.generate", "generate"), ),
                )
                self.assert_targets(
                    model,
                    "training",
                    (("mosstts.forward", "forward"), ),
                )

    def test_zonos_targets_both_incremental_inference_stages(self):
        model = _bare_module(ZonosForCausalLM)
        self.assert_targets(
            model,
            "inference",
            (
                ("zonos.prefill", "prefill"),
                ("zonos.decode_step", "decode_step"),
            ),
        )
        self.assert_targets(
            model,
            "training",
            (("zonos.forward", "forward"), ),
        )

    def test_vibevoice_fails_closed_for_unsupported_inference(self):
        offline = _bare_module(VibeVoiceForConditionalGeneration)
        self.assert_targets(offline, "inference", ())
        self.assert_targets(
            offline,
            "training",
            (("vibevoice.forward", "forward"), ),
        )

        realtime = _bare_module(VibeVoiceRealtimeForConditionalGeneration, )
        self.assert_targets(realtime, "inference", ())
        self.assert_targets(realtime, "training", ())

    def test_all_hooks_reject_unknown_modes(self):
        model_classes = (
            ConversationTTSModel,
            CosyVoiceNativeModel,
            GPTSoVITSRuntime,
            ParlerTTSForConditionalGeneration,
            MossDelayModel,
            MossOldLocalModel,
            MossLocalV15Model,
            MossRealtimeModel,
            ZonosForCausalLM,
            VibeVoiceForConditionalGeneration,
            VibeVoiceRealtimeForConditionalGeneration,
        )
        for model_class in model_classes:
            with self.subTest(model_class=model_class.__name__):
                with self.assertRaisesRegex(
                        ValueError,
                        "Unsupported optimization mode",
                ):
                    _bare_module(model_class, ).optimization_compile_targets("evaluation")


if __name__ == "__main__":
    unittest.main()
