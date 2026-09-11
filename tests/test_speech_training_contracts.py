import importlib.util
import unittest

from voicehub.outputs import SpeechTrainingOutput, TTSTrainingOutput
from voicehub.registry import ModelSpec, register_model_spec, unregister_model_spec
from voicehub.tasks import SpeechTask
from voicehub.training.adapters import (
    AudioClassificationTrainingAdapter,
    CTCTrainingAdapter,
    FrameClassificationTrainingAdapter,
    RNNTTrainingAdapter,
    SpeechSeq2SeqTrainingAdapter,
    TDTTrainingAdapter,
    UpstreamNativeTrainingAdapter,
)
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.collators import (
    AudioFieldSchema,
    DataCollatorForAudioTraining,
    DataCollatorForTTSTraining,
    TTSFieldSchema,
)
from voicehub.training.contracts import TrainingContext
from voicehub.training.datasets import SpeechDataset
from voicehub.training.specs import (
    ALL_MODEL_TRAINING_SPECS,
    MODEL_TRAINING_SPECS,
    ModelTrainingSpec,
    TrainingFamily,
    get_training_spec,
    list_training_specs,
    register_training_alias,
    register_training_spec,
    unregister_training_spec,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class SpeechTrainingContractTests(unittest.TestCase):

    EXPECTED_ADAPTERS = {
        TrainingFamily.CTC: CTCTrainingAdapter,
        TrainingFamily.SPEECH_SEQ2SEQ: SpeechSeq2SeqTrainingAdapter,
        TrainingFamily.RNNT: RNNTTrainingAdapter,
        TrainingFamily.TDT: TDTTrainingAdapter,
        TrainingFamily.AUDIO_CLASSIFICATION: AudioClassificationTrainingAdapter,
        TrainingFamily.FRAME_CLASSIFICATION: FrameClassificationTrainingAdapter,
        TrainingFamily.NATIVE_ASR_DISPATCH: UpstreamNativeTrainingAdapter,
        TrainingFamily.UPSTREAM_NATIVE: UpstreamNativeTrainingAdapter,
    }

    def test_audio_collator_names_are_canonical_compatibility_aliases(self):
        self.assertIs(TTSFieldSchema, AudioFieldSchema)
        self.assertIs(
            DataCollatorForTTSTraining,
            DataCollatorForAudioTraining,
        )
        schema = AudioFieldSchema(
            sequence_dim=-1,
            length_field="input_lengths",
            mask_field="input_mask",
        )
        collator = DataCollatorForAudioTraining(field_schemas={"input_values": schema}, )
        self.assertIs(collator.field_schemas["input_values"], schema)

    def test_new_family_values_are_stable(self):
        expected = {
            TrainingFamily.CTC: "ctc",
            TrainingFamily.SPEECH_SEQ2SEQ: "speech-sequence-to-sequence",
            TrainingFamily.RNNT: "rnnt",
            TrainingFamily.TDT: "tdt",
            TrainingFamily.AUDIO_CLASSIFICATION: "audio-classification",
            TrainingFamily.FRAME_CLASSIFICATION: "frame-classification",
            TrainingFamily.NATIVE_ASR_DISPATCH: "native-asr-dispatch",
            TrainingFamily.UPSTREAM_NATIVE: "upstream-native",
        }
        self.assertEqual(
            {family: family.value
             for family in expected},
            expected,
        )

    def test_auto_adapter_resolves_every_new_family_without_loading(self):
        for index, (family, adapter_class) in enumerate(self.EXPECTED_ADAPTERS.items()):
            with self.subTest(family=family):
                spec = ModelTrainingSpec(
                    model_type=f"speech-contract-{index}",
                    family=family,
                    task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
                )
                adapter = AutoTrainingAdapter.from_model(
                    object(),
                    spec=spec,
                )
                self.assertIsInstance(adapter, adapter_class)
                self.assertFalse(adapter.is_ready)

    def test_fallback_defaults_are_safe_for_speech_objectives(self):
        native_families = (
            TrainingFamily.CTC,
            TrainingFamily.RNNT,
            TrainingFamily.TDT,
            TrainingFamily.NATIVE_ASR_DISPATCH,
            TrainingFamily.UPSTREAM_NATIVE,
        )
        for index, family in enumerate(native_families):
            with self.subTest(family=family):
                spec = ModelTrainingSpec(
                    model_type=f"native-contract-{index}",
                    family=family,
                )
                self.assertIsNone(spec.fallback_objective)
                self.assertIsNone(spec.get_phase().fallback_objective)

        speech_seq2seq = ModelTrainingSpec(
            model_type="speech-seq2seq-contract",
            family=TrainingFamily.SPEECH_SEQ2SEQ,
        )
        self.assertEqual(
            speech_seq2seq.fallback_objective,
            "cross_entropy",
        )
        for family in (
                TrainingFamily.AUDIO_CLASSIFICATION,
                TrainingFamily.FRAME_CLASSIFICATION,
        ):
            spec = ModelTrainingSpec(
                model_type=f"{family.value}-contract",
                family=family,
            )
            self.assertEqual(spec.fallback_objective, "classification")

    def test_single_optimizer_composite_profiles_are_not_multi_optimizer_recipes(self):
        for model_type in ("vibevoice", "vad_pyannote_brouhaha"):
            with self.subTest(model_type=model_type):
                spec = get_training_spec(model_type)
                self.assertFalse(spec.separate_optimizers)
                self.assertEqual(
                    spec.get_phase().optimizer_names,
                    ("model", ),
                )

    def test_task_registry_keeps_legacy_tts_view_stable(self):
        spec = ModelTrainingSpec(
            model_type="speech-task-registry-contract",
            family=TrainingFamily.CTC,
            task="asr",
        )
        legacy_keys = tuple(MODEL_TRAINING_SPECS)
        register_training_spec(spec)
        try:
            self.assertEqual(tuple(MODEL_TRAINING_SPECS), legacy_keys)
            self.assertNotIn(spec.model_type, MODEL_TRAINING_SPECS)
            self.assertIs(ALL_MODEL_TRAINING_SPECS[spec.model_type], spec)
            self.assertIs(get_training_spec(spec.model_type), spec)
            self.assertIn(
                spec.model_type,
                AutoTrainingAdapter.available_models(),
            )
            self.assertIn(
                spec,
                list_training_specs(task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            )
            self.assertIn(spec, list_training_specs(task=None))
            self.assertNotIn(spec, list_training_specs())
        finally:
            unregister_training_spec(spec.model_type, missing_ok=True)

    def test_training_registration_rejects_cross_task_profiles(self):
        model_type = "speech-training-task-mismatch"
        register_model_spec(
            ModelSpec(
                model_type=model_type,
                module="tests._unused_speech_backend",
                class_name="_UnusedSpeechBackend",
                default_model_path="",
                install_extra="training",
                task=SpeechTask.VOICE_ACTIVITY_DETECTION,
            ))
        try:
            profile = ModelTrainingSpec(
                model_type=model_type,
                family=TrainingFamily.CTC,
                task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
            )
            with self.assertRaisesRegex(ValueError, "inference backend"):
                register_training_spec(profile)
            self.assertNotIn(model_type, ALL_MODEL_TRAINING_SPECS)
        finally:
            unregister_training_spec(model_type, missing_ok=True)
            unregister_model_spec(model_type, missing_ok=True)

    def test_training_aliases_cannot_shadow_canonical_model_types(self):
        inference_type = "speech-training-canonical-collision"
        profile_type = "speech-training-alias-target"
        register_model_spec(
            ModelSpec(
                model_type=inference_type,
                module="tests._unused_speech_backend",
                class_name="_UnusedSpeechBackend",
                default_model_path="",
                install_extra="training",
                task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
            ))
        register_training_spec(
            ModelTrainingSpec(
                model_type=profile_type,
                family=TrainingFamily.CTC,
                task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
            ))
        try:
            with self.assertRaisesRegex(
                    ValueError,
                    "registered inference model type",
            ):
                register_training_alias(inference_type, profile_type)
            with self.assertRaises(KeyError):
                get_training_spec(inference_type)
        finally:
            unregister_training_spec(profile_type, missing_ok=True)
            unregister_model_spec(inference_type, missing_ok=True)

    def test_training_model_types_cannot_shadow_existing_aliases(self):
        target = ModelTrainingSpec(
            model_type="speech-training-existing-alias-target",
            family=TrainingFamily.CTC,
        )
        register_training_spec(
            target,
            aliases=("speech-training-reserved-model-type", ),
        )
        try:
            conflicting = ModelTrainingSpec(
                model_type="speech-training-reserved-model-type",
                family=TrainingFamily.CTC,
            )
            with self.assertRaisesRegex(ValueError, "collides with an alias"):
                register_training_spec(conflicting)
            self.assertIs(
                get_training_spec("speech-training-reserved-model-type"),
                target,
            )
        finally:
            unregister_training_spec(target.model_type, missing_ok=True)

    def test_training_alias_cannot_repeat_its_canonical_model_type(self):
        profile = ModelTrainingSpec(
            model_type="speech-training-self-alias",
            family=TrainingFamily.CTC,
        )
        with self.assertRaisesRegex(ValueError, "identical"):
            register_training_spec(
                profile,
                aliases=(profile.model_type.upper(), ),
            )
        self.assertNotIn(profile.model_type, ALL_MODEL_TRAINING_SPECS)

    def test_builtin_speech_models_have_task_aware_training_profiles(self):
        from voicehub.registry import list_model_specs

        for task in (
                SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
                SpeechTask.VOICE_ACTIVITY_DETECTION,
        ):
            models = {spec.model_type for spec in list_model_specs(task=task)}
            profiles = {spec.model_type for spec in list_training_specs(task=task)}
            self.assertEqual(profiles, models)

    def test_transformers_providers_use_declarative_model_adapters(self):
        from voicehub.models.asr_transformers import TransformersASRConfig, TransformersASRForSpeechRecognition
        from voicehub.models.asr_transformers.training_asr_transformers import TransformersASRTrainingAdapter
        from voicehub.models.vad_transformers import TransformersVADConfig, TransformersVADForVoiceActivityDetection
        from voicehub.models.vad_transformers.training_vad_transformers import TransformersVADTrainingAdapter

        cases = (
            (
                TransformersASRForSpeechRecognition(TransformersASRConfig(architecture_family="ctc")),
                TransformersASRTrainingAdapter,
            ),
            (
                TransformersVADForVoiceActivityDetection(
                    TransformersVADConfig(architecture_family="frame-classification")),
                TransformersVADTrainingAdapter,
            ),
        )
        for model, expected_type in cases:
            with self.subTest(model=model.config.model_type):
                adapter = model.get_training_adapter()
                self.assertIsInstance(adapter, expected_type)
                self.assertFalse(model.is_loaded)

    def test_turnkey_adapter_builds_a_dependency_light_speech_dataset(self):
        from voicehub.models.asr_transformers import TransformersASRConfig, TransformersASRForSpeechRecognition

        model = TransformersASRForSpeechRecognition(TransformersASRConfig(architecture_family="ctc"))
        dataset = model.create_training_dataset(
            [{
                "audio": [0.0, 0.1],
                "sampling_rate": 16_000,
                "text": "hello",
            }],
            required_fields=("audio", "text"),
        )

        self.assertIsInstance(dataset, SpeechDataset)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            dataset.column_names,
            ("audio", "sampling_rate", "text"),
        )
        copied = dataset[0]
        copied["text"] = "changed"
        self.assertEqual(dataset[0]["text"], "hello")

    def test_speech_dataset_copies_nested_metadata_and_rejects_empty_required_values(self):
        source = [{
            "text": "hello",
            "metadata": {
                "speaker": "one",
                "tags": ["clean"],
            },
        }]
        dataset = SpeechDataset(source, required_fields=("text", ))
        source[0]["metadata"]["tags"].append("changed")
        materialized = dataset[0]
        materialized["metadata"]["tags"].append("also-changed")

        self.assertEqual(dataset[0]["metadata"]["tags"], ["clean"])
        with self.assertRaisesRegex(ValueError, "missing required"):
            SpeechDataset(
                [{
                    "text": "   "
                }],
                required_fields=("text", ),
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class SpeechTrainingTorchContractTests(unittest.TestCase):

    @staticmethod
    def _adapter(family):
        spec = ModelTrainingSpec(
            model_type=f"torch-contract-{family.value}",
            family=family,
        )
        return AutoTrainingAdapter.from_model(object(), spec=spec)

    def test_audio_collator_pads_waveforms_and_emits_explicit_mask(self):
        import torch

        collator = DataCollatorForAudioTraining(
            field_schemas={
                "input_values":
                AudioFieldSchema(
                    sequence_dim=-1,
                    length_field="input_lengths",
                    mask_field="input_mask",
                    pad_to_multiple_of=4,
                ),
            }, )
        batch = collator([
            {
                "input_values": torch.tensor([1.0, 2.0, 3.0]),
            },
            {
                "input_values": torch.tensor([4.0]),
            },
        ])

        self.assertEqual(tuple(batch["input_values"].shape), (2, 4))
        self.assertEqual(batch["input_lengths"].tolist(), [3, 1])
        self.assertEqual(
            batch["input_mask"].tolist(),
            [[True, True, True, False], [True, False, False, False]],
        )

    def test_audio_classification_uses_cross_entropy_fallback(self):
        import torch

        adapter = self._adapter(TrainingFamily.AUDIO_CLASSIFICATION)
        logits = torch.tensor([[3.0, -1.0], [-2.0, 4.0]])
        labels = torch.tensor([0, 1])

        actual = adapter.compute_objective(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits, labels)
        self.assertTrue(torch.allclose(actual, expected))

    def test_frame_binary_classification_excludes_masked_padding(self):
        import torch

        adapter = self._adapter(TrainingFamily.FRAME_CLASSIFICATION)
        logits = torch.tensor([
            [0.0, 2.0, 80.0],
            [-2.0, 0.0, -80.0],
        ])
        labels = torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ])
        frame_mask = torch.tensor([
            [True, True, False],
            [True, True, False],
        ])
        context = TrainingContext(
            phase=adapter.spec.get_phase(),
            inputs={
                "labels": labels,
                "frame_mask": frame_mask,
            },
        )

        actual = adapter.compute_phase_objective(
            logits,
            labels,
            context,
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[frame_mask],
            labels[frame_mask],
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_frame_binary_classification_ignores_collator_padding(self):
        import torch

        adapter = self._adapter(TrainingFamily.FRAME_CLASSIFICATION)
        logits = torch.tensor([
            [0.0, 2.0, 80.0],
            [-2.0, 0.0, -80.0],
        ])
        labels = torch.tensor([
            [0.0, 1.0, -100.0],
            [0.0, 1.0, -100.0],
        ])

        actual = adapter.compute_objective(logits, labels)
        valid = labels.ne(-100)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[valid],
            labels[valid],
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_frame_cross_entropy_combines_mask_and_ignore_index(self):
        import torch

        adapter = self._adapter(TrainingFamily.FRAME_CLASSIFICATION)
        logits = torch.tensor([
            [[3.0, 0.0], [0.0, 3.0], [20.0, -20.0]],
            [[0.0, 3.0], [3.0, 0.0], [-20.0, 20.0]],
        ])
        labels = torch.tensor([
            [0, 1, -100],
            [1, 0, 0],
        ])
        label_mask = torch.tensor([
            [True, True, True],
            [True, False, False],
        ])
        context = TrainingContext(
            phase=adapter.spec.get_phase(),
            inputs={
                "labels": labels,
                "label_mask": label_mask,
            },
        )

        actual = adapter.compute_phase_objective(
            logits,
            labels,
            context,
        )
        valid = label_mask & labels.ne(-100)
        expected = torch.nn.functional.cross_entropy(
            logits[valid],
            labels[valid],
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_transducer_families_never_invent_generic_objectives(self):
        import torch

        for family in (
                TrainingFamily.CTC,
                TrainingFamily.RNNT,
                TrainingFamily.TDT,
                TrainingFamily.UPSTREAM_NATIVE,
        ):
            with self.subTest(family=family):
                adapter = self._adapter(family)
                with self.assertRaisesRegex(ValueError, "backend-native loss"):
                    adapter.compute_objective(
                        torch.randn(2, 3),
                        torch.zeros(2, dtype=torch.long),
                    )

    def test_native_loss_takes_precedence_for_ctc_family(self):
        import torch

        class NativeCTCModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, input_values, labels):
                logits = input_values * self.scale
                return {
                    "loss": (logits - labels).square().mean(),
                    "logits": logits,
                }

        class Wrapper:

            def __init__(self):
                self.model = None

            def load_for_training(self):
                self.model = NativeCTCModel()

        spec = ModelTrainingSpec(
            model_type="native-ctc-loss-contract",
            family=TrainingFamily.CTC,
            module_paths=("model", ),
            task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        )
        adapter = AutoTrainingAdapter.from_model(
            Wrapper(),
            spec=spec,
        )
        input_values = torch.tensor([[1.0, 2.0]])
        labels = torch.tensor([[0.0, 0.0]])

        output = adapter(
            input_values=input_values,
            labels=labels,
        )

        self.assertTrue(torch.allclose(output.loss, torch.tensor(2.5)))
        self.assertIs(type(output), SpeechTrainingOutput)
        self.assertNotIsInstance(output, TTSTrainingOutput)


if __name__ == "__main__":
    unittest.main()
