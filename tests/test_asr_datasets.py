import ast
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import voicehub.training.asr_data_contracts as asr_data_contracts_module
import voicehub.training.asr_datasets as asr_datasets_module
from voicehub import ASROutput, PreTrainedASRModel, SpeechTask, VoiceHubConfig
from voicehub.training import (
    ASRDataArchitecture,
    ASRDataReadiness,
    ASRDataset,
    ASRDatasetSpec,
    ASRRecordVariant,
    EpochGroupedBatchSampler,
    get_asr_dataset_spec,
    get_training_spec,
    list_asr_dataset_specs,
    list_training_specs,
)
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily, register_training_spec, unregister_training_spec


def _normalize_extension_record(record, *, index):
    value = dict(record)
    value["language"] = value.get("language", "").strip().lower()
    value["normalizer_index"] = index
    return value


def _return_invalid_record(record, *, index):
    del record, index
    return None


def _build_extension_asr_dataset_spec():
    return ASRDatasetSpec(
        architecture=ASRDataArchitecture.CTC,
        variants=(
            ASRRecordVariant(
                name="extension-raw",
                required_fields=("audio", "text"),
            ),
            ASRRecordVariant(
                name="extension-ready",
                required_fields=("input_values", "labels"),
                preprocessed=True,
            ),
        ),
        sample_rate=22_050,
        description="Extension-owned ASR dataset contract.",
    )


def _return_invalid_asr_dataset_spec():
    return {"architecture": "ctc"}


class ASRDatasetContractTests(unittest.TestCase):

    def test_training_profiles_select_dataset_specs_without_a_provider_map(self):
        training_specs = list_training_specs(task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, )
        self.assertTrue(all(spec.dataset_spec_factory for spec in training_specs))

        source_path = Path(asr_data_contracts_module.__file__)
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn("_MODEL_DATA_OVERRIDES", source)
        tree = ast.parse(source)
        model_types = {spec.model_type for spec in training_specs}
        provider_keyed_dicts = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                key.value
                for key in node.keys if isinstance(key, ast.Constant) and key.value in model_types
            }
            if keys:
                provider_keyed_dicts.append((node.lineno, sorted(keys)))
        self.assertEqual(provider_keyed_dicts, [])

    def test_extension_dataset_spec_factory_needs_no_shared_provider_edit(self):
        model_type = "future-asr-dataset-contract"
        training_spec = ModelTrainingSpec(
            model_type=model_type,
            family=TrainingFamily.CTC,
            task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
            support=TrainingSupport.NATIVE,
            dataset_spec_factory=f"{__name__}:_build_extension_asr_dataset_spec",
        )
        register_training_spec(training_spec)
        try:
            dataset_spec = get_asr_dataset_spec(model_type)
        finally:
            unregister_training_spec(model_type, missing_ok=True)

        self.assertEqual(dataset_spec.model_type, model_type)
        self.assertEqual(dataset_spec.sample_rate, 22_050)
        self.assertEqual(
            tuple(variant.name for variant in dataset_spec.variants),
            ("extension-raw", "extension-ready"),
        )
        self.assertIs(dataset_spec.readiness, ASRDataReadiness.INTEGRATED)
        self.assertEqual(dataset_spec.training_support, "native")

    def test_dataset_spec_factory_resolution_fails_actionably(self):
        model_type = "future-missing-asr-dataset-contract"
        training_spec = ModelTrainingSpec(
            model_type=model_type,
            family=TrainingFamily.CTC,
            task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
            dataset_spec_factory=f"{__name__}:missing_asr_dataset_spec_factory",
        )
        register_training_spec(training_spec)
        try:
            with self.assertRaisesRegex(
                    ImportError,
                    "Could not resolve ASR dataset spec factory.*missing_asr_dataset_spec_factory",
            ):
                get_asr_dataset_spec(model_type)
        finally:
            unregister_training_spec(model_type, missing_ok=True)

    def test_dataset_spec_factory_protocol_fails_actionably(self):
        cases = (
            (
                "future-non-callable-asr-dataset-contract",
                "voicehub.training.asr_data_contracts:_TRANSCRIPT_FIELDS",
                "must be callable",
            ),
            (
                "future-invalid-asr-dataset-contract",
                f"{__name__}:_return_invalid_asr_dataset_spec",
                "returned dict; expected ASRDatasetSpec",
            ),
        )
        for model_type, factory_path, message in cases:
            with self.subTest(model_type=model_type):
                training_spec = ModelTrainingSpec(
                    model_type=model_type,
                    family=TrainingFamily.CTC,
                    task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
                    dataset_spec_factory=factory_path,
                )
                register_training_spec(training_spec)
                try:
                    with self.assertRaisesRegex(TypeError, message):
                        get_asr_dataset_spec(model_type)
                finally:
                    unregister_training_spec(model_type, missing_ok=True)

    def test_dataset_spec_factory_keeps_framework_imports_lazy(self):
        code = "\nimport json\nimport sys\nfrom voicehub.training import get_asr_dataset_spec\nget_asr_dataset_spec('asr_qwen3')\nprint(json.dumps({name: name in sys.modules for name in ('torch', 'transformers')}))\n"
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(result.stdout),
            {
                "torch": False,
                "transformers": False
            },
        )

    def test_model_specific_normalization_is_declared_by_dataset_specs(self):
        sensevoice = get_asr_dataset_spec("asr_funasr")
        seamless = get_asr_dataset_spec("asr_seamless_m4t_v2")

        self.assertEqual(
            dict(sensevoice.field_aliases),
            {
                "emo_target": "emotion",
                "event_target": "event",
                "source": "audio",
                "target": "text",
                "text_language": "language",
                "with_or_wo_itn": "use_itn",
            },
        )
        self.assertEqual(
            sensevoice.record_normalizer,
            "voicehub.models.asr_funasr.native.data:normalize_record",
        )
        self.assertEqual(sensevoice.record_normalizer_phase, "after-aliases")
        self.assertEqual(
            dict(seamless.field_aliases),
            {"target_lang": "target_language"},
        )
        self.assertEqual(
            seamless.record_normalizer,
            "voicehub.models.asr_seamless_m4t_v2.native.data:normalize_record",
        )
        self.assertEqual(seamless.record_normalizer_phase, "before-aliases")

    def test_dataset_spec_normalizer_metadata_validates_fail_closed(self):
        values = {
            "architecture": ASRDataArchitecture.CTC,
            "variants": (ASRRecordVariant(name="raw", required_fields=("audio", "text")), ),
        }
        with self.assertRaisesRegex(ValueError, "module:attribute"):
            ASRDatasetSpec(**values, record_normalizer="not-an-import-path")
        with self.assertRaisesRegex(ValueError, "record_normalizer_phase"):
            ASRDatasetSpec(**values, record_normalizer_phase="before-aliases")
        with self.assertRaisesRegex(ValueError, "repeats source"):
            ASRDatasetSpec(
                **values,
                field_aliases=(("recording", "audio"), ("recording", "input_values")),
            )

    def test_extension_normalizer_uses_the_shared_pipeline(self):
        spec = ASRDatasetSpec(
            architecture=ASRDataArchitecture.CTC,
            variants=(ASRRecordVariant(name="raw", required_fields=("audio", "text")), ),
            model_type="future-asr",
            field_aliases={
                "recording": "audio",
                "transcript": "text",
            },
            record_normalizer=f"{__name__}:_normalize_extension_record",
        )
        with mock.patch.object(
                asr_datasets_module,
                "get_asr_dataset_spec",
                return_value=spec,
        ):
            dataset = ASRDataset(
                [{
                    "recording": "sample.wav",
                    "transcript": "Hello.",
                    "language": " EN ",
                }],
                model_type=spec.model_type,
            )

        self.assertEqual(dataset[0]["audio"], "sample.wav")
        self.assertEqual(dataset[0]["text"], "Hello.")
        self.assertEqual(dataset[0]["language"], "en")
        self.assertEqual(dataset[0]["normalizer_index"], 0)

    def test_normalizer_resolution_and_output_fail_actionably(self):
        variant = ASRRecordVariant(name="raw", required_fields=("audio", "text"))
        missing = ASRDatasetSpec(
            architecture=ASRDataArchitecture.CTC,
            variants=(variant, ),
            model_type="missing-normalizer",
            record_normalizer=f"{__name__}:missing_record_normalizer",
        )
        invalid = ASRDatasetSpec(
            architecture=ASRDataArchitecture.CTC,
            variants=(variant, ),
            model_type="invalid-normalizer",
            record_normalizer=f"{__name__}:_return_invalid_record",
        )
        record = {"audio": "sample.wav", "text": "Hello."}

        with mock.patch.object(
                asr_datasets_module,
                "get_asr_dataset_spec",
                return_value=missing,
        ):
            with self.assertRaisesRegex(ImportError, "missing_record_normalizer"):
                ASRDataset([record], model_type=missing.model_type)
        with mock.patch.object(
                asr_datasets_module,
                "get_asr_dataset_spec",
                return_value=invalid,
        ):
            with self.assertRaisesRegex(TypeError, "expected a mapping"):
                ASRDataset([record], model_type=invalid.model_type)

    def test_dataset_contract_listing_keeps_normalizer_modules_lazy(self):
        code = "\nimport json\nimport sys\nfrom voicehub.training import list_asr_dataset_specs\nlist_asr_dataset_specs()\nprint(json.dumps(sorted(\n    name for name in sys.modules\n    if name in {\n        'voicehub.models.asr_funasr.native.data',\n        'voicehub.models.asr_seamless_m4t_v2.native.data',\n    }\n)))\n"
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(json.loads(result.stdout), [])

    def test_shared_dataset_pipeline_does_not_branch_on_model_names(self):
        source_path = Path(asr_datasets_module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        model_types = {
            spec.model_type
            for spec in list_training_specs(task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, )
        }
        violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            matched = sorted({
                value.value
                for value in ast.walk(node) if isinstance(value, ast.Constant) and value.value in model_types
            })
            if matched:
                violations.append((node.lineno, matched))

        self.assertEqual(violations, [])

    def test_every_asr_profile_has_a_raw_finetuning_contract(self):
        training_specs = list_training_specs(task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, )
        dataset_specs = list_asr_dataset_specs()

        self.assertEqual(len(training_specs), 23)
        self.assertEqual(
            [spec.model_type for spec in dataset_specs],
            [spec.model_type for spec in training_specs],
        )
        for training_spec in training_specs:
            with self.subTest(model_type=training_spec.model_type):
                dataset_spec = get_asr_dataset_spec(training_spec.model_type)
                self.assertEqual(
                    training_spec.dataset_spec,
                    dataset_spec,
                )
                self.assertIs(dataset_spec.readiness, ASRDataReadiness.INTEGRATED)
                self.assertTrue(dataset_spec.accepts_raw_records)
                self.assertTrue(dataset_spec.preprocessed_variants)

                record = {
                    "audio": "sample.wav",
                    "text": "A fine-tuning transcript.",
                }
                if training_spec.model_type in {"asr_cohere", "asr_funasr"}:
                    record["language"] = "en"
                dataset = ASRDataset(
                    [record],
                    model_type=training_spec.model_type,
                )
                self.assertEqual(dataset.model_type, training_spec.model_type)
                self.assertNotEqual(dataset.variant_names, ("unchecked", ))

    def test_contract_primitives_are_public_and_validate_dependencies(self):
        variant = ASRRecordVariant(
            name="conditioned",
            required_fields=("audio", "text"),
            requires=(("prompt", ("language", )), ),
        )
        spec = ASRDatasetSpec(
            architecture=ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
            variants=(variant, ),
            sample_rate=16_000,
        )

        self.assertTrue(variant.matches({
            "audio": "sample.wav",
            "text": "Hello.",
        }))
        self.assertEqual(
            variant.missing({
                "audio": "sample.wav",
                "text": "Hello.",
                "prompt": "Transcribe",
            }),
            ("prompt requires language", ),
        )
        self.assertEqual(
            spec.match_variant({
                "audio": "sample.wav",
                "text": "Hello.",
            }),
            "conditioned",
        )
        self.assertIs(
            ASRDataArchitecture.coerce("seq2seq"),
            ASRDataArchitecture.SPEECH_SEQUENCE_TO_SEQUENCE,
        )

    def test_preprocessed_variant_wins_when_source_fields_are_retained(self):
        spec = ASRDatasetSpec(
            architecture="seq2seq",
            variants=(
                ASRRecordVariant(
                    name="raw",
                    required_fields=("audio", "text"),
                ),
                ASRRecordVariant(
                    name="model-ready",
                    required_fields=("input_features", "labels"),
                    preprocessed=True,
                ),
            ),
        )

        self.assertEqual(
            spec.match_variant({
                "audio": "trace.wav",
                "text": "Trace transcript.",
                "input_features": [[0.0]],
                "labels": [1],
            }),
            "model-ready",
        )

    def test_qwen_vibe_and_granite_have_exact_multimodal_variants(self):
        cases = (
            (
                "asr_qwen3",
                {
                    "audio": "qwen.wav",
                    "text": "Qwen transcript.",
                    "context": "Meeting notes",
                },
                "raw-audio",
            ),
            (
                "asr_qwen3",
                {
                    "input_ids": [1, 2],
                    "attention_mask": [1, 1],
                    "input_features": [[0.1]],
                    "feature_attention_mask": [1],
                    "labels": [-100, 2],
                },
                "qwen3-model-ready",
            ),
            (
                "asr_vibevoice",
                {
                    "audio":
                    "vibe.wav",
                    "segments": [{
                        "speaker": "Speaker 1",
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Structured transcript.",
                    }],
                },
                "segmented-audio",
            ),
            (
                "asr_vibevoice",
                {
                    "audio":
                    "vibe.wav",
                    "text":
                    ('[{"speaker":"Speaker 1","start":0.0,'
                     '"end":1.0,"text":"Serialized transcript."}]'),
                },
                "serialized-audio",
            ),
            (
                "asr_vibevoice",
                {
                    "input_ids": [1],
                    "attention_mask": [1],
                    "input_values": [0.0, 0.1],
                    "padding_mask": [1, 1],
                    "labels": [1],
                },
                "vibevoice-model-ready",
            ),
            (
                "asr_granite_speech",
                {
                    "audio": "granite.wav",
                    "text": "Granite transcript.",
                    "prompt": "Transcribe in English.",
                },
                "raw-audio",
            ),
            (
                "asr_granite_speech",
                {
                    "input_ids": [1],
                    "attention_mask": [1],
                    "input_features": [[0.1]],
                    "input_features_mask": [1],
                    "labels": [1],
                },
                "granite-model-ready",
            ),
        )
        for model_type, record, expected_variant in cases:
            with self.subTest(
                    model_type=model_type,
                    expected_variant=expected_variant,
            ):
                dataset = ASRDataset([record], model_type=model_type)
                self.assertEqual(dataset.variant_names, (expected_variant, ))
                self.assertIs(
                    dataset.architecture,
                    ASRDataArchitecture.PROMPTED_MULTIMODAL,
                )

        self.assertEqual(get_asr_dataset_spec("asr_vibevoice").sample_rate, 24_000)
        with self.assertRaisesRegex(ValueError, "forbidden field language"):
            ASRDataset(
                [{
                    "audio": "granite.wav",
                    "text": "Put language guidance in the prompt.",
                    "language": "en",
                }],
                model_type="asr_granite_speech",
            )
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            ASRDataset(
                [{
                    "audio": "vibe.wav",
                    "segments": [{
                        "text": "Structured",
                    }],
                    "text": "Ambiguous serialized target.",
                }],
                model_type="asr_vibevoice",
            )

    def test_rnnt_tdt_and_native_framework_variants_are_exact(self):
        cases = (
            (
                "asr_nemotron",
                ASRDataArchitecture.RNNT,
                {
                    "input_features": [[0.1]],
                    "attention_mask": [1],
                    "prompt_ids": [1],
                    "labels": [2],
                    "label_lengths": 1,
                    "decoder_input_ids": [0, 2],
                },
                "nemotron-rnnt-model-ready",
            ),
            (
                "asr_parakeet_tdt",
                ASRDataArchitecture.TDT,
                {
                    "input_features": [[0.1]],
                    "attention_mask": [1],
                    "labels": [2],
                    "decoder_input_ids": [0, 2],
                },
                "parakeet-tdt-model-ready",
            ),
            (
                "asr_nemo",
                ASRDataArchitecture.CTC,
                {
                    "input_signal": [[0.0, 0.1]],
                    "input_signal_length": [2],
                    "labels": [[1]],
                    "label_lengths": [1],
                },
                "nemo-ctc-waveform-model-ready",
            ),
            (
                "asr_nemo",
                ASRDataArchitecture.CTC,
                {
                    "processed_signal": [[[0.1]]],
                    "processed_signal_length": [1],
                    "labels": [[1]],
                    "label_lengths": [1],
                },
                "nemo-ctc-feature-model-ready",
            ),
            (
                "asr_speechbrain",
                ASRDataArchitecture.HYBRID_CTC_ATTENTION,
                {
                    "waveforms": [[0.0, 0.1]],
                    "waveform_lengths": [2],
                    "tokens_bos": [[1, 2]],
                    "tokens_eos": [[2, 3]],
                    "token_lengths": [2],
                    "ctc_tokens": [[2]],
                    "ctc_token_lengths": [1],
                },
                "speechbrain-model-ready",
            ),
            (
                "asr_funasr",
                ASRDataArchitecture.CTC,
                {
                    "audio_values": [0.0, 0.1],
                    "text": "SenseVoice transcript.",
                    "language": "en",
                    "emotion": "neutral",
                    "event": "speech",
                    "use_itn": True,
                },
                "raw-audio",
            ),
            (
                "asr_funasr",
                ASRDataArchitecture.CTC,
                {
                    "features": [[0.1]],
                    "transcript": "Feature transcript.",
                    "language": "en",
                },
                "sensevoice-feature-transcript",
            ),
            (
                "asr_funasr",
                ASRDataArchitecture.CTC,
                {
                    "features": [[0.1]],
                    "feature_lengths": [1],
                    "labels": [[1]],
                    "label_lengths": [1],
                },
                "sensevoice-model-ready",
            ),
            (
                "asr_espnet",
                ASRDataArchitecture.HYBRID_CTC_ATTENTION,
                {
                    "features": [[0.1]],
                    "text": "ESPnet feature transcript.",
                },
                "espnet-feature-transcript",
            ),
            (
                "asr_espnet",
                ASRDataArchitecture.HYBRID_CTC_ATTENTION,
                {
                    "waveforms": [[0.0, 0.1]],
                    "waveform_lengths": [2],
                    "labels": [[1]],
                    "label_lengths": [1],
                },
                "espnet-waveform-model-ready",
            ),
            (
                "asr_espnet",
                ASRDataArchitecture.HYBRID_CTC_ATTENTION,
                {
                    "features": [[0.1]],
                    "feature_lengths": [1],
                    "labels": [[1]],
                    "label_lengths": [1],
                },
                "espnet-feature-model-ready",
            ),
            (
                "asr_wenet",
                ASRDataArchitecture.HYBRID_CTC_ATTENTION,
                {
                    "input_signal": [[0.0, 0.1]],
                    "input_signal_length": [2],
                    "labels": [[1]],
                    "label_lengths": [1],
                },
                "wenet-waveform-model-ready",
            ),
            (
                "asr_wenet",
                ASRDataArchitecture.HYBRID_CTC_ATTENTION,
                {
                    "features": [[[0.1]]],
                    "feature_lengths": [1],
                    "labels": [[1]],
                    "label_lengths": [1],
                },
                "wenet-feature-model-ready",
            ),
        )
        for model_type, architecture, record, expected_variant in cases:
            with self.subTest(
                    model_type=model_type,
                    expected_variant=expected_variant,
            ):
                dataset = ASRDataset([record], model_type=model_type)
                self.assertIs(dataset.architecture, architecture)
                self.assertEqual(dataset.variant_names, (expected_variant, ))

    def test_special_contracts_reject_incomplete_model_ready_records(self):
        invalid_records = (
            (
                "asr_qwen3",
                {
                    "input_ids": [1],
                    "attention_mask": [1],
                    "input_features": [[0.1]],
                    "labels": [1],
                },
            ),
            (
                "asr_nemotron",
                {
                    "input_features": [[0.1]],
                    "attention_mask": [1],
                    "labels": [1],
                    "decoder_input_ids": [0, 1],
                },
            ),
            (
                "asr_parakeet_tdt",
                {
                    "input_features": [[0.1]],
                    "labels": [1],
                    "decoder_input_ids": [0, 1],
                },
            ),
            (
                "asr_speechbrain",
                {
                    "waveforms": [[0.0]],
                    "labels": [[1]],
                },
            ),
            (
                "asr_wenet",
                {
                    "input_signal": [[0.0]],
                    "labels": [[1]],
                },
            ),
        )
        for model_type, record in invalid_records:
            with self.subTest(model_type=model_type):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    ASRDataset([record], model_type=model_type)


class ASRDatasetManifestTests(unittest.TestCase):

    def test_empty_audio_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "audio.*non-empty"):
            ASRDataset(
                [{
                    "audio": " ",
                    "text": "No source audio.",
                }],
                architecture="ctc",
            )

    def test_jsonl_aliases_and_relative_paths_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "clips" / "sample.wav"
            audio.parent.mkdir()
            audio.touch()
            manifest = root / "records.jsonl"
            manifest.write_text(
                json.dumps({
                    "id": "sample",
                    "audio_filepath": "clips/sample.wav",
                    "transcription": "Hello from JSON Lines.",
                    "lang": "en",
                    "sample_rate": 16_000,
                }) + "\n",
                encoding="utf-8",
            )

            dataset = ASRDataset.from_manifest(
                manifest,
                model_type="asr_cohere",
                validate_files=True,
            )

        self.assertEqual(dataset.variant_names, ("raw-audio", ))
        self.assertEqual(dataset[0]["audio"], str(audio.resolve()))
        self.assertEqual(dataset[0]["text"], "Hello from JSON Lines.")
        self.assertEqual(dataset[0]["language"], "en")
        self.assertEqual(dataset[0]["sampling_rate"], 16_000)
        self.assertNotIn("audio_filepath", dataset[0])
        self.assertNotIn("transcription", dataset[0])

    def test_json_objects_lists_and_nemo_json_lines_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_manifest = root / "object.json"
            object_manifest.write_text(
                json.dumps({
                    "audio": "one.wav",
                    "text": "One.",
                }),
                encoding="utf-8",
            )
            list_manifest = root / "list.json"
            list_manifest.write_text(
                json.dumps([
                    {
                        "audio": "one.wav",
                        "text": "One.",
                    },
                    {
                        "audio": "two.wav",
                        "text": "Two.",
                    },
                ]),
                encoding="utf-8",
            )
            nemo_manifest = root / "manifest.json"
            nemo_manifest.write_text(
                "\n".join(
                    json.dumps(record) for record in (
                        {
                            "audio_filepath": "one.wav",
                            "text": "One.",
                            "duration": 1.0,
                        },
                        {
                            "audio_filepath": "two.wav",
                            "text": "Two.",
                            "duration": 2.0,
                        },
                    )) + "\n",
                encoding="utf-8",
            )

            object_dataset = ASRDataset.from_manifest(
                object_manifest,
                model_type="asr_nemo",
            )
            list_dataset = ASRDataset.from_manifest(
                list_manifest,
                model_type="asr_nemo",
            )
            nemo_dataset = ASRDataset.from_manifest(
                nemo_manifest,
                model_type="asr_nemo",
            )

        self.assertEqual(len(object_dataset), 1)
        self.assertEqual(len(list_dataset), 2)
        self.assertEqual(len(nemo_dataset), 2)
        self.assertEqual(nemo_dataset[1]["duration"], 2.0)
        self.assertEqual(
            nemo_dataset[0]["audio"],
            str((root / "one.wav").resolve()),
        )

    def test_csv_and_tsv_aliases_resolve_paths_and_parse_json_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_manifest = root / "records.csv"
            csv_manifest.write_text(
                "wav_path,sentence,lang,punctuation\n"
                'english.wav,English sentence.,en,false\n',
                encoding="utf-8",
            )
            tsv_manifest = root / "records.tsv"
            tsv_manifest.write_text(
                "file\ttarget_text\ttarget_lang\n"
                "german.wav\tDeutscher Satz.\tdeu\n",
                encoding="utf-8",
            )

            cohere = ASRDataset.from_manifest(
                csv_manifest,
                model_type="asr_cohere",
            )
            seamless = ASRDataset.from_manifest(
                tsv_manifest,
                model_type="asr_seamless_m4t_v2",
            )

        self.assertEqual(cohere[0]["audio"], str((root / "english.wav").resolve()))
        self.assertEqual(cohere[0]["text"], "English sentence.")
        self.assertEqual(cohere[0]["punctuation"], False)
        self.assertEqual(
            seamless[0]["audio"],
            str((root / "german.wav").resolve()),
        )
        self.assertEqual(seamless[0]["target_language"], "deu")

    def test_json_and_tabular_manifests_reject_ambiguous_json_before_dataset_construction(self):
        ambiguities = (
            (
                "duplicate",
                '{"score":"discarded-secret","score":1}',
                r"Duplicate JSON object key 'score'",
            ),
            (
                "constant",
                '{"score":NaN}',
                r"non-finite JSON constant 'NaN'",
            ),
            (
                "overflow",
                '{"score":1e400}',
                r"score.*non-finite",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, metadata, diagnostic in ambiguities:
                record = f'{{"text":"Safe record","audio":"sample.wav","metadata":{metadata}}}'
                escaped_metadata = metadata.replace('"', '""')
                documents = (
                    ("json", record),
                    ("jsonl", record + "\n"),
                    (
                        "csv",
                        "text,audio,metadata\n" + f'Safe record,sample.wav,"{escaped_metadata}"\n',
                    ),
                )
                for suffix, document in documents:
                    with self.subTest(name=name, suffix=suffix):
                        manifest = root / f"{name}.{suffix}"
                        manifest.write_text(document, encoding="utf-8")
                        with self.assertRaises(ValueError) as captured:
                            ASRDataset.from_manifest(
                                manifest,
                                architecture="ctc",
                                validate=False,
                            )
                        rendered = str(captured.exception)
                        self.assertIn(str(manifest.resolve()), rendered)
                        self.assertRegex(rendered, diagnostic)
                        self.assertNotIn("discarded-secret", rendered)

    def test_sensevoice_and_seamless_upstream_records_are_accepted_directly(self):
        sensevoice = ASRDataset(
            [{
                "source": "sense.wav",
                "target": "SenseVoice transcript.",
                "text_language": "<|en|>",
                "emo_target": "<|NEUTRAL|>",
                "event_target": "<|Speech|>",
                "with_or_wo_itn": "<|woitn|>",
            }],
            model_type="asr_funasr",
        )
        seamless = ASRDataset(
            [{
                "source": {
                    "id": "source-1",
                    "audio_local_path": "seamless.wav",
                    "sampling_rate": 16_000,
                    "lang": "eng",
                },
                "target": {
                    "id": "target-1",
                    "text": "Seamless transcript.",
                    "lang": "deu",
                },
            }],
            model_type="asr_seamless_m4t_v2",
        )

        self.assertEqual(sensevoice.variant_names, ("raw-audio", ))
        self.assertEqual(sensevoice[0]["audio"], "sense.wav")
        self.assertEqual(sensevoice[0]["text"], "SenseVoice transcript.")
        self.assertEqual(sensevoice[0]["language"], "en")
        self.assertEqual(sensevoice[0]["emotion"], "neutral")
        self.assertEqual(sensevoice[0]["event"], "speech")
        self.assertIs(sensevoice[0]["use_itn"], False)
        restored_sensevoice = pickle.loads(pickle.dumps(sensevoice))
        self.assertEqual(restored_sensevoice[0], sensevoice[0])
        self.assertEqual(seamless.variant_names, ("raw-audio", ))
        self.assertEqual(seamless[0]["audio"], "seamless.wav")
        self.assertEqual(seamless[0]["text"], "Seamless transcript.")
        self.assertEqual(seamless[0]["source_language"], "eng")
        self.assertEqual(seamless[0]["target_language"], "deu")
        self.assertEqual(seamless[0]["sampling_rate"], 16_000)

    def test_declared_record_normalizers_preserve_failure_behavior(self):
        with self.assertRaisesRegex(ValueError, "withitn"):
            ASRDataset(
                [{
                    "source": "sense.wav",
                    "target": "Transcript.",
                    "text_language": "<|en|>",
                    "with_or_wo_itn": "<|maybe|>",
                }],
                model_type="asr_funasr",
            )
        with self.assertRaisesRegex(TypeError, "must both be mappings"):
            ASRDataset(
                [{
                    "source": {
                        "audio": "seamless.wav"
                    },
                    "target": "Transcript.",
                }],
                model_type="asr_seamless_m4t_v2",
            )
        with self.assertRaisesRegex(ValueError, "canonical field 'text'"):
            ASRDataset(
                [{
                    "source": {
                        "audio": "seamless.wav"
                    },
                    "target": {
                        "text": "Nested transcript.",
                        "lang": "eng",
                    },
                    "text": "Conflicting transcript.",
                }],
                model_type="asr_seamless_m4t_v2",
            )

    def test_tabular_numeric_sampling_rate_and_duration_are_coerced(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "records.csv"
            manifest.write_text(
                "audio,text,sampling_rate,duration\n"
                "sample.wav,Transcript.,16000,1.25\n",
                encoding="utf-8",
            )

            dataset = ASRDataset.from_manifest(
                manifest,
                model_type="asr_whisper",
            )

        self.assertEqual(dataset[0]["sampling_rate"], 16_000)
        self.assertEqual(dataset[0]["duration"], 1.25)

    def test_from_audio_folder_pairs_sidecars_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "speaker-a"
            nested.mkdir()
            audio = nested / "utterance.wav"
            audio.touch()
            audio.with_suffix(".txt").write_text(
                "  A sidecar transcript.  \n",
                encoding="utf-8",
            )

            dataset = ASRDataset.from_audio_folder(
                root,
                model_type="asr_whisper",
                metadata={
                    "speaker_id": "speaker-a",
                    "language": "en",
                },
            )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["audio"], str(audio.resolve()))
        self.assertEqual(dataset[0]["text"], "A sidecar transcript.")
        self.assertEqual(dataset[0]["speaker_id"], "speaker-a")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "missing.wav").touch()
            with self.assertRaisesRegex(FileNotFoundError, "sidecar"):
                ASRDataset.from_audio_folder(
                    root,
                    model_type="asr_whisper",
                )

    def test_from_kaldi_supports_plain_files_and_rejects_shell_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "wav" / "utterance.wav"
            audio.parent.mkdir()
            audio.touch()
            (root / "wav.scp").write_text(
                "utt-1 wav/utterance.wav\n",
                encoding="utf-8",
            )
            (root / "text").write_text(
                "utt-1 Kaldi transcript.\n",
                encoding="utf-8",
            )

            dataset = ASRDataset.from_kaldi(
                root,
                model_type="asr_espnet",
                validate_files=True,
            )
            self.assertEqual(dataset[0]["id"], "utt-1")
            self.assertEqual(dataset[0]["text"], "Kaldi transcript.")
            self.assertEqual(dataset[0]["audio"], str(audio.resolve()))

            (root / "wav.scp").write_text(
                "utt-1 sox source.flac -t wav - |\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "shell pipeline"):
                ASRDataset.from_kaldi(
                    root,
                    model_type="asr_espnet",
                )

    def test_validate_files_checks_flat_and_nested_audio_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "present.wav"
            audio.touch()
            flat = ASRDataset(
                [{
                    "audio": "present.wav",
                    "text": "Present.",
                }],
                model_type="asr_whisper",
                root=root,
                validate_files=True,
            )
            nested = ASRDataset(
                [{
                    "audio": {
                        "path": "present.wav",
                        "sampling_rate": 16_000,
                    },
                    "text": "Nested.",
                }],
                model_type="asr_whisper",
                root=root,
                validate_files=True,
            )

            self.assertEqual(flat[0]["audio"], str(audio.resolve()))
            self.assertEqual(nested[0]["audio"]["path"], str(audio.resolve()))
            with self.assertRaises(FileNotFoundError):
                ASRDataset(
                    [{
                        "audio": "missing.wav",
                        "text": "Missing.",
                    }],
                    model_type="asr_whisper",
                    root=root,
                    validate_files=True,
                )

    def test_custom_aliases_normalize_and_collisions_are_rejected(self):
        dataset = ASRDataset(
            [{
                "recording": "sample.wav",
                "orthography": "Custom fields.",
            }],
            model_type="asr_whisper",
            aliases={
                "recording": "audio",
                "orthography": "text",
            },
        )

        self.assertEqual(dataset[0]["audio"], "sample.wav")
        self.assertEqual(dataset[0]["text"], "Custom fields.")
        with self.assertRaisesRegex(ValueError, "both alias"):
            ASRDataset(
                [{
                    "audio": "canonical.wav",
                    "audio_path": "alias.wav",
                    "text": "Collision.",
                }],
                model_type="asr_whisper",
            )
        with self.assertRaisesRegex(ValueError, "both alias"):
            ASRDataset(
                [{
                    "audio": "canonical.wav",
                    "recording": "alias.wav",
                    "text": "Collision.",
                }],
                model_type="asr_whisper",
                aliases={"recording": "audio"},
            )

    def test_grouped_split_is_deterministic_and_leakage_safe(self):
        records = [{
            "id": f"record-{index}",
            "audio": f"{index}.wav",
            "text": f"Transcript {index}.",
            "speaker_id": f"speaker-{index // 2}",
        } for index in range(8)]
        dataset = ASRDataset(
            records,
            architecture=ASRDataArchitecture.CTC,
        )

        train, validation = dataset.train_test_split(
            validation_fraction=0.25,
            seed=7,
            group_by="speaker_id",
        )
        repeated_train, repeated_validation = dataset.train_test_split(
            validation_fraction=0.25,
            seed=7,
            group_by="speaker_id",
        )

        train_speakers = {record["speaker_id"] for record in train}
        validation_speakers = {record["speaker_id"] for record in validation}
        self.assertFalse(train_speakers.intersection(validation_speakers))
        self.assertEqual(
            [record["id"] for record in train],
            [record["id"] for record in repeated_train],
        )
        self.assertEqual(
            [record["id"] for record in validation],
            [record["id"] for record in repeated_validation],
        )

    def test_jsonl_round_trip_writes_portable_relative_audio_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "clips" / "sample.wav"
            audio.parent.mkdir()
            audio.touch()
            dataset = ASRDataset(
                [{
                    "audio": str(audio),
                    "text": "Portable transcript.",
                }],
                model_type="asr_whisper",
            )

            manifest = dataset.to_jsonl(root / "prepared.jsonl")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            restored = ASRDataset.from_manifest(
                manifest,
                model_type="asr_whisper",
                validate_files=True,
            )

        self.assertEqual(payload["audio"], "clips/sample.wav")
        self.assertEqual(restored[0]["audio"], str(audio.resolve()))
        self.assertEqual(restored[0]["text"], "Portable transcript.")

    def test_jsonl_serialization_failure_does_not_truncate_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "records.jsonl"
            destination.write_text("keep-me\n", encoding="utf-8")
            dataset = ASRDataset(
                [
                    {
                        "audio": "one.wav",
                        "text": "Serializable.",
                    },
                    {
                        "audio": "two.wav",
                        "metadata": object(),
                        "text": "Not serializable.",
                    },
                ],
                architecture="ctc",
                validate=False,
            )

            with self.assertRaisesRegex(TypeError, "record 1"):
                dataset.to_jsonl(destination)

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "keep-me\n",
            )

    def test_jsonl_rejects_credentials_before_destination_mutation(self):
        credential = "must-not-appear-in-errors"
        dataset = ASRDataset(
            [{
                "audio": "sample.wav",
                "text": "Unsafe manifest record.",
                "metadata": {
                    "api_key": credential,
                },
            }],
            architecture="ctc",
            validate=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.jsonl"
            existing.write_text("keep-me\n", encoding="utf-8")
            missing_parent = root / "new" / "records.jsonl"

            for destination in (existing, missing_parent):
                with self.assertRaisesRegex(
                        ValueError,
                        r"ASR dataset record 0.*metadata\.api_key",
                ) as captured:
                    dataset.to_jsonl(destination)
                self.assertNotIn(credential, str(captured.exception))

            self.assertEqual(existing.read_text(encoding="utf-8"), "keep-me\n")
            self.assertFalse(missing_parent.parent.exists())

    def test_manifest_loader_rejects_credentials_and_keeps_safe_metadata(self):
        credential = "must-not-appear-in-errors"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe = root / "unsafe.jsonl"
            unsafe.write_text(
                json.dumps({
                    "audio": "sample.wav",
                    "text": "Unsafe manifest record.",
                    "metadata": {
                        "authorization": credential,
                    },
                }) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                    ValueError,
                    r"ASR manifest record 0.*metadata\.authorization",
            ) as captured:
                ASRDataset.from_manifest(
                    unsafe,
                    architecture="ctc",
                    validate=False,
                )

            safe = root / "safe.jsonl"
            safe.write_text(
                json.dumps({
                    "audio": "sample.wav",
                    "text": "Safe manifest record.",
                    "metadata": {
                        "token_count": 512,
                    },
                }) + "\n",
                encoding="utf-8",
            )
            restored = ASRDataset.from_manifest(
                safe,
                architecture="ctc",
                validate=False,
            )

        self.assertNotIn(credential, str(captured.exception))
        self.assertEqual(restored[0]["metadata"]["token_count"], 512)

    def test_fingerprint_covers_content_order_and_transform_version(self):
        records = [
            {
                "audio": "one.wav",
                "text": "One.",
            },
            {
                "audio": "two.wav",
                "text": "Two.",
            },
        ]
        original = ASRDataset(records, architecture="ctc")
        reordered = ASRDataset(list(reversed(records)), architecture="ctc")
        changed = ASRDataset(
            [
                records[0],
                {
                    **records[1],
                    "text": "Changed.",
                },
            ],
            architecture="ctc",
        )

        self.assertNotEqual(
            original.resume_fingerprint()["content_sha256"],
            reordered.resume_fingerprint()["content_sha256"],
        )
        self.assertNotEqual(
            original.resume_fingerprint()["content_sha256"],
            changed.resume_fingerprint()["content_sha256"],
        )

        def annotate(record):
            return {
                **record,
                "revision": 1,
            }

        unversioned = ASRDataset(
            records,
            architecture="ctc",
            transform=annotate,
        )
        version_one = ASRDataset(
            records,
            architecture="ctc",
            transform=annotate,
            transform_fingerprint="annotate-v1",
        )
        version_two = ASRDataset(
            records,
            architecture="ctc",
            transform=annotate,
            transform_fingerprint="annotate-v2",
        )
        with self.assertRaisesRegex(ValueError, "transform_fingerprint"):
            unversioned.resume_fingerprint()
        self.assertNotEqual(
            version_one.resume_fingerprint()["content_sha256"],
            version_two.resume_fingerprint()["content_sha256"],
        )

    def test_dataset_is_pickle_safe_for_worker_processes(self):
        dataset = ASRDataset(
            [{
                "audio": "worker.wav",
                "text": "Worker-safe.",
                "metadata": {
                    "source": "unit-test",
                },
            }],
            model_type="asr_whisper",
        )

        restored = pickle.loads(pickle.dumps(dataset))

        self.assertEqual(restored[0], dataset[0])
        self.assertEqual(restored.variant_names, dataset.variant_names)
        self.assertEqual(
            restored.resume_fingerprint(),
            dataset.resume_fingerprint(),
        )


class ASRPreparedRuntimeContractTests(unittest.TestCase):

    def test_wenet_phase_accepts_cached_frontend_modality(self):
        phase = get_training_spec("asr_wenet").get_phase()

        self.assertEqual(
            phase.required_inputs,
            ("labels", "label_lengths"),
        )

    def test_nemo_and_wenet_cached_variants_bypass_raw_preprocessing(self):
        import torch

        from voicehub.models.asr_nemo import NeMoASRConfig, NeMoASRForSpeechRecognition
        from voicehub.models.asr_wenet import WeNetASRConfig, WeNetASRForSpeechRecognition

        nemo = NeMoASRForSpeechRecognition(
            NeMoASRConfig(),
            device="cpu",
            lazy_load=True,
        )
        nemo_waveform = {
            "input_signal": torch.zeros(1, 16),
            "input_signal_length": torch.tensor([16]),
            "labels": torch.tensor([[1, 2]]),
            "label_lengths": torch.tensor([2]),
        }
        nemo_features = {
            "processed_signal": torch.zeros(1, 64, 4),
            "processed_signal_length": torch.tensor([4]),
            "labels": torch.tensor([[1, 2]]),
            "label_lengths": torch.tensor([2]),
        }
        self.assertEqual(
            set(nemo.prepare_training_inputs(nemo_waveform, phase="ctc")),
            set(nemo_waveform),
        )
        self.assertEqual(
            set(nemo.prepare_training_inputs(nemo_features, phase="ctc")),
            set(nemo_features),
        )
        self.assertFalse(nemo.is_loaded)

        wenet = WeNetASRForSpeechRecognition(
            WeNetASRConfig(),
            device="cpu",
            lazy_load=True,
        )
        wenet_waveform = {
            "input_signal": torch.zeros(1, 16),
            "input_signal_length": torch.tensor([16]),
            "labels": torch.tensor([[1, 2]]),
            "label_lengths": torch.tensor([2]),
        }
        wenet_features = {
            "features": torch.zeros(1, 4, 80),
            "feature_lengths": torch.tensor([4]),
            "labels": torch.tensor([[1, 2]]),
            "label_lengths": torch.tensor([2]),
        }
        self.assertEqual(
            set(wenet.prepare_training_inputs(wenet_waveform, phase="hybrid")),
            set(wenet_waveform),
        )
        self.assertEqual(
            set(wenet.prepare_training_inputs(wenet_features, phase="hybrid")),
            set(wenet_features),
        )
        self.assertFalse(wenet.is_loaded)

    def test_specialized_datasets_preserve_cached_records(self):
        from voicehub.models.asr_espnet.native.training import ESPnetASRTrainingDataset
        from voicehub.models.asr_native.speechbrain_training import SpeechBrainASRTrainingDataset

        speechbrain_record = {
            "waveforms": [[0.0, 0.1]],
            "waveform_lengths": [2],
            "tokens_bos": [[1, 2]],
            "tokens_eos": [[2, 3]],
            "token_lengths": [2],
            "ctc_tokens": [[2]],
            "ctc_token_lengths": [1],
        }
        self.assertEqual(
            SpeechBrainASRTrainingDataset([speechbrain_record])[0],
            speechbrain_record,
        )

        espnet_records = (
            {
                "features": [[0.1]],
                "text": "FEATURE TRANSCRIPT",
            },
            {
                "waveforms": [[0.0, 0.1]],
                "waveform_lengths": [2],
                "labels": [[1]],
                "label_lengths": [1],
            },
            {
                "features": [[[0.1]]],
                "feature_lengths": [1],
                "labels": [[1]],
                "label_lengths": [1],
            },
        )
        dataset = ESPnetASRTrainingDataset(espnet_records)
        self.assertEqual(len(dataset), 3)
        for index, record in enumerate(espnet_records):
            self.assertEqual(dataset[index], record)


class ASRGroupedBatchSamplerTests(unittest.TestCase):

    def test_cohere_batches_are_language_and_punctuation_homogeneous(self):
        records = []
        for language, punctuation in (
            ("en", True),
            ("en", False),
            ("tr", True),
        ):
            for index in range(2):
                records.append({
                    "id": f"{language}-{punctuation}-{index}",
                    "audio": f"{language}-{punctuation}-{index}.wav",
                    "text": "Transcript.",
                    "language": language,
                    "punctuation": punctuation,
                })
        dataset = ASRDataset(records, model_type="asr_cohere")
        sampler = dataset.create_batch_sampler(
            batch_size=2,
            seed=19,
            shuffle=True,
            drop_last=False,
        )

        self.assertIsInstance(sampler, EpochGroupedBatchSampler)
        self.assertTrue(dataset.requires_homogeneous_batches)
        self.assertEqual(len(sampler), 3)
        for batch in sampler:
            keys = {dataset.batch_group_key(dataset._records[index]) for index in batch}
            self.assertEqual(len(keys), 1)

    def test_seamless_batches_use_target_language_aliases(self):
        dataset = ASRDataset(
            [
                {
                    "audio": "english-1.wav",
                    "text": "One.",
                    "target_lang": "eng",
                },
                {
                    "audio": "english-2.wav",
                    "text": "Two.",
                    "target_language": "eng",
                },
                {
                    "audio": "turkish-1.wav",
                    "text": "Bir.",
                    "language": "tur",
                },
                {
                    "audio": "turkish-2.wav",
                    "text": "İki.",
                    "lang": "tur",
                },
            ],
            model_type="asr_seamless_m4t_v2",
        )
        sampler = dataset.create_batch_sampler(
            batch_size=2,
            seed=3,
            shuffle=False,
            drop_last=False,
        )

        self.assertIsInstance(sampler, EpochGroupedBatchSampler)
        self.assertEqual(len(sampler), 2)
        for batch in sampler:
            keys = {dataset.batch_group_key(dataset._records[index]) for index in batch}
            self.assertEqual(len(keys), 1)

    def test_sampler_state_restores_exact_epoch_order(self):
        dataset = ASRDataset(
            [{
                "audio": f"{index}.wav",
                "text": f"Transcript {index}.",
                "language": "en" if index < 4 else "tr",
                "punctuation": index % 2 == 0,
            } for index in range(8)],
            model_type="asr_cohere",
        )
        original = dataset.create_batch_sampler(
            batch_size=2,
            seed=13,
            shuffle=True,
            drop_last=False,
        )
        restored = dataset.create_batch_sampler(
            batch_size=2,
            seed=13,
            shuffle=True,
            drop_last=False,
        )
        original.set_epoch(4)
        state = original.state_dict()
        restored.load_state_dict(state)

        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(list(restored), list(original))

        incompatible = dataset.create_batch_sampler(
            batch_size=1,
            seed=13,
            shuffle=True,
            drop_last=False,
        )
        with self.assertRaisesRegex(ValueError, "batch_size differs"):
            incompatible.load_state_dict(state)

        ordinary = ASRDataset(
            [{
                "audio": "ordinary.wav",
                "text": "No grouping required.",
            }],
            model_type="asr_whisper",
        )
        self.assertIsNone(
            ordinary.create_batch_sampler(
                batch_size=1,
                seed=0,
                shuffle=True,
                drop_last=False,
            ), )


class ASRModelManifestIntegrationTests(unittest.TestCase):

    class _CapturingAdapter:

        def __init__(self):
            self.records = None
            self.kwargs = None

        def create_dataset(self, records, **kwargs):
            self.records = records
            self.kwargs = kwargs
            return records

    class _Model(PreTrainedASRModel):

        def __init__(self, config):
            super().__init__(config, device="cpu")
            self.adapter = (ASRModelManifestIntegrationTests._CapturingAdapter())

        def get_training_adapter(self):
            return self.adapter

        def _load_pretrained_model(self):
            raise AssertionError("Dataset creation must not load model weights.")

        def _transcribe(self, audio, **kwargs):
            return ASROutput(text="dummy")

    def test_model_accepts_manifest_path_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "records.jsonl"
            manifest.write_text(
                json.dumps({
                    "audio_filepath": "hello.wav",
                    "transcript": "Hello.",
                }) + "\n",
                encoding="utf-8",
            )
            config = VoiceHubConfig(name_or_path="dummy")
            config.model_type = "asr_whisper"
            model = self._Model(config)

            dataset = model.create_training_dataset(
                manifest,
                cache_features=True,
            )

        self.assertIsInstance(dataset, ASRDataset)
        self.assertFalse(model.is_loaded)
        self.assertEqual(model.adapter.kwargs, {"cache_features": True})
        self.assertEqual(dataset[0]["text"], "Hello.")
        self.assertEqual(
            dataset[0]["audio"],
            str((manifest.parent / "hello.wav").resolve()),
        )

    def test_model_data_options_normalize_in_memory_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = VoiceHubConfig(name_or_path="dummy")
            config.model_type = "asr_whisper"
            model = self._Model(config)

            dataset = model.create_training_dataset(
                [{
                    "recording": "hello.wav",
                    "orthography": "Hello.",
                }],
                data_root=root,
                data_aliases={
                    "recording": "audio",
                    "orthography": "text",
                },
            )

        self.assertIsInstance(dataset, ASRDataset)
        self.assertFalse(model.is_loaded)
        self.assertEqual(model.adapter.kwargs, {})
        self.assertEqual(dataset[0]["text"], "Hello.")
        self.assertEqual(
            dataset[0]["audio"],
            str((root / "hello.wav").resolve()),
        )

    def test_model_normalizes_common_in_memory_aliases_by_default(self):
        config = VoiceHubConfig(name_or_path="dummy")
        config.model_type = "asr_whisper"
        model = self._Model(config)

        dataset = model.create_training_dataset([{
            "audio_path": "hello.wav",
            "sentence": "Hello.",
        }])

        self.assertIsInstance(dataset, ASRDataset)
        self.assertEqual(dataset[0]["audio"], "hello.wav")
        self.assertEqual(dataset[0]["text"], "Hello.")


if __name__ == "__main__":
    unittest.main()
