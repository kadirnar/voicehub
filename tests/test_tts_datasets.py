import ast
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import voicehub.training.data_contracts as data_contracts_module
from voicehub import PreTrainedTTSModel, TTSOutput, VoiceHubConfig
from voicehub.training import (
    TTSDataArchitecture,
    TTSDataReadiness,
    TTSDataset,
    TTSDatasetSpec,
    TTSRecordVariant,
    get_tts_dataset_spec,
    list_training_specs,
    list_tts_dataset_specs,
)
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily, register_training_spec, unregister_training_spec


def _build_extension_tts_dataset_spec():
    return TTSDatasetSpec(
        architecture=TTSDataArchitecture.VITS,
        variants=(
            TTSRecordVariant(
                name="extension-raw",
                required_fields=("text", "audio"),
            ),
            TTSRecordVariant(
                name="extension-ready",
                required_fields=("input_ids", "audio_values"),
                preprocessed=True,
            ),
        ),
        sample_rate=22_050,
        description="Extension-owned TTS dataset contract.",
    )


def _return_invalid_tts_dataset_spec():
    return {"architecture": "vits"}


class TTSDatasetContractTests(unittest.TestCase):

    def test_training_profiles_select_dataset_specs_without_a_provider_map(self):
        training_specs = list_training_specs()
        self.assertEqual(len(training_specs), 34)
        self.assertTrue(all(spec.dataset_spec_factory for spec in training_specs))

        source_path = Path(data_contracts_module.__file__)
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
            if len(keys) > 1:
                provider_keyed_dicts.append((node.lineno, sorted(keys)))
        self.assertEqual(provider_keyed_dicts, [])

    def test_extension_dataset_spec_factory_needs_no_shared_provider_edit(self):
        model_type = "future-tts-dataset-contract"
        training_spec = ModelTrainingSpec(
            model_type=model_type,
            family=TrainingFamily.VITS,
            support=TrainingSupport.NATIVE,
            dataset_spec_factory=f"{__name__}:_build_extension_tts_dataset_spec",
        )
        register_training_spec(training_spec)
        try:
            dataset_spec = get_tts_dataset_spec(model_type)
        finally:
            unregister_training_spec(model_type, missing_ok=True)

        self.assertEqual(dataset_spec.model_type, model_type)
        self.assertEqual(dataset_spec.sample_rate, 22_050)
        self.assertEqual(
            tuple(variant.name for variant in dataset_spec.variants),
            ("extension-raw", "extension-ready"),
        )
        self.assertIs(dataset_spec.readiness, TTSDataReadiness.INTEGRATED)
        self.assertEqual(dataset_spec.training_support, "native")

    def test_dataset_spec_factory_resolution_fails_actionably(self):
        cases = (
            (
                "future-missing-tts-dataset-contract",
                f"{__name__}:missing_tts_dataset_spec_factory",
                ImportError,
                "Could not resolve TTS dataset spec factory.*missing_tts_dataset_spec_factory",
            ),
            (
                "future-non-callable-tts-dataset-contract",
                "voicehub.training.data_contracts:_TRAINING_FAMILY_TO_DATA_ARCHITECTURE",
                TypeError,
                "must be callable",
            ),
            (
                "future-invalid-tts-dataset-contract",
                f"{__name__}:_return_invalid_tts_dataset_spec",
                TypeError,
                "returned dict; expected TTSDatasetSpec",
            ),
        )
        for model_type, factory_path, error_type, message in cases:
            with self.subTest(model_type=model_type):
                training_spec = ModelTrainingSpec(
                    model_type=model_type,
                    family=TrainingFamily.VITS,
                    dataset_spec_factory=factory_path,
                )
                register_training_spec(training_spec)
                try:
                    with self.assertRaisesRegex(error_type, message):
                        get_tts_dataset_spec(model_type)
                finally:
                    unregister_training_spec(model_type, missing_ok=True)

    def test_every_tts_profile_has_an_architecture_dataset_contract(self):
        training_specs = list_training_specs()
        dataset_specs = list_tts_dataset_specs()

        self.assertEqual(
            [spec.model_type for spec in dataset_specs],
            [spec.model_type for spec in training_specs],
        )
        for training_spec in training_specs:
            with self.subTest(model_type=training_spec.model_type):
                self.assertIs(
                    training_spec.dataset_spec.architecture,
                    get_tts_dataset_spec(training_spec.model_type).architecture,
                )
                self.assertTrue(training_spec.dataset_spec.variants)

    def test_special_training_families_have_distinct_data_contracts(self):
        self.assertIs(
            get_tts_dataset_spec("orpheustts").architecture,
            TTSDataArchitecture.CODEC_LM,
        )
        self.assertIs(
            get_tts_dataset_spec("f5tts").architecture,
            TTSDataArchitecture.DIFFUSION,
        )
        self.assertIs(
            get_tts_dataset_spec("melotts").architecture,
            TTSDataArchitecture.VITS,
        )
        self.assertEqual(get_tts_dataset_spec("dia").sample_rate, 44_100)
        self.assertEqual(get_tts_dataset_spec("qwen3tts").sample_rate, 24_000)

    def test_model_field_aliases_are_inspectable_contract_metadata(self):
        self.assertEqual(
            get_tts_dataset_spec("cosyvoice").field_aliases,
            (("audio_path", "audio_path"), ),
        )
        self.assertEqual(
            get_tts_dataset_spec("qwen3tts").field_aliases,
            (
                ("reference_audio", "ref_audio"),
                ("reference_audio_path", "ref_audio"),
                ("speaker_audio", "ref_audio"),
            ),
        )

    def test_field_alias_contract_is_normalized_and_validated(self):
        variant = TTSRecordVariant(name="raw", required_fields=("text", ))
        spec = TTSDatasetSpec(
            architecture="codec-lm",
            variants=(variant, ),
            field_aliases={" prompt ": " text "},
        )
        self.assertEqual(spec.field_aliases, (("prompt", "text"), ))

        with self.assertRaisesRegex(ValueError, "repeats source"):
            TTSDatasetSpec(
                architecture="codec-lm",
                variants=(variant, ),
                field_aliases=(("prompt", "text"), ("prompt", "input_ids")),
            )
        with self.assertRaisesRegex(ValueError, "targets must be non-empty"):
            TTSDatasetSpec(
                architecture="codec-lm",
                variants=(variant, ),
                field_aliases=(("prompt", " "), ),
            )

    def test_shared_tts_dataset_has_no_registered_model_literals(self):
        source = (Path(__file__).parents[1] / "voicehub" / "training" /
                  "tts_datasets.py").read_text(encoding="utf-8")
        literals = {
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        registered = {spec.model_type for spec in list_training_specs()}
        self.assertFalse(literals & registered)

    def test_model_contracts_distinguish_raw_preprocessed_and_unavailable_data(self):
        self.assertIs(
            get_tts_dataset_spec("dia").readiness,
            TTSDataReadiness.INTEGRATED,
        )
        self.assertTrue(get_tts_dataset_spec("dia").accepts_raw_records)
        self.assertIs(
            get_tts_dataset_spec("f5tts").readiness,
            TTSDataReadiness.PREPROCESSED,
        )
        self.assertFalse(get_tts_dataset_spec("f5tts").accepts_raw_records)
        self.assertIs(
            get_tts_dataset_spec("melotts").readiness,
            TTSDataReadiness.PREPROCESSED,
        )
        self.assertIs(
            get_tts_dataset_spec("kokoro").readiness,
            TTSDataReadiness.PREPROCESSED,
        )

    def test_xtts_contract_requires_native_preencoded_tokens_and_conditioning(self):
        contract = get_tts_dataset_spec("xtts")
        cond_mels = TTSDataset(
            [{
                "text_inputs": [1, 2, 3],
                "text_lengths": 3,
                "audio_codes": [4, 5],
                "wav_lengths": 44_100,
                "cond_mels": [[[0.1, 0.2]]],
            }],
            model_type="xtts",
        )
        cond_latents = TTSDataset(
            [{
                "text_inputs": [1, 2, 3],
                "text_lengths": 3,
                "audio_codes": [4, 5],
                "wav_lengths": 44_100,
                "cond_latents": [[0.1, 0.2]],
                "cond_idxs": [0, 2],
                "cond_lens": 2,
            }],
            model_type="xtts",
        )

        self.assertIs(contract.architecture, TTSDataArchitecture.HYBRID)
        self.assertIs(contract.readiness, TTSDataReadiness.PREPROCESSED)
        self.assertFalse(contract.accepts_raw_records)
        self.assertEqual(contract.sample_rate, 22_050)
        self.assertEqual(cond_mels.variant_names, ("native-gpt-tokens", ))
        self.assertEqual(cond_latents.variant_names, ("native-gpt-tokens", ))
        with self.assertRaisesRegex(ValueError, "cond_mels or cond_latents"):
            TTSDataset(
                [{
                    "text_inputs": [1, 2, 3],
                    "text_lengths": 3,
                    "audio_codes": [4, 5],
                    "wav_lengths": 44_100,
                }],
                model_type="xtts",
            )
        with self.assertRaisesRegex(ValueError, "native-gpt-tokens"):
            TTSDataset(
                [{
                    "text": "hello",
                    "audio": "clips/example.wav",
                    "language": "en",
                }],
                model_type="xtts",
            )

    def test_model_contract_overrides_match_native_record_boundaries(self):
        cases = (
            (
                "chatterbox",
                {
                    "text": "hello",
                    "audio": "clip.wav",
                },
                "t3-raw",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "cosyvoice",
                {
                    "text": "hello",
                    "speech_tokens": [1, 2],
                },
                "llm-record",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "conversationtts",
                {
                    "text": "hello",
                    "audio": "clip.wav",
                },
                "raw-text-audio",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "f5tts",
                {
                    "input_values": [0.0, 0.1],
                    "input_ids": [1, 2],
                },
                "waveform-vocab",
                TTSDataReadiness.PREPROCESSED,
            ),
            (
                "gptsovits",
                {
                    "phoneme_ids": [1, 2],
                    "semantic_ids": [3, 4],
                    "bert_features": [[0.1], [0.2]],
                },
                "s1-preprocessed",
                TTSDataReadiness.PREPROCESSED,
            ),
            (
                "melotts",
                {
                    "input_ids": [1],
                    "tone_ids": [0],
                    "language_ids": [0],
                    "bert_features": [[0.1]],
                    "ja_bert_features": [[0.2]],
                    "spectrogram": [[0.3]],
                    "audio_values": [0.0],
                    "speaker_id": 0,
                },
                "explicit-features",
                TTSDataReadiness.PREPROCESSED,
            ),
            (
                "outetts",
                {
                    "input_ids": [1, 2],
                    "labels": [-100, 2],
                },
                "tokenized",
                TTSDataReadiness.PREPROCESSED,
            ),
            (
                "mosstts",
                {
                    "text": "hello",
                    "speech_tokens": [[1, 2]],
                },
                "preencoded-rvq",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "parlertts",
                {
                    "description": "A calm voice",
                    "audio_values": [0.0, 0.1],
                },
                "waveform-teacher-forcing",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "zonos2",
                {
                    "text": "hello",
                    "audio": "clip.wav",
                },
                "raw-audio",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "irodoritts",
                {
                    "text": "hello",
                    "target_latent": [[0.1]],
                },
                "preencoded-latent",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "voxcpm",
                {
                    "text": "hello",
                    "audio_features": [[0.1]],
                },
                "audio-features",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "omnivoice",
                {
                    "text": "hello",
                    "audio_tokens": [[1, 2]],
                },
                "audio-tokens",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "higgstts",
                {
                    "text": "hello",
                    "audio": "clip.wav",
                },
                "raw-audio",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "fishtts",
                {
                    "tokens": [[1, 2]],
                    "labels": [[1, 2]],
                },
                "semantic-tokens",
                TTSDataReadiness.PREPROCESSED,
            ),
            (
                "vits",
                {
                    "text": "hello",
                    "audio": "clip.wav",
                },
                "raw-adversarial",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "vui",
                {
                    "input_ids": [1, 2],
                    "audio_codes": [[3, 4]],
                },
                "codec-batch",
                TTSDataReadiness.PREPROCESSED,
            ),
            (
                "kokoro",
                {
                    "phonemes": "həlˈoʊ",
                    "ref_s": [0.1],
                    "durations": [1, 2],
                    "audio_values": [0.0],
                },
                "full-preprocessed",
                TTSDataReadiness.PREPROCESSED,
            ),
            (
                "openvoice",
                {
                    "source_audio": "source.wav",
                    "target_audio": "target.wav",
                },
                "paired-waveforms",
                TTSDataReadiness.INTEGRATED,
            ),
            (
                "styletts2",
                {
                    "input_ids": [0, 1, 2],
                    "alignments": [[1.0]],
                    "normalized_mel": [[0.1]],
                    "reference_mel": [[0.1]],
                    "f0_targets": [1.0],
                    "noise_targets": [[0.0]],
                    "audio_values": [0.0],
                },
                "explicit-features",
                TTSDataReadiness.PREPROCESSED,
            ),
            (
                "supertonic",
                {
                    "text": "hello",
                    "style_ttl": [[0.1]],
                    "style_dp": [[0.2]],
                    "target_duration": 0.5,
                },
                "text-style-tensors",
                TTSDataReadiness.PREPROCESSED,
            ),
            (
                "bark",
                {
                    "input_ids": [1, 2],
                    "labels": [2, 3],
                    "training_phase": "coarse",
                },
                "causal-stage",
                TTSDataReadiness.PREPROCESSED,
            ),
            (
                "inflecttts",
                {
                    "input_ids": [1, 2],
                    "spectrogram": [[0.1]],
                    "audio_values": [0.0],
                },
                "explicit-features",
                TTSDataReadiness.PREPROCESSED,
            ),
        )
        for model_type, record, variant, readiness in cases:
            with self.subTest(model_type=model_type):
                dataset = TTSDataset([record], model_type=model_type)
                self.assertEqual(dataset.variant_names, (variant, ))
                self.assertIs(dataset.spec.readiness, readiness)

    def test_cosyvoice_contract_accepts_raw_audio_and_precomputed_tokens(self):
        contract = get_tts_dataset_spec("cosyvoice")
        raw = TTSDataset(
            [{
                "text": "Merhaba",
                "speech_audio": [0.0, 0.1, -0.1],
                "speech_sampling_rate": 16_000,
            }],
            model_type="cosyvoice",
        )
        path_backed = TTSDataset(
            [{
                "text": "Hello",
                "audio_path": "speaker.wav",
            }],
            model_type="cosyvoice",
        )
        tokens = TTSDataset(
            [{
                "text": "Bonjour",
                "speech_tokens": [1, 2, 3],
            }],
            model_type="cosyvoice",
        )

        self.assertIs(contract.readiness, TTSDataReadiness.INTEGRATED)
        self.assertTrue(contract.accepts_raw_records)
        self.assertEqual(raw.variant_names, ("llm-raw-audio", ))
        self.assertEqual(path_backed.variant_names, ("llm-raw-audio", ))
        self.assertEqual(tokens.variant_names, ("llm-record", ))
        with self.assertRaisesRegex(ValueError, "speech_audio requires one of"):
            TTSDataset(
                [{
                    "text": "Missing source rate",
                    "speech_audio": [0.0, 0.1],
                }],
                model_type="cosyvoice",
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            TTSDataset(
                [{
                    "text": "Ambiguous source",
                    "speech_tokens": [1, 2],
                    "waveform": [0.0, 0.1],
                    "sample_rate": 16_000,
                }],
                model_type="cosyvoice",
            )

    def test_stale_tts_record_shapes_are_rejected(self):
        invalid = (
            ("outetts", {
                "audio": "clip.wav"
            }),
            ("melotts", {
                "input_ids": [1],
                "spectrogram": [[0.1]],
                "audio_values": [0.0],
            }),
            ("fishtts", {
                "input_ids": [[1, 2]],
                "labels": [[1, 2]],
            }),
            ("mosstts", {
                "text": "hello",
                "audio": "clip.wav",
                "speech_tokens": [[1, 2]],
            }),
            ("conversationtts", {
                "text": "hello",
                "audio": "clip.wav",
                "audio_codes": [[1, 2]],
            }),
            ("parlertts", {
                "description": "A calm voice",
                "audio_values": [0.0],
                "input_values": [0.0],
            }),
        )
        for model_type, record in invalid:
            with self.subTest(model_type=model_type):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    TTSDataset([record], model_type=model_type)

    def test_higgs_reference_contract_enforces_conditional_metadata(self):
        dataset = TTSDataset(
            [{
                "text": "Target",
                "audio_codes": [[1, 2]],
                "reference_audio": "reference.wav",
                "reference_text": "Reference",
            }],
            model_type="higgstts",
            root="/tmp/corpus",
        )

        self.assertEqual(dataset.variant_names, ("audio-codes", ))
        self.assertEqual(
            dataset[0]["reference_audio"],
            str(Path("/tmp/corpus/reference.wav").resolve()),
        )
        self.assertNotIn("ref_audio", dataset[0])
        with self.assertRaisesRegex(ValueError, "reference_audio requires"):
            TTSDataset(
                [{
                    "text": "Target",
                    "audio_codes": [[1, 2]],
                    "reference_audio": "reference.wav",
                }],
                model_type="higgstts",
            )
        with self.assertRaisesRegex(ValueError, "reference_text requires"):
            TTSDataset(
                [{
                    "text": "Target",
                    "audio_codes": [[1, 2]],
                    "reference_text": "Reference",
                }],
                model_type="higgstts",
            )

    def test_csm_grouped_audio_contract_requires_unambiguous_segmentation(self):
        audios = TTSDataset(
            [{
                "texts": ["one", "two"],
                "speaker_ids": [0, 1],
                "audios": ["one.wav", "two.wav"],
            }],
            model_type="csm",
        )
        concatenated = TTSDataset(
            [{
                "texts": ["one", "two"],
                "speaker_ids": [0, 1],
                "audio": "joined.wav",
                "audio_cut_idxs": [[0, 100], [100, 200]],
            }],
            model_type="csm",
        )

        self.assertEqual(audios.variant_names, ("grouped-audios", ))
        self.assertEqual(
            concatenated.variant_names,
            ("grouped-concatenated", ),
        )
        with self.assertRaisesRegex(ValueError, "audio_cut_idxs"):
            TTSDataset(
                [{
                    "texts": ["one", "two"],
                    "speaker_ids": [0, 1],
                    "audio": "joined.wav",
                }],
                model_type="csm",
            )

    def test_model_defaults_never_advertise_unintegrated_raw_preprocessing(self):
        with self.assertRaisesRegex(ValueError, "flow-batch"):
            TTSDataset(
                [{
                    "text": "Raw data is architecture-valid only.",
                    "audio": "sample.wav",
                }],
                model_type="echo",
            )

    def test_coerce_revalidates_existing_datasets_for_the_target_model(self):
        generic = TTSDataset(
            [{
                "text": "Raw sequence data",
                "audio": "sample.wav",
            }],
            architecture="sequence-to-sequence",
        )
        unchecked = TTSDataset(
            [{
                "text": "Missing target audio"
            }],
            model_type="dia",
            validate=False,
        )

        with self.assertRaisesRegex(ValueError, "flow-batch"):
            TTSDataset.coerce(generic, model_type="echo")
        with self.assertRaisesRegex(ValueError, "does not match"):
            TTSDataset.coerce(unchecked, model_type="dia")
        with tempfile.TemporaryDirectory() as directory:
            unvalidated_file = TTSDataset(
                [{
                    "text": "Missing file",
                    "audio": "missing.wav",
                }],
                model_type="dia",
                root=directory,
            )
            with self.assertRaises(FileNotFoundError):
                TTSDataset.coerce(
                    unvalidated_file,
                    model_type="dia",
                    validate_files=True,
                )

    def test_dataset_contract_imports_remain_framework_lazy(self):
        script = (
            "import sys;from voicehub.training import get_tts_dataset_spec;print(get_tts_dataset_spec('f5tts').architecture.value,'torch' in sys.modules,'numpy' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "diffusion False False")

    def test_invalid_record_lists_every_supported_variant(self):
        with self.assertRaisesRegex(
                ValueError,
                "raw-audio.*tokenized",
        ):
            TTSDataset(
                [{
                    "text": "missing audio"
                }],
                model_type="orpheustts",
            )


class TTSDatasetManifestTests(unittest.TestCase):

    def test_jsonl_aliases_and_paths_are_normalized_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio" / "sample.wav"
            audio.parent.mkdir()
            audio.touch()
            manifest = root / "records.jsonl"
            manifest.write_text(
                json.dumps({
                    "id": "sample",
                    "transcript": "Hello.",
                    "audio_path": "audio/sample.wav",
                    "speaker_id": "speaker-a",
                }) + "\n",
                encoding="utf-8",
            )

            dataset = TTSDataset.from_manifest(
                manifest,
                model_type="dia",
                validate_files=True,
            )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.variant_names, ("raw-audio", ))
        self.assertEqual(dataset[0]["text"], "Hello.")
        self.assertEqual(dataset[0]["audio"], str(audio.resolve()))
        self.assertNotIn("transcript", dataset[0])
        self.assertNotIn("audio_path", dataset[0])

    def test_csv_parses_json_code_arrays_and_reference_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "records.csv"
            manifest.write_text(
                "text,audio_codes,reference_audio\n"
                'Hello,"[[1,2],[3,4]]",reference.wav\n',
                encoding="utf-8",
            )
            dataset = TTSDataset.from_manifest(
                manifest,
                model_type="qwen3tts",
            )

        self.assertEqual(dataset[0]["audio_codes"], [[1, 2], [3, 4]])
        self.assertEqual(dataset[0]["ref_audio"], str((root / "reference.wav").resolve()))
        self.assertEqual(dataset.variant_names, ("single-speaker-sft", ))

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
                    ("json", f"[{record}]"),
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
                            TTSDataset.from_manifest(
                                manifest,
                                architecture="vits",
                                validate=False,
                            )
                        rendered = str(captured.exception)
                        self.assertIn(str(manifest.resolve()), rendered)
                        self.assertRegex(rendered, diagnostic)
                        self.assertNotIn("discarded-secret", rendered)

    def test_ljspeech_layout_uses_normalized_text_and_audio_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.csv").write_text(
                "LJ001-0001|Raw text.|Normalized text.\n",
                encoding="utf-8",
            )
            dataset = TTSDataset.from_ljspeech(
                root,
                model_type="speecht5",
            )

        self.assertEqual(dataset[0]["id"], "LJ001-0001")
        self.assertEqual(dataset[0]["text"], "Normalized text.")
        self.assertEqual(
            dataset[0]["audio"],
            str((root / "wavs" / "LJ001-0001.wav").resolve()),
        )

    def test_nested_audio_paths_are_resolved_validated_and_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "wavs" / "nested.wav"
            audio.parent.mkdir()
            audio.touch()
            dataset = TTSDataset(
                [{
                    "text": "Nested",
                    "audio": {
                        "path": "wavs/nested.wav",
                        "sampling_rate": 16_000,
                    },
                }],
                model_type="dia",
                root=root,
                validate_files=True,
            )
            manifest = dataset.to_jsonl(root / "portable.jsonl")
            payload = json.loads(manifest.read_text(encoding="utf-8"))

            self.assertEqual(
                dataset[0]["audio"]["path"],
                str(audio.resolve()),
            )
            self.assertEqual(payload["audio"]["path"], "wavs/nested.wav")
            with self.assertRaises(FileNotFoundError):
                TTSDataset(
                    [{
                        "text": "Missing",
                        "audio": {
                            "path": "wavs/missing.wav",
                            "sampling_rate": 16_000,
                        },
                    }],
                    model_type="dia",
                    root=root,
                    validate_files=True,
                )

    def test_conversation_audio_paths_are_resolved_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "turns" / "one.wav"
            audio.parent.mkdir()
            audio.touch()

            def conversation(path):
                return [{
                    "role": "user",
                    "content": [{
                        "type": "audio",
                        "audio": {
                            "path": path,
                        },
                    }],
                }]

            dataset = TTSDataset(
                [{
                    "conversation": conversation("turns/one.wav"),
                }],
                model_type="csm",
                root=root,
                validate_files=True,
            )
            manifest = dataset.to_jsonl(root / "conversation.jsonl")
            payload = json.loads(manifest.read_text(encoding="utf-8"))

            resolved = dataset[0]["conversation"][0]["content"][0]["audio"]["path"]
            portable = payload["conversation"][0]["content"][0]["audio"]["path"]
            self.assertEqual(resolved, str(audio.resolve()))
            self.assertEqual(portable, "turns/one.wav")
            with self.assertRaises(FileNotFoundError):
                TTSDataset(
                    [{
                        "messages": conversation("turns/missing.wav"),
                    }],
                    model_type="csm",
                    root=root,
                    validate_files=True,
                )

    def test_grouped_split_is_deterministic_and_leakage_safe(self):
        records = [{
            "id": f"record-{index}",
            "text": f"Text {index}",
            "audio": f"{index}.wav",
            "speaker_id": f"speaker-{index // 2}",
        } for index in range(8)]
        dataset = TTSDataset(
            records,
            architecture="codec-lm",
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

    def test_fingerprint_covers_content_and_order(self):
        records = [
            {
                "text": "First",
                "audio": "first.wav",
            },
            {
                "text": "Second",
                "audio": "second.wav",
            },
        ]
        original = TTSDataset(records, architecture="diffusion")
        reordered = TTSDataset(list(reversed(records)), architecture="diffusion")
        changed = TTSDataset(
            [
                records[0],
                {
                    **records[1],
                    "text": "Changed",
                },
            ],
            architecture="diffusion",
        )

        self.assertNotEqual(
            original.resume_fingerprint()["content_sha256"],
            reordered.resume_fingerprint()["content_sha256"],
        )
        self.assertNotEqual(
            original.resume_fingerprint()["content_sha256"],
            changed.resume_fingerprint()["content_sha256"],
        )

    def test_transform_requires_an_explicit_resume_fingerprint(self):

        def annotate(record):
            return {
                **record,
                "revision": 1,
            }

        records = [{
            "text": "Transformed",
            "audio": "sample.wav",
        }]
        unversioned = TTSDataset(
            records,
            architecture="diffusion",
            transform=annotate,
        )
        version_one = TTSDataset(
            records,
            architecture="diffusion",
            transform=annotate,
            transform_fingerprint="annotate-v1",
        )
        version_two = TTSDataset(
            records,
            architecture="diffusion",
            transform=annotate,
            transform_fingerprint="annotate-v2",
        )

        with self.assertRaisesRegex(ValueError, "transform_fingerprint"):
            unversioned.resume_fingerprint()
        self.assertNotEqual(
            version_one.resume_fingerprint()["content_sha256"],
            version_two.resume_fingerprint()["content_sha256"],
        )

    def test_fingerprint_rejects_non_string_mapping_keys(self):
        dataset = TTSDataset(
            [{
                "text": "Typed keys",
                "audio": "sample.wav",
                "metadata": {
                    1: "integer",
                    "1": "string",
                },
            }],
            architecture="diffusion",
        )

        with self.assertRaisesRegex(TypeError, "string mapping keys"):
            dataset.resume_fingerprint()

    def test_dataset_is_pickle_safe_for_worker_processes(self):
        dataset = TTSDataset(
            [{
                "text": "Worker-safe",
                "audio": "worker.wav",
            }],
            architecture="sequence-to-sequence",
        )

        restored = pickle.loads(pickle.dumps(dataset))

        self.assertEqual(restored[0], dataset[0])
        self.assertEqual(restored.resume_fingerprint(), dataset.resume_fingerprint())

    def test_jsonl_round_trip_writes_relative_audio_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            dataset = TTSDataset(
                [{
                    "text": "Hello",
                    "audio": str(audio),
                }],
                architecture="vits",
            )
            manifest = dataset.to_jsonl(root / "prepared.jsonl")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            restored = TTSDataset.from_manifest(
                manifest,
                architecture="vits",
            )

        self.assertEqual(payload["audio"], "audio.wav")
        self.assertEqual(restored[0]["audio"], str(audio.resolve()))
        self.assertEqual(restored[0]["text"], "Hello")

    def test_jsonl_rejects_credentials_before_destination_mutation(self):
        credential = "must-not-appear-in-errors"
        dataset = TTSDataset(
            [{
                "text": "Unsafe manifest record",
                "audio": "sample.wav",
                "metadata": {
                    "api_key": credential,
                },
            }],
            architecture="vits",
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
                        r"TTS dataset record 0.*metadata\.api_key",
                ) as captured:
                    dataset.to_jsonl(destination)
                self.assertNotIn(credential, str(captured.exception))

            self.assertEqual(existing.read_text(encoding="utf-8"), "keep-me\n")
            self.assertFalse(missing_parent.parent.exists())

    def test_jsonl_serialization_failure_does_not_truncate_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "records.jsonl"
            destination.write_text("keep-me\n", encoding="utf-8")
            dataset = TTSDataset(
                [
                    {
                        "text": "Serializable record",
                        "audio": "one.wav",
                    },
                    {
                        "text": "Invalid record",
                        "audio": "two.wav",
                        "metadata": object(),
                    },
                ],
                architecture="vits",
                validate=False,
            )

            with self.assertRaisesRegex(TypeError, "record 1"):
                dataset.to_jsonl(destination)

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "keep-me\n",
            )

    def test_manifest_loader_rejects_credentials_and_keeps_safe_metadata(self):
        credential = "must-not-appear-in-errors"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe = root / "unsafe.jsonl"
            unsafe.write_text(
                json.dumps({
                    "text": "Unsafe manifest record",
                    "audio": "sample.wav",
                    "metadata": {
                        "authorization": credential,
                    },
                }) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                    ValueError,
                    r"TTS manifest record 0.*metadata\.authorization",
            ) as captured:
                TTSDataset.from_manifest(
                    unsafe,
                    architecture="vits",
                    validate=False,
                )

            safe = root / "safe.jsonl"
            safe.write_text(
                json.dumps({
                    "text": "Safe manifest record",
                    "audio": "sample.wav",
                    "metadata": {
                        "token_count": 512,
                    },
                }) + "\n",
                encoding="utf-8",
            )
            restored = TTSDataset.from_manifest(
                safe,
                architecture="vits",
                validate=False,
            )

        self.assertNotIn(credential, str(captured.exception))
        self.assertEqual(restored[0]["metadata"]["token_count"], 512)


class TTSModelManifestIntegrationTests(unittest.TestCase):

    class _CapturingAdapter:

        def __init__(self):
            self.records = None
            self.kwargs = None

        def create_dataset(self, records, **kwargs):
            self.records = records
            self.kwargs = kwargs
            return records

    class _Model(PreTrainedTTSModel):

        def __init__(self, config):
            super().__init__(config, device="cpu")
            self.adapter = TTSModelManifestIntegrationTests._CapturingAdapter()

        def get_training_adapter(self):
            return self.adapter

        def _load_pretrained_model(self):
            raise AssertionError("Dataset creation must not load model weights.")

        def _generate(self, text: str, **kwargs):
            return TTSOutput(audio=[0.0], sample_rate=24_000)

    def test_model_accepts_manifest_path_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "records.jsonl"
            manifest.write_text(
                json.dumps({
                    "text": "Hello",
                    "audio": "hello.wav",
                }) + "\n",
                encoding="utf-8",
            )
            config = VoiceHubConfig(name_or_path="dummy")
            config.model_type = "orpheustts"
            model = self._Model(config)

            dataset = model.create_training_dataset(
                manifest,
                completion_only=True,
            )

        self.assertIsInstance(dataset, TTSDataset)
        self.assertFalse(model.is_loaded)
        self.assertEqual(model.adapter.kwargs, {"completion_only": True})
        self.assertEqual(
            model.adapter.records[0]["audio"],
            str((manifest.parent / "hello.wav").resolve()),
        )

    def test_model_data_options_normalize_in_memory_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = VoiceHubConfig(name_or_path="dummy")
            config.model_type = "orpheustts"
            model = self._Model(config)

            dataset = model.create_training_dataset(
                [{
                    "audio_path": "hello.wav",
                    "transcript": "Hello",
                }],
                data_root=root,
            )

        self.assertIsInstance(dataset, TTSDataset)
        self.assertFalse(model.is_loaded)
        self.assertEqual(model.adapter.kwargs, {})
        self.assertEqual(dataset[0]["text"], "Hello")
        self.assertEqual(
            dataset[0]["audio"],
            str((root / "hello.wav").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
