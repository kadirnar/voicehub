from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path, PureWindowsPath

from scripts.scaffold_model import (
    IMPLEMENTATION_STATUS,
    MODEL_PAGE_SECTIONS,
    READY_STATUS,
    TASKS,
    ScaffoldError,
    _display_relative_path,
    check_model_scaffold,
    create_model_scaffold,
    main,
    render_builtin_catalog_fragments,
    scaffold_files,
)
from voicehub.models.manifests import discover_builtin_model_manifests
from voicehub.registry import discover_manifest_model_specs
from voicehub.training.specs import discover_manifest_training_specs


class ModelScaffoldTests(unittest.TestCase):

    def test_displayed_paths_are_platform_independent(self):
        root = PureWindowsPath("C:/workspace/voicehub")
        path = root / "voicehub/models/auroratts/runtime.py"

        self.assertEqual(
            _display_relative_path(path, root),
            "voicehub/models/auroratts/runtime.py",
        )

    def _files(self, *, task="tts"):
        return scaffold_files(
            model_type="auroratts",
            class_prefix="AuroraTTS",
            task=task,
            checkpoint="acme/aurora-base",
            source_url="https://github.com/acme/aurora-tts",
            source_revision="0123456789abcdef0123456789abcdef01234567",
            license_id="Apache-2.0",
            license_text="Authoritative upstream license fixture.",
            aliases=("aurora-tts", ),
        )

    def _complete_scaffold(
        self,
        root: Path,
        *,
        task: str = "tts",
        registry_source: str | None = None,
        training_source: str | None = None,
    ) -> None:
        create_model_scaffold(root, self._files(task=task))
        task_template = TASKS[task]
        task_enum = {
            "tts": "TEXT_TO_SPEECH",
            "asr": "AUTOMATIC_SPEECH_RECOGNITION",
            "vad": "VOICE_ACTIVITY_DETECTION",
            "codec": "AUDIO_CODEC",
        }[task]
        model_class = "AuroraTTS" + task_template.model_suffix
        model_path = root / "voicehub/models/auroratts/modeling.py"
        model_path.write_text(
            model_path.read_text(encoding="utf-8").replace(
                f"IMPLEMENTATION_STATUS = {IMPLEMENTATION_STATUS!r}",
                f"IMPLEMENTATION_STATUS = {READY_STATUS!r}",
            ),
            encoding="utf-8",
        )
        source_path = root / "voicehub/models/auroratts/source/SOURCE.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["checkpoint"]["revision"] = "89abcdef0123456789abcdef0123456789abcdef"
        source_path.write_text(
            json.dumps(source, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if registry_source is None:
            registry_source = textwrap.dedent(
                f'''\
                _MODEL_SPECS = (
                    ModelSpec(
                        "auroratts",
                        "voicehub.models.auroratts.modeling",
                        {model_class!r},
                        "acme/aurora-base",
                        capabilities=({task_template.task!r},),
                        config_module="voicehub.models.auroratts.configuration",
                        config_class="AuroraTTSConfig",
                        task=SpeechTask.{task_enum},
                    ),
                )
                _BUILTIN_MODEL_ALIASES = {{"aurora-tts": "auroratts"}}
                ''')
        registry_path = root / "voicehub/models/catalog.py"
        registry_path.write_text(registry_source, encoding="utf-8")
        if training_source is None:
            training_source = textwrap.dedent(
                f'''\
                _BUILTIN_TRAINING_SPECS = (
                    _profile(
                        "auroratts",
                        TrainingFamily.{task_template.training_family},
                        task=SpeechTask.{task_enum},
                    ),
                )
                ''')
        training_path = root / "voicehub/training/specs.py"
        training_path.parent.mkdir(parents=True, exist_ok=True)
        training_path.write_text(training_source, encoding="utf-8")
        (root / "mkdocs.yml").write_text(
            "nav:\n  - auroratts: models/providers/auroratts.md\n",
            encoding="utf-8",
        )

    def test_every_task_renders_the_shared_contract_file_set(self):
        expected = {
            "tts": (
                "PreTrainedTTSModel",
                "TTSOutput",
                "_generate",
                "AutoModelForTextToSpeech.register(",
                "TrainingFamily.ACOUSTIC",
                "components=()",
            ),
            "asr": (
                "PreTrainedASRModel",
                "ASROutput",
                "_transcribe",
                "AutoModelForSpeechRecognition.register(",
                "TrainingFamily.CTC",
                "components=()",
            ),
            "vad": (
                "PreTrainedVADModel",
                "VADOutput",
                "_detect",
                "AutoModelForVoiceActivityDetection.register(",
                "TrainingFamily.AUDIO_CLASSIFICATION",
                "components=()",
            ),
        }
        expected["codec"] = (
            "PreTrainedCodecModel", "_encode", "_decode", "AutoModelForAudioCodec.register(",
            "TrainingFamily.AUDIO_CODEC")
        required_paths = {
            Path("voicehub/models/auroratts/__init__.py"),
            Path("voicehub/models/auroratts/configuration.py"),
            Path("voicehub/models/auroratts/modeling.py"),
            Path("voicehub/models/auroratts/runtime.py"),
            Path("voicehub/models/auroratts/registration.py"),
            Path("voicehub/models/auroratts/model-integration.json"),
            Path("voicehub/models/auroratts/source/SOURCE.json"),
            Path("voicehub/models/auroratts/source/THIRD_PARTY_LICENSE"),
            Path("tests/test_auroratts.py"),
            Path("docs/models/providers/auroratts.md"),
        }

        for task, fragments in expected.items():
            with self.subTest(task=task):
                files = self._files(task=task)
                self.assertEqual(set(files), required_paths)
                python_source = "\n".join(source for path, source in files.items() if path.suffix == ".py")
                for path, source in files.items():
                    if path.suffix == ".py":
                        compile(source, str(path), "exec")
                for fragment in fragments:
                    self.assertIn(fragment, python_source)

                page = files[Path("docs/models/providers/auroratts.md")]
                positions = tuple(page.index(f"## {heading}") for heading in MODEL_PAGE_SECTIONS)
                self.assertEqual(positions, tuple(sorted(positions)))
                self.assertIn("Unverified scaffold", page)

    def test_inputs_reject_unsafe_or_ambiguous_values(self):
        cases = (
            ({
                "model_type": "../escape"
            }, "model_type"),
            ({
                "class_prefix": "aurora_tts"
            }, "class_prefix"),
            ({
                "source_url": "http://example.com/source"
            }, "HTTPS"),
            ({
                "source_revision": "main"
            }, "immutable"),
            ({
                "license_text": "  "
            }, "license_text"),
            ({
                "aliases": ("aurora-tts", "aurora-tts")
            }, "duplicates"),
            ({
                "aliases": ("auroratts", )
            }, "must not equal"),
        )
        base = {
            "model_type": "auroratts",
            "class_prefix": "AuroraTTS",
            "task": "tts",
            "checkpoint": "acme/aurora-base",
            "source_url": "https://github.com/acme/aurora-tts",
            "source_revision": "0123456789abcdef0123456789abcdef01234567",
            "license_id": "Apache-2.0",
            "license_text": "License fixture",
            "aliases": ("aurora-tts", ),
        }

        for changes, message in cases:
            with self.subTest(changes=changes):
                values = {**base, **changes}
                with self.assertRaisesRegex(ScaffoldError, message):
                    scaffold_files(**values)

    def test_creation_never_overwrites_an_existing_artifact(self):
        files = self._files()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            created = create_model_scaffold(root, files)
            license_path = (root / "voicehub/models/auroratts/source/THIRD_PARTY_LICENSE")
            self.assertEqual(len(created), len(files))
            self.assertEqual(
                license_path.read_text(encoding="utf-8"),
                "Authoritative upstream license fixture.\n",
            )

            with self.assertRaisesRegex(ScaffoldError, "Refusing to overwrite"):
                create_model_scaffold(root, files)
            self.assertEqual(
                license_path.read_text(encoding="utf-8"),
                "Authoritative upstream license fixture.\n",
            )

    def test_checker_reports_each_material_omission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_model_scaffold(root, self._files())
            (root / "voicehub/models/auroratts/runtime.py").unlink()
            (root / "voicehub/models/auroratts/source/THIRD_PARTY_LICENSE").write_text(
                "",
                encoding="utf-8",
            )
            test_path = root / "tests/test_auroratts.py"
            test_path.write_text(
                test_path.read_text(encoding="utf-8").replace(
                    "apply_optimization_plan",
                    "missing_optimization_contract",
                ),
                encoding="utf-8",
            )
            page_path = root / "docs/models/providers/auroratts.md"
            page_path.write_text(
                page_path.read_text(encoding="utf-8").replace(
                    "## Quickstart",
                    "## Missing quickstart",
                ),
                encoding="utf-8",
            )
            registry_path = root / "voicehub/models/catalog.py"
            registry_path.write_text("MODEL_SPECS = ()\n", encoding="utf-8")

            errors = "\n".join(check_model_scaffold(root, "auroratts"))

        for fragment in (
                "missing voicehub/models/auroratts/runtime.py",
                "IMPLEMENTATION_STATUS is 'replace-me'",
                "checkpoint.revision must be replaced",
                "THIRD_PARTY_LICENSE must contain",
                "contract coverage marker 'apply_optimization_plan'",
                "common section order",
                "mkdocs.yml is missing",
                "built-in registry discovery is missing",
                "missing voicehub/training/specs.py",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, errors)

    def test_checker_accepts_a_structurally_completed_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_scaffold(root)

            self.assertEqual(check_model_scaffold(root, "auroratts"), ())

    def test_checker_accepts_builtin_contracts_for_every_speech_task(self):
        for task in TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._complete_scaffold(root, task=task)

                self.assertEqual(check_model_scaffold(root, "auroratts"), ())

    def test_checker_preserves_the_external_registration_path_without_central_catalogs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_scaffold(root)
            (root / "voicehub/models/catalog.py").unlink()
            (root / "voicehub/training/specs.py").unlink()

            self.assertEqual(check_model_scaffold(root, "auroratts"), ())

    def test_checker_rejects_quoted_names_without_a_model_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_scaffold(
                root,
                registry_source=textwrap.dedent(
                    '''\
                    # "auroratts" is documentation, not a registry declaration.
                    _MODEL_SPECS = ()
                    _BUILTIN_MODEL_ALIASES = {"aurora-tts": "auroratts"}
                    '''),
            )

            errors = "\n".join(check_model_scaffold(root, "auroratts"))

        self.assertIn("built-in registry discovery is missing", errors)

    def test_checker_reports_each_mismatched_builtin_contract_field(self):
        registry_source = textwrap.dedent(
            '''\
            _MODEL_SPECS = (
                ModelSpec(
                    "auroratts",
                    "voicehub.models.wrong.modeling_wrong",
                    "WrongModel",
                    "acme/wrong-checkpoint",
                    config_module="voicehub.models.wrong.configuration_wrong",
                    config_class="WrongConfig",
                    task=SpeechTask.VOICE_ACTIVITY_DETECTION,
                ),
            )
            _BUILTIN_MODEL_ALIASES = {"aurora-tts": "auroratts"}
            ''')
        training_source = textwrap.dedent(
            '''\
            _BUILTIN_TRAINING_SPECS = (
                _profile(
                    "auroratts",
                    TrainingFamily.ACOUSTIC,
                    task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
                ),
            )
            ''')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_scaffold(
                root,
                registry_source=registry_source,
                training_source=training_source,
            )

            errors = "\n".join(check_model_scaffold(root, "auroratts"))

        for fragment in (
                "ModelSpec module must be",
                "ModelSpec class_name must be",
                "ModelSpec default_model_path must be",
                "ModelSpec config_module must be",
                "ModelSpec config_class must be",
                "ModelSpec task must be 'text-to-speech'",
                "training profile task must be 'text-to-speech'",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, errors)

    def test_checker_requires_manifest_aliases_and_builtin_training_profile(self):
        registry_source = textwrap.dedent(
            '''\
            _MODEL_SPECS = (
                ModelSpec(
                    "auroratts",
                    "voicehub.models.auroratts.modeling",
                    "AuroraTTSForTextToSpeech",
                    "acme/aurora-base",
                    config_module="voicehub.models.auroratts.configuration",
                    config_class="AuroraTTSConfig",
                ),
            )
            _BUILTIN_MODEL_ALIASES = {}
            ''')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_scaffold(
                root,
                registry_source=registry_source,
                training_source="_BUILTIN_TRAINING_SPECS = ()\n",
            )

            errors = "\n".join(check_model_scaffold(root, "auroratts"))

        self.assertIn("built-in alias 'aurora-tts' must target 'auroratts'", errors)
        self.assertIn("built-in training profile is missing", errors)

    def test_catalog_fragments_complete_every_builtin_speech_task_without_mutation(self):
        for task in TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._complete_scaffold(root, task=task)
                before = {
                    path.relative_to(root): path.read_bytes()
                    for path in root.rglob("*") if path.is_file()
                }

                first = render_builtin_catalog_fragments(root, "auroratts")
                second = render_builtin_catalog_fragments(root, "auroratts")

                self.assertEqual(first, second)
                self.assertEqual(
                    before,
                    {path.relative_to(root): path.read_bytes()
                     for path in root.rglob("*") if path.is_file()},
                )
                registry_source = (
                    "_MODEL_SPECS = (\n" + textwrap.indent(first.model_spec, "    ") +
                    ")\n_BUILTIN_MODEL_ALIASES = {\n" + first.aliases + "}\n")
                training_source = (
                    "_BUILTIN_TRAINING_SPECS = (\n" + textwrap.indent(first.training_spec, "    ") + ")\n")
                ast.parse(registry_source, filename="generated-registry.py")
                ast.parse(training_source, filename="generated-training.py")
                (root / "voicehub/models/catalog.py").write_text(
                    registry_source,
                    encoding="utf-8",
                )
                (root / "voicehub/training/specs.py").write_text(
                    training_source,
                    encoding="utf-8",
                )

                self.assertEqual(check_model_scaffold(root, "auroratts"), ())

    def test_incomplete_manifest_is_not_discovered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_model_scaffold(root, self._files())
            manifest_path = root / "voicehub/models/auroratts/model-integration.json"

            self.assertEqual(
                discover_builtin_model_manifests(root / "voicehub/models"),
                (),
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["builtin"] = True
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    ValueError,
                    "IMPLEMENTATION_STATUS = 'ready'",
            ):
                discover_builtin_model_manifests(root / "voicehub/models")

    def test_inactive_ambiguous_manifest_stays_undiscovered(self):
        documents = (
            '{"builtin":false,"audit":"discarded-secret","audit":1}',
            '{"builtin":false,"audit":NaN}',
            '{"builtin":false,"audit":1e400}',
        )
        for document in documents:
            with self.subTest(document=document), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package = root / "voicehub/models/auroratts"
                package.mkdir(parents=True)
                (package / "model-integration.json").write_text(
                    document,
                    encoding="utf-8",
                )

                self.assertEqual(
                    discover_builtin_model_manifests(root / "voicehub/models"),
                    (),
                )

    def test_active_manifest_and_source_reject_ambiguous_json_before_discovery(self):
        ambiguities = (
            (
                "duplicate",
                r"Duplicate JSON object key 'model_type'",
            ),
            (
                "constant",
                r"non-finite JSON constant 'NaN'",
            ),
            (
                "overflow",
                r"\$\.audit.*non-finite",
            ),
        )
        for artifact in ("model-integration.json", "SOURCE.json"):
            for ambiguity, diagnostic in ambiguities:
                with self.subTest(
                        artifact=artifact,
                        ambiguity=ambiguity,
                ), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._complete_scaffold(root)
                    package = root / "voicehub/models/auroratts"
                    manifest_path = package / "model-integration.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["builtin"] = True
                    manifest_path.write_text(
                        json.dumps(manifest, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    source_path = package / "source/SOURCE.json"
                    target = manifest_path if artifact == "model-integration.json" else source_path
                    document = json.dumps(
                        json.loads(target.read_text(encoding="utf-8")),
                        sort_keys=True,
                    )
                    if ambiguity == "duplicate":
                        document = document.replace(
                            '"model_type": "auroratts"',
                            '"model_type": "discarded-secret", "model_type": "auroratts"',
                            1,
                        )
                    else:
                        value = "NaN" if ambiguity == "constant" else "1e400"
                        document = document[:-1] + f', "audit": {value}}}'
                    target.write_text(document + "\n", encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, diagnostic) as discovered:
                        discover_builtin_model_manifests(root / "voicehub/models")
                    discovery_error = str(discovered.exception)
                    self.assertIn(str(target), discovery_error)
                    self.assertNotIn("discarded-secret", discovery_error)

                    checker_errors = "\n".join(check_model_scaffold(root, "auroratts"))
                    self.assertIn(artifact, checker_errors)
                    self.assertRegex(checker_errors, diagnostic)
                    self.assertNotIn("discarded-secret", checker_errors)

                    if artifact == "model-integration.json":
                        with self.assertRaisesRegex(ScaffoldError, diagnostic) as catalog:
                            render_builtin_catalog_fragments(root, "auroratts")
                        self.assertNotIn("discarded-secret", str(catalog.exception))

    def test_activated_manifest_needs_no_central_catalog_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_scaffold(root)
            manifest_path = root / "voicehub/models/auroratts/model-integration.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["builtin"] = True
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (root / "voicehub/models/catalog.py").write_text(
                "_MODEL_SPECS = ()\n_BUILTIN_MODEL_ALIASES = {}\n",
                encoding="utf-8",
            )
            (root / "voicehub/training/specs.py").write_text(
                "_BUILTIN_TRAINING_SPECS = ()\n",
                encoding="utf-8",
            )
            before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

            manifests = discover_builtin_model_manifests(root / "voicehub/models")
            model_specs = discover_manifest_model_specs(root / "voicehub/models")
            training_specs = discover_manifest_training_specs(root / "voicehub/models")

            self.assertEqual(check_model_scaffold(root, "auroratts"), ())
            self.assertEqual(len(manifests), 1)
            self.assertEqual(manifests[0].aliases, ("aurora-tts", ))
            self.assertEqual(model_specs[0].model_type, "auroratts")
            self.assertEqual(model_specs[0].class_name, "AuroraTTSForTextToSpeech")
            self.assertEqual(training_specs[0].model_type, "auroratts")
            self.assertEqual(training_specs[0].support.value, "inference-only")
            self.assertEqual(
                before,
                {path.relative_to(root): path.read_bytes()
                 for path in root.rglob("*") if path.is_file()},
            )

            script = textwrap.dedent(
                f'''\
                import json
                import sys
                from pathlib import Path
                from voicehub.registry import discover_manifest_model_specs
                from voicehub.training.specs import discover_manifest_training_specs

                root = Path({str(root / "voicehub/models")!r})
                print(json.dumps({{
                    "models": len(discover_manifest_model_specs(root)),
                    "training": len(discover_manifest_training_specs(root)),
                    "torch_imported": "torch" in sys.modules,
                    "model_package_imported": any(
                        name.startswith("voicehub.models.auroratts") for name in sys.modules
                    ),
                }}, sort_keys=True))
                ''')
            completed = subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(completed.stdout),
                {
                    "model_package_imported": False,
                    "models": 1,
                    "torch_imported": False,
                    "training": 1,
                },
            )

    def test_activated_manifest_rejects_unexpressed_training_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_model_scaffold(root, self._files())
            manifest_path = root / "voicehub/models/auroratts/model-integration.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["builtin"] = True
            manifest["training"]["support"] = "native"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                    ValueError,
                    "register richer training metadata explicitly",
            ):
                discover_builtin_model_manifests(root / "voicehub/models")

    def test_activated_manifest_rejects_duplicate_central_declarations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_scaffold(root)
            manifest_path = root / "voicehub/models/auroratts/model-integration.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["builtin"] = True
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            errors = "\n".join(check_model_scaffold(root, "auroratts"))

            self.assertIn("duplicates a central ModelSpec", errors)
            self.assertIn("duplicates central aliases", errors)
            self.assertIn("duplicates a central training profile", errors)

    def test_catalog_renderer_reports_invalid_manifest_instead_of_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_model_scaffold(root, self._files())
            manifest_path = root / "voicehub/models/auroratts/model-integration.json"
            manifest_path.write_text("{not-json}\n", encoding="utf-8")

            with self.assertRaisesRegex(ScaffoldError, "invalid model-integration.json"):
                render_builtin_catalog_fragments(root, "auroratts")

    def test_each_task_registration_runs_without_central_rewrites_or_torch(self):
        cases = {
            "tts": (
                "AutoModelForTextToSpeech",
                "text-to-speech",
                "AutoModelForSpeechRecognition",
            ),
            "asr": (
                "AutoModelForSpeechRecognition",
                "automatic-speech-recognition",
                "AutoModelForVoiceActivityDetection",
            ),
            "vad": (
                "AutoModelForVoiceActivityDetection",
                "voice-activity-detection",
                "AutoModelForTextToSpeech",
            ),
        }
        for task, (factory, task_value, wrong_factory) in cases.items():
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                create_model_scaffold(root, self._files(task=task))
                script = textwrap.dedent(
                    f'''\
                    import sys
                    from pathlib import Path

                    root = Path({str(root)!r})
                    import voicehub
                    import voicehub.models

                    voicehub.__path__.append(str(root / "voicehub"))
                    voicehub.models.__path__.append(str(root / "voicehub" / "models"))

                    from voicehub import AutoModel, {factory}, {wrong_factory}
                    from voicehub.models.auroratts.registration import (
                        register_auroratts,
                        unregister_auroratts,
                    )

                    try:
                        spec = register_auroratts()
                        assert spec.model_type == "auroratts"
                        assert spec.task.value == {task_value!r}
                        model = {factory}.from_pretrained(
                            "acme/aurora-base",
                            model_type="auroratts",
                            device="cpu",
                            lazy_load=True,
                        )
                        generic_model = AutoModel.from_pretrained(
                            "acme/aurora-base",
                            model_type="auroratts",
                            device="cpu",
                            lazy_load=True,
                        )
                        assert model.config.model_type == "auroratts"
                        assert type(generic_model) is type(model)
                        assert not model.is_loaded
                        assert not generic_model.is_loaded
                        try:
                            {wrong_factory}.from_pretrained(
                                "acme/aurora-base",
                                model_type="auroratts",
                                device="cpu",
                                lazy_load=True,
                            )
                        except ValueError as exc:
                            assert "cannot be loaded by" in str(exc)
                            assert {factory!r} in str(exc)
                        else:
                            raise AssertionError("wrong-task factory accepted the scaffold")
                        assert "torch" not in sys.modules
                    finally:
                        unregister_auroratts()
                ''')
                completed = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=Path(__file__).resolve().parents[1],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_creates_an_explicitly_incomplete_scaffold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            license_path = root / "UPSTREAM_LICENSE"
            license_path.write_text("License fixture\n", encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main([
                    "create",
                    "--model-type",
                    "auroratts",
                    "--class-prefix",
                    "AuroraTTS",
                    "--task",
                    "tts",
                    "--checkpoint",
                    "acme/aurora-base",
                    "--source-url",
                    "https://github.com/acme/aurora-tts",
                    "--source-revision",
                    "0123456789abcdef0123456789abcdef01234567",
                    "--license-id",
                    "Apache-2.0",
                    "--license-file",
                    str(license_path),
                    "--output-root",
                    str(root),
                ])
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                check_result = main([
                    "check",
                    "--model-type",
                    "auroratts",
                    "--output-root",
                    str(root),
                ])

        self.assertEqual(result, 0)
        self.assertIn("INCOMPLETE", stdout.getvalue())
        self.assertIn("ModelSpec, aliases, and training profile", stdout.getvalue())
        self.assertEqual(check_result, 1)
        self.assertIn("IMPLEMENTATION_STATUS", stderr.getvalue())
        self.assertIn("checkpoint.revision", stderr.getvalue())

    def test_cli_renders_named_catalog_insertion_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_model_scaffold(root, self._files(task="asr"))
            before = tuple(sorted(path.relative_to(root) for path in root.rglob("*")))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main([
                    "catalog",
                    "--model-type",
                    "auroratts",
                    "--output-root",
                    str(root),
                ])
            after = tuple(sorted(path.relative_to(root) for path in root.rglob("*")))
            import_probe = textwrap.dedent(
                f'''\
                import sys
                from scripts.scaffold_model import main

                result = main([
                    "catalog",
                    "--model-type", "auroratts",
                    "--output-root", {str(root)!r},
                ])
                assert result == 0
                assert "voicehub" not in sys.modules
                assert "torch" not in sys.modules
                ''')
            completed = subprocess.run(
                [sys.executable, "-c", import_probe],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

        output = stdout.getvalue()
        self.assertEqual(result, 0)
        self.assertEqual(before, after)
        self.assertIn("voicehub/models/catalog.py :: _MODEL_SPECS", output)
        self.assertIn("voicehub/training/specs.py :: _BUILTIN_TRAINING_SPECS", output)
        self.assertIn("SpeechTask.AUTOMATIC_SPEECH_RECOGNITION", output)
        self.assertIn("'aurora-tts': 'auroratts'", output)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
