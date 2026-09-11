from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from voicehub.policies.architecture_dependencies import (
    collect_native_import_closure,
    collect_native_runtime_paths,
    discover_native_runtime_directories,
    inspect_native_imports,
    inspect_native_runtime,
    require_native_runtime_independence,
)
from voicehub.registry import list_model_specs

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "voicehub"


class NativeDependencyPolicyTests(unittest.TestCase):

    @staticmethod
    def _module_path(module_name: str) -> Path:
        prefix = "voicehub."
        if not module_name.startswith(prefix):
            raise AssertionError(f"Native registry module is outside VoiceHub: {module_name!r}")
        relative = Path(*module_name[len(prefix):].split("."))
        module_file = (PACKAGE_ROOT / relative).with_suffix(".py")
        if module_file.is_file():
            return module_file
        package_file = PACKAGE_ROOT / relative / "__init__.py"
        if package_file.is_file():
            return package_file
        raise AssertionError(f"Native registry module does not resolve to source: {module_name!r}")

    def test_native_runtime_uses_only_stdlib_voicehub_and_torch(self):
        violations = inspect_native_runtime(PACKAGE_ROOT)
        self.assertEqual(
            violations,
            (),
            "\n".join(str(violation) for violation in violations),
        )
        require_native_runtime_independence(PACKAGE_ROOT)

    def test_migrated_public_providers_are_inside_the_native_boundary(self):
        violations = inspect_native_runtime(
            PACKAGE_ROOT,
            directories=(
                "models/asr_hubert",
                "models/asr_granite_speech",
                "models/asr_parakeet_tdt",
                "models/asr_nemotron",
                "models/asr_cohere",
                "models/asr_seamless_m4t_v2",
                "models/asr_vibevoice",
                "models/asr_medasr",
                "models/asr_transformers",
                "models/asr_wavlm",
                "models/asr_whisper_native",
                "models/dia",
                "models/fishtts/__init__.py",
                'models/fishtts/configuration.py',
                'models/fishtts/modeling.py',
                'models/fishtts/modeling.py',
                "models/fishtts/training.py",
                "models/higgstts/__init__.py",
                'models/higgstts/configuration.py',
                "models/higgstts/inference.py",
                'models/higgstts/modeling.py',
                "models/higgstts/training.py",
                "models/openvoice/__init__.py",
                'models/openvoice/configuration.py',
                'models/openvoice/modeling.py',
                'models/openvoice/modeling.py',
                "models/openvoice/training.py",
                "models/openvoice/source/openvoice/models.py",
                "models/openvoice/source/openvoice/modules.py",
                "models/openvoice/source/openvoice/commons.py",
                "models/openvoice/source/openvoice/attentions.py",
                "models/openvoice/source/openvoice/transforms.py",
                "models/vits",
                "models/vibevoice/__init__.py",
                'models/vibevoice/configuration.py',
                'models/vibevoice/modeling.py',
                'models/vibevoice/modeling.py',
                "models/vibevoice/training.py",
                "models/voxcpm/__init__.py",
                'models/voxcpm/configuration.py',
                'models/voxcpm/modeling.py',
                'models/voxcpm/modeling.py',
                "models/voxcpm/training.py",
                'models/voxcpm/native',
                "models/vad_pyannote",
                "models/vad_pyannote_brouhaha",
                "models/vad_pyannote_segmentation",
                "models/vad_silero",
                "models/vad_funasr",
                "models/vad_nemo",
                "models/vad_sherpa_onnx",
                "models/vad_webrtc",
            ),
        )
        for path in (
                PACKAGE_ROOT / "models/asr_native/configuration.py",
                PACKAGE_ROOT / "models/asr_native/faster_whisper.py",
                PACKAGE_ROOT / "models/asr_native/openai_whisper.py",
                PACKAGE_ROOT / "models/asr_native/whisper_compat.py",
        ):
            violations += inspect_native_imports(path)
        self.assertEqual(
            violations,
            (),
            "\n".join(str(violation) for violation in violations),
        )

    def test_every_native_registry_facade_is_inside_the_policy_boundary(self):
        covered = set(collect_native_runtime_paths(PACKAGE_ROOT))
        uncovered = {}
        for spec in list_model_specs(task=None):
            if not spec.is_voicehub_native:
                continue
            paths = {
                self._module_path(spec.module),
                self._module_path(spec.config_module),
            }
            missing = tuple(sorted(paths - covered))
            if missing:
                uncovered[spec.model_type] = tuple(str(path.relative_to(PACKAGE_ROOT)) for path in missing)

        self.assertEqual(uncovered, {})

    def test_every_model_package_facade_is_discovered_without_a_provider_list(self):
        discovered = set(discover_native_runtime_directories(PACKAGE_ROOT))
        expected = {
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in (PACKAGE_ROOT / "models").glob("*/*.py")
        }

        self.assertTrue(expected)
        self.assertEqual(expected - discovered, set())

    def test_new_model_facade_joins_the_default_policy_without_a_central_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "voicehub"
            model_package = root / "models" / "future_speech"
            model_package.mkdir(parents=True)
            (root / "__init__.py").write_text("", encoding="utf-8")
            (root / "models" / "__init__.py").write_text("", encoding="utf-8")
            (model_package / "__init__.py").write_text("", encoding="utf-8")
            model_path = model_package / "modeling_future_speech.py"
            model_path.write_text("import transformers\n", encoding="utf-8")

            discovered = discover_native_runtime_directories(root)
            violations = inspect_native_runtime(root)

        self.assertIn(
            "models/future_speech/modeling_future_speech.py",
            discovered,
        )
        self.assertEqual(
            tuple(violation.module for violation in violations),
            ("transformers", ),
        )

    def test_every_declared_architecture_component_is_inside_the_policy_boundary(self, ):
        from voicehub.runtime import list_architecture_specs

        covered = set(collect_native_runtime_paths(PACKAGE_ROOT))
        uncovered = {}
        for spec in list_architecture_specs():
            for component_name, reference in spec.component_references.items():
                if reference.module == "torch" or reference.module.startswith("torch."):
                    continue
                try:
                    component_path = self._module_path(reference.module)
                except AssertionError:
                    uncovered.setdefault(spec.architecture_id, {})[component_name] = reference.module
                    continue
                if component_path not in covered:
                    uncovered.setdefault(spec.architecture_id,
                                         {})[component_name] = str(component_path.relative_to(PACKAGE_ROOT))

        self.assertEqual(uncovered, {})

    def test_outetts_active_runtime_is_covered_but_dormant_source_is_not(self):
        covered = set(collect_native_runtime_paths(PACKAGE_ROOT))
        expected = {
            *(PACKAGE_ROOT / 'models/outetts/native').glob("*.py"),
            PACKAGE_ROOT / "models/outetts/__init__.py",
            PACKAGE_ROOT / 'models/outetts/configuration.py',
            PACKAGE_ROOT / "models/outetts/inference.py",
            PACKAGE_ROOT / 'models/outetts/modeling.py',
            PACKAGE_ROOT / "models/outetts/training.py",
        }

        self.assertTrue(expected.issubset(covered))
        self.assertNotIn(
            PACKAGE_ROOT / "models/outetts/source/outetts/interface.py",
            covered,
        )

    def test_fishtts_active_graph_is_covered_but_provider_source_is_not(self):
        covered = set(collect_native_runtime_paths(PACKAGE_ROOT))
        expected = {
            *(PACKAGE_ROOT / 'models/fishtts/native').glob("*.py"),
            PACKAGE_ROOT / "models/fishtts/__init__.py",
            PACKAGE_ROOT / 'models/fishtts/configuration.py',
            PACKAGE_ROOT / 'models/fishtts/modeling.py',
            PACKAGE_ROOT / 'models/fishtts/modeling.py',
            PACKAGE_ROOT / "models/fishtts/training.py",
        }

        self.assertTrue(expected.issubset(covered))
        self.assertNotIn(
            PACKAGE_ROOT / "models/fishtts/source/fish_speech/tokenizer.py",
            covered,
        )

    def test_mosstts_active_graph_is_covered_but_legacy_source_is_not(self):
        covered = set(collect_native_runtime_paths(PACKAGE_ROOT))
        expected = {
            *(PACKAGE_ROOT / 'models/mosstts/native').glob("*.py"),
            PACKAGE_ROOT / "models/mosstts/__init__.py",
            PACKAGE_ROOT / 'models/mosstts/configuration.py',
            PACKAGE_ROOT / 'models/mosstts/modeling.py',
            PACKAGE_ROOT / 'models/mosstts/modeling.py',
            PACKAGE_ROOT / "models/mosstts/training.py",
        }

        self.assertTrue(expected.issubset(covered))
        self.assertNotIn(
            PACKAGE_ROOT / "models/mosstts/source/__init__.py",
            covered,
        )

    def test_melotts_active_graph_is_covered_but_provider_frontends_are_not(self):
        covered = set(collect_native_import_closure(PACKAGE_ROOT))
        expected = {
            *(PACKAGE_ROOT / 'models/melotts/native').glob("*.py"),
            PACKAGE_ROOT / "models/melotts/__init__.py",
            PACKAGE_ROOT / 'models/melotts/configuration.py',
            PACKAGE_ROOT / 'models/melotts/modeling.py',
            PACKAGE_ROOT / 'models/melotts/modeling.py',
            PACKAGE_ROOT / "models/melotts/training.py",
            PACKAGE_ROOT / "models/melotts/source/melo/models.py",
            PACKAGE_ROOT / "models/melotts/source/melo/modules.py",
            PACKAGE_ROOT / "models/melotts/source/melo/attentions.py",
            PACKAGE_ROOT / "models/melotts/source/melo/commons.py",
            PACKAGE_ROOT / "models/melotts/source/melo/transforms.py",
            *(PACKAGE_ROOT / "models/melotts/source/melo/monotonic_align").glob("*.py"),
        }

        self.assertTrue(expected.issubset(covered))
        self.assertNotIn(
            PACKAGE_ROOT / "models/melotts/source/melo/api.py",
            covered,
        )
        self.assertNotIn(
            PACKAGE_ROOT / "models/melotts/source/melo/mel_processing.py",
            covered,
        )
        self.assertNotIn(
            PACKAGE_ROOT / "models/melotts/source/melo/train.py",
            covered,
        )

    def test_openvoice_active_converter_excludes_optional_upstream_frontends(self):
        covered = set(collect_native_import_closure(PACKAGE_ROOT))
        expected = {
            *(PACKAGE_ROOT / 'models/openvoice/native').glob("*.py"),
            PACKAGE_ROOT / "models/openvoice/__init__.py",
            PACKAGE_ROOT / 'models/openvoice/configuration.py',
            PACKAGE_ROOT / 'models/openvoice/modeling.py',
            PACKAGE_ROOT / 'models/openvoice/modeling.py',
            PACKAGE_ROOT / "models/openvoice/training.py",
            PACKAGE_ROOT / "models/openvoice/source/openvoice/models.py",
            PACKAGE_ROOT / "models/openvoice/source/openvoice/modules.py",
            PACKAGE_ROOT / "models/openvoice/source/openvoice/commons.py",
            PACKAGE_ROOT / "models/openvoice/source/openvoice/attentions.py",
            PACKAGE_ROOT / "models/openvoice/source/openvoice/transforms.py",
        }

        self.assertTrue(expected.issubset(covered))
        self.assertNotIn(
            PACKAGE_ROOT / "models/openvoice/source/openvoice/api.py",
            covered,
        )
        self.assertNotIn(
            PACKAGE_ROOT / "models/openvoice/source/openvoice/se_extractor.py",
            covered,
        )

    def test_static_and_literal_dynamic_external_imports_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.py"
            source.write_text(
                "import transformers\n"
                "from diffusers import Scheduler\n"
                "import_optional('nemo.collections.asr')\n",
                encoding="utf-8",
            )

            violations = inspect_native_imports(source)

            self.assertEqual(
                tuple(violation.module for violation in violations),
                ("transformers", "diffusers", "nemo.collections.asr"),
            )

    def test_unresolved_dynamic_imports_are_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dynamic.py"
            source.write_text(
                "from importlib import import_module\n"
                "module_name = input()\n"
                "import_module(module_name)\n",
                encoding="utf-8",
            )

            violations = inspect_native_imports(source)

        self.assertEqual(
            tuple(violation.module for violation in violations),
            ("<dynamic:import_module>", ),
        )

    def test_package_initializers_can_implement_lazy_voicehub_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "voicehub"
            package = root / "native"
            package.mkdir(parents=True)
            (root / "__init__.py").write_text("", encoding="utf-8")
            (package / "__init__.py").write_text(
                "from importlib import import_module\n"
                "def __getattr__(name):\n"
                "    return import_module('voicehub.native.' + name)\n",
                encoding="utf-8",
            )
            (package / "model.py").write_text("import torch\n", encoding="utf-8")

            violations = inspect_native_runtime(
                root,
                directories=("native/model.py", ),
            )

        self.assertEqual(violations, ())

    def test_foundational_and_relative_imports_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "good.py"
            source.write_text(
                "from __future__ import annotations\n"
                "from dataclasses import dataclass\n"
                "from .local import Model\n"
                "from voicehub.processing import ModelBatch\n"
                "import torch\n",
                encoding="utf-8",
            )
            self.assertEqual(inspect_native_imports(source), ())

    def test_runtime_inspection_accepts_an_explicit_file_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "runtime.py"
            source.write_text(
                "import transformers\n",
                encoding="utf-8",
            )

            violations = inspect_native_runtime(
                root,
                directories=("runtime.py", ),
            )

        self.assertEqual(
            tuple(violation.module for violation in violations),
            ("transformers", ),
        )

    def test_runtime_inspection_includes_ancestor_package_initializers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "voicehub"
            module = root / "native" / "nested" / "model.py"
            module.parent.mkdir(parents=True)
            (root / "__init__.py").write_text("", encoding="utf-8")
            (root / "native" / "__init__.py").write_text(
                "import voicehub\n",
                encoding="utf-8",
            )
            (root / "native" / "nested" / "__init__.py").write_text(
                "import torchaudio\n",
                encoding="utf-8",
            )
            module.write_text("import torch\n", encoding="utf-8")

            violations = inspect_native_runtime(
                root,
                directories=("native/nested/model.py", ),
            )

        self.assertEqual(
            tuple(violation.module for violation in violations),
            ("torchaudio", ),
        )

    def test_runtime_inspection_follows_transitive_internal_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "voicehub"
            native = root / "native"
            native.mkdir(parents=True)
            (root / "__init__.py").write_text("", encoding="utf-8")
            (native / "__init__.py").write_text("", encoding="utf-8")
            (native / "model.py").write_text(
                "from . import relative_helper\n"
                "from importlib import import_module\n"
                "import voicehub.shared_helper\n"
                "import_module('voicehub.dynamic_helper')\n",
                encoding="utf-8",
            )
            (native / "relative_helper.py").write_text(
                "import torchaudio\n",
                encoding="utf-8",
            )
            (root / "shared_helper.py").write_text(
                "import transformers\n",
                encoding="utf-8",
            )
            (root / "dynamic_helper.py").write_text(
                "import safetensors\n",
                encoding="utf-8",
            )

            closure = collect_native_import_closure(
                root,
                directories=("native/model.py", ),
            )
            violations = inspect_native_runtime(
                root,
                directories=("native/model.py", ),
            )

        self.assertEqual(
            {path.name
             for path in closure},
            {
                "__init__.py",
                "dynamic_helper.py",
                "model.py",
                "relative_helper.py",
                "shared_helper.py",
            },
        )
        self.assertEqual(
            tuple(violation.module for violation in violations),
            ("safetensors", "torchaudio", "transformers"),
        )

    def test_wandb_exception_is_scoped_to_the_public_integration_seam(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "voicehub"
            root.mkdir()
            (root / "training").mkdir()
            (root / "__init__.py").write_text("", encoding="utf-8")
            (root / 'training/integrations.py').write_text(
                "import wandb\n"
                "import transformers\n",
                encoding="utf-8",
            )
            (root / "runtime.py").write_text(
                "import wandb\n",
                encoding="utf-8",
            )

            integration_violations = inspect_native_runtime(
                root,
                directories=('training/integrations.py', ),
            )
            runtime_violations = inspect_native_runtime(
                root,
                directories=("runtime.py", ),
            )

        self.assertEqual(
            tuple(violation.module for violation in integration_violations),
            ("transformers", ),
        )
        self.assertEqual(
            tuple(violation.module for violation in runtime_violations),
            ("wandb", ),
        )


if __name__ == "__main__":
    unittest.main()
