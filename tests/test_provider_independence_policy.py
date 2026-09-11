from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from voicehub.policies.provider_independence import (
    collect_shared_python_paths,
    inspect_shared_provider_branches,
    require_shared_provider_independence,
)
from voicehub.registry import ModelSpec, register_model_spec, unregister_model_spec

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "voicehub"


class ProviderIndependencePolicyTests(unittest.TestCase):

    def test_all_shared_layers_are_provider_independent(self):
        violations = inspect_shared_provider_branches(PACKAGE_ROOT)

        self.assertEqual(
            violations,
            (),
            "\n".join(str(violation) for violation in violations),
        )
        require_shared_provider_independence(PACKAGE_ROOT)

    def test_every_shared_python_file_joins_the_default_policy(self):
        discovered = set(collect_shared_python_paths(PACKAGE_ROOT))
        expected = {
            path
            for path in PACKAGE_ROOT.rglob("*.py")
            if path.relative_to(PACKAGE_ROOT).parts[0] not in {'architectures', "models"}
        }

        self.assertEqual(discovered, expected)

    def test_runtime_extension_joins_the_default_policy_without_a_central_edit(self):
        extension = ModelSpec(
            model_type="future_speech",
            module="extension.modeling",
            class_name="FutureSpeechModel",
            default_model_path="acme/future-speech",
        )
        register_model_spec(extension, aliases=("future-speech", ))
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "voicehub"
                root.mkdir()
                (root / "shared.py").write_text(
                    "enabled = model_type == 'future-speech'\n",
                    encoding="utf-8",
                )

                violations = inspect_shared_provider_branches(root)

                self.assertEqual(
                    [(violation.provider, violation.construct) for violation in violations],
                    [("future-speech", "comparison")],
                )
        finally:
            unregister_model_spec(extension.model_type)

    def test_policy_detects_condition_syntax_but_allows_declarative_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "voicehub"
            (root / "models" / "auroratts").mkdir(parents=True)
            (root / "models/auroratts/native").mkdir(parents=True)
            (root / "metadata.py").write_text(
                "declaration = ModelSpec(model_type='auroratts')\n",
                encoding="utf-8",
            )
            branch_source = textwrap.dedent(
                """
                def dispatch(model_type, values):
                    direct = model_type == "auroratts"
                    if model_type == "auroratts":
                        return direct
                    label = "alias" if model_type.startswith("aurora-tts") else "other"
                    while model_type in {"auroratts"}:
                        break
                    assert model_type != "aurora-tts"
                    selected = [value for value in values if model_type == "auroratts"]
                    match values:
                        case [value] if model_type == "auroratts":
                            return value
                    match model_type:
                        case "aurora-tts":
                            return selected
                    return label
                """)
            (root / "shared.py").write_text(branch_source, encoding="utf-8")
            for local_path in (
                    root / "models" / "auroratts" / "runtime.py",
                    root / "models/auroratts/native/modeling.py",
            ):
                local_path.write_text(branch_source, encoding="utf-8")

            violations = inspect_shared_provider_branches(
                root,
                provider_names={"auroratts", "aurora-tts"},
            )

            self.assertEqual({violation.path for violation in violations}, {"shared.py"})
            self.assertEqual(
                {violation.construct
                 for violation in violations},
                {
                    "assertion",
                    "comparison",
                    "comprehension condition",
                    "conditional expression",
                    "if condition",
                    "match case",
                    "match guard",
                    "while condition",
                },
            )
            with self.assertRaisesRegex(
                    RuntimeError,
                    "Shared VoiceHub behavior must use capabilities",
            ):
                require_shared_provider_independence(
                    root,
                    provider_names={"auroratts", "aurora-tts"},
                )


if __name__ == "__main__":
    unittest.main()
