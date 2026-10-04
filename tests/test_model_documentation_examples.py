"""Generated usage examples must satisfy each native runtime's input
boundary."""

import ast
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "scripts"


class _Opaque(str):
    """Stand-in for a non-literal example value such as a loaded feature."""


def _documentation():
    return runpy.run_path(str(SCRIPTS / "model_documentation.py"))


def _profile(model_type: str):
    return _documentation()["INFERENCE_PROFILES"][model_type]


def _example_arguments(profile) -> dict[str, object]:
    """Evaluate literal example arguments; others become opaque prepared
    values."""
    arguments = {}
    for argument in profile.arguments:
        name, _, expression = argument.partition("=")
        try:
            arguments[name] = ast.literal_eval(expression)
        except ValueError:
            arguments[name] = _Opaque(expression)
    return arguments


class ModelDocumentationExampleTest(unittest.TestCase):

    def test_melotts_example_passes_every_required_linguistic_feature(self):
        from voicehub.architectures.melotts.frontend import validate_feature_mapping

        validate_feature_mapping(_example_arguments(_profile("melotts")))

    def test_gptsovits_example_reaches_the_prepared_native_path(self):
        from voicehub.architectures.gptsovits.runtime import TTS
        from voicehub.models.gptsovits.inference import GPTSoVITSForTextToSpeech

        profile = _profile("gptsovits")
        request_arguments = {
            "text_language": "",
            "speaker_audio_path": "",
            "prompt_language": "",
            "prompt_text": "",
            "speed": 1.0,
            "seed": 42,
            "batch_size": 1,
            "text_split_method": "cut5",
            "parallel_inference": True,
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
        }
        request_arguments.update(_example_arguments(profile))
        request = GPTSoVITSForTextToSpeech._build_request(profile.text, **request_arguments)
        runtime = SimpleNamespace(synthesize_prepared=lambda **_: (32_000, "audio"))

        self.assertEqual(list(TTS.run(runtime, request)), [(32_000, "audio")])
        # The registered lj1995 release is a PyTorch pickle container.
        self.assertIn(
            'config=AutoConfig.for_model("gptsovits", trust_pickle_checkpoint=True)',
            profile.load_arguments,
        )

    def test_zonos_example_supplies_phonemes_and_no_unencodable_reference(self):
        from voicehub.architectures.zonos.frontend import resolve_phonemes
        from voicehub.architectures.zonos.metadata import ZONOS_SPEAKER_SAFE_CHECKPOINT_PUBLISHED

        profile = _profile("zonos")
        arguments = _example_arguments(profile)
        _, frontend = resolve_phonemes(
            profile.text,
            language=arguments["language"],
            phonemes=arguments.get("phonemes"),
            frontend=None,
        )

        self.assertEqual(frontend, "precomputed-phonemes")
        if not ZONOS_SPEAKER_SAFE_CHECKPOINT_PUBLISHED:
            self.assertNotIn("speaker_audio_path", arguments)

    def test_xtts_example_loads_a_converted_directory(self):
        import voicehub.architectures.xtts2 as xtts2
        from voicehub import get_model_spec

        spec = get_model_spec("xtts")
        checkpoint = _documentation()["checkpoint_documentation"](spec)

        self.assertNotEqual(checkpoint.example, spec.default_model_path)
        self.assertIn("convert_trusted_legacy_xtts2_checkpoint", checkpoint.note)
        self.assertIn(f"{checkpoint.example}/model.safetensors", checkpoint.note)
        self.assertTrue(callable(xtts2.convert_trusted_legacy_xtts2_checkpoint))

    def test_styletts2_example_loads_the_released_checkpoint_with_opt_in(self):
        from voicehub import get_model_spec

        documentation = _documentation()
        checkpoint = documentation["checkpoint_documentation"](get_model_spec("styletts2"))
        profile = documentation["INFERENCE_PROFILES"]["styletts2"]

        self.assertEqual(Path(checkpoint.example).suffix, ".pth")
        self.assertIn(
            'config=AutoConfig.for_model("styletts2", trust_pickle_checkpoint=True)',
            profile.load_arguments,
        )
        self.assertNotIn("not a drop-in", profile.input_note)
        self.assertNotIn("not a native VoiceHub directory", checkpoint.hugging_face_status)

    def test_cosyvoice_cites_the_cosyvoice3_paper(self):
        references = runpy.run_path(str(SCRIPTS / "documentation_references.py"))["MODEL_REFERENCES"]

        self.assertEqual(
            [paper.url for paper in references["cosyvoice"].papers],
            ["https://arxiv.org/abs/2505.17589"],
        )


if __name__ == "__main__":
    unittest.main()
