import ast
import re
import unittest
from collections import Counter
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from voicehub.registry import list_model_specs

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_INFERENCE_EXTRAS = {
    "asr-vad",
    "asr-espnet",
    "asr-funasr",
    "asr-nemo",
    "asr-speechbrain",
    "asr-training",
    "asr-transformers",
    "asr-wenet",
    "faster-whisper",
    "openai-whisper",
    "vad-funasr",
    "vad-nemo",
    "vad-pyannote",
    "vad-silero",
    "vad-silero-onnx",
    "vad-speechbrain",
    "vad-transformers",
    "vad-webrtc",
    "whisperx",
}
EXPECTED_INFERENCE_ABI_REQUIREMENTS = ("torch>=2.8,<2.9", )
FORBIDDEN_EXTERNAL_RUNTIME_DISTRIBUTIONS = {
    "accelerate",
    "apache-tvm-ffi",
    "argbind",
    "cached-path",
    "cloudpickle",
    "diffusers",
    "encodec",
    "fiddle",
    "huggingface-hub",
    "hyperpyyaml",
    "librosa",
    "local-attention",
    "msgpack",
    "modelscope",
    "ninja",
    "numba",
    "numpy",
    "onnxruntime",
    "peft",
    "protobuf",
    "pynini",
    "pyzmq",
    "rjieba",
    "sacremoses",
    "safetensors",
    "scikit-learn",
    "sentencepiece",
    "split-lang",
    "tiktoken",
    "tokenizers",
    "torchaudio",
    "torchdiffeq",
    "torchtune",
    "torchvision",
    "transformers",
    "x-transformers",
}
EXPECTED_TRAINING_REQUIREMENTS = (
    "ema-pytorch",
    "datasets",
    "evaluate",
    "jiwer",
    "pyarrow",
    "pyworld",
    "wandb>=0.19",
)
REQUIRED_DISTRIBUTION_FILES = (
    "voicehub/py.typed",
    "voicehub/models/outetts/native/default_speaker.json",
    "voicehub/models/conversationtts/source/conversationtts/llama3_2/tokenizer.json",
    ("voicehub/models/chatterbox/source/perth/perth_net/pretrained/implicit/"
     "perth_net_250000.pth.tar"),
    "voicehub/kernels/csrc/activations.cpp",
)


def _read_optional_dependencies() -> dict[str, list[str]]:
    """Read the simple array-valued optional-dependency table.

    The project supports Python 3.10, where ``tomllib`` is unavailable.
    Keeping this test helper scoped to the table's existing multiline-
    array format avoids making a TOML parser a runtime or test
    dependency.
    """
    source = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = source.split("[project.optional-dependencies]", 1)[1]
    section = section.split("\n[", 1)[0]
    extras: dict[str, list[str]] = {}
    current_name: str | None = None
    current_values: list[str] = []

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if current_name is None:
            match = re.fullmatch(r"([A-Za-z0-9_-]+)\s*=\s*\[", line)
            if match is None:
                raise AssertionError(
                    "Optional dependencies must use readable multiline arrays; "
                    f"could not parse: {raw_line!r}")
            current_name = match.group(1)
            continue
        if line == "]":
            extras[current_name] = current_values
            current_name = None
            current_values = []
            continue
        if not line.endswith(","):
            raise AssertionError(
                "Optional dependency entries must use trailing commas; "
                f"could not parse: {raw_line!r}")
        value = ast.literal_eval(line[:-1])
        if not isinstance(value, str):
            raise TypeError(f"Optional dependency entries must be strings: {raw_line!r}")
        current_values.append(value)

    if current_name is not None:
        raise AssertionError(f"Optional dependency array {current_name!r} was not closed.")
    return extras


def _read_project_dependencies() -> list[str]:
    source = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = source.split("[project]", 1)[1].split("\n[", 1)[0]
    array = project.split("dependencies = [", 1)[1].split("\n]", 1)[0]
    dependencies = []
    for raw_line in array.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.endswith(","):
            raise AssertionError(
                "Project dependencies must use a readable multiline array "
                f"with trailing commas; could not parse: {raw_line!r}")
        value = ast.literal_eval(line[:-1])
        if not isinstance(value, str):
            raise TypeError(f"Project dependency entries must be strings: {raw_line!r}")
        dependencies.append(value)
    return dependencies


def _distribution_name(requirement: str) -> str:
    return canonicalize_name(Requirement(requirement).name)


def _requirements_by_distribution(requirements: list[str]) -> dict[str, Requirement]:
    return {_distribution_name(requirement): Requirement(requirement) for requirement in requirements}


def _normalized_requirements(requirements) -> set[str]:
    return {str(Requirement(requirement)) for requirement in requirements}


class PackagingMetadataTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extras = _read_optional_dependencies()
        cls.dependencies = _read_project_dependencies()

    def test_only_training_is_a_public_runtime_extra(self):
        self.assertEqual(set(self.extras), {"docs", "test", "training"})
        self.assertEqual(
            FORBIDDEN_INFERENCE_EXTRAS & self.extras.keys(),
            set(),
            "All TTS, ASR, and VAD inference dependencies must ship in the "
            "default installation; only training has a runtime feature extra.",
        )

    def test_every_inference_model_uses_the_default_installation(self):
        for spec in list_model_specs(task=None):
            with self.subTest(model_type=spec.model_type):
                self.assertIsNone(spec.install_extra)

    def test_default_install_has_the_supported_inference_abi(self):
        normalized = _normalized_requirements(self.dependencies)
        self.assertEqual(
            normalized,
            _normalized_requirements(EXPECTED_INFERENCE_ABI_REQUIREMENTS),
        )
        aggregate_requirements = _requirements_by_distribution(self.dependencies)
        self.assertNotIn(
            "wenet",
            aggregate_requirements,
            "WeNet's inference runtime is vendored because no matching PyPI "
            "distribution exists.",
        )
        self.assertNotIn(
            "espnet",
            aggregate_requirements,
            "The audited ESPnet Transformer runtime is implemented natively.",
        )
        self.assertNotIn(
            "espnet-model-zoo",
            aggregate_requirements,
            "Native artifact resolution replaces ESPnet's model-zoo client.",
        )
        self.assertEqual(
            FORBIDDEN_EXTERNAL_RUNTIME_DISTRIBUTIONS
            & aggregate_requirements.keys(),
            set(),
            "VoiceHub-native graphs must not reintroduce external model, "
            "tokenizer, checkpoint, DSP, or provider runtimes.",
        )
        self.assertNotIn("descript-audiotools", aggregate_requirements)
        self.assertNotIn(
            "pyannote-audio",
            aggregate_requirements,
            "PyanNet is implemented by VoiceHub and must not reinstall "
            "pyannote.audio.",
        )
        self.assertNotIn(
            "torchcodec",
            aggregate_requirements,
            "VoiceHub-native audio loading must not require TorchCodec.",
        )

    def test_training_extra_has_exact_shared_trainer_contract(self):
        self.assertEqual(
            _normalized_requirements(self.extras["training"]),
            _normalized_requirements(EXPECTED_TRAINING_REQUIREMENTS),
        )

    def test_aggregate_extras_are_flat_and_deduplicated(self):
        for extra in ("training", "docs", "test"):
            with self.subTest(extra=extra):
                distributions = [_distribution_name(requirement) for requirement in self.extras[extra]]
                duplicates = {name for name, count in Counter(distributions).items() if count > 1}
                self.assertEqual(duplicates, set())
                self.assertNotIn("voicehub", distributions)

    def test_test_extra_does_not_install_an_external_trainer(self):
        distributions = {_distribution_name(requirement) for requirement in self.extras["test"]}
        self.assertIn("build", distributions)
        self.assertNotIn(
            'trainer',
            distributions,
            "Inference imports and VoiceHub's native trainer must not depend "
            "on Coqui's Python-version-limited trainer distribution.",
        )

    def test_test_extra_keeps_numba_compatible_with_supported_python(self):
        requirements = _requirements_by_distribution(self.extras["test"])
        self.assertIn("numba", requirements)
        self.assertEqual(str(requirements["numba"].specifier), ">=0.59")

    def test_wandb_is_training_only(self):
        inference_distributions = {_distribution_name(requirement) for requirement in self.dependencies}
        training_distributions = {_distribution_name(requirement) for requirement in self.extras["training"]}

        self.assertNotIn("wandb", inference_distributions)
        self.assertIn("wandb", training_distributions)

    def test_pep517_build_configuration_supports_wheels_and_sdists(self):
        source = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('build-backend = "setuptools.build_meta"', source)
        self.assertIn('requires = ["setuptools>=77", "wheel"]', source)
        self.assertIn('name = "voicehub"', source)
        self.assertIn('where = ["."]', source)
        self.assertIn('include = ["voicehub*"]', source)
        self.assertIn("include-package-data = true", source)

    def test_manifest_includes_required_runtime_data(self):
        manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn("graft voicehub", manifest)
        for relative_path in REQUIRED_DISTRIBUTION_FILES:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((REPOSITORY_ROOT / relative_path).is_file())

    def test_distribution_check_covers_all_install_modes(self):
        source = (REPOSITORY_ROOT / "scripts" / "check_distribution.py").read_text(encoding="utf-8")
        install_modes = set()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and isinstance(key.value, str) and
                        isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and
                        value.func.id == "install_and_probe"):
                    install_modes.add(key.value)

        self.assertEqual(install_modes, {"wheel", "sdist", "editable"})


if __name__ == "__main__":
    unittest.main()
