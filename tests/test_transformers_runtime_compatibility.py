import importlib
import importlib.util
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _modules_are_available(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


HIGGS_RUNTIME_AVAILABLE = _modules_are_available(
    "safetensors",
    "torch",
    "transformers",
)
XTTS_RUNTIME_AVAILABLE = _modules_are_available(
    "fsspec",
    "numpy",
    "torch",
    "torchaudio",
    "transformers",
)
PARLER_RUNTIME_AVAILABLE = _modules_are_available("torch", "transformers")


def _assert_imports_without_dependencies(
    *,
    modules: tuple[str, ...],
    blocked: tuple[str, ...],
) -> None:
    script = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import sys

        blocked = {blocked!r}
        modules = {modules!r}

        class RejectDormantProviderDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.', 1)[0] in blocked:
                    raise AssertionError(
                        f'Native runtime imported dormant dependency {{fullname}}')
                return None

        sys.meta_path.insert(0, RejectDormantProviderDependency())
        for module_name in modules:
            importlib.import_module(module_name)
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)


@unittest.skipUnless(
    HIGGS_RUNTIME_AVAILABLE,
    "HiggsTTS runtime dependencies are optional",
)
class HiggsTransformersCompatibilityTests(unittest.TestCase):

    def test_llama_attention_factory_uses_the_installed_transformers_api(self):
        import torch
        from transformers import LlamaConfig

        runtime = importlib.import_module(
            "voicehub.models.higgstts.source.boson_multimodal.model."
            "higgs_audio.modeling_higgs_audio")
        config = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        config._attn_implementation = "eager"

        attention = runtime._build_llama_attention(config, 3, "eager")

        self.assertIsInstance(attention, torch.nn.Module)
        self.assertEqual(attention.layer_idx, 3)


@unittest.skipUnless(
    XTTS_RUNTIME_AVAILABLE,
    "XTTS runtime dependencies are optional",
)
class XTTSTransformersCompatibilityTests(unittest.TestCase):

    def test_streaming_and_tortoise_modules_import(self):
        stream_runtime = importlib.import_module(
            "voicehub.models.xtts.source.TTS.tts.layers.xtts.stream_generator")
        tortoise_runtime = importlib.import_module(
            "voicehub.models.xtts.source.TTS.tts.layers.tortoise.arch_utils")

        config = stream_runtime.StreamGenerationConfig(do_stream=True)
        self.assertTrue(config.do_stream)
        self.assertTrue(callable(tortoise_runtime.TypicalLogitsWarper))

    def test_legacy_beam_types_are_loaded_only_when_available(self):
        stream_runtime = importlib.import_module(
            "voicehub.models.xtts.source.TTS.tts.layers.xtts.stream_generator")

        try:
            scorer = stream_runtime._load_legacy_generation_symbol("BeamSearchScorer")
        except RuntimeError as error:
            self.assertIn("Use Transformers 4", str(error))
        else:
            self.assertEqual(scorer.__name__, "BeamSearchScorer")

    def test_typical_logits_warper_preserves_tensor_shape(self):
        import torch

        tortoise_runtime = importlib.import_module(
            "voicehub.models.xtts.source.TTS.tts.layers.tortoise.arch_utils")
        warper = tortoise_runtime.TypicalLogitsWarper(mass=0.9)
        input_ids = torch.tensor([[1, 2]])
        scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

        filtered_scores = warper(input_ids, scores)

        self.assertEqual(filtered_scores.shape, scores.shape)


@unittest.skipUnless(
    PARLER_RUNTIME_AVAILABLE,
    "ParlerTTS runtime dependencies are unavailable",
)
class ParlerTransformersCompatibilityTests(unittest.TestCase):

    def test_logits_processor_uses_public_torch_isin_fallback(self):
        import torch

        runtime = importlib.import_module("voicehub.models.parlertts.source.parler_tts.logits_processors")
        processor = runtime.ParlerTTSLogitsProcessor(
            eos_token_id=3,
            num_codebooks=2,
            batch_size=1,
        )
        scores = torch.zeros((2, 5))

        output = processor(torch.tensor([[1, 3], [1, 2]]), scores)

        self.assertEqual(output.shape, scores.shape)


class NativeProviderDependencyCompatibilityTests(unittest.TestCase):

    def test_native_melotts_imports_without_dormant_language_frontends(self):
        _assert_imports_without_dependencies(
            modules=(
                "voicehub.models.melotts.native.frontend",
                "voicehub.models.melotts.native.runtime",
                "voicehub.models.melotts.modeling",
            ),
            blocked=(
                "MeCab",
                "pykakasi",
                "transformers",
                "unidic_lite",
            ),
        )

    def test_native_outetts_imports_without_dormant_backend_dependencies(self):
        _assert_imports_without_dependencies(
            modules=(
                "voicehub.models.outetts.native.runtime",
                "voicehub.models.outetts.inference",
            ),
            blocked=(
                "llama_cpp",
                "loguru",
                "polars",
                "transformers",
            ),
        )


if __name__ == "__main__":
    unittest.main()
