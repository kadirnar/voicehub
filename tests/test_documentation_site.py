import ast
import io
import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPOSITORY_ROOT / "docs"
SITE_CONFIG_PATH = REPOSITORY_ROOT / "mkdocs.yml"
NOTEBOOK_PATHS = tuple(sorted((REPOSITORY_ROOT / "notebooks").glob("*.ipynb")))
EXPECTED_NOTEBOOK_FILENAMES = {
    "data_preparation.ipynb",
    "inference.ipynb",
    "training.ipynb",
    "tts_workflow.ipynb",
}
NOTEBOOKS_README_PATH = REPOSITORY_ROOT / "notebooks" / "README.md"
MODEL_NOTEBOOK_DIR = REPOSITORY_ROOT / "notebooks" / "models"
MODEL_NOTEBOOK_GALLERY_PATH = MODEL_NOTEBOOK_DIR / "README.md"
MODEL_NOTEBOOK_GENERATOR_PATH = REPOSITORY_ROOT / "scripts" / "generate_model_notebooks.py"
MODEL_PAGE_DIR = DOCS_ROOT / "models" / "providers"
MODEL_PAGE_INDEX_PATH = MODEL_PAGE_DIR / "index.md"
MODEL_PAGE_GENERATOR_PATH = REPOSITORY_ROOT / "scripts" / "generate_model_pages.py"
OPTIMIZATION_PAGE_DIR = DOCS_ROOT / "optimizations"
OPTIMIZATION_PAGE_INDEX_PATH = OPTIMIZATION_PAGE_DIR / "index.md"
OPTIMIZATION_PAGE_GENERATOR_PATH = (REPOSITORY_ROOT / "scripts" / "generate_optimization_pages.py")
DOCUMENTATION_DOM_CHECK_PATH = REPOSITORY_ROOT / "scripts" / "check_documentation_dom.py"
DOCUMENTATION_VISUAL_CHECK_PATH = REPOSITORY_ROOT / "scripts" / "check_documentation_visual.py"
DOCUMENTATION_VISUAL_SHARD_CHECK_PATH = (REPOSITORY_ROOT / "scripts" / "check_documentation_visual_shards.py")
DOCUMENTATION_SCREENSHOT_BASELINES_PATH = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "documentation_screenshot_signatures.json")
DOCUMENTATION_LINUX_SCREENSHOT_BASELINES_PATH = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "documentation_screenshot_signatures_linux.json")
ADDING_MODEL_PATH = DOCS_ROOT / "project" / "adding-a-model.md"
INSTALLATION_PATH = DOCS_ROOT / "getting-started" / "installation.md"
QUICKSTART_PATH = DOCS_ROOT / "getting-started" / "quickstart.md"
INFERENCE_GUIDE_PATH = DOCS_ROOT / "guides" / "inference.md"
TRAINER_OVERVIEW_PATH = DOCS_ROOT / "guides" / "trainer.md"
OPTIMIZATION_OVERVIEW_PATH = DOCS_ROOT / "guides" / "optimization-overview.md"
OPTIONAL_BACKENDS_PATH = DOCS_ROOT / "guides" / "optional-backends.md"
MODEL_API_PATH = DOCS_ROOT / "reference" / "models.md"
NOTEBOOK_GALLERY_PATH = DOCS_ROOT / "guides" / "notebook.md"
README_PATH = REPOSITORY_ROOT / "README.md"
HOME_PATH = DOCS_ROOT / "index.md"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
THEME_OVERRIDE_PATH = REPOSITORY_ROOT / "overrides" / "main.html"
HEADER_OVERRIDE_PATH = REPOSITORY_ROOT / "overrides" / "partials" / "header.html"
SEARCH_OVERRIDE_PATH = REPOSITORY_ROOT / "overrides" / "partials" / "search.html"
LANGUAGE_OVERRIDE_PATH = REPOSITORY_ROOT / "overrides" / "partials" / "alternate.html"
PALETTE_OVERRIDE_PATH = REPOSITORY_ROOT / "overrides" / "partials" / "palette.html"
SOURCE_OVERRIDE_PATH = REPOSITORY_ROOT / "overrides" / "partials" / "source.html"
STYLESHEET_PATH = DOCS_ROOT / "stylesheets" / "extra.css"
HEADER_CONTROL_SCRIPT_PATH = DOCS_ROOT / "javascripts" / "header-controls.js"
PAGE_ACTION_SCRIPT_PATH = DOCS_ROOT / "javascripts" / "page-actions.js"
MOBILE_DRAWER_SCRIPT_PATH = DOCS_ROOT / "javascripts" / "mobile-drawer.js"
PAGE_ACTIONS_OVERRIDE_PATH = REPOSITORY_ROOT / "overrides" / "partials" / "actions.html"
PUBLIC_SITE_URL = "https://kadirnar.github.io/voicehub/"
LOCALIZED_HOME_LOCALES = ("ar", "de", "es", "fr", "ja", "ko", "pt", "ru", "tr", "zh")
TOP_LEVEL_NAVIGATION = (
    "Get started",
    "Models",
    "Train",
    "Optimize",
)
GUIDE_PATHS = (
    DOCS_ROOT / "getting-started" / "quickstart.md",
    DOCS_ROOT / "guides" / "inference.md",
    DOCS_ROOT / "guides" / "speech-recognition.md",
    DOCS_ROOT / "guides" / "voice-activity-detection.md",
    DOCS_ROOT / "guides" / "optimization-overview.md",
    DOCS_ROOT / "guides" / "tts-optimization.md",
    DOCS_ROOT / "guides" / "data-preparation.md",
    DOCS_ROOT / "guides" / "speech-data.md",
    DOCS_ROOT / "guides" / "training.md",
    DOCS_ROOT / "guides" / "notebook.md",
)
CONCISE_GUIDE_PATHS = (
    DOCS_ROOT / "guides" / "inference.md",
    DOCS_ROOT / "guides" / "speech-recognition.md",
    DOCS_ROOT / "guides" / "voice-activity-detection.md",
    DOCS_ROOT / "guides" / "training.md",
    DOCS_ROOT / "guides" / "optimization-overview.md",
    DOCS_ROOT / "guides" / "tts-optimization.md",
)
PROCESS_PAGE_STEPS = (
    (DOCS_ROOT / "guides" / "index.md", 7),
    (DOCS_ROOT / "guides" / "data-preparation.md", 6),
    (ADDING_MODEL_PATH, 8),
)
NAVIGATION_PATHS = (
    "index.md",
    "getting-started/installation.md",
    "getting-started/quickstart.md",
    "guides/inference.md",
    "guides/data-preparation.md",
    "guides/trainer.md",
    "guides/training.md",
    "guides/optimization-overview.md",
    "optimizations/index.md",
    "optimizations/compile.md",
    "optimizations/hqq.md",
    "models/providers/index.md",
    "models/training-support.md",
    "project/adding-a-model.md",
    "reference/models.md",
)
PUBLIC_ROUTES = (
    "getting-started/installation/",
    "getting-started/quickstart/",
    "guides/training/",
    "models/tts-capabilities/",
    "models/asr-vad-support/",
    "models/training-support/",
    "models/providers/",
    "optimizations/",
    "reference/api/",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HTML_HREF = re.compile(r"""href=["']([^"']+)["']""")
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
MODEL_PAGE_SECTIONS = (
    "Usage",
    "Overview",
    "Paper and GitHub",
    "Configuration",
    "Processing",
    "Inference",
    "Training and optimization",
    "Checkpoints, provenance, license, and limitations",
    "Public API",
)
OPTIMIZATION_PAGE_SECTIONS = (
    "Use",
    "Support",
    "Paper and GitHub",
    "Verify",
)


def _cell_source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _local_link_path(raw_target):
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return Path(unquote(parsed.path))


class DocumentationSiteTests(unittest.TestCase):

    def setUp(self):
        self.notebooks = {path: nbformat.read(path, as_version=4) for path in NOTEBOOK_PATHS}

    def test_notebooks_are_clean_and_structurally_valid(self):
        self.assertEqual(
            {path.name
             for path in self.notebooks},
            EXPECTED_NOTEBOOK_FILENAMES,
        )
        for path, notebook in self.notebooks.items():
            with self.subTest(notebook=path.name):
                nbformat.validate(notebook)
                self.assertEqual(notebook["nbformat"], 4)
                self.assertGreaterEqual(notebook["nbformat_minor"], 5)
                cells = notebook["cells"]
                self.assertTrue(cells)
                self.assertLessEqual(
                    len(cells),
                    16,
                    f"{path.name} should remain a short top-to-bottom workflow.",
                )
                notebook_source = "\n".join(_cell_source(cell) for cell in cells)
                self.assertIn(
                    "https://colab.research.google.com/github/"
                    "kadirnar/voicehub/blob/main/notebooks/"
                    f"{path.name}",
                    notebook_source,
                )
                if path.name != "tts_workflow.ipynb":
                    self.assertIn(
                        'importlib.util.find_spec("voicehub") is None',
                        notebook_source,
                    )

                cell_ids = [cell.get("id") for cell in cells]
                self.assertTrue(all(cell_ids))
                self.assertEqual(len(cell_ids), len(set(cell_ids)))

                for cell in cells:
                    self.assertIn(cell["cell_type"], {"code", "markdown"})
                    self.assertIsInstance(_cell_source(cell), str)
                    if cell["cell_type"] == "code":
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])
                    else:
                        for raw_target in MARKDOWN_LINK.findall(_cell_source(cell)):
                            local_path = _local_link_path(raw_target)
                            if local_path is None:
                                continue
                            with self.subTest(
                                    notebook=path.name,
                                    target=raw_target,
                            ):
                                self.assertTrue(
                                    (path.parent / local_path).exists(),
                                    f"Broken notebook link {raw_target!r} "
                                    f"in {path.name}",
                                )

    def test_hugging_face_models_have_generated_notebooks(self):
        generator = runpy.run_path(str(MODEL_NOTEBOOK_GENERATOR_PATH))
        checkpoint_documentation = generator["checkpoint_documentation"]
        hub_specs = generator["hub_model_specs"]()
        expected_paths = {MODEL_NOTEBOOK_DIR / f"{spec.model_type}.ipynb": spec for spec in hub_specs}
        self.assertEqual(
            set(MODEL_NOTEBOOK_DIR.glob("*.ipynb")),
            set(expected_paths),
        )
        self.assertTrue(expected_paths)

        gallery = MODEL_NOTEBOOK_GALLERY_PATH.read_text(encoding="utf-8")
        for path, spec in expected_paths.items():
            with self.subTest(model_type=spec.model_type):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                self.assertLessEqual(len(notebook["cells"]), 8)
                self.assertEqual(
                    notebook["metadata"]["voicehub"]["model_type"],
                    spec.model_type,
                )
                source = "\n".join(_cell_source(cell) for cell in notebook["cells"])
                checkpoint = checkpoint_documentation(spec)
                self.assertTrue(checkpoint.is_hugging_face)
                self.assertIn(checkpoint.url, source)
                self.assertIn(
                    "https://colab.research.google.com/github/"
                    "kadirnar/voicehub/blob/main/notebooks/models/"
                    f"{path.name}",
                    source,
                )
                self.assertIn(f"[View]({path.name})", gallery)
                namespace = {"__name__": "__main__"}
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        source = _cell_source(cell)
                        ast.parse(
                            source,
                            filename=f"{path.name}:{cell['id']}",
                        )
                        if "smoke-safe" in cell["metadata"].get("tags", ()):
                            with redirect_stdout(io.StringIO()):
                                exec(  # noqa: S102 - execute generated smoke cells
                                    compile(
                                        source,
                                        f"{path.name}:{cell['id']}",
                                        "exec",
                                    ),
                                    namespace,
                                )
                self.assertFalse(namespace["RUN_INFERENCE"])
                self.assertEqual(namespace["MODEL_TYPE"], spec.model_type)
                self.assertEqual(namespace["CHECKPOINT"], spec.default_model_path)

        generated_files = generator["generated_files"]()
        self.assertEqual(generator["check_generated_files"](generated_files), ())

    def test_every_registered_model_has_a_generated_guide(self):
        from voicehub import list_model_specs

        notebook_generator = runpy.run_path(str(MODEL_NOTEBOOK_GENERATOR_PATH))
        checkpoint_documentation = notebook_generator["checkpoint_documentation"]
        generator = runpy.run_path(str(MODEL_PAGE_GENERATOR_PATH))
        module_source_path = generator["_module_source_path"]
        references = generator["MODEL_REFERENCES"]
        specs = tuple(list_model_specs(task=None))
        self.assertEqual(set(references), {spec.model_type for spec in specs})
        expected_paths = {MODEL_PAGE_DIR / f"{spec.model_type}.md": spec for spec in specs}
        self.assertEqual(
            set(MODEL_PAGE_DIR.glob("*.md")),
            {*expected_paths, MODEL_PAGE_INDEX_PATH},
        )
        self.assertTrue(expected_paths)

        index = MODEL_PAGE_INDEX_PATH.read_text(encoding="utf-8")
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        for path, spec in expected_paths.items():
            with self.subTest(model_type=spec.model_type):
                source = path.read_text(encoding="utf-8")
                self.assertIn(f"# {spec.display_name}", source)
                self.assertIn(f"`{spec.model_type}`", source)
                sections = tuple(
                    line.removeprefix("## ") for line in source.splitlines() if line.startswith("## "))
                self.assertEqual(sections, MODEL_PAGE_SECTIONS)
                self.assertLessEqual(
                    len(source.splitlines()),
                    230,
                    f"{path.name} should link shared workflows instead of repeating them.",
                )
                self.assertIn(
                    'git+https://github.com/kadirnar/voicehub.git@main',
                    source,
                )
                self.assertNotIn(
                    "linked release record before treating a checkpoint path as verified",
                    source,
                )
                self.assertIn(spec.task.value.replace("-", " "), source.lower())
                self.assertIn(spec.training.support.value, source)
                self.assertIn("Checkpoint status", source)
                self.assertIn("Source provenance", source)
                self.assertIn("available_optimization_passes", source)
                self.assertIn("## Paper and GitHub", source)
                self.assertIn("- **Paper:**", source)
                self.assertIn(references[spec.model_type].github.url, source)
                self.assertIn(spec.config_class, source)
                for fragment in (
                        "### Limitations",
                        "Optional dependency extra",
                        "Hardware and runtime",
                        "Real-checkpoint evidence",
                        "https://github.com/kadirnar/voicehub/blob/main/",
                ):
                    self.assertIn(fragment, source)
                for module in (spec.config_module, spec.module):
                    source_path = module_source_path(module)
                    self.assertTrue((REPOSITORY_ROOT / source_path).is_file())
                    self.assertIn(source_path.as_posix(), source)
                self.assertIn(f"[`{spec.display_name}`]({path.name})", index)
                self.assertEqual(
                    config.count(f"models/providers/{path.name}"),
                    1,
                    f"{spec.model_type} should appear once in the Models sidebar",
                )
                if spec.default_model_path:
                    self.assertIn(spec.default_model_path, source)
                if checkpoint_documentation(spec).is_hugging_face:
                    self.assertIn(
                        "https://colab.research.google.com/github/"
                        "kadirnar/voicehub/blob/main/notebooks/models/"
                        f"{spec.model_type}.ipynb",
                        source,
                    )
                examples = PYTHON_BLOCK.findall(source)
                self.assertGreaterEqual(len(examples), 4)
                quickstart = source.split("## Usage", 1)[1].split(
                    "## Overview",
                    1,
                )[0]
                self.assertTrue(PYTHON_BLOCK.findall(quickstart))
                for example_index, example in enumerate(examples, start=1):
                    ast.parse(
                        textwrap.dedent(example),
                        filename=f"{path.name}:python-block-{example_index}",
                    )

        generated_files = generator["generated_files"]()
        self.assertEqual(generator["check_generated_files"](generated_files), ())
        self.assertIn("- Text to speech:", config)
        self.assertIn("- Automatic speech recognition:", config)
        self.assertIn("- Voice activity detection:", config)

    def test_model_index_lists_every_registered_model(self):
        from voicehub import list_model_specs

        specs = tuple(list_model_specs(task=None))
        self.assertEqual(len(specs), 68)
        index = MODEL_PAGE_INDEX_PATH.read_text(encoding="utf-8")
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")

        headings = (
            "# Model list",
            "## Text to speech",
            "## Automatic speech recognition",
            "## Voice activity detection",
        )
        positions = tuple(index.index(heading) for heading in headings)
        self.assertEqual(positions, tuple(sorted(positions)))
        for fragment in (
                "from voicehub import list_model_specs",
                "for model in list_model_specs():",
                "model.display_name",
                "training matrix",
                "optimization catalog",
        ):
            self.assertIn(fragment, index)
        self.assertIn("- Model list: models/providers/index.md", config)
        for example_index, example in enumerate(PYTHON_BLOCK.findall(index), start=1):
            ast.parse(
                textwrap.dedent(example),
                filename=f"models/providers/index.md:python-block-{example_index}",
            )

        for spec in specs:
            with self.subTest(model_type=spec.model_type):
                self.assertTrue(spec.display_name[0].isupper())
                self.assertNotEqual(spec.display_name, spec.model_type)
                page = (MODEL_PAGE_DIR / f"{spec.model_type}.md").read_text(encoding="utf-8")
                self.assertIn(f"# {spec.display_name}", page)
                self.assertIn(f"[`{spec.display_name}`]({spec.model_type}.md)", index)
                self.assertIn(
                    f'- "{spec.display_name}": models/providers/{spec.model_type}.md',
                    config,
                )

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "MODEL_INDEX_ROUTE",
                "MODEL_INDEX_HEADINGS",
                "MODEL_INDEX_TABLE_ROWS",
                "def _validate_model_index_state(",
                "def _validate_model_index_page_copy(",
                '"model_index_cases"',
                '"model_index_interaction_cases"',
        ):
            self.assertIn(fragment, checker)

    def test_speecht5_model_detail_matches_transformers_contract(self):
        source = (MODEL_PAGE_DIR / "speecht5.md").read_text(encoding="utf-8")
        headings = (
            "# SpeechT5",
            "## Usage",
            "## Overview",
            "## Paper and GitHub",
            "## Configuration",
            "## Processing",
            "## Inference",
            "## Training and optimization",
            "## Checkpoints, provenance, license, and limitations",
            "### Limitations",
            "## Public API",
            "### `SpeechT5Config`",
            "### `SpeechT5ForTextToSpeech`",
        )
        positions = tuple(source.index(heading) for heading in headings)
        self.assertEqual(positions, tuple(sorted(positions)))
        self.assertEqual(
            next(line for line in source.splitlines() if line.startswith("## ")),
            "## Usage",
        )

        for fragment in (
                "from voicehub import AutoConfig",
                "from voicehub import AutoProcessor",
                "AutoModelForTextToSpeech.from_pretrained(",
                "SpeechT5Config(**config_kwargs)",
                "AutoProcessor.from_pretrained(",
                "model_type='speecht5'",
                "Normalized output",
                "Optional dependency extra",
                "Checkpoint status",
                "Hardware and runtime",
                "Real-checkpoint evidence",
                "voicehub/models/speecht5/configuration_speecht5.py",
                "voicehub/models/speecht5/modeling_speecht5.py",
        ):
            self.assertIn(fragment, source)

        examples = PYTHON_BLOCK.findall(source)
        self.assertGreaterEqual(len(examples), 4)
        for example_index, example in enumerate(examples, start=1):
            ast.parse(
                textwrap.dedent(example),
                filename=f"speecht5.md:python-block-{example_index}",
            )

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "SPEECHT5_ROUTE",
                "SPEECHT5_HEADINGS",
                "SPEECHT5_TABLE_ROWS",
                "def _validate_speecht5_state(",
                "def _validate_speecht5_page_copy(",
                '"speecht5_cases"',
                '"speecht5_interaction_cases"',
        ):
            self.assertIn(fragment, checker)

    def test_model_guides_reference_bundled_source_manifests(self):
        from voicehub import list_model_specs

        generator = runpy.run_path(str(MODEL_PAGE_GENERATOR_PATH))
        source_provenance = generator["_source_provenance"]
        expected_examples = {
            "asr_nemo": "voicehub/architectures/nemo_ctc/SOURCE.json",
            "asr_speechbrain": ("voicehub/architectures/speechbrain_asr/SOURCE.json"),
            "asr_wenet": "voicehub/architectures/wenet_u2pp/SOURCE.json",
            "bark": "voicehub/architectures/bark/SOURCE.json",
            "vad_webrtc": "voicehub/architectures/webrtc_vad/SOURCE.json",
            "vits": "voicehub/architectures/vits/SOURCE.json",
        }

        for spec in list_model_specs(task=None):
            with self.subTest(model_type=spec.model_type):
                rendered = source_provenance(spec)
                if rendered.startswith("`"):
                    relative = rendered.strip("`")
                    self.assertTrue((REPOSITORY_ROOT / relative).is_file())
                    page = (MODEL_PAGE_DIR / f"{spec.model_type}.md").read_text(encoding="utf-8")
                    self.assertIn(rendered, page)

                if not spec.is_voicehub_native:
                    continue
                architecture_manifests = []
                for reference in spec.native_architecture.component_references.values():
                    module_path = REPOSITORY_ROOT / Path(*reference.module.split("."))
                    package = (module_path if module_path.is_dir() else module_path.with_suffix(".py").parent)
                    manifest = package / "SOURCE.json"
                    if manifest.is_file():
                        architecture_manifests.append(manifest)
                if architecture_manifests:
                    self.assertFalse(rendered.startswith("No integration-specific"))

        specs = {spec.model_type: spec for spec in list_model_specs(task=None)}
        for model_type, expected in expected_examples.items():
            with self.subTest(example=model_type):
                self.assertEqual(source_provenance(specs[model_type]), f"`{expected}`")

    def test_model_page_source_discovery_remains_backend_lazy(self):
        code = """
import json
import runpy
import sys

generator = runpy.run_path("scripts/generate_model_pages.py")
generator["generated_files"]()
blocked = ("nemo", "safetensors", "sentencepiece", "torch", "transformers")
print(json.dumps({name: name in sys.modules for name in blocked}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(completed.stdout.strip().splitlines()[-1]),
            {
                "nemo": False,
                "safetensors": False,
                "sentencepiece": False,
                "torch": False,
                "transformers": False,
            },
        )

    def test_external_archive_checkpoint_records_the_verified_mirror(self):
        from voicehub import get_model_spec

        generator = runpy.run_path(str(MODEL_NOTEBOOK_GENERATOR_PATH))
        spec = get_model_spec("asr_wenet")
        checkpoint = generator["checkpoint_documentation"](spec)
        source_record = json.loads(
            (REPOSITORY_ROOT / "voicehub/architectures/wenet_u2pp/SOURCE.json").read_text(encoding="utf-8"))
        page = (MODEL_PAGE_DIR / "asr_wenet.md").read_text(encoding="utf-8")
        index = MODEL_PAGE_INDEX_PATH.read_text(encoding="utf-8")
        gallery = MODEL_NOTEBOOK_GALLERY_PATH.read_text(encoding="utf-8")

        self.assertEqual(checkpoint.provider, "external-archive")
        self.assertEqual(checkpoint.example, "path/to/converted-wenet-u2pp")
        self.assertIn("2026-08-04", checkpoint.status)
        self.assertIn("openspeech/wenet-models", checkpoint.status)
        self.assertIn("github.com/wenet-e2e/wenet/blob/", checkpoint.url)
        self.assertFalse(checkpoint.is_hugging_face)
        self.assertEqual(spec.license.upstream, checkpoint.url)
        self.assertEqual(
            source_record["artifact"]["availability"]["status"],
            "available-via-verified-mirror",
        )
        self.assertEqual(source_record["artifact"]["availability"]["http_status"], 404)
        self.assertEqual(
            source_record["artifact"]["availability"]["mirror"]["revision"],
            "90acd57d17169a15d5ceab462c6e7db3bd003921",
        )
        self.assertEqual(
            source_record["artifact"]["availability"]["mirror"]["repository"],
            "openspeech/wenet-models",
        )
        self.assertEqual(source_record["artifact"]["availability"]["source_listing"], checkpoint.url)
        self.assertIn(checkpoint.example, page)
        self.assertIn(checkpoint.status, page)
        self.assertIn(checkpoint.url, page)
        self.assertIn(checkpoint.url, index)
        self.assertNotIn("https://huggingface.co/wenet/gigaspeech", page)
        self.assertNotIn("asr_wenet.ipynb", page)
        self.assertNotIn("asr_wenet.ipynb", gallery)
        self.assertFalse((MODEL_NOTEBOOK_DIR / "asr_wenet.ipynb").exists())

    def test_notebook_code_cells_compile_and_execute_in_smoke_mode(self):
        namespaces = {}
        for path, notebook in self.notebooks.items():
            namespace = {
                "__name__": "__main__",
            }
            output = io.StringIO()
            original_directory = Path.cwd()
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                try:
                    with redirect_stdout(output):
                        for cell in notebook["cells"]:
                            if cell["cell_type"] != "code":
                                continue
                            source = _cell_source(cell)
                            ast.parse(
                                source,
                                filename=f"{path.name}:{cell['id']}",
                            )
                            tags = set(cell["metadata"].get("tags", ()))
                            if "smoke-safe" not in tags:
                                continue
                            self.assertTrue(
                                tags.isdisjoint({
                                    "requires-model",
                                    "requires-training",
                                    "requires-audio-runtime",
                                    "writes-data",
                                    "requires-data",
                                    "setup",
                                    "optional-colab",
                                }))
                            exec(  # noqa: S102 - execute opt-in notebook smoke cells
                                compile(
                                    source,
                                    f"{path.name}:{cell['id']}",
                                    "exec",
                                ),
                                namespace,
                            )
                    self.assertEqual(list(Path(directory).iterdir()), [])
                finally:
                    os.chdir(original_directory)
            namespaces[path.name] = namespace

        workflow = namespaces["tts_workflow.ipynb"]
        self.assertFalse(workflow["RUN_INFERENCE"])
        self.assertFalse(workflow["RUN_TRAINING"])
        self.assertFalse(workflow["RUN_POST_TRAINING_INFERENCE"])
        self.assertEqual(workflow["MODEL_TYPE"], "dia")
        self.assertEqual(workflow["training_spec"].model_type, "dia")
        self.assertFalse(workflow["manifest_loaded"])
        self.assertIs(workflow["records"], workflow["template_records"])
        self.assertTrue(workflow["validation_errors"])
        self.assertTrue(workflow["train_records"])
        self.assertTrue(workflow["validation_records"])
        train_sessions = {record["session_id"] for record in workflow["train_records"]}
        validation_sessions = {record["session_id"] for record in workflow["validation_records"]}
        self.assertTrue(train_sessions.isdisjoint(validation_sessions))
        self.assertGreaterEqual(len(workflow["EVALUATION_TEXT"].split()), 55)
        self.assertNotIn("workflow_trainer", workflow)
        self.assertNotIn("baseline_output", workflow)
        self.assertNotIn("fine_tuned_output", workflow)

        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                '{"id":"one","text":"Authorized","audio":"audio/one.wav"}\n',
                encoding="utf-8",
            )
            loaded = workflow["load_manifest_records"](manifest)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0]["audio"],
            str((manifest.parent / "audio" / "one.wav").resolve()),
        )

        inference = namespaces["inference.ipynb"]
        self.assertFalse(inference["RUN_TTS"])
        self.assertFalse(inference["RUN_TTS_OPTIMIZATION"])
        self.assertFalse(inference["RUN_ASR"])
        self.assertFalse(inference["RUN_VAD"])
        self.assertEqual(sum(inference["task_counts"].values()), len(inference["catalog"]))
        self.assertEqual(len(inference["TTS_SAMPLES"]), 3)
        self.assertGreaterEqual(
            min(len(text.split()) for text in inference["TTS_SAMPLES"]),
            55,
        )
        self.assertNotIn("tts_output", inference)
        self.assertNotIn("asr_output", inference)
        self.assertNotIn("vad_output", inference)

        data = namespaces["data_preparation.ipynb"]
        self.assertFalse(data["WRITE_MANIFESTS"])
        self.assertFalse(data["RUN_AUDIO_VALIDATION"])
        self.assertFalse(data["RUN_MODEL_PREPARATION"])
        self.assertEqual(len(data["tts_source"]), 4)
        self.assertEqual(len(data["asr_source"]), 4)
        self.assertEqual(len(data["vad_source"]), 2)
        self.assertTrue(data["tts_train_groups"].isdisjoint(data["tts_validation_groups"], ))
        self.assertTrue(data["asr_train_groups"].isdisjoint(data["asr_validation_groups"], ))
        self.assertEqual(data["tts_contract"].model_type, "dia")
        self.assertEqual(data["asr_contract"].model_type, "asr_wav2vec2")

        training = namespaces["training.ipynb"]
        self.assertFalse(training["RUN_TRAINING"])
        self.assertFalse(training["RUN_RELOAD"])
        self.assertEqual(training["MODEL_TYPE"], "dia")
        self.assertEqual(training["training_spec"].model_type, "dia")
        self.assertEqual(training["smoke_arguments"].max_steps, 1)
        self.assertNotIn("active_trainer", training)
        self.assertNotIn("active_model", training)

    def test_site_sources_and_navigation_exist(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn(f"site_url: {PUBLIC_SITE_URL}", config)

        for relative_path in NAVIGATION_PATHS:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((DOCS_ROOT / relative_path).is_file())
                self.assertIn(relative_path, config)

        self.assertFalse((DOCS_ROOT / "tts_workflow.md").exists())

    def test_homepage_matches_current_transformers_representative_contract(self):
        source = HOME_PATH.read_text(encoding="utf-8")
        normalized_source = " ".join(source.split())
        headings = tuple(line for line in source.splitlines() if line.startswith(("# ", "## ", "### ")))
        self.assertEqual(
            headings,
            (
                "# VoiceHub",
                "## Features",
                "## Design",
                "## Learn",
            ),
        )
        self.assertEqual(source.count("-   **"), 13)
        for fragment in (
                "[Inference](guides/inference.md)",
                "[Trainer](guides/trainer.md)",
                "[generate](reference/api.md#generation)",
                "!!! tip",
                "configuration, model, and processor",
                "**68 integrations**",
                "**34 TTS backends**",
                "**23 ASR providers**",
                "**11 VAD providers**",
                '<div class="grid cards" markdown>',
                "https://github.com/kadirnar/voicehub/actions/workflows/ci.yml",
                "https://github.com/kadirnar/voicehub/actions/workflows/docs.yml",
                "https://github.com/kadirnar/voicehub/blob/main/pyproject.toml",
                "https://github.com/kadirnar/voicehub/blob/main/LICENSE",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, normalized_source)

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "HOME_ROUTE",
                "HOME_HEADINGS",
                "HOME_FEATURE_TARGETS",
                "HOME_CARD_TARGETS",
                "HOME_BADGE_TARGETS",
                "def _validate_home_state(",
                "def _validate_home_page_copy(",
                '"home_cases"',
                '"home_interaction_cases"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, checker)

    def test_installation_matches_current_transformers_workflow(self):
        source = INSTALLATION_PATH.read_text(encoding="utf-8")
        headings = tuple(line for line in source.splitlines() if line.startswith(("# ", "## ", "### ")))
        self.assertEqual(
            headings,
            (
                "# Installation",
                "## Create an environment",
                "## Install",
                "### Editable checkout",
                "## Verify",
                "## Cache and offline mode",
            ),
        )
        for fragment in (
                '=== "Linux"',
                '=== "macOS"',
                '=== "Windows"',
                ".venv\\Scripts\\Activate.ps1",
                "python -m pip install \"voicehub @ git+https://github.com/kadirnar/voicehub.git@main\"",
                "python -m pip install \"voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@main\"",
                "git clone https://github.com/kadirnar/voicehub.git",
                "python -m pip install -e \".[test,training,docs]\"",
                "VOICEHUB_OFFLINE=1",
                "local_files_only=True",
                "https://pytorch.org/get-started/locally/",
        ):
            self.assertIn(fragment, source)
        self.assertEqual(source.count('=== "'), 6)

        examples = PYTHON_BLOCK.findall(source)
        self.assertEqual(len(examples), 1)
        for example_index, example in enumerate(examples, start=1):
            ast.parse(
                textwrap.dedent(example),
                filename=f"getting-started/installation.md:python-block-{example_index}",
            )

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "INSTALLATION_ROUTE",
                "INSTALLATION_HEADINGS",
                "INSTALLATION_EXTERNAL_TARGETS",
                "INSTALLATION_INTERNAL_TARGETS",
                "def _validate_installation_state(",
                "def _validate_installation_code_copy(",
                "def _validate_installation_page_copy(",
                '"installation_cases"',
                '"installation_code_interaction_cases"',
                '"installation_page_interaction_cases"',
        ):
            self.assertIn(fragment, checker)

    def test_navigation_uses_the_compact_product_architecture(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        navigation = config.split("nav:\n", 1)[1].split("\nplugins:", 1)[0]
        labels = tuple(
            line.removeprefix("  - ").split(":", 1)[0] for line in navigation.splitlines()
            if line.startswith("  - "))
        self.assertEqual(labels, TOP_LEVEL_NAVIGATION)

        sections = {}
        for index, label in enumerate(TOP_LEVEL_NAVIGATION):
            start = f"  - {label}:"
            end = (
                f"  - {TOP_LEVEL_NAVIGATION[index + 1]}:" if index + 1 < len(TOP_LEVEL_NAVIGATION) else None)
            section = navigation.split(start, 1)[1]
            sections[label] = section if end is None else section.split(end, 1)[0]

        get_started = sections["Get started"]
        self.assertIn("index.md", get_started)
        self.assertIn("getting-started/installation.md", get_started)
        self.assertIn("getting-started/quickstart.md", get_started)
        self.assertIn("guides/inference.md", get_started)
        self.assertIn("models/providers/index.md", sections["Models"])
        self.assertIn("reference/models.md", sections["Models"])
        self.assertIn("project/adding-a-model.md", sections["Models"])
        self.assertIn("guides/trainer.md", sections["Train"])
        self.assertIn("guides/training.md", sections["Train"])
        self.assertIn("models/training-support.md", sections["Train"])
        self.assertIn("guides/optimization-overview.md", sections["Optimize"])
        self.assertIn("optimizations/index.md", sections["Optimize"])
        self.assertNotIn("models/xtts2.md", navigation)

        stale_labels = ("Home", "Quick Start", "API Reference", "Contributing")
        for locale in LOCALIZED_HOME_LOCALES:
            with self.subTest(locale=locale):
                locale_block = config.split(f"        - locale: {locale}\n", 1)[1]
                locale_block = locale_block.split("        - locale:", 1)[0]
                for label in TOP_LEVEL_NAVIGATION:
                    self.assertRegex(locale_block, rf"(?m)^            {re.escape(label)}:")
                for label in stale_labels:
                    self.assertNotRegex(locale_block, rf"(?m)^            {re.escape(label)}:")

    def test_every_visible_navigation_label_has_a_translation_in_every_locale(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        navigation = config.split("nav:\n", 1)[1].split("\nplugins:", 1)[0]
        required_labels = set()
        for line in navigation.splitlines():
            match = re.match(r'^\s+- (?:(?:"([^"]+)")|([^:]+)):(?:\s+(.+))?$', line)
            if match is None:
                continue
            label = match.group(1) or match.group(2).strip()
            target = (match.group(3) or "").strip()
            if target.startswith("models/providers/") and target != "models/providers/index.md":
                continue
            required_labels.add(label)

        self.assertLessEqual(len(required_labels), 31)
        for locale in LOCALIZED_HOME_LOCALES:
            with self.subTest(locale=locale):
                locale_block = config.split(f"        - locale: {locale}\n", 1)[1]
                locale_block = locale_block.split("        - locale:", 1)[0]
                translated_labels = {
                    match.group(1)
                    for match in re.finditer(r"(?m)^            ([^:#][^:]*):\s+.+$", locale_block)
                }
                self.assertEqual(required_labels - translated_labels, set())

    def test_every_model_guide_is_listed_in_the_primary_models_navigation(self):
        from voicehub import list_model_specs

        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        models_navigation = config.split("  - Models:\n", 1)[1].split(
            "  - Train:\n",
            1,
        )[0]
        model_guides = models_navigation.split(
            "      # BEGIN GENERATED MODEL GUIDE NAVIGATION",
            1,
        )[1].split("      # END GENERATED MODEL GUIDE NAVIGATION", 1)[0]

        self.assertIn("- Model list: models/providers/index.md", models_navigation)
        self.assertIn("- Models API: reference/models.md", models_navigation)
        self.assertIn("- Add a model: project/adding-a-model.md", models_navigation)
        self.assertIn("- Text to speech:", model_guides)
        self.assertIn("- Automatic speech recognition:", model_guides)
        self.assertIn("- Voice activity detection:", model_guides)
        self.assertLess(
            models_navigation.index("- Model list:"),
            models_navigation.index("# BEGIN GENERATED MODEL GUIDE NAVIGATION"),
        )

        for spec in list_model_specs(task=None):
            entry = f"models/providers/{spec.model_type}.md"
            with self.subTest(model_type=spec.model_type):
                self.assertEqual(config.count(entry), 1)
                self.assertIn(entry, model_guides)

        dom_checker = DOCUMENTATION_DOM_CHECK_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'expanded_branches=("Models", )',
            dom_checker,
        )
        self.assertIn(
            'expanded_branches=("Models", "Text to speech")',
            dom_checker,
        )

    def test_every_model_guide_declares_evidence_bounded_language_support(self):
        from voicehub import SpeechTask, list_model_specs
        from voicehub.models.language_support import model_language_support

        index = MODEL_PAGE_INDEX_PATH.read_text(encoding="utf-8")
        self.assertEqual(index.count('<div class="vh-model-catalog" markdown>'), 3)
        self.assertEqual(index.count("| Model | Languages | Default checkpoint | Training | Notebook |"), 3)

        for spec in list_model_specs(task=None):
            with self.subTest(model_type=spec.model_type):
                support = model_language_support(spec)
                page = (MODEL_PAGE_DIR / f"{spec.model_type}.md").read_text(encoding="utf-8")
                self.assertIn(f"# {spec.display_name} {{.vh-model-title}}", page)
                self.assertIn("| Languages |", page)
                self.assertIn("### Language support", page)
                if support.kind == "enumerated":
                    self.assertTrue(support.codes)
                    for code in support.codes:
                        self.assertIn(f"`{code}`", page)
                    self.assertIn('<details class="vh-language-support" markdown>', page)
                elif support.kind == "not-text-conditioned":
                    self.assertIs(spec.task, SpeechTask.VOICE_ACTIVITY_DETECTION)
                    self.assertIn("Not text-language conditioned", page)
                    self.assertIn("does not select a spoken language", page)
                else:
                    self.assertFalse(support.codes)
                    self.assertIn("Checkpoint-defined; not exhaustively enumerated", page)
                    self.assertIn("does not claim one exhaustive language list", page)

    def test_model_and_optimization_highlights_use_semantic_routes(self):
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")

        for selector in (
                ".md-typeset .vh-model-title",
                ".md-typeset .vh-language-support",
                ".md-typeset .vh-model-catalog tbody td:first-child code",
                ".md-typeset code.vh-optimization-term",
                ".md-nav__link.vh-model-link",
                ".md-sidebar--primary a.md-nav__link.vh-model-link",
                ".md-nav__link.vh-optimization-link",
                "body.vh-optimization-page .md-typeset h1",
        ):
            self.assertIn(selector, stylesheet)
        for fragment in (
                "const initializeSemanticHighlights = () => {",
                'document.body.classList.toggle("vh-optimization-page", isOptimizationPage)',
                '"vh-model-link",',
                'link.classList.toggle("vh-optimization-link"',
                'document.querySelectorAll(".md-typeset code:not(pre code)")',
                'term.classList.add("vh-optimization-term")',
                "initializeSemanticHighlights();",
        ):
            self.assertIn(fragment, script)

        model_link_rule = stylesheet.split(
            ".md-sidebar--primary a.md-nav__link.vh-model-link {",
            1,
        )[1].split("}", 1)[0]
        self.assertIn("width: calc(100% + 0.45rem);", model_link_rule)
        self.assertIn("padding: 0.3rem 0.45rem;", model_link_rule)

    def test_theme_uses_reference_neutral_surfaces_with_voicehub_accents(self):
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        mark = (DOCS_ROOT / "assets" / "voicehub-mark.svg").read_text(encoding="utf-8")
        repository_mark = (REPOSITORY_ROOT / "assets" / "logo.svg").read_text(encoding="utf-8")

        self.assertEqual(mark, repository_mark)

        for legacy_fragment in (
                "--vh-indigo",
                "--vh-header-start",
                "--vh-header-end",
                "--vh-header-fuchsia",
                "--vh-header-coral",
                "--vh-header-tangerine",
                "--vh-header-sunshine",
                "--vh-header-mint",
                "--vh-header-violet",
                "--vh-header-glow",
                "--vh-page-glow-coral",
                "--vh-page-glow-sunshine",
                "--vh-page-glow-mint",
                "--vh-page-glow-violet",
                "#2563eb",
                "#1e40af",
                "#2878ed",
                "#1d5fd1",
                "#0891b2",
                "#1fb6c9",
                "rgba(74, 144, 255",
                "rgba(125, 211, 252",
        ):
            with self.subTest(legacy_fragment=legacy_fragment):
                self.assertNotIn(legacy_fragment, stylesheet)
                self.assertNotIn(legacy_fragment, mark)

        header = stylesheet.split(".md-header {", 1)[1].split("}", 1)[0]
        self.assertIn("border-bottom: 1px solid var(--vh-line);", header)
        self.assertIn("background: var(--vh-surface);", header)
        self.assertIn("box-shadow: none;", header)
        self.assertIn("color: var(--vh-ink);", header)
        self.assertNotIn("gradient", header)

        body = stylesheet.split("body {", 1)[1].split("}", 1)[0]
        self.assertIn("background-image: none;", body)
        self.assertIn('"Source Sans 3", "Source Sans Pro"', body)

        active_navigation = stylesheet.split(".md-nav__link--active {", 1)[1].split("}", 1)[0]
        self.assertIn("background: var(--vh-active-bg);", active_navigation)
        self.assertIn("color: var(--vh-active-text);", active_navigation)
        self.assertIn("box-shadow: none;", active_navigation)

        back_to_top_state = stylesheet.split(
            ".md-top:is(:hover, :focus-visible) {",
            1,
        )[1].split("}", 1)[0]
        self.assertIn("background: var(--vh-surface-soft);", back_to_top_state)
        self.assertIn("color: var(--vh-accent);", back_to_top_state)
        self.assertEqual(mark.count("<stop"), 2)
        self.assertEqual(mark.count("<rect"), 0)
        self.assertEqual(mark.count("<path"), 1)
        self.assertEqual(mark.count("<circle"), 0)
        self.assertIn('fill="url(#voicehub-signal)"', mark)
        self.assertIn('fill-rule="evenodd"', mark)
        for color in ("#ff4f68", "#ff8a3d"):
            self.assertIn(f'stop-color="{color}"', mark)
        for rejected_color in ("#111827", "#f8fafc", "#4f46e5", "#14b8a6"):
            self.assertNotIn(rejected_color, mark)

    def test_model_api_reference_matches_transformers_contract(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        models_navigation = config.split("  - Models:\n", 1)[1].split("  - Train:\n", 1)[0]
        self.assertIn("- Models API: reference/models.md", models_navigation)
        self.assertEqual(config.count("reference/models.md"), 1)
        for locale in LOCALIZED_HOME_LOCALES:
            with self.subTest(locale=locale):
                locale_block = config.split(f"        - locale: {locale}\n", 1)[1]
                locale_block = locale_block.split("        - locale:", 1)[0]
                self.assertRegex(locale_block, r"(?m)^            Models API:")

        source = MODEL_API_PATH.read_text(encoding="utf-8")
        normalized_source = " ".join(source.split())
        headings = tuple(line for line in source.splitlines() if line.startswith("#"))
        self.assertEqual(
            headings,
            (
                "# Models",
                "## `PreTrainedSpeechModel`",
                "## Task-specific pretrained models",
                "## Model outputs",
                "## Loading, saving, and sharing",
            ),
        )
        for fragment in (
                "PreTrainedSpeechModel",
                "PreTrainedTTSModel",
                "PreTrainedAudioModel",
                "PreTrainedASRModel",
                "PreTrainedVADModel",
                "TTSOutput",
                "ASROutput",
                "VADOutput",
                "from_pretrained",
                "save_pretrained",
                "load_for_training",
                "validate_training_support",
                "https://github.com/kadirnar/voicehub/blob/main/voicehub/modeling_utils.py",
                "https://github.com/kadirnar/voicehub/blob/main/voicehub/audio_modeling_utils.py",
                "https://github.com/kadirnar/voicehub/blob/main/voicehub/modeling_outputs.py",
                "[full API reference](api.md)",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)
        self.assertIn("does not expose a public `push_to_hub()` method", normalized_source)
        examples = PYTHON_BLOCK.findall(source)
        self.assertGreaterEqual(len(examples), 2)
        for example_index, example in enumerate(examples, start=1):
            ast.parse(
                textwrap.dedent(example),
                filename=f"reference/models.md:python-block-{example_index}",
            )
        self.assertLessEqual(len(source.splitlines()), 240)

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "MODEL_API_ROUTE",
                "MODEL_API_HEADINGS",
                "MODEL_API_SOURCE_TARGETS",
                "MODEL_API_INTERNAL_TARGETS",
                "def _validate_model_api_state(",
                "def _validate_model_api_page_copy(",
                '"model_api_cases"',
                '"model_api_interaction_cases"',
        ):
            self.assertIn(fragment, checker)

        parity = (DOCS_ROOT / "project" / "transformers-parity.md").read_text(encoding="utf-8")
        api_inventory = parity.split("## API main-class route inventory\n", 1)[1].split("\n## ", 1)[0]
        upstream_titles = (
            "Auto Classes",
            "Backbones",
            "Callbacks",
            "Configuration",
            "Continuous batching",
            "Data Collator",
            "Logging",
            "Models",
            "Text Generation",
            "Optimization",
            "Model outputs",
            "PEFT",
            "Pipelines",
            "Processors",
            "Exporters",
            "Quantization",
            "Tokenizer",
            "Trainer",
            "DeepSpeed",
            "ExecuTorch",
            "Feature Extractor",
            "Image Processor",
            "Video Processor",
            "Kernels",
        )
        for title in upstream_titles:
            with self.subTest(upstream_title=title):
                self.assertEqual(api_inventory.count(f"| {title} |"), 1)

    def test_trainer_overview_matches_transformers_representative_contract(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        training_navigation = config.split("  - Train:\n", 1)[1].split(
            "  - Optimize:\n",
            1,
        )[0]
        overview_entry = "- Trainer overview: guides/trainer.md"
        fine_tuning_entry = "- Fine-tuning: guides/training.md"
        self.assertIn(overview_entry, training_navigation)
        self.assertIn(fine_tuning_entry, training_navigation)
        self.assertLess(
            training_navigation.index(overview_entry),
            training_navigation.index(fine_tuning_entry),
        )
        self.assertEqual(config.count("guides/trainer.md"), 1)
        self.assertEqual(config.count("guides/training.md"), 1)

        source = TRAINER_OVERVIEW_PATH.read_text(encoding="utf-8")
        normalized_source = " ".join(source.split())
        self.assertTrue(source.startswith("---\n"))
        self.assertIn("description:", source.split("---", 2)[1])
        headings = tuple(line for line in source.splitlines() if line.startswith("#"))
        self.assertEqual(headings, ("# Trainer", "## Next steps"))
        for fragment in (
                "`Trainer`",
                "`TrainingArguments`",
                "training and evaluation loop",
                "model-owned objective",
                "batching",
                "gradient accumulation",
                "evaluation",
                "checkpoint",
                "[fine-tuning tutorial](training.md)",
                "[Trainer architecture](../concepts/trainer.md)",
                "[training support matrix](../models/training-support.md)",
                "[data preparation guide](data-preparation.md)",
        ):
            self.assertIn(fragment, normalized_source)
        self.assertLessEqual(len(source.splitlines()), 45)
        self.assertNotIn("```", source)
        self.assertNotIn("<table", source)

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "TRAINER_ROUTE",
                "TRAINER_HEADINGS",
                "TRAINER_NEXT_STEP_PATHS",
                "def _validate_trainer_state(",
                "def _validate_trainer_page_copy(",
                '"trainer_cases"',
                '"trainer_interaction_cases"',
        ):
            self.assertIn(fragment, checker)

    def test_optimization_overview_matches_transformers_representative_contract(self):
        from voicehub.optimization import OPTIMIZATION_PASSES

        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        optimization_navigation = config.split(
            "  - Optimize:\n",
            1,
        )[1].split("\nplugins:", 1)[0]
        overview_entry = "- Overview: guides/optimization-overview.md"
        workflow_entry = "- Optimization catalog: optimizations/index.md"
        self.assertIn(overview_entry, optimization_navigation)
        self.assertIn(workflow_entry, optimization_navigation)
        self.assertLess(
            optimization_navigation.index(overview_entry),
            optimization_navigation.index(workflow_entry),
        )
        self.assertEqual(config.count("guides/optimization-overview.md"), 1)
        self.assertEqual(config.count("optimizations/index.md"), 1)

        source = OPTIMIZATION_OVERVIEW_PATH.read_text(encoding="utf-8")
        normalized_source = " ".join(source.split())
        self.assertTrue(source.startswith("---\n"))
        self.assertIn("description:", source.split("---", 2)[1])
        headings = tuple(line for line in source.splitlines() if line.startswith("#"))
        self.assertEqual(
            headings,
            (
                "# Optimization overview",
                "## Compilation",
                "## Attention backends",
                "## Kernels",
                "## Diffusion caching",
                "## Diffusion sampling",
                "## Boundaries",
                "## Next steps",
            ),
        )
        for pass_name in OPTIMIZATION_PASSES.list():
            with self.subTest(pass_name=pass_name):
                self.assertEqual(source.count(f"`{pass_name}`"), 1)
        for fragment in (
                "available_optimization_passes()",
                "apply_optimization_plan(",
                "optimization_manifest(",
                "restore_optimization_plan(",
                "validation happens before mutation",
                "no registry-wide public quantization pass",
                "Parallelism is a training or serving topology",
                "Continuous batching belongs to a serving scheduler",
                "[TTS optimization workflow](tts-optimization.md)",
                "[optimization API](../reference/api.md#optimization)",
        ):
            self.assertIn(fragment, normalized_source)
        examples = PYTHON_BLOCK.findall(source)
        self.assertEqual(len(examples), 1)
        ast.parse(
            textwrap.dedent(examples[0]),
            filename="guides/optimization-overview.md:python-block-1",
        )
        self.assertLessEqual(len(source.splitlines()), 150)

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "OPTIMIZATION_ROUTE",
                "OPTIMIZATION_HEADINGS",
                "OPTIMIZATION_PASS_NAMES",
                "OPTIMIZATION_NEXT_STEP_TARGETS",
                "def _validate_optimization_state(",
                "def _validate_optimization_page_copy(",
                '"optimization_cases"',
                '"optimization_interaction_cases"',
        ):
            self.assertIn(fragment, checker)

    def test_every_optimization_has_a_generated_guide_and_sidebar_link(self):
        from voicehub.optimization import OPTIMIZATION_PASSES

        generator = runpy.run_path(str(OPTIMIZATION_PAGE_GENERATOR_PATH))
        guides = tuple(generator["OPTIMIZATION_GUIDES"])
        expected_paths = {OPTIMIZATION_PAGE_DIR / f"{guide.slug}.md": guide for guide in guides}
        self.assertEqual(
            set(OPTIMIZATION_PAGE_DIR.glob("*.md")),
            {*expected_paths, OPTIMIZATION_PAGE_INDEX_PATH},
        )
        self.assertEqual(len(guides), 11)
        self.assertEqual(
            {guide.registry_name
             for guide in guides if guide.registry_name},
            set(OPTIMIZATION_PASSES.list()),
        )

        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        index = OPTIMIZATION_PAGE_INDEX_PATH.read_text(encoding="utf-8")
        for path, guide in expected_paths.items():
            with self.subTest(optimization=guide.slug):
                source = path.read_text(encoding="utf-8")
                headings = tuple(
                    line.removeprefix("## ") for line in source.splitlines() if line.startswith("## "))
                self.assertEqual(headings, OPTIMIZATION_PAGE_SECTIONS)
                self.assertLessEqual(len(source.splitlines()), 65)
                self.assertIn("- **Paper:**", source)
                self.assertIn("- **Upstream GitHub:**", source)
                self.assertIn("- **VoiceHub source:**", source)
                for reference in guide.github:
                    self.assertIn(reference.url, source)
                    self.assertTrue(reference.url.startswith("https://github.com/"))
                if guide.papers:
                    for reference in guide.papers:
                        self.assertIn(reference.url, source)
                else:
                    self.assertIn(
                        "No dedicated upstream research paper is published",
                        source,
                    )
                self.assertIn(f"[{guide.title}]({path.name})", index)
                self.assertEqual(
                    config.count(f"optimizations/{path.name}"),
                    1,
                    f"{guide.slug} should appear once in the optimization sidebar",
                )
                if guide.registry_name:
                    self.assertIn(f"`{guide.registry_name}`", source)
                    self.assertIn(guide.pass_id, source)
                    self.assertIn("restore_optimization_plan", source)
                if guide.source_install:
                    self.assertIn(guide.source_install, source)
                    self.assertIn("separate environment", source)

        files = generator["generated_files"]()
        self.assertEqual(generator["check_generated_files"](files), ())
        self.assertNotIn("- Optimization passes:", config.split("nav:\n", 1)[1].split("\nplugins:", 1)[0])
        self.assertNotIn(
            "- Optional source backends:",
            config.split("nav:\n", 1)[1].split("\nplugins:", 1)[0])

    def test_optional_backends_are_source_pinned_and_fail_closed(self):
        source = OPTIONAL_BACKENDS_PATH.read_text(encoding="utf-8")
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn('- "HQQ": optimizations/hqq.md', config)
        self.assertIn('- "GemLite": optimizations/gemlite.md', config)
        self.assertIn('- "audio.cpp": optimizations/audio-cpp.md', config)
        self.assertIn('- "vLLM": optimizations/vllm.md', config)
        self.assertIn('- "SGLang": optimizations/sglang.md', config)
        for fragment in (
                "dropbox/hqq.git@d88a488ec8aa2d58362ef2038a52bca862db2e74",
                "dropbox/gemlite.git@3dc52c3115fee49a09d00fd9e470ef6396885949",
                "git checkout 748c5e28f6a7228b8f38ad7142ca97d29584544b",
                "not a Python optimization pass",
                "does not report them as applied public passes",
                "real-checkpoint evidence",
                "list_llm_backend_support",
                "does not silently fall back",
        ):
            self.assertIn(fragment, source)

        self.assertNotIn("pip install hqq\n", source)
        self.assertNotIn("pip install gemlite\n", source)

    def test_model_contribution_matches_current_modular_transformers_contract(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        models_navigation = config.split("  - Models:\n", 1)[1].split(
            "  - Train:\n",
            1,
        )[0]
        add_model_entry = "- Add a model: project/adding-a-model.md"
        self.assertIn(add_model_entry, models_navigation)
        self.assertNotIn("- Add a TTS model:", config)
        self.assertEqual(config.count("project/adding-a-model.md"), 1)

        for locale in LOCALIZED_HOME_LOCALES:
            with self.subTest(locale=locale):
                locale_block = config.split(f"        - locale: {locale}\n", 1)[1]
                locale_block = locale_block.split("        - locale:", 1)[0]
                self.assertRegex(locale_block, r"(?m)^            Add a model:")

        source = ADDING_MODEL_PATH.read_text(encoding="utf-8")
        normalized_source = " ".join(source.split())
        self.assertIn(
            "https://huggingface.co/docs/transformers/main/en/modular_transformers",
            source,
        )
        self.assertIn("legacy `add_new_model` guide", normalized_source)
        self.assertIn('class="vh-process vh-process--eight"', source)
        titles = (
            "Create the package",
            "Record provenance and license",
            "Define the config",
            "Implement the task wrapper",
            "Register once",
            "Declare training and optimization support",
            "Test the contract",
            "Generate the model page",
        )
        headings = tuple(f"## {index}. {title}" for index, title in enumerate(titles, start=1))
        positions = tuple(source.index(heading) for heading in headings)
        self.assertEqual(positions, tuple(sorted(positions)))
        for step in headings:
            with self.subTest(step=step):
                self.assertIn(f"| {step.removeprefix('## ')} |", source)
        for fragment in (
                "explicit standalone files",
                "composition",
                "voicehub/models/<model_type>/",
                "voicehub/architectures/<model_type>/",
                "tests/test_<model_type>.py",
                "docs/models/providers/<model_type>.md",
                "generated navigation block in `mkdocs.yml`",
        ):
            self.assertIn(fragment, normalized_source)

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "CONTRIBUTION_ROUTE",
                "CONTRIBUTION_HEADINGS",
                "CONTRIBUTION_PROCESS_LABELS",
                "CONTRIBUTION_FINAL_TARGETS",
                "def _validate_contribution_state(",
                "def _validate_contribution_page_copy(",
                '"contribution_cases"',
                '"contribution_interaction_cases"',
        ):
            self.assertIn(fragment, checker)

    def test_quickstart_matches_transformers_representative_contract(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("- Quickstart: getting-started/quickstart.md", config)
        self.assertNotIn("- First generation: getting-started/quickstart.md", config)

        quickstart = QUICKSTART_PATH.read_text(encoding="utf-8")
        headings = tuple(line for line in quickstart.splitlines() if line.startswith(("# ", "## ", "### ")))
        self.assertEqual(
            headings,
            (
                "# Quickstart",
                "## Set up",
                "## Pretrained models",
                "## Inference",
                "## Trainer",
                "## Next steps",
            ),
        )
        introduction = quickstart.split("## Set up", 1)[0]
        for outcome in (
                "load a pretrained speech model",
                "run inference with `pipeline()`",
                "inspect training support before constructing a `Trainer`",
        ):
            self.assertIn(outcome, introduction)
        for fragment in (
                '=== "Linux"',
                '=== "macOS"',
                '=== "Windows"',
                '=== "Text to speech"',
                '=== "Automatic speech recognition"',
                '=== "Voice activity detection"',
                "from voicehub import pipeline",
                'task="text-to-speech"',
                'task="automatic-speech-recognition"',
                'task="voice-activity-detection"',
                "!!! tip",
                "[Installation](installation.md)",
                "[Inference guide](../guides/inference.md)",
                "[training guide](../guides/training.md)",
                "[Model list](../models/providers/index.md)",
                "[Train](../guides/trainer.md)",
                "[Optimize](../guides/optimization-overview.md)",
        ):
            self.assertIn(fragment, quickstart)
        self.assertEqual(quickstart.count('=== "'), 6)
        self.assertEqual(quickstart.count("!!! tip"), 2)

        examples = PYTHON_BLOCK.findall(quickstart)
        self.assertGreaterEqual(len(examples), 6)
        for example_index, example in enumerate(examples, start=1):
            ast.parse(
                textwrap.dedent(example),
                filename=f"getting-started/quickstart.md:python-block-{example_index}",
            )

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "QUICKSTART_ROUTE",
                "QUICKSTART_HEADINGS",
                "QUICKSTART_TAB_LABELS",
                "QUICKSTART_EXTERNAL_TARGETS",
                "QUICKSTART_INTERNAL_TARGETS",
                "def _validate_quickstart_state(",
                "def _validate_quickstart_tabs(",
                "def _validate_quickstart_page_copy(",
                '"quickstart_cases"',
                '"quickstart_interaction_cases"',
                '"quickstart_page_interaction_cases"',
        ):
            self.assertIn(fragment, checker)

        quickstart_tabs = checker.split(
            "def _validate_quickstart_tabs(",
            1,
        )[1].split("def _validate_quickstart_page_copy", 1)[0]
        self.assertIn("for step_index in range(1, target_index + 1):", quickstart_tabs)
        self.assertIn('inputs.first.evaluate("input => input.focus({preventScroll: true})")', quickstart_tabs)
        self.assertIn("step_source.element_handle()", quickstart_tabs)
        self.assertIn('input.ownerDocument.addEventListener("keydown"', quickstart_tabs)
        self.assertIn('page.keyboard.press("ArrowRight")', quickstart_tabs)
        self.assertIn("input.dataset.vhArrowDefaultPrevented === 'true'", quickstart_tabs)
        self.assertIn("step_target.element_handle()", quickstart_tabs)
        self.assertIn("document.activeElement === input", quickstart_tabs)
        self.assertIn("const focusRequest = input.dataset.vhFocusRequest", quickstart_tabs)
        self.assertIn("input.dataset.vhFocusSettled === focusRequest", quickstart_tabs)
        self.assertIn("requestAnimationFrame", quickstart_tabs)
        self.assertIn("const stableTolerance = 0.25", quickstart_tabs)
        self.assertIn("const viewportTolerance = 1", quickstart_tabs)
        self.assertNotIn("target.focus()", quickstart_tabs)

        quickstart_interaction = checker.split(
            "if relative_path == QUICKSTART_ROUTE:",
            3,
        )[3].split("if relative_path == INFERENCE_ROUTE:", 1)[0]
        self.assertIn('page.reload(wait_until="networkidle")', quickstart_interaction)
        self.assertIn("_reset_quickstart_tabs(page)", quickstart_interaction)

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        self.assertIn("scroll-margin: 1rem;", stylesheet)
        for declaration in (
                "--vh-content-gutter: 24px;",
                "margin-inline: var(--vh-content-gutter);",
                "font-size: 0.84rem;",
                "font-size: 1.2rem;",
                "line-height: 1.6rem;",
                "font-size: 1rem;",
                "line-height: 1.4rem;",
        ):
            self.assertIn(declaration, stylesheet)
        desktop_shell = stylesheet.split("@media screen and (min-width: 60em) {", 1)[1]
        self.assertIn("--vh-content-gutter: 48px;", desktop_shell)
        mobile_shell = stylesheet.split("@media screen and (max-width: 59.984375em) {", 1)[1]
        self.assertIn("margin-inline: -12px;", mobile_shell)
        self.assertIn(".md-typeset .tabbed-labels {", mobile_shell)
        self.assertIn("width: calc(100% + 2rem);", mobile_shell)

        self.assertTrue(PAGE_ACTIONS_OVERRIDE_PATH.is_file())
        actions = PAGE_ACTIONS_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn("data-vh-copy-page", actions)
        self.assertIn("Copy page", actions)
        self.assertIn('aria-live="polite"', actions)
        self.assertTrue(PAGE_ACTION_SCRIPT_PATH.is_file())
        script = PAGE_ACTION_SCRIPT_PATH.read_text(encoding="utf-8")
        for fragment in (
                "initializePageActions",
                'querySelector("[data-vh-copy-page]")',
                "const copyPage = async ({ restoreKeyboardFocus = false } = {}) =>",
                "navigator.clipboard.writeText",
                'label.textContent = "Copied"',
                'button.addEventListener("keydown"',
                'event.key !== "Enter" && event.key !== " "',
                "event.preventDefault()",
                "void copyPage({ restoreKeyboardFocus: true })",
                "const keyboardActivation = event.detail === 0",
                "void copyPage({ restoreKeyboardFocus: keyboardActivation })",
                "let copyInProgress = false",
                'button.setAttribute("aria-busy", "true")',
                'document.execCommand("copy")',
                "button.focus({ preventScroll: true })",
                'button.classList.add("focus-visible")',
                'button.classList.remove("focus-visible")',
        ):
            self.assertIn(fragment, script)
        self.assertEqual(script.count("event.preventDefault()"), 1)
        self.assertEqual(script.count("void copyPage("), 2)
        self.assertNotIn("button.disabled", script)
        self.assertIn(
            ".md-content__button.vh-copy-page:is(:focus-visible, .focus-visible)",
            stylesheet,
        )
        self.assertIn('.md-content__button.vh-copy-page[aria-busy="true"]', stylesheet)
        self.assertIn("- javascripts/page-actions.js", config)

    def test_inference_guide_matches_transformers_representative_contract(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("- Inference: guides/inference.md", config)
        self.assertNotIn("- TTS inference: guides/inference.md", config)
        self.assertEqual(config.count("            Inference:"), len(LOCALIZED_HOME_LOCALES))

        guide_index = (DOCS_ROOT / "guides" / "index.md").read_text(encoding="utf-8")
        self.assertIn("[Inference guide](inference.md)", guide_index)
        self.assertNotIn("[TTS inference guide](inference.md)", guide_index)
        self.assertIn(
            "[model list](https://kadirnar.github.io/voicehub/models/providers/)",
            README_PATH.read_text(encoding="utf-8"),
        )

        guide = INFERENCE_GUIDE_PATH.read_text(encoding="utf-8")
        headings = (
            "# Inference",
            "## Tasks",
            "### Text to speech",
            "### Automatic speech recognition",
            "### Voice activity detection",
            "## Parameters",
            "### Device",
            "### Batch inference",
            "### Task-specific parameters",
            "## Chunking and streaming",
            "## Large inputs",
            "## Large models",
            "## Save and reload",
            "## Troubleshooting",
        )
        positions = tuple(guide.index(heading) for heading in headings)
        self.assertEqual(positions, tuple(sorted(positions)))
        for fragment in (
                "from voicehub import pipeline",
                'task="text-to-speech"',
                'task="automatic-speech-recognition"',
                'task="voice-activity-detection"',
                "TTSOutput",
                "ASROutput",
                "VADOutput",
                "duration < 10",
                "does not provide a universal vectorized batch contract",
        ):
            self.assertIn(fragment, guide)

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "INFERENCE_ROUTE",
                "INFERENCE_HEADINGS",
                "def _validate_inference_state(",
                "def _validate_inference_code_copy(",
                '"inference_cases"',
                '"inference_interaction_cases"',
        ):
            self.assertIn(fragment, checker)

    def test_header_controls_follow_transformers_product_order(self):
        header = HEADER_OVERRIDE_PATH.read_text(encoding="utf-8")
        control_order = ("product", "search", "version", "language", "theme", "source")
        positions = tuple(header.index(f'data-vh-header-control="{name}"') for name in control_order)
        self.assertEqual(positions, tuple(sorted(positions)))

        self.assertIn("data-vh-version-control", header)
        self.assertIn('aria-label="Documentation version"', header)
        self.assertNotIn('aria-haspopup="menu"', header)
        self.assertNotIn('role="menu"', header)
        self.assertNotIn('role="menuitem"', header)
        self.assertIn('aria-expanded="false"', header)
        self.assertIn("vh-header-product__compact", header)
        self.assertIn("config.extra.docs_version.label }} ·", header)
        self.assertIn("Release candidate status", header)
        self.assertIn("Published package", header)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        for fragment in ("toggle", "pointerdown", "Enter", "Escape", "aria-expanded"):
            self.assertIn(fragment, script)

        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("- javascripts/header-controls.js", config)
        self.assertIn("docs_version:", config)
        self.assertIn('release: "0.3.0"', config)
        self.assertIn('published: "0.1.6"', config)

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        self.assertIn(".md-header__button:is(:focus-visible, .focus-visible)", stylesheet)
        self.assertIn('[dir="rtl"] .vh-header-version__menu', stylesheet)

    def test_desktop_documentation_controls_use_transformers_left_rail(self):
        header = HEADER_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn('class="vh-global-brand"', header)
        self.assertIn('class="vh-doc-rail-controls"', header)
        self.assertIn('class="vh-doc-rail-utility"', header)

        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertNotRegex(config, r"(?m)^\s+- navigation\.tabs(?:\.sticky)?$")

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        for declaration in (
                "--vh-global-header-height: 3.25rem;",
                "--vh-doc-rail-width: 13.5rem;",
                "--vh-doc-rail-controls-height: 7.25rem;",
        ):
            self.assertIn(declaration, stylesheet)
        desktop_shell = stylesheet.split(
            "@media screen and (min-width: 60em) {",
            1,
        )[1].split(
            "@media screen and (min-width: 60em) and (max-width: 76.234375em)",
            1,
        )[0]
        for selector in (
                ".vh-doc-rail-controls {",
                ".vh-doc-rail-utility {",
                ".md-sidebar--primary {",
                ".md-sidebar--secondary {",
                ".md-main__inner {",
        ):
            self.assertIn(selector, desktop_shell)
        self.assertIn(
            "top: calc(var(--vh-global-header-height) - var(--vh-shell-scroll-offset));",
            desktop_shell,
        )
        self.assertIn("width: var(--vh-doc-rail-width);", desktop_shell)
        self.assertIn("height: var(--vh-doc-rail-controls-height);", desktop_shell)
        self.assertIn("padding-top: var(--vh-doc-rail-controls-height);", desktop_shell)
        self.assertIn("max-width: none;", desktop_shell)
        for declaration in (
                "grid-template-columns: minmax(0, 1fr) 2.4rem 1.7rem 2.4rem;",
                "column-gap: 0.3rem;",
                "height: 1.5rem;",
                "transform: none;",
        ):
            self.assertIn(declaration, desktop_shell)
        self.assertIn(
            ".vh-doc-rail-utility .vh-source-link .md-source__repository {",
            desktop_shell,
        )

    def test_desktop_documentation_shell_tracks_the_reference_scroll_offset(self):
        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        for fragment in (
                "initializeShellScrollTracking",
                "Math.min(window.scrollY, header.offsetHeight)",
                'document.documentElement.style.setProperty("--vh-shell-scroll-offset"',
                'window.addEventListener("scroll"',
                "requestAnimationFrame(updateShellScrollOffset)",
                "{ passive: true }",
        ):
            self.assertIn(fragment, script)

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        desktop_shell = stylesheet.split(
            "@media screen and (min-width: 60em) {",
            1,
        )[1].split(
            "@media screen and (min-width: 60em) and (max-width: 76.234375em)",
            1,
        )[0]
        for declaration in (
                "transform: translateY(calc(-1 * var(--vh-shell-scroll-offset)));",
                "top: calc(var(--vh-global-header-height) - var(--vh-shell-scroll-offset)) !important;",
                "height: calc(100vh - var(--vh-global-header-height) + var(--vh-shell-scroll-offset));",
                "height: auto !important;",
                "flex: 1 1 auto;",
                "z-index: auto;",
                ".vh-doc-rail-controls .md-header__topic:first-child {",
                ".vh-doc-rail-controls .md-search__form:focus-within {",
        ):
            self.assertIn(declaration, desktop_shell)

        tablet_shell = stylesheet.split(
            "@media screen and (min-width: 60em) and (max-width: 76.234375em)",
            1,
        )[1].split("@media screen and (max-width: 44.984375em)", 1)[0]
        self.assertIn("top: var(--vh-doc-rail-controls-height);", tablet_shell)
        self.assertIn(
            "height: calc(100% - var(--vh-doc-rail-controls-height)) !important;",
            tablet_shell,
        )

    def test_search_dialog_matches_transformers_interaction_contract(self):
        header = HEADER_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn("data-vh-search-trigger", header)
        self.assertIn('<button\n            type="button"', header)
        self.assertNotIn('role="button"', header)
        self.assertIn('aria-controls="__search"', header)
        self.assertIn('aria-expanded="false"', header)

        search = SEARCH_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn("data-vh-search-shortcut", search)
        self.assertIn("data-vh-search-shortcut-primary", search)
        self.assertIn("vh-search-shortcut__expanded", search)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        for fragment in (
                "initializeSearchControl",
                'querySelector("#__search")',
                'querySelector("[data-vh-search-trigger]")',
                'trigger.addEventListener("click"',
                'event.key.toLowerCase() === "k"',
                'event.key === "Escape"',
                "event.stopImmediatePropagation()",
                'document.body.classList.toggle("vh-search-open"',
                'trigger.setAttribute("aria-expanded"',
                "focusClosedSearchTarget",
                "closeFocusTarget.focus()",
        ):
            self.assertIn(fragment, script)

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        self.assertIn("#__search:checked ~ .md-header .md-search", stylesheet)
        self.assertIn("top: 3.2rem;", stylesheet)
        self.assertIn("width: min(25rem, calc(100vw - 2rem));", stylesheet)
        self.assertIn("body.vh-search-open", stylesheet)
        self.assertIn(
            "body.vh-search-open .md-header__inner > :not(.vh-doc-rail-controls)",
            stylesheet,
        )
        self.assertIn(
            "body.vh-search-open .vh-doc-rail-controls > :not(.md-search)",
            stylesheet,
        )
        self.assertIn("height: 100vh;", stylesheet)
        self.assertIn("top: 64px;", stylesheet)
        self.assertIn("right: 16px;", stylesheet)
        self.assertIn("left: 16px;", stylesheet)
        self.assertIn("height: 72px;", stylesheet)
        self.assertIn(".vh-search-shortcut", stylesheet)
        self.assertIn(".md-search__form:focus-within", stylesheet)

    def test_language_control_matches_transformers_native_select_contract(self):
        header = HEADER_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn('data-vh-header-control="language"', header)

        language = LANGUAGE_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn("data-vh-language-select", language)
        self.assertIn('aria-label="{{ lang.t(\'select.language\') }}"', language)
        self.assertIn("alt.lang | upper", language)
        self.assertIn("alt.lang == config.theme.language", language)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("initializeLanguageControl", script)
        self.assertIn('querySelector("[data-vh-language-select]")', script)
        self.assertIn('select.addEventListener("keydown"', script)
        self.assertIn('event.key === "ArrowDown"', script)
        self.assertIn("select.selectedIndex = nextIndex", script)
        self.assertIn('select.addEventListener("change"', script)
        self.assertIn('sessionStorage.setItem(paletteTransferKey', script)
        self.assertIn('sessionStorage.removeItem(paletteTransferKey)', script)
        self.assertIn("window.location.assign(select.value)", script)

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        self.assertIn(".vh-language-select", stylesheet)
        language_style = stylesheet.split(".vh-language-select {", 1)[1].split("}", 1)[0]
        self.assertIn("width: 2.4rem;", language_style)
        self.assertIn("height: 1.3rem;", language_style)
        self.assertIn("color: var(--vh-ink);", language_style)
        self.assertIn(".vh-language-select:focus-visible", stylesheet)
        self.assertIn('[data-vh-header-control="language"],', stylesheet)

    def test_theme_and_source_controls_match_transformers_compact_contract(self):
        palette = PALETTE_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn('data-vh-theme-toggle', palette)
        self.assertIn('aria-label="{{ option.toggle.name }}"', palette)
        self.assertIn('type="button"', palette)
        self.assertIn('data-vh-theme-target', palette)
        self.assertNotIn('role="button"', palette)

        source = SOURCE_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn('class="md-source vh-source-link"', source)
        self.assertIn('aria-label="Open VoiceHub source repository"', source)
        self.assertIn('data-md-component="source"', source)

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        theme_style = stylesheet.split(".md-header__button.vh-theme-toggle {", 1)[1].split("}", 1)[0]
        self.assertIn("width: 1.7rem;", theme_style)
        self.assertIn("height: 1.2rem;", theme_style)
        self.assertIn("margin: 0;", theme_style)
        self.assertIn(".md-header__button.vh-theme-toggle[hidden]", stylesheet)
        source_style = stylesheet.split(".vh-source-link {", 1)[1].split("}", 1)[0]
        self.assertIn("width: 2.75rem;", source_style)
        self.assertIn("height: 0.8rem;", source_style)
        self.assertIn(".vh-source-link:is(:focus-visible, .focus-visible)", stylesheet)
        self.assertIn('[data-vh-header-control="theme"],', stylesheet)
        self.assertIn('[data-vh-header-control="source"] {', stylesheet)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("initializeThemeControl", script)
        self.assertIn("initializeSourceControl", script)
        self.assertIn('querySelector(\'[data-vh-header-control="theme"]\')', script)
        self.assertIn('querySelector(\'[data-vh-header-control="source"] a[href]\')', script)
        self.assertIn("toggle instanceof HTMLButtonElement", script)
        self.assertIn("toggle.dataset.vhThemeTarget", script)
        self.assertIn("target.checked = true", script)
        self.assertIn('addEventListener("change", focusVisibleToggle)', script)
        self.assertIn('querySelector("[data-vh-theme-toggle]:not([hidden])")', script)
        self.assertIn("visibleToggle.focus()", script)
        self.assertIn("window.location.assign(link.href)", script)

    def test_main_workflow_guides_stay_concise(self):
        for path in CONCISE_GUIDE_PATHS:
            with self.subTest(path=path):
                line_count = len(path.read_text(encoding="utf-8").splitlines())
                self.assertLessEqual(
                    line_count,
                    250,
                    f"{path.name} should remain a concise user workflow.",
                )

    def test_qwen3_decoding_example_uses_supported_options(self):
        source = (DOCS_ROOT / "guides" / "speech-recognition.md").read_text(encoding="utf-8")
        section = source.split(
            "## Decoding configuration",
            1,
        )[1].split("## Output", 1)[0]

        self.assertIn('hotwords=("VoiceHub",)', section)
        self.assertIn("batch_size=1", section)
        self.assertNotIn("return_timestamps=", section)
        self.assertNotIn("chunk_length_s=", section)
        self.assertNotIn("stride_length_s=", section)
        self.assertNotIn("batch_size=4", section)

    def test_rtx_4090_report_tracks_asr_vad_manifest(self):
        result_path = (REPOSITORY_ROOT / "benchmarks" / "asr_vad_rtx4090_2026-07-31.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        report = (DOCS_ROOT / "guides" / "rtx-4090-speech-benchmarks.md").read_text(encoding="utf-8")

        self.assertIn(result_path.relative_to(REPOSITORY_ROOT).as_posix(), report)
        moonshine = next(
            measurement for measurement in result["asr_measurements"]
            if measurement["model_type"] == "asr_moonshine")
        fp32 = moonshine["profiles"][0]
        self.assertIn(
            (f"{fp32['mean_seconds'] * 1000:.2f} / "
             f"{fp32['median_seconds'] * 1000:.2f} ms"),
            report,
        )
        whisper = next(
            measurement for measurement in result["asr_measurements"]
            if measurement["model_type"] == "asr_whisper")
        for profile in whisper["profiles"]:
            self.assertIn(
                (f"{profile['mean_seconds'] * 1000:.2f} / "
                 f"{profile['median_seconds'] * 1000:.2f} ms"),
                report,
            )
        sherpa = next(
            measurement for measurement in result["vad_measurements"]
            if measurement["model_type"] == "vad_sherpa_onnx")
        baseline = sherpa["profiles"][0]
        self.assertIn(
            (f"{baseline['mean_seconds'] * 1000:,.2f} / "
             f"{baseline['median_seconds'] * 1000:,.2f} ms"),
            report,
        )

    def test_every_asr_training_profile_is_in_model_and_training_docs(self):
        from voicehub import SpeechTask, list_training_specs

        model_types = {
            spec.model_type
            for spec in list_training_specs(task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION)
        }
        self.assertEqual(len(model_types), 23)
        pages = (
            DOCS_ROOT / "guides" / "speech-recognition.md",
            DOCS_ROOT / "models" / "asr-vad-support.md",
            DOCS_ROOT / "models" / "training-support.md",
        )
        for page in pages:
            with self.subTest(page=page):
                source = page.read_text(encoding="utf-8")
                documented = []
                for line in source.splitlines():
                    if not line.startswith("|"):
                        continue
                    first_cell = line.split("|", 2)[1]
                    match = re.search(r"`(asr_[a-z0-9_]+)`", first_cell)
                    if match:
                        documented.append(match.group(1))
                self.assertEqual(set(documented), model_types)
                self.assertEqual(len(documented), len(model_types))

    def test_multilingual_homepages_and_configuration_are_complete(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        theme_override = THEME_OVERRIDE_PATH.read_text(encoding="utf-8")
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        self.assertIn("mkdocs-static-i18n==1.3.1", PYPROJECT_PATH.read_text(encoding="utf-8"))
        self.assertIn("docs_structure: suffix", config)
        self.assertIn("fallback_to_default: true", config)
        self.assertIn("reconfigure_material: true", config)
        self.assertIn("reconfigure_search: true", config)
        self.assertIn("pymdownx.slugs.slugify", config)
        self.assertNotIn("navigation.instant", config)
        self.assertIn("i18n_page_locale != i18n_file_locale", theme_override)
        self.assertIn('class="vh-translation-fallback"', theme_override)
        self.assertIn('lang="{{ i18n_file_locale }}" dir="ltr"', theme_override)
        self.assertIn('[dir="rtl"] .vh-doc-teaser', stylesheet)
        self.assertIn(".md-tabs__item--active > .md-tabs__link", stylesheet)
        self.assertNotIn(".md-tabs__link--active", stylesheet)
        self.assertEqual(stylesheet.count(".md-typeset .vh-process:not([hidden]) {"), 2)
        self.assertIn("grid-template-columns: repeat(2, minmax(0, 1fr))", stylesheet)
        self.assertNotIn(".vh-flow-diagram", stylesheet)
        self.assertNotIn("name: mermaid", config)

        for locale in LOCALIZED_HOME_LOCALES:
            with self.subTest(locale=locale):
                self.assertIn(f"- locale: {locale}", config)
                localized_home = DOCS_ROOT / f"index.{locale}.md"
                self.assertTrue(localized_home.is_file())
                localized_source = localized_home.read_text(encoding="utf-8")
                self.assertIn('<div class="vh-doc-home" markdown>', localized_source)
                self.assertIn('<div class="grid cards" markdown>', localized_source)

    def test_homepages_keep_the_transformers_shell_visible(self):
        parity_inventory = (DOCS_ROOT / "project" / "transformers-parity.md").read_text(encoding="utf-8")
        self.assertIn("b3a36037d3feb22e3f0174b3dd4248fcc0f0f722", parity_inventory)
        self.assertIn("/docs/transformers/main/en/index", parity_inventory)

        homepages = (DOCS_ROOT / "index.md", ) + tuple(
            DOCS_ROOT / f"index.{locale}.md" for locale in LOCALIZED_HOME_LOCALES)
        for homepage in homepages:
            with self.subTest(homepage=homepage):
                source = homepage.read_text(encoding="utf-8")
                frontmatter = source.split("---\n", 2)[1]
                self.assertNotIn("hide:", frontmatter)
                self.assertNotIn("navigation", frontmatter)
                self.assertNotIn("toc", frontmatter)
                expected_mark_path = (
                    "assets/voicehub-mark.svg" if homepage == DOCS_ROOT /
                    "index.md" else "../assets/voicehub-mark.svg")
                self.assertEqual(source.count(f'src="{expected_mark_path}"'), 2)

    def test_desktop_and_tablet_shell_collapse_inactive_navigation_branches(self):
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        desktop_shell = stylesheet.split(
            "@media screen and (min-width: 60em) {",
            1,
        )[1].split(
            "@media screen and (min-width: 60em) and (max-width: 76.234375em)",
            1,
        )[0]
        tablet_shell = stylesheet.split(
            "@media screen and (min-width: 60em) and (max-width: 76.234375em)",
            1,
        )[1].split("@media screen and (max-width: 44.984375em)", 1)[0]

        self.assertIn('.md-header__button[data-vh-drawer-trigger]', desktop_shell)
        self.assertIn('[dir="ltr"] .md-sidebar--primary', desktop_shell)
        self.assertIn("position: sticky", desktop_shell)
        self.assertIn("width: var(--vh-doc-rail-width)", desktop_shell)
        self.assertIn(".md-sidebar--primary .md-sidebar__scrollwrap", desktop_shell)
        self.assertIn("overflow-y: auto", desktop_shell)
        navigation_button = desktop_shell.split(
            ".md-sidebar--primary button.md-nav__link[data-vh-nav-toggle] {",
            1,
        )[1].split("}", 1)[0]
        self.assertIn("pointer-events: auto;", navigation_button)
        self.assertIn("cursor: pointer;", stylesheet)
        collapsed_navigation = desktop_shell.split(
            ".md-nav__toggle ~ .md-nav {",
            1,
        )[1].split("}", 1)[0]
        self.assertIn("display: none;", collapsed_navigation)
        self.assertNotIn("opacity", collapsed_navigation)
        self.assertNotIn("visibility", collapsed_navigation)
        expanded_navigation = desktop_shell.split(
            ".md-nav__toggle.md-toggle--indeterminate ~ .md-nav,",
            1,
        )[1].split("}", 1)[0]
        self.assertIn(".md-nav__toggle:checked ~ .md-nav {", expanded_navigation)
        self.assertIn("display: block;", expanded_navigation)
        self.assertIn(".md-nav--primary > .md-nav__title", tablet_shell)
        self.assertIn(".md-sidebar--secondary:not([hidden])", tablet_shell)
        self.assertIn("display: none", tablet_shell)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        primary_navigation = script.split(
            "const initializePrimaryNavigationControl = () => {",
            1,
        )[1].split("const initializeTableOfContentsTracking = () => {", 1)[0]
        for fragment in (
                'querySelector(".md-sidebar--primary")',
                'querySelectorAll("label.md-nav__link[for]")',
                'document.createElement("button")',
                'button.dataset.vhNavToggle = toggle.id',
                'button.setAttribute("aria-controls", panel.id)',
                'button.setAttribute("aria-expanded", String(toggle.checked))',
                'toggle.addEventListener("change", synchronizeExpandedState)',
                'button.addEventListener("click"',
                "event.preventDefault()",
                "event.stopImmediatePropagation()",
                "toggle.checked = !toggle.checked",
                'toggle.dispatchEvent(new Event("change", { bubbles: true }))',
        ):
            self.assertIn(fragment, primary_navigation)
        self.assertNotIn("label.click()", primary_navigation)
        self.assertIn("initializePrimaryNavigationControl();", script)

        for fragment in (
                "> :is(label, button).md-nav__link--active {",
                "+ a.md-nav__link--active {",
                "background: var(--vh-active-bg);",
                "color: var(--vh-active-text);",
                "display: none;",
        ):
            self.assertIn(fragment, stylesheet)

    def test_rendered_representative_navigation_has_a_ci_contract(self):
        self.assertTrue(DOCUMENTATION_DOM_CHECK_PATH.is_file())
        checker = DOCUMENTATION_DOM_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "REPRESENTATIVE_ROUTES",
                "TOP_LEVEL_NAVIGATION",
                '"getting-started/installation/index.html"',
                '"getting-started/quickstart/index.html"',
                '"guides/inference/index.html"',
                '"models/providers/index.html"',
                '"models/providers/speecht5/index.html"',
                '"guides/trainer/index.html"',
                '"guides/optimization-overview/index.html"',
                '"project/adding-a-model/index.html"',
                '"reference/models/index.html"',
                '"md-sidebar--primary"',
                '"md-nav__link--active"',
                '"aria-expanded"',
                '"checked"',
        ):
            self.assertIn(fragment, checker)

        command = "python scripts/check_documentation_dom.py site"
        for workflow_name in ("docs.yml", "release.yml"):
            workflow = (REPOSITORY_ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
            self.assertIn(command, workflow)
            self.assertLess(
                workflow.index("mkdocs build --strict --clean --site-dir site"),
                workflow.index(command),
            )

    def test_representative_routes_have_a_responsive_visual_ci_contract(self):
        self.assertTrue(DOCUMENTATION_VISUAL_CHECK_PATH.is_file())
        self.assertTrue(DOCUMENTATION_VISUAL_SHARD_CHECK_PATH.is_file())
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "REPRESENTATIVE_ROUTES",
                "VIEWPORTS",
                "PALETTES",
                "sync_playwright",
                '"width": 1440',
                '"width": 1024',
                '"width": 390',
                '"article_x": 318',
                '"article_width": 804',
                '"article_width": 658',
                '"article_x": 24',
                '"article_width": 342',
                '"header_height": 65',
                '"header_height": 64',
                '"rgb(255, 255, 255)"',
                '"rgb(17, 24, 39)"',
                '"rgb(11, 15, 25)"',
                '"rgb(243, 244, 246)"',
                '".md-sidebar--primary"',
                '".md-sidebar--secondary"',
                '"a.md-nav__link--active"',
                '"visibleActiveLabels"',
                '"overflow"',
                '"--viewport"',
                '"--palette"',
                "selected_viewports",
                "selected_palette_names",
        ):
            self.assertIn(fragment, checker)

        shard_checker = DOCUMENTATION_VISUAL_SHARD_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "ThreadPoolExecutor(max_workers=len(selected_viewports))",
                '"--viewport"',
                "EXPECTED_TOTALS",
                "VIEWPORT_SPECIFIC_EXPECTATIONS",
                "MINIMUM_FOCUS_STEPS_BY_VIEWPORT",
                "_validate_viewport_summary",
                "_validate_viewport_palette_summary",
                '"cases": 60',
                '"keyboard_cases": 330',
                '"screenshot_cases": 60',
                'totals.get("focus_steps", 0) < 4200',
                "if result.returncode:",
        ):
            self.assertIn(fragment, shard_checker)

        self.assertIn(
            '"playwright==1.62.0"',
            PYPROJECT_PATH.read_text(encoding="utf-8"),
        )
        command = "python scripts/check_documentation_visual_shards.py site"
        install_command = "python -m playwright install --with-deps chromium"
        for workflow_name in ("docs.yml", "release.yml"):
            workflow = (REPOSITORY_ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
            self.assertIn(install_command, workflow)
            self.assertIn(command, workflow)
            self.assertLess(
                workflow.index("mkdocs build --strict --clean --site-dir site"),
                workflow.index(command),
            )

    def test_representative_routes_have_a_screenshot_pixel_regression_contract(self):
        manifests = []
        for path in (
                DOCUMENTATION_SCREENSHOT_BASELINES_PATH,
                DOCUMENTATION_LINUX_SCREENSHOT_BASELINES_PATH,
        ):
            self.assertTrue(path.is_file())
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(len(manifest["cases"]), 60)
            manifests.append(manifest)

        darwin_manifest, linux_manifest = manifests
        self.assertEqual(darwin_manifest["playwright"], linux_manifest["playwright"])
        self.assertEqual(darwin_manifest["chromium"], linux_manifest["chromium"])
        self.assertEqual(darwin_manifest["cases"].keys(), linux_manifest["cases"].keys())
        for key, darwin_case in darwin_manifest["cases"].items():
            linux_case = linux_manifest["cases"][key]
            for field in ("width", "height", "hash_bits"):
                self.assertEqual(darwin_case[field], linux_case[field])

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "SCREENSHOT_BASELINES_PATHS",
                '"darwin"',
                '"linux"',
                "sys.platform",
                "SCREENSHOT_SIGNATURE_WIDTH",
                "SCREENSHOT_MAX_HAMMING_RATIO",
                "Image.open(BytesIO(screenshot))",
                "ImageFilter.GaussianBlur",
                "page.screenshot(",
                "_screenshot_signature",
                "_compare_screenshot_signature",
                '"screenshot_cases"',
                '"--update-screenshot-baselines"',
        ):
            self.assertIn(fragment, checker)

        hamming_ratio_match = re.search(
            r"^SCREENSHOT_MAX_HAMMING_RATIO = (?P<ratio>[0-9.]+)$",
            checker,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(hamming_ratio_match)
        self.assertLess(float(hamming_ratio_match.group("ratio")), 0.10)

        pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")
        self.assertIn('"pillow==12.3.0"', pyproject)
        docs_workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
        self.assertIn('"pillow==12.3.0"', docs_workflow)
        self.assertEqual(docs_workflow.count("--update-screenshot-baselines"), 1)
        self.assertIn(
            "> documentation-screenshot-signatures-linux.json",
            docs_workflow,
        )
        self.assertIn("name: documentation-screenshot-signatures-linux", docs_workflow)
        self.assertIn("name: documentation-site", docs_workflow)
        self.assertIn("viewport: [desktop, tablet, mobile]", docs_workflow)
        self.assertIn("palette: [default, slate]", docs_workflow)
        self.assertIn('--viewport "${{ matrix.viewport }}"', docs_workflow)
        self.assertIn('--palette "${{ matrix.palette }}"', docs_workflow)
        self.assertIn("needs: [build, screenshots, visual]", docs_workflow)
        self.assertEqual(
            docs_workflow.count("github.event.pull_request.head.repo.full_name == github.repository"),
            2,
        )
        self.assertNotIn(
            "if: github.event_name != 'pull_request' && github.ref == 'refs/heads/main'",
            docs_workflow,
        )

        release_workflow = (REPOSITORY_ROOT / ".github" / "workflows" /
                            "release.yml").read_text(encoding="utf-8")
        self.assertNotIn("--update-screenshot-baselines", release_workflow)

    def test_visual_viewport_shard_aggregation_fails_closed(self):
        namespace = runpy.run_path(
            str(DOCUMENTATION_VISUAL_SHARD_CHECK_PATH),
            run_name="voicehub_visual_shards_test",
        )
        viewport_names = namespace["VIEWPORT_NAMES"]
        expected_totals = namespace["EXPECTED_TOTALS"]
        summaries = {}
        for index, viewport in enumerate(viewport_names):
            summary = {
                "axe_core": "axe-core test",
                "palettes": 2,
                "representative_routes": 10,
                "focus_steps": 4500 if index == 0 else 0,
            }
            for field, total in expected_totals.items():
                summary[field] = 1 if field == "viewports" else (total if index == 0 else 0)
            summaries[viewport] = summary

        aggregate = namespace["_aggregate_summaries"](summaries)
        self.assertEqual(aggregate["totals"]["cases"], 60)
        self.assertEqual(aggregate["totals"]["keyboard_cases"], 330)
        self.assertEqual(aggregate["totals"]["viewports"], 3)

        incomplete = {viewport: dict(summary) for viewport, summary in summaries.items()}
        incomplete["mobile"]["cases"] = 1
        with self.assertRaisesRegex(
                namespace["DocumentationVisualShardError"],
                "Aggregated visual contract coverage differs",
        ):
            namespace["_aggregate_summaries"](incomplete)

        missing = dict(summaries)
        del missing["tablet"]
        with self.assertRaisesRegex(
                namespace["DocumentationVisualShardError"],
                "Viewport shard inventory differs",
        ):
            namespace["_aggregate_summaries"](missing)

        expected_viewport_summary = namespace["_expected_viewport_summary"]
        expected_viewport_palette_summary = namespace["_expected_viewport_palette_summary"]
        minimum_focus_steps = namespace["MINIMUM_FOCUS_STEPS_BY_VIEWPORT"]
        minimum_palette_focus_steps = namespace["MINIMUM_FOCUS_STEPS_BY_VIEWPORT_PALETTE"]
        validate_viewport_summary = namespace["_validate_viewport_summary"]
        validate_viewport_palette_summary = namespace["_validate_viewport_palette_summary"]
        for viewport in viewport_names:
            with self.subTest(viewport=viewport):
                self.assertEqual(
                    sum(minimum_palette_focus_steps[viewport].values()),
                    minimum_focus_steps[viewport],
                )
                summary = {
                    "axe_core": "axe-core test",
                    "palettes": 2,
                    "representative_routes": 10,
                    "focus_steps": minimum_focus_steps[viewport],
                    **expected_viewport_summary(viewport),
                }
                self.assertEqual(validate_viewport_summary(viewport, summary), summary)

                incomplete_viewport = dict(summary)
                incomplete_viewport["cases"] -= 1
                with self.assertRaisesRegex(
                        namespace["DocumentationVisualShardError"],
                        f"{viewport} visual contract coverage differs",
                ):
                    validate_viewport_summary(viewport, incomplete_viewport)

                for palette in namespace["PALETTE_NAMES"]:
                    palette_summary = {
                        "axe_core": "axe-core test",
                        "palettes": 1,
                        "representative_routes": 10,
                        "focus_steps": minimum_palette_focus_steps[viewport][palette],
                        **expected_viewport_palette_summary(viewport, palette),
                    }
                    self.assertEqual(
                        validate_viewport_palette_summary(viewport, palette, palette_summary),
                        palette_summary,
                    )

                    incomplete_palette = dict(palette_summary)
                    incomplete_palette["screenshot_cases"] -= 1
                    with self.assertRaisesRegex(
                            namespace["DocumentationVisualShardError"],
                            f"{viewport}/{palette} visual contract coverage differs",
                    ):
                        validate_viewport_palette_summary(viewport, palette, incomplete_palette)

    def test_representative_routes_have_a_rendered_axe_accessibility_contract(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "from axe_playwright_python.sync_playwright import Axe",
                "def _validate_accessibility(",
                "axe.run(page)",
                '"accessibility_cases"',
                '"axe_core"',
        ):
            self.assertIn(fragment, checker)

        pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")
        docs_workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
        for source in (pyproject, docs_workflow):
            self.assertIn('"axe-playwright-python==0.1.8"', source)

        header = HEADER_OVERRIDE_PATH.read_text(encoding="utf-8")
        palette = PALETTE_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn('<button\n      type="button"\n      class="md-header__button md-icon"', header)
        self.assertIn("data-vh-search-trigger", header)
        self.assertIn("data-vh-theme-target", palette)
        self.assertNotIn('role="button"', header)
        self.assertNotIn('role="button"', palette)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('document.createElement("button")', script)
        self.assertIn('output.setAttribute("aria-hidden", String(!expanded))', script)
        self.assertIn("output.inert = !expanded", script)
        self.assertIn("scrollwrap.tabIndex = expanded ? 0 : -1", script)
        self.assertIn('nav.setAttribute("aria-label", `Code block ${index + 1} actions`)', script)
        self.assertIn("const initializeScrollableRegions", script)
        self.assertIn("region.tabIndex = 0", script)
        self.assertIn(
            'region.setAttribute("aria-label", `Scrollable table ${index + 1}`)',
            script,
        )
        self.assertIn('querySelectorAll(".md-typeset pre > code")', script)
        self.assertIn('region.dataset.vhScrollableCode = "true"', script)
        self.assertIn(
            'region.setAttribute("aria-label", `Scrollable code block ${index + 1}`)',
            script,
        )
        self.assertIn('querySelectorAll(".md-typeset .tabbed-labels")', script)
        self.assertIn('region.dataset.vhScrollableTabs = "true"', script)
        self.assertIn("const keepLabelInViewport = (label) =>", script)
        self.assertIn("const viewportMargin = 4", script)
        self.assertIn("const labelsBounds = labels.getBoundingClientRect();", script)
        self.assertIn("const scrollElementBy = (element, { left = 0, top = 0 }) =>", script)
        self.assertIn('element.style.scrollBehavior = "auto";', script)
        self.assertIn("element.style.scrollBehavior = previousScrollBehavior;", script)
        self.assertIn("scrollElementBy(labels, { left:", script)
        self.assertIn("document.scrollingElement", script)
        self.assertIn("let focusRequestId = 0;", script)
        self.assertIn("const focusSettleFrames = 6;", script)
        self.assertIn("const revealLabel = (input, label, requestId) =>", script)
        self.assertIn("const isCurrentRequest = () =>", script)
        self.assertIn(
            "input.dataset.vhFocusRequest === String(requestId)",
            script,
        )
        self.assertIn("if (!isCurrentRequest()) return;", script)
        self.assertIn("requestAnimationFrame(() => settleLabel(frame + 1));", script)
        self.assertIn("input.dataset.vhFocusSettled = String(requestId);", script)
        self.assertIn("const requestId = ++focusRequestId;", script)
        self.assertIn("input.dataset.vhFocusRequest = String(requestId);", script)
        self.assertIn("delete input.dataset.vhFocusSettled;", script)
        self.assertIn("revealLabel(input, label, requestId);", script)
        self.assertIn('tabSet.addEventListener("keydown", (event) => {', script)
        self.assertIn("event.preventDefault();", script)
        self.assertIn("nextInput.focus({ preventScroll: true });", script)
        self.assertLess(
            script.index('nextInput.dispatchEvent(new Event("change", { bubbles: true }));'),
            script.index("nextInput.focus({ preventScroll: true });"),
        )
        self.assertIn('nextInput.dispatchEvent(new Event("input", { bubbles: true }))', script)
        self.assertIn('nextInput.dispatchEvent(new Event("change", { bubbles: true }))', script)
        self.assertIn(
            'region.setAttribute("aria-label", `Scrollable options ${index + 1}`)',
            script,
        )

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        for fragment in (
                ".md-typeset p a,",
                "--md-code-hl-operator-color: #7e22ce;",
                "--md-code-hl-variable-color: #665c63;",
                "--md-code-hl-constant-color: #fbbf24;",
                "--md-code-hl-keyword-color: #e9d5ff;",
                "--md-code-hl-number-color: #fb7185;",
                ".md-copyright,",
                ".md-typeset__table:focus-visible {",
                ".md-typeset pre > code:focus-visible,",
                ".md-typeset .tabbed-labels:focus-visible {",
                ".md-top {",
        ):
            self.assertIn(fragment, stylesheet)

    def test_shared_shell_expanded_states_have_a_rendered_axe_contract(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "INTERACTIVE_ACCESSIBILITY_STATES",
                "def _prepare_interactive_accessibility_state(",
                '"search-open"',
                '"search-results"',
                '"search-empty"',
                '"version-open"',
                '"branch-open"',
                '"drawer-open"',
                "Number.parseFloat(getComputedStyle(searchInner).opacity) >= 0.999",
                '"interactive_accessibility_cases"',
        ):
            self.assertIn(fragment, checker)

        header = HEADER_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertNotIn('aria-haspopup="menu"', header)
        self.assertNotIn('role="menu"', header)
        self.assertNotIn('role="menuitem"', header)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        for fragment in (
                'querySelectorAll(".md-search-result article pre > code")',
                'region.setAttribute("aria-label", `Search result code ${index + 1}`)',
                "new MutationObserver(normalizeScrollableRegions)",
                'panel.setAttribute("aria-label"',
                'panel.removeAttribute("aria-labelledby")',
        ):
            self.assertIn(fragment, script)

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        self.assertIn(".md-search-result__meta {", stylesheet)
        self.assertIn("  opacity: 1;", stylesheet)
        self.assertIn(".md-search-result article pre > code:focus-visible", stylesheet)
        self.assertIn(
            ".md-sidebar--primary .md-nav:not(.md-nav--primary) > .md-nav__title",
            stylesheet,
        )

    def test_native_keyboard_navigation_has_a_rendered_ci_contract(self):
        header = HEADER_OVERRIDE_PATH.read_text(encoding="utf-8")
        for fragment in (
                "data-vh-drawer-trigger",
                '<button\n      type="button"',
                'aria-expanded="false"',
        ):
            self.assertIn(fragment, header)

        search = SEARCH_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn(
            '<div class="md-search__scrollwrap" tabindex="-1" data-md-scrollfix>',
            search,
        )

        palette = PALETTE_OVERRIDE_PATH.read_text(encoding="utf-8")
        self.assertIn('tabindex="-1"', palette)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        primary_navigation = script.split(
            "const initializePrimaryNavigationControl = () => {",
            1,
        )[1].split("const initializeTableOfContentsTracking = () => {", 1)[0]
        for fragment in (
                'document.createElement("button")',
                'button.addEventListener("click"',
                "event.stopImmediatePropagation()",
                "toggle.checked = !toggle.checked",
                'toggle.dispatchEvent(new Event("change", { bubbles: true }))',
                'window.matchMedia("(max-width: 59.984375em)")',
                "input.tabIndex = expanded || !mobileViewport.matches ? 0 : -1",
                'mobileViewport.addEventListener("change", synchronizeExpandedState)',
        ):
            self.assertIn(fragment, script)
        self.assertNotIn("label.click()", primary_navigation)

        drawer_script = MOBILE_DRAWER_SCRIPT_PATH.read_text(encoding="utf-8")
        for fragment in (
                'querySelector("[data-vh-drawer-trigger]")',
                'querySelector(".md-sidebar--primary")',
                'window.matchMedia("(max-width: 59.984375em)")',
                "navigation.inert = mobileViewport.matches && !drawer.checked",
                'trigger.setAttribute("aria-expanded", String(expanded))',
                'trigger.addEventListener("click"',
                "event.stopImmediatePropagation()",
                "firstFocusTarget.focus()",
                'event.key !== "Escape"',
                "trigger.focus()",
        ):
            self.assertIn(fragment, drawer_script)

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        for selector in (
                ".md-sidebar--primary .md-nav__button:focus-visible,",
                ".md-sidebar--primary .md-nav__button.focus-visible,",
                ".md-sidebar--primary button.md-nav__link[data-vh-nav-toggle]:focus-visible,",
                ".md-sidebar--primary button.md-nav__link[data-vh-nav-toggle].focus-visible,",
        ):
            self.assertIn(selector, stylesheet)

        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "DESKTOP_KEYBOARD_FOCUS_PREFIX",
                "TABLET_KEYBOARD_FOCUS_PREFIX",
                "KEYBOARD_FOCUS_PREFIX",
                "MOBILE_KEYBOARD_FOCUS_PREFIX",
                "DRAWER_ACTIVATION_CASES",
                "_focus_prefix_for_viewport",
                "_validate_focus_cycle",
                "_validate_root_branch_activation",
                "_validate_mobile_drawer_activation",
                'page.keyboard.press("Tab")',
                '"branch:Models"',
                '"header:drawer"',
                '"keyboard_cases"',
                '"focus_steps"',
                '("Enter", "default")',
                '("Space", "slate")',
        ):
            self.assertIn(fragment, checker)

        command = "python scripts/check_documentation_visual_shards.py site"
        for workflow_name in ("docs.yml", "release.yml"):
            workflow = (REPOSITORY_ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
            self.assertIn("Validate responsive documentation and keyboard behavior", workflow)
            self.assertIn(command, workflow)

    def test_all_representative_routes_have_complete_native_focus_cycles(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "DESKTOP_KEYBOARD_FOCUS_PREFIX",
                "TABLET_KEYBOARD_FOCUS_PREFIX",
                "MOBILE_KEYBOARD_FOCUS_PREFIX",
                "_focus_prefix_for_viewport",
                "focusable_count = page.locator(",
                "focus_cycle_cases = 0",
                "focus_cycle_cases += 1",
                '"focus_cycle_cases": focus_cycle_cases',
                '"keyboard_activation_cases": keyboard_activation_cases',
        ):
            self.assertIn(fragment, checker)

        matrix = checker.split("for palette in selected_palette_names:", 1)[1].split(
            'page.set_viewport_size({"width": 1440, "height": 900})',
            1,
        )[0]
        self.assertIn("focus_steps += _validate_focus_cycle(", matrix)
        self.assertIn("_focus_prefix_for_viewport(viewport)", matrix)
        self.assertNotIn("_validate_keyboard_focus_cycle", checker)
        self.assertNotIn("_validate_mobile_keyboard_focus_cycle", checker)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("initializeContentTabFocus", script)
        self.assertIn('label.classList.add("vh-content-tab--focus")', script)
        self.assertIn('label.classList.remove("vh-content-tab--focus")', script)
        self.assertIn("initializeSequentialFocusBoundary", script)
        self.assertIn("document.activeElement !== document.body", script)
        self.assertIn("skipLink.focus()", script)

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        content_tab_focus = stylesheet.split(
            ".md-typeset .tabbed-labels > label.vh-content-tab--focus {",
            1,
        )[1].split("}", 1)[0]
        self.assertIn("outline: 2px solid var(--vh-accent);", content_tab_focus)
        self.assertIn("outline-offset: 2px;", content_tab_focus)

    def test_all_visible_root_navigation_branches_activate_and_restore(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                'ROOT_BRANCH_ACTIVATION_METHOD_BY_PALETTE = {',
                '"default": "keyboard"',
                '"slate": "pointer"',
                "def _validate_root_branch_activation(",
                "root_branch_activation_cases = 0",
                "root_branch_pointer_activation_cases = 0",
                "root_branch_keyboard_activation_cases = 0",
                "root_branch_interaction_accessibility_cases = 0",
                "for viewport in selected_non_mobile_viewports:",
                "for branch_label in TOP_LEVEL_NAVIGATION:",
                '"root_branch_activation_cases": root_branch_activation_cases',
                '"root_branch_pointer_activation_cases": root_branch_pointer_activation_cases',
                '"root_branch_keyboard_activation_cases": root_branch_keyboard_activation_cases',
                '"root_branch_interaction_accessibility_cases": '
                "root_branch_interaction_accessibility_cases",
        ):
            self.assertIn(fragment, checker)

        activation = checker.split(
            "def _validate_root_branch_activation(",
            1,
        )[1].split("def _validate_mobile_drawer_activation", 1)[0]
        for fragment in (
                '"nav.md-nav--primary > ul.md-nav__list > li.md-nav__item > "',
                '"button.md-nav__link[data-vh-nav-toggle]"',
                "TOP_LEVEL_NAVIGATION",
                'getAttribute("aria-controls")',
                'getAttribute("aria-expanded")',
                'getAttribute("aria-label")',
                "document.activeElement === button",
                'page.keyboard.press("Enter")',
                "button.click()",
                '"Inference"',
                "location.pathname",
                "document.body.dataset.mdColorScheme",
                'getComputedStyle(panel).display',
                '_rendered_state(page)["overflow"] != 0',
                "_validate_accessibility(axe, page",
        ):
            self.assertIn(fragment, activation)

    def test_deep_model_navigation_branches_activate_and_remain_sticky(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                "SPEECHT5_NESTED_BRANCH_STATES = (",
                '(("Models", "Text to speech"), True)',
                '(("Models", "Text to speech", "SpeechT5"), False)',
                '(("Models", "Automatic speech recognition"), False)',
                '(("Models", "Voice activity detection"), False)',
                "NESTED_BRANCH_ACTIVATION_METHOD_BY_PALETTE = {",
                "def _validate_nested_branch_activation(",
                "nested_branch_activation_cases = 0",
                "nested_branch_pointer_activation_cases = 0",
                "nested_branch_keyboard_activation_cases = 0",
                "nested_branch_interaction_accessibility_cases = 0",
                "for branch_path, expected_initial_expanded in SPEECHT5_NESTED_BRANCH_STATES:",
                '"nested_branch_activation_cases": nested_branch_activation_cases',
                '"nested_branch_pointer_activation_cases": '
                "nested_branch_pointer_activation_cases",
                '"nested_branch_keyboard_activation_cases": '
                "nested_branch_keyboard_activation_cases",
                '"nested_branch_interaction_accessibility_cases": '
                "nested_branch_interaction_accessibility_cases",
        ):
            self.assertIn(fragment, checker)

        activation = checker.split(
            "def _validate_nested_branch_activation(",
            1,
        )[1].split("def _validate_mobile_drawer_activation", 1)[0]
        for fragment in (
                "SPEECHT5_ROUTE",
                "branch_path",
                'getAttribute("aria-controls")',
                'getAttribute("aria-expanded")',
                "document.activeElement === button",
                'page.keyboard.press("Enter")',
                "button.click()",
                '"SpeechT5"',
                "location.pathname",
                "document.body.dataset.mdColorScheme",
                'getComputedStyle(panel).display',
                'getComputedStyle(navigation).position',
                "window.scrollTo(0, 320)",
                '"--vh-shell-scroll-offset"',
                '_rendered_state(page)["overflow"] != 0',
                "_validate_accessibility(axe, page",
        ):
            self.assertIn(fragment, activation)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        for fragment in (
                'panel.querySelectorAll("nav.md-nav")',
                "nestedPanel.getAttribute(\"aria-label\")",
                "`${sectionName} subsection ${nestedIndex + 1}: ${nestedName}`",
        ):
            self.assertIn(fragment, script)

    def test_left_navigation_marks_active_and_keyboard_focus_states(self):
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        active_state = stylesheet.split(".md-nav__link--active {", 1)[1].split("}", 1)[0]
        focus_selector = (".md-nav__link[href]:focus-visible,\n"
                          ".md-nav__link[href].focus-visible {")
        focus_state = stylesheet.split(focus_selector, 1)[1].split("}", 1)[0]

        self.assertIn("background:", active_state)
        self.assertIn("box-shadow:", active_state)
        self.assertIn("border-radius:", active_state)
        self.assertIn(".md-nav__link[href].focus-visible", stylesheet)
        self.assertIn("outline: 2px solid var(--vh-accent)", focus_state)
        self.assertIn("outline-offset: 2px", focus_state)

    def test_right_table_of_contents_tracks_the_active_heading(self):
        config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("- navigation.tracking", config)
        self.assertIn("- toc.follow", config)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        toc_tracking = script.split(
            "const initializeTableOfContentsTracking = () => {",
            1,
        )[1].split("const initializeVersionControl = () => {", 1)[0]
        for fragment in (
                'tableOfContents.addEventListener("click"',
                'event.target.closest(\'.md-nav__link[href^="#"]\')',
                'window.history.replaceState(window.history.state, "", link.hash)',
                'trackedLink.classList.toggle("md-nav__link--active", trackedLink === link)',
                'window.addEventListener("scroll", scheduleSettledAnchor, { passive: true })',
                'window.removeEventListener("scroll", scheduleSettledAnchor)',
                'window.setTimeout(preserveSettledAnchor, 500)',
                'tableOfContents.addEventListener("keydown"',
                'event.key !== "Enter" || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey',
                "event.preventDefault()",
                "link.click()",
                "link.focus({ preventScroll: true })",
        ):
            self.assertIn(fragment, toc_tracking)

        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        self.assertIn(".md-typeset :target {", stylesheet)
        target_state = stylesheet.split(".md-typeset :target {", 1)[1].split("}", 1)[0]
        self.assertIn(
            "--md-scroll-margin: calc(",
            target_state,
        )
        self.assertIn(
            "var(--vh-global-header-height) + var(--md-scroll-offset) - 1px",
            target_state,
        )

        selector = ".md-nav--secondary .md-nav__link--active {"
        self.assertIn(selector, stylesheet)
        active_state = stylesheet.split(selector, 1)[1].split("}", 1)[0]
        for declaration in (
                "margin-inline: 0;",
                "padding: 0;",
                "background: transparent;",
                "box-shadow: none;",
                "color: var(--vh-accent);",
                "font-weight: 700;",
        ):
            self.assertIn(declaration, active_state)

    def test_all_representative_table_of_contents_activate_by_pointer_and_keyboard(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                'TOC_ACTIVATION_METHODS = ("pointer", "keyboard")',
                "def _validate_table_of_contents_activation(",
                "toc_activation_cases = 0",
                "toc_pointer_activation_cases = 0",
                "toc_keyboard_activation_cases = 0",
                "toc_interaction_accessibility_cases = 0",
                "for relative_path in REPRESENTATIVE_ROUTES:",
                "for activation_method in TOC_ACTIVATION_METHODS:",
                '"toc_activation_cases": toc_activation_cases',
                '"toc_pointer_activation_cases": toc_pointer_activation_cases',
                '"toc_keyboard_activation_cases": toc_keyboard_activation_cases',
                '"toc_interaction_accessibility_cases": toc_interaction_accessibility_cases',
        ):
            self.assertIn(fragment, checker)

        activation = checker.split(
            "def _validate_table_of_contents_activation(",
            1,
        )[1].split("def _reset_quickstart_tabs", 1)[0]
        for fragment in (
                '".md-sidebar--secondary a.md-nav__link[href^=\'#\']"',
                'page.keyboard.press("Enter")',
                "target_link.click()",
                "window.location.hash",
                "md-nav__link--active",
                "document.activeElement === link",
                "target?.getBoundingClientRect()",
                "header?.getBoundingClientRect()",
                '_rendered_state(page)["overflow"] != 0',
        ):
            self.assertIn(fragment, activation)

    def test_all_representative_routes_activate_and_close_the_visible_search_control(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                'SEARCH_ACTIVATION_METHOD_BY_VIEWPORT = {',
                '"desktop": "keyboard"',
                '"tablet": "keyboard"',
                '"mobile": "pointer"',
                "def _validate_search_activation(",
                "search_activation_cases = 0",
                "search_pointer_activation_cases = 0",
                "search_keyboard_activation_cases = 0",
                "search_interaction_accessibility_cases = 0",
                "for relative_path in REPRESENTATIVE_ROUTES:",
                "SEARCH_ACTIVATION_METHOD_BY_VIEWPORT[viewport[\"name\"]]",
                '"search_activation_cases": search_activation_cases',
                '"search_pointer_activation_cases": search_pointer_activation_cases',
                '"search_keyboard_activation_cases": search_keyboard_activation_cases',
                '"search_interaction_accessibility_cases": search_interaction_accessibility_cases',
        ):
            self.assertIn(fragment, checker)

        activation = checker.split(
            "def _validate_search_activation(",
            1,
        )[1].split("def _validate_table_of_contents_activation", 1)[0]
        for fragment in (
                '"[data-vh-search-trigger]"',
                '".md-search__input"',
                '".md-search__output"',
                '".md-search__scrollwrap"',
                'page.keyboard.press("Control+K")',
                "trigger.click()",
                'page.keyboard.press("Escape")',
                'getAttribute("aria-expanded")',
                'getAttribute("aria-hidden")',
                ".inert",
                "document.activeElement",
                '_rendered_state(page)["overflow"] != 0',
        ):
            self.assertIn(fragment, activation)

        script = HEADER_CONTROL_SCRIPT_PATH.read_text(encoding="utf-8")
        search_control = script.split(
            "const initializeSearchControl = () => {",
            1,
        )[1].split("const initializeLanguageControl = () => {", 1)[0]
        self.assertIn("const closeFocusTarget = mobileViewport.matches ? trigger : input", search_control)
        self.assertIn("if (!restoreFocus) return", search_control)
        self.assertIn('event.key === "Tab" && document.activeElement === input', search_control)
        self.assertIn("restoreFocus = false", search_control)
        self.assertIn("closeFocusTarget.focus()", search_control)

    def test_all_representative_routes_activate_and_close_the_version_control(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                'VERSION_ACTIVATION_METHOD_BY_PALETTE = {',
                '"default": "keyboard"',
                '"slate": "pointer"',
                "def _validate_version_activation(",
                "version_activation_cases = 0",
                "version_pointer_activation_cases = 0",
                "version_keyboard_activation_cases = 0",
                "version_interaction_accessibility_cases = 0",
                "for relative_path in REPRESENTATIVE_ROUTES:",
                "VERSION_ACTIVATION_METHOD_BY_PALETTE[palette]",
                '"version_activation_cases": version_activation_cases',
                '"version_pointer_activation_cases": version_pointer_activation_cases',
                '"version_keyboard_activation_cases": version_keyboard_activation_cases',
                '"version_interaction_accessibility_cases": version_interaction_accessibility_cases',
        ):
            self.assertIn(fragment, checker)

        activation = checker.split(
            "def _validate_version_activation(",
            1,
        )[1].split("def _validate_search_activation", 1)[0]
        for fragment in (
                '"[data-vh-version-control]"',
                '"summary"',
                '".vh-header-version__menu"',
                '".md-select__link"',
                'summary.click()',
                'page.keyboard.press("Enter")',
                'page.keyboard.press("Escape")',
                'getAttribute("aria-expanded")',
                'getAttribute("aria-current")',
                "document.activeElement === summary",
                '_rendered_state(page)["overflow"] != 0',
        ):
            self.assertIn(fragment, activation)

    def test_all_visible_representative_language_controls_switch_locale(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                'LANGUAGE_ACTIVATION_METHOD_BY_PALETTE = {',
                '"default": "keyboard"',
                '"slate": "pointer"',
                'LANGUAGE_TARGET_BY_PALETTE = {',
                '"default": "tr"',
                '"slate": "ar"',
                "def _validate_language_activation(",
                "language_activation_cases = 0",
                "language_pointer_activation_cases = 0",
                "language_keyboard_activation_cases = 0",
                "language_interaction_accessibility_cases = 0",
                "for viewport in selected_non_mobile_viewports:",
                "for relative_path in REPRESENTATIVE_ROUTES:",
                'route_url = f"{base_url}{_localized_route_path(relative_path, \'en\')}"',
                '"language_activation_cases": language_activation_cases',
                '"language_pointer_activation_cases": language_pointer_activation_cases',
                '"language_keyboard_activation_cases": language_keyboard_activation_cases',
                '"language_interaction_accessibility_cases": language_interaction_accessibility_cases',
        ):
            self.assertIn(fragment, checker)

        activation = checker.split(
            "def _validate_language_activation(",
            1,
        )[1].split("def _validate_version_activation", 1)[0]
        for fragment in (
                '"[data-vh-language-select]"',
                'select.locator("option")',
                "select.click()",
                "select.select_option(label=target_locale.upper())",
                'page.keyboard.press("ArrowDown")',
                "page.expect_navigation(",
                "document.documentElement.lang",
                "document.documentElement.dir",
                ".selectedOptions",
                '_rendered_state(page)["overflow"] != 0',
        ):
            self.assertIn(fragment, activation)

    def test_all_visible_representative_theme_controls_switch_palette(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                'THEME_ACTIVATION_METHOD_BY_PALETTE = {',
                '"default": "keyboard"',
                '"slate": "pointer"',
                'THEME_TARGET_BY_PALETTE = {',
                '"default": "slate"',
                '"slate": "default"',
                "def _validate_theme_activation(",
                "theme_activation_cases = 0",
                "theme_pointer_activation_cases = 0",
                "theme_keyboard_activation_cases = 0",
                "theme_interaction_accessibility_cases = 0",
                "for viewport in selected_non_mobile_viewports:",
                "for relative_path in REPRESENTATIVE_ROUTES:",
                '"theme_activation_cases": theme_activation_cases',
                '"theme_pointer_activation_cases": theme_pointer_activation_cases',
                '"theme_keyboard_activation_cases": theme_keyboard_activation_cases',
                '"theme_interaction_accessibility_cases": theme_interaction_accessibility_cases',
        ):
            self.assertIn(fragment, checker)

        activation = checker.split(
            "def _validate_theme_activation(",
            1,
        )[1].split("def _validate_language_activation", 1)[0]
        for fragment in (
                '"[data-vh-theme-toggle]:not([hidden])"',
                'page.keyboard.press("Enter")',
                "toggle.click()",
                "document.body.dataset.mdColorScheme",
                "document.activeElement === visibleToggle",
                "location.pathname",
                '_rendered_state(page)["overflow"] != 0',
                "_validate_accessibility(axe, page",
        ):
            self.assertIn(fragment, activation)

    def test_all_visible_representative_source_controls_open_repository(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        for fragment in (
                'SOURCE_REPOSITORY_URL = "https://github.com/kadirnar/voicehub"',
                'SOURCE_ACTIVATION_METHOD_BY_PALETTE = {',
                '"default": "keyboard"',
                '"slate": "pointer"',
                "def _validate_source_activation(",
                "source_activation_cases = 0",
                "source_pointer_activation_cases = 0",
                "source_keyboard_activation_cases = 0",
                "source_interaction_accessibility_cases = 0",
                "for viewport in selected_non_mobile_viewports:",
                "for relative_path in REPRESENTATIVE_ROUTES:",
                '"source_activation_cases": source_activation_cases',
                '"source_pointer_activation_cases": source_pointer_activation_cases',
                '"source_keyboard_activation_cases": source_keyboard_activation_cases',
                '"source_interaction_accessibility_cases": source_interaction_accessibility_cases',
        ):
            self.assertIn(fragment, checker)

        activation = checker.split(
            "def _validate_source_activation(",
            1,
        )[1].split("def _validate_theme_activation", 1)[0]
        for fragment in (
                '[data-vh-header-control="source"] a[href]',
                '"Open VoiceHub source repository"',
                "page.route(",
                "SOURCE_REPOSITORY_URL,",
                "page.expect_navigation(",
                'page.keyboard.press("Enter")',
                "link.click()",
                "page.unroute(SOURCE_REPOSITORY_URL)",
                "document.activeElement === link",
                'location.pathname',
                '_rendered_state(page)["overflow"] != 0',
                "_validate_accessibility(axe, page",
        ):
            self.assertIn(fragment, activation)

    def test_all_representative_page_actions_activate_exact_targets(self):
        checker = DOCUMENTATION_VISUAL_CHECK_PATH.read_text(encoding="utf-8")
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        for fragment in (
                ':is(.md-content__button[rel="edit"], .md-footer__link, .md-top)'
                ":is(:focus-visible, .focus-visible)",
                "outline: 2px solid var(--vh-accent)",
                "outline-offset: 2px",
        ):
            self.assertIn(fragment, stylesheet)
        for fragment in (
                "REPRESENTATIVE_PAGE_ACTIONS = {",
                "PAGE_ACTION_METHOD_BY_PALETTE = {",
                '"default": "keyboard"',
                '"slate": "pointer"',
                "def _validate_page_actions(",
                "page_action_cases = 0",
                "page_action_edit_activations = 0",
                "page_action_footer_activations = 0",
                "page_action_back_to_top_activations = 0",
                "page_action_pointer_cases = 0",
                "page_action_keyboard_cases = 0",
                "page_action_interaction_accessibility_cases = 0",
                "for viewport in selected_viewports:",
                "for relative_path in REPRESENTATIVE_ROUTES:",
                '"page_action_cases": page_action_cases',
                '"page_action_edit_activations": page_action_edit_activations',
                '"page_action_footer_activations": page_action_footer_activations',
                '"page_action_back_to_top_activations": page_action_back_to_top_activations',
                '"page_action_pointer_cases": page_action_pointer_cases',
                '"page_action_keyboard_cases": page_action_keyboard_cases',
                '"page_action_interaction_accessibility_cases": '
                "page_action_interaction_accessibility_cases",
        ):
            self.assertIn(fragment, checker)

        activation = checker.split(
            "def _validate_page_actions(",
            1,
        )[1].split("def _validate_source_activation", 1)[0]
        for fragment in (
                '.md-content__button[rel="edit"]',
                ".md-footer__link--prev",
                ".md-footer__link--next",
                'button[data-md-component="top"]',
                "page.route(",
                "page.expect_navigation(",
                'page.keyboard.press("Enter")',
                ".click()",
                "window.scrollY === 0",
                "document.activeElement === action",
                "location.pathname",
                '_rendered_state(page)["overflow"] != 0',
                "_validate_accessibility(axe, page",
        ):
            self.assertIn(fragment, activation)

    def test_mobile_drawer_overlay_click_target_stays_outside_the_panel(self):
        stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
        mobile_overlay = stylesheet.split(
            "@media screen and (max-width: 59.984375em)",
            1,
        )[1].split(
            "@media screen and (min-width: 60em) and (max-width: 76.234375em)",
            1,
        )[0]

        self.assertIn('[dir="ltr"] [data-md-toggle="drawer"]:checked ~ .md-overlay', mobile_overlay)
        self.assertIn('[dir="rtl"] [data-md-toggle="drawer"]:checked ~ .md-overlay', mobile_overlay)
        self.assertIn("left: 12.1rem", mobile_overlay)
        self.assertIn("right: 12.1rem", mobile_overlay)
        self.assertEqual(mobile_overlay.count("width: calc(100% - 12.1rem)"), 2)

    def test_mobile_drawer_escape_dismissal_is_loaded(self):
        site_config = SITE_CONFIG_PATH.read_text(encoding="utf-8")
        script = MOBILE_DRAWER_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("javascripts/mobile-drawer.js", site_config)
        self.assertIn('document.addEventListener("keydown"', script)
        self.assertIn('event.key !== "Escape"', script)
        self.assertIn('document.getElementById("__drawer")', script)
        self.assertIn("!drawer.checked", script)
        self.assertIn("event.preventDefault()", script)
        self.assertIn("setDrawerState(false)", script)
        self.assertIn('drawer.dispatchEvent(new Event("change", { bubbles: true }))', script)

    def test_process_overviews_are_readable_without_horizontal_scrolling(self):
        for page_path, expected_steps in PROCESS_PAGE_STEPS:
            with self.subTest(page=page_path):
                source = page_path.read_text(encoding="utf-8")
                self.assertIn('<ol class="vh-process ', source)
                self.assertIn('role="list"', source)
                self.assertEqual(
                    source.count('class="vh-process__number"'),
                    expected_steps,
                )
                self.assertIn('class="vh-process__detail"', source)
                self.assertNotIn("vh-flow-diagram", source)
                self.assertNotIn("```mermaid", source)
                self.assertNotIn("tabindex=", source)

    def test_model_contribution_template_covers_the_definition_of_done(self):
        source = ADDING_MODEL_PATH.read_text(encoding="utf-8")

        required_paths = (
            "voicehub/models/auroratts/",
            "configuration_auroratts.py",
            "modeling_auroratts.py",
            "registration.py",
            "runtime.py",
            "SOURCE.json",
            "THIRD_PARTY_LICENSE",
            "tests/test_auroratts.py",
            "docs/models/providers/auroratts.md",
            "mkdocs.yml",
        )
        required_contracts = (
            "PreTrainedTTSModel",
            "PreTrainedASRModel",
            "PreTrainedVADModel",
            "TTSOutput",
            "ASROutput",
            "VADOutput",
            "ArchitectureSpec",
            "ModelSpec",
            "ModelTrainingSpec",
            '"builtin": true',
            "model-integration.json",
            "voicehub/models/registry.py",
            "voicehub/training/specs.py",
            "_profile(",
            "inference-only",
            "apply_optimization_plan",
            "restore_optimization_plan",
            "scripts/scaffold_model.py create",
            "scripts/scaffold_model.py catalog",
            "scripts/scaffold_model.py check",
            "scripts/generate_model_pages.py --check",
            "scripts/check_distribution.py",
            "unverified",
            "hardware-limited",
        )
        for fragment in (*required_paths, *required_contracts):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

        self.assertNotIn("Add optional metadata", source)
        self.assertIn("authoritative license text", source)
        self.assertIn("never overwrites an existing", source)
        self.assertIn("completion gate", source)
        self.assertIn("quoted name", source)
        self.assertIn("unsupported-hardware tests", source)
        self.assertIn("generated navigation entry", source)

        examples = PYTHON_BLOCK.findall(source)
        self.assertGreaterEqual(len(examples), 4)
        for index, example in enumerate(examples, start=1):
            ast.parse(
                textwrap.dedent(example),
                filename=f"adding-a-model.md:python-block-{index}",
            )

    def test_internal_markdown_links_resolve(self):
        for source_path in DOCS_ROOT.rglob("*.md"):
            source = source_path.read_text(encoding="utf-8")
            raw_targets = MARKDOWN_LINK.findall(source) + HTML_HREF.findall(source)
            for raw_target in raw_targets:
                local_path = _local_link_path(raw_target)
                if local_path is None:
                    continue
                resolved = (source_path.parent / local_path).resolve()
                candidates = (resolved, )
                if urlsplit(raw_target).path.endswith("/"):
                    candidates = (
                        resolved / "index.md",
                        resolved.with_suffix(".md"),
                    )
                with self.subTest(source=source_path, target=raw_target):
                    self.assertTrue(
                        any(candidate.exists() for candidate in candidates),
                        f"Broken documentation link {raw_target!r} in {source_path}",
                    )

    def test_public_navigation_uses_rendered_site_routes(self):
        readme = README_PATH.read_text(encoding="utf-8")
        project_metadata = PYPROJECT_PATH.read_text(encoding="utf-8")
        notebook_source = "\n".join(
            _cell_source(cell) for notebook in self.notebooks.values() for cell in notebook["cells"])
        public_content = f"{readme}\n{project_metadata}\n{notebook_source}"

        for route in PUBLIC_ROUTES:
            with self.subTest(route=route):
                self.assertIn(f"{PUBLIC_SITE_URL}{route}", public_content)

        self.assertNotIn(
            "github.com/kadirnar/voicehub/blob/main/docs/",
            public_content,
        )
        self.assertNotIn(
            "github.com/kadirnar/voicehub/tree/main/docs",
            public_content,
        )

    def test_every_notebook_is_linked_from_each_gallery(self):
        notebooks_readme = NOTEBOOKS_README_PATH.read_text(encoding="utf-8")
        docs_gallery = NOTEBOOK_GALLERY_PATH.read_text(encoding="utf-8")

        for filename in EXPECTED_NOTEBOOK_FILENAMES:
            github_url = ("https://github.com/kadirnar/voicehub/blob/main/"
                          f"notebooks/{filename}")
            colab_url = (
                "https://colab.research.google.com/github/"
                "kadirnar/voicehub/blob/main/"
                f"notebooks/{filename}")
            with self.subTest(notebook=filename):
                self.assertIn(f"]({filename})", notebooks_readme)
                self.assertIn(colab_url, notebooks_readme)
                self.assertIn(github_url, docs_gallery)
                self.assertIn(colab_url, docs_gallery)

    def test_model_pages_cover_every_registry_entry(self):
        from voicehub import AutoInferenceModel, list_model_specs

        catalog = MODEL_PAGE_INDEX_PATH.read_text(encoding="utf-8")
        tts_matrix = (DOCS_ROOT / "models" / "tts-capabilities.md").read_text(encoding="utf-8", )
        speech_matrix = (DOCS_ROOT / "models" / "asr-vad-support.md").read_text(encoding="utf-8", )
        training_matrix = (DOCS_ROOT / "models" / "training-support.md").read_text(encoding="utf-8")

        for model_spec in AutoInferenceModel.available_models():
            with self.subTest(model_type=model_spec.model_type):
                self.assertIn(
                    f"| [`{model_spec.display_name}`]({model_spec.model_type}.md) |",
                    catalog,
                )
                self.assertEqual(
                    tts_matrix.count(f"| `{model_spec.model_type}` |"),
                    1,
                )
                self.assertIn(f"(`{model_spec.model_type}`)", training_matrix)

        for model_spec in list_model_specs(task=None):
            if model_spec.task.value == "text-to-speech":
                continue
            with self.subTest(model_type=model_spec.model_type):
                self.assertIn(f"| `{model_spec.model_type}` |", speech_matrix)

    def test_homepage_registry_counts_match_the_runtime_catalog(self):
        from voicehub import list_model_specs

        specs = list_model_specs(task=None)
        counts = {
            task: sum(spec.task.value == task for spec in specs)
            for task in (
                "text-to-speech",
                "automatic-speech-recognition",
                "voice-activity-detection",
            )
        }
        homepage = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")

        self.assertIn(f"**{len(specs)} integrations**", homepage)
        self.assertIn(
            f"**{counts['text-to-speech']} TTS backends**",
            homepage,
        )
        self.assertIn(
            f"**{counts['automatic-speech-recognition']} ASR\nproviders**",
            homepage,
        )
        self.assertIn(
            f"**{counts['voice-activity-detection']} VAD providers**",
            homepage,
        )

    def test_inference_registry_uses_the_default_installation(self):
        from voicehub import list_model_specs

        metadata = PYPROJECT_PATH.read_text(encoding="utf-8")
        optional_dependencies = metadata.split(
            "[project.optional-dependencies]",
            1,
        )[1].split("[tool.setuptools]", 1)[0]
        declared_extras = set(
            re.findall(
                r"^([a-z0-9][a-z0-9-]*) = \[$",
                optional_dependencies,
                re.MULTILINE,
            ))

        self.assertEqual(declared_extras, {"docs", "test", "training"})
        for model_spec in list_model_specs(task=None):
            with self.subTest(model_type=model_spec.model_type):
                self.assertIsNone(model_spec.install_extra)

    def test_guide_python_examples_compile(self):
        example_count = 0
        for guide_path in GUIDE_PATHS:
            guide = guide_path.read_text(encoding="utf-8")
            examples = PYTHON_BLOCK.findall(guide)
            self.assertTrue(examples, f"No Python examples found in {guide_path}")
            example_count += len(examples)
            for index, source in enumerate(examples, start=1):
                ast.parse(
                    textwrap.dedent(source),
                    filename=f"{guide_path.name}:python-block-{index}",
                )

        self.assertGreaterEqual(example_count, len(GUIDE_PATHS))

    def test_readme_python_examples_compile(self):
        examples = PYTHON_BLOCK.findall(README_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(examples), 3)
        for index, source in enumerate(examples, start=1):
            ast.parse(
                textwrap.dedent(source),
                filename=f"README.md:python-block-{index}",
            )

    def test_quickstart_models_are_registered_without_runtime_imports(self):
        from voicehub import get_model_spec

        expected = {
            "parlertts": (
                "text-to-speech",
                "parler-tts/parler-tts-mini-v1",
            ),
            "asr_qwen3": (
                "automatic-speech-recognition",
                "Qwen/Qwen3-ASR-0.6B",
            ),
            "vad_silero": (
                "voice-activity-detection",
                "safestack/silero-vad",
            ),
        }
        for model_type, (task, checkpoint) in expected.items():
            with self.subTest(model_type=model_type):
                spec = get_model_spec(model_type)
                self.assertEqual(spec.task.value, task)
                self.assertEqual(spec.default_model_path, checkpoint)


if __name__ == "__main__":
    unittest.main()
