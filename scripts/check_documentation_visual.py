#!/usr/bin/env python3
"""Validate responsive geometry, screenshot pixels, accessibility, and keyboard
behavior."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

from check_documentation_dom import REPRESENTATIVE_ROUTES as DOM_REPRESENTATIVE_ROUTES
from check_documentation_dom import TOP_LEVEL_NAVIGATION

try:
    from axe_playwright_python.sync_playwright import Axe
except ImportError as error:  # pragma: no cover - exercised by the documented setup boundary
    raise SystemExit(
        "axe-playwright-python is required for rendered accessibility checks. Install the docs extra."
    ) from error

try:
    from playwright.sync_api import Page, sync_playwright
except ImportError as error:  # pragma: no cover - exercised by the documented setup boundary
    raise SystemExit(
        "Playwright is required for visual documentation checks. "
        "Install the docs extra and run `python -m playwright install chromium`.") from error

try:
    from PIL import Image, ImageFilter, ImageStat
except ImportError as error:  # pragma: no cover - exercised by the documented setup boundary
    raise SystemExit(
        "Pillow is required for screenshot regression checks. Install the docs extra.") from error

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_BASELINES_PATHS = {
    "darwin": REPOSITORY_ROOT / "tests" / "fixtures" / "documentation_screenshot_signatures.json",
    "linux": REPOSITORY_ROOT / "tests" / "fixtures" / "documentation_screenshot_signatures_linux.json",
}
SCREENSHOT_SIGNATURE_WIDTH = 64
SCREENSHOT_MAX_HAMMING_RATIO = 0.08
SCREENSHOT_MAX_MEAN_CHANNEL_DELTA = 6.0
SCREENSHOT_SCHEMA_VERSION = 1
REPRESENTATIVE_ROUTES = {
    route: expectation
    for route, expectation in DOM_REPRESENTATIVE_ROUTES.items() if route != "optimizations/compile/index.html"
}


def _platform_screenshot_baselines_path() -> Path:
    try:
        return SCREENSHOT_BASELINES_PATHS[sys.platform]
    except KeyError as error:
        raise DocumentationVisualError(
            f"No reviewed screenshot baseline is available for platform {sys.platform!r}. "
            "Pass --screenshot-baselines explicitly or capture and review one with "
            "--update-screenshot-baselines.") from error


VIEWPORTS = (
    {
        "name": "desktop",
        "width": 1440,
        "height": 900,
        "article_x": 318,
        "article_width": 804,
        "header_height": 65,
        "primary_x": 0,
        "primary_width": 270,
        "secondary_x": 1170,
        "secondary_width": 270,
    },
    {
        "name": "tablet",
        "width": 1024,
        "height": 768,
        "article_x": 318,
        "article_width": 658,
        "header_height": 65,
        "primary_x": 0,
        "primary_width": 270,
        "secondary_x": 0,
        "secondary_width": 0,
    },
    {
        "name": "mobile",
        "width": 390,
        "height": 844,
        "article_x": 24,
        "article_width": 342,
        "header_height": 64,
        "primary_x": -242,
        "primary_width": 242,
        "secondary_x": 0,
        "secondary_width": 0,
    },
)
VIEWPORTS_BY_NAME = {viewport["name"]: viewport for viewport in VIEWPORTS}

PALETTES = {
    "default": {
        "background": "rgb(255, 255, 255)",
        "text": "rgb(17, 24, 39)",
    },
    "slate": {
        "background": "rgb(11, 15, 25)",
        "text": "rgb(243, 244, 246)",
    },
}

HOME_ROUTE = "index.html"
HOME_HEADINGS = (
    ("H1", "VoiceHub"),
    ("H2", "Find a model for your language and task"),
    ("H2", "Features"),
    ("H2", "Design"),
    ("H2", "Learn"),
)
HOME_TOC = (
    "Find a model for your language and task",
    "Features",
    "Design",
    "Learn",
)
HOME_MODEL_TARGETS = (
    "/models/providers/",
    "/models/training-support/",
)
HOME_MODEL_STATS = (
    "68 Models",
    "34 TTS",
    "23 ASR",
    "11 VAD",
)
HOME_FEATURE_TARGETS = (
    "/guides/inference/",
    "/guides/trainer/",
    "/reference/api/#generation",
)
HOME_CARD_TARGETS = (
    "/getting-started/quickstart/",
    "/guides/inference/",
    "/guides/speech-recognition/",
    "/guides/voice-activity-detection/",
    "/guides/data-preparation/",
    "/guides/training/",
    "/models/",
    "/models/asr-vad-support/",
    "/models/training-support/",
    "/guides/notebook/",
    "/reference/api/",
    "/concepts/architecture/",
    "/project/adding-a-model/",
)
HOME_BADGE_TARGETS = (
    "https://github.com/kadirnar/voicehub/actions/workflows/ci.yml",
    "https://github.com/kadirnar/voicehub/actions/workflows/docs.yml",
    "https://github.com/kadirnar/voicehub/blob/main/pyproject.toml",
    "https://github.com/kadirnar/voicehub/blob/main/LICENSE",
)
INFERENCE_ROUTE = "guides/inference/index.html"
INFERENCE_HEADINGS = (
    ("H1", "Inference"),
    ("H2", "Tasks"),
    ("H3", "Text to speech"),
    ("H3", "Automatic speech recognition"),
    ("H3", "Voice activity detection"),
    ("H2", "Parameters"),
    ("H3", "Device"),
    ("H3", "Batch inference"),
    ("H3", "Task-specific parameters"),
    ("H2", "Chunking and streaming"),
    ("H2", "Large inputs"),
    ("H2", "Large models"),
    ("H2", "Save and reload"),
    ("H2", "Troubleshooting"),
)
KEYBOARD_ROUTE = INFERENCE_ROUTE
INSTALLATION_ROUTE = "getting-started/installation/index.html"
INSTALLATION_HEADINGS = (
    ("H1", "Installation"),
    ("H2", "Create an environment"),
    ("H2", "Install"),
    ("H3", "Editable checkout"),
    ("H2", "Verify"),
    ("H2", "Cache and offline mode"),
)
INSTALLATION_EXTERNAL_TARGETS = ("https://pytorch.org/get-started/locally/", )
INSTALLATION_INTERNAL_TARGETS = ()
MODEL_INDEX_ROUTE = "models/providers/index.html"
MODEL_INDEX_HEADINGS = (
    ("H1", "Model list"),
    ("H2", "Find the right speech model"),
    ("H2", "Search the registry in Python"),
)
MODEL_INDEX_TOC = ("Search the registry in Python", )
MODEL_INDEX_MINIMUM_INTERSECTING_CARDS = {
    "desktop": 3,
    "tablet": 2,
    "mobile": 1,
}
SPEECHT5_ROUTE = "models/providers/speecht5/index.html"
SPEECHT5_HEADINGS = (
    ("H1", "SpeechT5"),
    ("H2", "Model facts"),
    ("H2", "Usage"),
    ("H2", "Overview"),
    ("H3", "Language support"),
    ("H2", "Paper and GitHub"),
    ("H2", "Configuration"),
    ("H2", "Processing"),
    ("H2", "Inference"),
    ("H3", "Input and output contract"),
    ("H2", "Training and optimization"),
    ("H3", "Training contract"),
    ("H2", "Checkpoints, provenance, license, and limitations"),
    ("H3", "Limitations"),
    ("H2", "Public API"),
    ("H3", "SpeechT5Config"),
    ("H3", "SpeechT5ForTextToSpeech"),
)
SPEECHT5_TOC = ()
SPEECHT5_TABLE_ROWS = (7, 3, 4, 2, 6, 1, 10, 8)
SPEECHT5_ARTICLE_WIDTHS = {
    "desktop": 1074,
    "tablet": 658,
    "mobile": 342,
}
SPEECHT5_TABS = (
    ("usage", "#usage", "Usage", None),
    ("model-card", "#overview", "Model card", "location"),
    ("sources", "#paper-and-github", "Sources", None),
    ("training", "#training-and-optimization", "Training", None),
    (
        "checkpoint",
        "#checkpoints-provenance-license-and-limitations",
        "Checkpoint",
        None,
    ),
    ("api", "#public-api", "Public API", None),
)
SPEECHT5_ACTIONS = (
    ("use", "#usage"),
    ("checkpoint", "https://huggingface.co/microsoft/speecht5_tts"),
    ("paper", "https://arxiv.org/abs/2110.07205"),
    ("github", "https://github.com/microsoft/SpeechT5"),
    (
        "source",
        "https://github.com/kadirnar/voicehub/blob/main/"
        "voicehub/models/speecht5/modeling_speecht5.py",
    ),
    (
        "colab",
        "https://colab.research.google.com/github/kadirnar/voicehub/blob/main/"
        "notebooks/models/speecht5.ipynb",
    ),
)
SPEECHT5_FACT_LABELS = (
    "Task",
    "Parameters",
    "Architecture",
    "Runtime",
    "Languages",
    "Capabilities",
    "Training",
    "License",
    "Default checkpoint",
)
TOC_ROUTES = tuple(route for route in REPRESENTATIVE_ROUTES if route != SPEECHT5_ROUTE)
PAGE_ACTION_ROUTES = tuple(REPRESENTATIVE_ROUTES)
QUICKSTART_ROUTE = "getting-started/quickstart/index.html"
QUICKSTART_HEADINGS = (
    ("H1", "Quickstart"),
    ("H2", "Set up"),
    ("H2", "Pretrained models"),
    ("H2", "Inference"),
    ("H2", "Trainer"),
    ("H2", "Next steps"),
)
QUICKSTART_TAB_LABELS = (
    ("Linux", "macOS", "Windows"),
    ("Text to speech", "Automatic speech recognition", "Voice activity detection"),
)
QUICKSTART_EXTERNAL_TARGETS = ()
QUICKSTART_INTERNAL_TARGETS = (
    "/getting-started/installation/",
    "/getting-started/quickstart/#trainer",
    "/guides/inference/",
    "/guides/training/",
    "/models/providers/",
    "/guides/trainer/",
    "/guides/optimization-overview/",
)
TRAINER_ROUTE = "guides/trainer/index.html"
TRAINER_HEADINGS = (
    ("H1", "Trainer"),
    ("H2", "Next steps"),
)
TRAINER_NEXT_STEP_PATHS = (
    "/guides/training/",
    "/concepts/trainer/",
    "/models/training-support/",
    "/guides/data-preparation/",
)
OPTIMIZATION_ROUTE = "guides/optimization-overview/index.html"
OPTIMIZATION_HEADINGS = (
    ("H1", "Optimization overview"),
    ("H2", "Compilation"),
    ("H2", "Attention backends"),
    ("H2", "Kernels"),
    ("H2", "Diffusion caching"),
    ("H2", "Diffusion sampling"),
    ("H2", "Boundaries"),
    ("H2", "Next steps"),
)
OPTIMIZATION_PASS_NAMES = (
    "codec-kernels",
    "compile",
    "custom-kernels",
    "diffusion-cache",
    "diffusion-sampling",
    "flash-attention-4",
)
OPTIMIZATION_NEXT_STEP_TARGETS = (
    "/guides/tts-optimization/",
    "/guides/optional-backends/",
    "/guides/codec-optimization/",
    "/guides/diffusion-optimization/",
    "/reference/api/#optimization",
    "/project/adding-an-optimization/",
)
CONTRIBUTION_ROUTE = "project/adding-a-model/index.html"
CONTRIBUTION_HEADINGS = (
    ("H1", "Add a model"),
    ("H2", "1. Create the package"),
    ("H2", "2. Record provenance and license"),
    ("H2", "3. Define the config"),
    ("H2", "4. Implement the task wrapper"),
    ("H2", "5. Register once"),
    ("H2", "6. Declare training and optimization support"),
    ("H2", "7. Test the contract"),
    ("H2", "8. Generate the model page"),
    ("H2", "Completion evidence"),
)
CONTRIBUTION_PROCESS_LABELS = (
    "Create",
    "Audit",
    "Configure",
    "Wrap",
    "Register",
    "Support",
    "Test",
    "Document",
)
CONTRIBUTION_FINAL_TARGETS = (
    "/project/adding-speech-provider/",
    "/project/adding-an-optimization/",
)
MODEL_API_ROUTE = "reference/models/index.html"
MODEL_API_HEADINGS = (
    ("H1", "Models"),
    ("H2", "PreTrainedSpeechModel"),
    ("H2", "Task-specific pretrained models"),
    ("H2", "Model outputs"),
    ("H2", "Loading, saving, and sharing"),
)
MODEL_API_SOURCE_TARGETS = (
    "https://github.com/kadirnar/voicehub/blob/main/voicehub/modeling_utils.py",
    "https://github.com/kadirnar/voicehub/blob/main/voicehub/modeling_utils.py",
    "https://github.com/kadirnar/voicehub/blob/main/voicehub/audio_modeling_utils.py",
    "https://github.com/kadirnar/voicehub/blob/main/voicehub/modeling_outputs.py",
)
MODEL_API_INTERNAL_TARGETS = (
    "/models/providers/",
    "/reference/api/",
    "/reference/api/#save-load-and-resume-boundaries",
)
DESKTOP_KEYBOARD_FOCUS_PREFIX = (
    "skip:Skip to content",
    "header:logo",
    "header:search",
    "header:version",
    "header:language",
    "header:theme",
    "header:source",
)
TABLET_KEYBOARD_FOCUS_PREFIX = (
    "skip:Skip to content",
    "header:search",
    "header:version",
    "header:language",
    "header:theme",
    "header:source",
)
KEYBOARD_FOCUS_PREFIX = DESKTOP_KEYBOARD_FOCUS_PREFIX + (
    "branch:Get started",
    "primary:Overview",
    "primary:Installation",
    "primary:Quickstart",
    "primary:Inference",
    "branch:Models",
    "branch:Train",
    "branch:Optimize",
    "toc:Tasks",
    "toc:Text to speech",
    "toc:Automatic speech recognition",
    "toc:Voice activity detection",
    "toc:Parameters",
    "toc:Device",
    "toc:Batch inference",
    "toc:Task-specific parameters",
    "toc:Chunking and streaming",
    "toc:Large inputs",
    "toc:Large models",
    "toc:Save and reload",
    "toc:Troubleshooting",
)
ROOT_BRANCH_ACTIVATION_METHOD_BY_PALETTE = {
    "default": "keyboard",
    "slate": "pointer",
}
SPEECHT5_NESTED_BRANCH_STATES = (
    (("Models", "Text to speech"), True),
    (("Models", "Text to speech", "SpeechT5"), False),
    (("Models", "Automatic speech recognition"), False),
    (("Models", "Voice activity detection"), False),
)
NESTED_BRANCH_ACTIVATION_METHOD_BY_PALETTE = {
    "default": "keyboard",
    "slate": "pointer",
}
MOBILE_KEYBOARD_FOCUS_PREFIX = (
    "skip:Skip to content",
    "header:drawer",
    "header:search-trigger",
    "header:version",
)
DRAWER_ACTIVATION_CASES = (("Enter", "default"), ("Space", "slate"))
TOC_ACTIVATION_METHODS = ("pointer", "keyboard")
SEARCH_ACTIVATION_METHOD_BY_VIEWPORT = {
    "desktop": "keyboard",
    "tablet": "keyboard",
    "mobile": "pointer",
}
VERSION_ACTIVATION_METHOD_BY_PALETTE = {
    "default": "keyboard",
    "slate": "pointer",
}
SOURCE_REPOSITORY_URL = "https://github.com/kadirnar/voicehub"
SOURCE_ACTIVATION_METHOD_BY_PALETTE = {
    "default": "keyboard",
    "slate": "pointer",
}
REPRESENTATIVE_PAGE_ACTIONS = {
    HOME_ROUTE: {
        "edit": "https://github.com/kadirnar/voicehub/edit/main/docs/index.md",
        "previous": None,
        "next": ("/getting-started/installation/", "Next: Installation"),
    },
    INSTALLATION_ROUTE: {
        "edit": "https://github.com/kadirnar/voicehub/edit/main/docs/getting-started/installation.md",
        "previous": ("/", "Previous: Overview"),
        "next": ("/getting-started/quickstart/", "Next: Quickstart"),
    },
    QUICKSTART_ROUTE: {
        "edit": "https://github.com/kadirnar/voicehub/edit/main/docs/getting-started/quickstart.md",
        "previous": ("/getting-started/installation/", "Previous: Installation"),
        "next": ("/guides/inference/", "Next: Inference"),
    },
    INFERENCE_ROUTE: {
        "edit": "https://github.com/kadirnar/voicehub/edit/main/docs/guides/inference.md",
        "previous": ("/getting-started/quickstart/", "Previous: Quickstart"),
        "next": ("/models/providers/", "Next: Model list"),
    },
    MODEL_INDEX_ROUTE: {
        "edit": "https://github.com/kadirnar/voicehub/edit/main/docs/models/providers/index.md",
        "previous": ("/guides/inference/", "Previous: Inference"),
        "next": ("/models/providers/bark/", "Next: Bark"),
    },
    SPEECHT5_ROUTE: {
        "edit": "https://github.com/kadirnar/voicehub/edit/main/docs/models/providers/speecht5.md",
        "previous": ("/models/providers/qwen3tts/", "Previous: Qwen3TTS"),
        "next": ("/models/providers/styletts2/", "Next: StyleTTS2"),
    },
    TRAINER_ROUTE: {
        "edit": "https://github.com/kadirnar/voicehub/edit/main/docs/guides/trainer.md",
        "previous": ("/project/adding-a-model/", "Previous: Add a model"),
        "next": ("/guides/training/", "Next: Fine-tuning"),
    },
    OPTIMIZATION_ROUTE: {
        "edit": "https://github.com/kadirnar/voicehub/edit/main/docs/guides/optimization-overview.md",
        "previous": ("/guides/data-preparation/", "Previous: Data preparation"),
        "next": ("/optimizations/", "Next: Optimization catalog"),
    },
    CONTRIBUTION_ROUTE: {
        "edit": "https://github.com/kadirnar/voicehub/edit/main/docs/project/adding-a-model.md",
        "previous": ("/reference/models/", "Previous: Models API"),
        "next": ("/guides/trainer/", "Next: Trainer overview"),
    },
    MODEL_API_ROUTE: {
        "edit": "https://github.com/kadirnar/voicehub/edit/main/docs/reference/models.md",
        "previous": ("/models/providers/vad_webrtc/", "Previous: WebRTCVAD"),
        "next": ("/project/adding-a-model/", "Next: Add a model"),
    },
}
PAGE_ACTION_METHOD_BY_PALETTE = {
    "default": "keyboard",
    "slate": "pointer",
}
THEME_ACTIVATION_METHOD_BY_PALETTE = {
    "default": "keyboard",
    "slate": "pointer",
}
THEME_TARGET_BY_PALETTE = {
    "default": "slate",
    "slate": "default",
}
LANGUAGE_ACTIVATION_METHOD_BY_PALETTE = {
    "default": "keyboard",
    "slate": "pointer",
}
LANGUAGE_TARGET_BY_PALETTE = {
    "default": "tr",
    "slate": "ar",
}
LANGUAGE_LOCALES = ("en", "tr", "es", "fr", "de", "pt", "zh", "ja", "ko", "ru", "ar")
INTERACTIVE_ACCESSIBILITY_STATES = (
    "search-open",
    "search-results",
    "search-empty",
    "version-open",
    "branch-open",
    "drawer-open",
)
FOCUSABLE_SELECTOR = (
    "a[href], area[href], button, input:not([type=hidden]), select, textarea, summary, iframe, "
    "object, embed, audio[controls], video[controls], "
    "[contenteditable]:not([contenteditable='false']), [tabindex]:not([tabindex='-1'])")


class DocumentationVisualError(RuntimeError):
    """Raised when rendered geometry contradicts the visual contract."""


class _QuietRequestHandler(SimpleHTTPRequestHandler):

    def log_message(self, format: str, *args: object) -> None:
        return

    def translate_path(self, path: str) -> str:
        if path == "/voicehub" or path.startswith("/voicehub/"):
            path = path.removeprefix("/voicehub") or "/"
        return super().translate_path(path)


def _route_url(base_url: str, relative_path: str) -> str:
    route = relative_path.removesuffix("index.html")
    return f"{base_url}/{route}"


def _screenshot_case_key(relative_path: str, viewport: str, palette: str) -> str:
    return f"{relative_path}|{viewport}|{palette}"


def _screenshot_signature(screenshot: bytes, viewport: dict[str, Any]) -> dict[str, Any]:
    image = Image.open(BytesIO(screenshot)).convert("RGB")
    expected_size = (viewport["width"], viewport["height"])
    if image.size != expected_size:
        raise DocumentationVisualError(f"Screenshot size is {image.size!r}, expected {expected_size!r}.")

    blurred = image.filter(ImageFilter.GaussianBlur(radius=2.0))
    signature_height = max(
        1,
        round(SCREENSHOT_SIGNATURE_WIDTH * image.height / image.width),
    )
    grayscale = blurred.convert("L").resize(
        (SCREENSHOT_SIGNATURE_WIDTH + 1, signature_height),
        Image.Resampling.LANCZOS,
    )
    pixels = grayscale.load()
    hash_value = 0
    hash_bits = SCREENSHOT_SIGNATURE_WIDTH * signature_height
    for y in range(signature_height):
        for x in range(SCREENSHOT_SIGNATURE_WIDTH):
            hash_value = (hash_value << 1) | int(pixels[x, y] > pixels[x + 1, y])
    hash_width = math.ceil(hash_bits / 4)
    mean_rgb = [round(channel, 3) for channel in ImageStat.Stat(blurred).mean]
    return {
        "height": image.height,
        "hash": f"{hash_value:0{hash_width}x}",
        "hash_bits": hash_bits,
        "mean_rgb": mean_rgb,
        "width": image.width,
    }


def _load_screenshot_baselines(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DocumentationVisualError(
            f"Screenshot baseline manifest is missing: {path}. Run with "
            "--update-screenshot-baselines and review the generated JSON.") from error
    except json.JSONDecodeError as error:
        raise DocumentationVisualError(
            f"Screenshot baseline manifest is invalid JSON: {path}: {error}.") from error

    if manifest.get("schema_version") != SCREENSHOT_SCHEMA_VERSION:
        raise DocumentationVisualError(
            f"Screenshot baseline schema is {manifest.get('schema_version')!r}, "
            f"expected {SCREENSHOT_SCHEMA_VERSION}.")
    if not isinstance(manifest.get("cases"), dict):
        raise DocumentationVisualError("Screenshot baseline manifest must contain a cases mapping.")
    return manifest


def _compare_screenshot_signature(
    case: str,
    actual: dict[str, Any],
    expected: dict[str, Any] | None,
) -> None:
    if expected is None:
        raise DocumentationVisualError(f"{case}: screenshot baseline is missing.")
    for field in ("width", "height", "hash_bits"):
        if actual[field] != expected.get(field):
            raise DocumentationVisualError(
                f"{case}: screenshot {field} is {actual[field]!r}, "
                f"expected {expected.get(field)!r}.")

    try:
        hamming_distance = (int(actual["hash"], 16) ^ int(expected["hash"], 16)).bit_count()
    except (KeyError, TypeError, ValueError) as error:
        raise DocumentationVisualError(f"{case}: screenshot hash baseline is invalid.") from error
    hamming_ratio = hamming_distance / actual["hash_bits"]
    if hamming_ratio > SCREENSHOT_MAX_HAMMING_RATIO:
        raise DocumentationVisualError(
            f"{case}: screenshot perceptual hash differs by {hamming_distance} bits "
            f"({hamming_ratio:.3%}), above {SCREENSHOT_MAX_HAMMING_RATIO:.1%}.")

    expected_mean = expected.get("mean_rgb")
    if not isinstance(expected_mean, list) or len(expected_mean) != 3:
        raise DocumentationVisualError(f"{case}: screenshot mean RGB baseline is invalid.")
    channel_deltas = [
        abs(actual_channel - expected_channel)
        for actual_channel, expected_channel in zip(actual["mean_rgb"], expected_mean, strict=True)
    ]
    if max(channel_deltas) > SCREENSHOT_MAX_MEAN_CHANNEL_DELTA:
        raise DocumentationVisualError(
            f"{case}: screenshot mean RGB deltas are {channel_deltas!r}, above "
            f"{SCREENSHOT_MAX_MEAN_CHANNEL_DELTA:.1f}.")


def _set_palette(page: Page, palette: str) -> None:
    selector = f"input[data-md-color-scheme='{palette}']"
    page.locator(selector).evaluate(
        """input => {
          input.checked = true;
          input.dispatchEvent(new Event("change", { bubbles: true }));
        }""")
    page.wait_for_function(
        "palette => document.body.dataset.mdColorScheme === palette",
        arg=palette,
    )
    page.evaluate(
        "() => new Promise(resolve => { "
        "requestAnimationFrame(() => requestAnimationFrame(resolve)); "
        "})")


def _reset_keyboard_focus(page: Page) -> None:
    page.wait_for_timeout(50)
    page.evaluate(
        """() => {
          const originalTabindex = document.body.getAttribute("tabindex");
          document.body.setAttribute("tabindex", "-1");
          document.body.focus();
          if (originalTabindex === null) document.body.removeAttribute("tabindex");
          else document.body.setAttribute("tabindex", originalTabindex);
        }""")
    page.wait_for_function("document.activeElement === document.body")


def _set_keyboard_palette(page: Page, palette: str) -> None:
    _set_palette(page, palette)
    page.reload(wait_until="networkidle")
    page.wait_for_function(
        "palette => document.body.dataset.mdColorScheme === palette",
        arg=palette,
    )


def _rendered_state(page: Page) -> dict[str, Any]:
    return page.evaluate(
        """() => {
          const rectangle = selector => {
            const element = document.querySelector(selector);
            if (!element) return null;
            const bounds = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return {
              x: bounds.x,
              y: bounds.y,
              width: bounds.width,
              height: bounds.height,
              display: style.display,
            };
          };
          const visible = element => {
            const bounds = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return bounds.width > 0 && bounds.height > 0 &&
              style.display !== "none" && style.visibility !== "hidden";
          };
          const primary = document.querySelector(".md-sidebar--primary");
          const inactiveFocusables = Array.from(
            primary?.querySelectorAll("input.md-nav__toggle:not(:checked)") || []
          ).flatMap(toggle => {
            const panel = Array.from(toggle.parentElement?.children || [])
              .find(child => child.classList.contains("md-nav"));
            return Array.from(panel?.querySelectorAll(
              "a[href], button, input:not([type=hidden]), select, textarea, summary, " +
              "[tabindex]:not([tabindex='-1'])"
            ) || []).filter(visible);
          });
          const checkedBranches = Array.from(
            primary?.querySelectorAll("input.md-nav__toggle:checked") || []
          ).filter(toggle => toggle.id.startsWith("__nav_"))
            .map(toggle => primary.querySelector(
              `button.md-nav__link[data-vh-nav-toggle='${toggle.id}']`
            )?.textContent?.trim());
          return {
            scheme: document.body.dataset.mdColorScheme,
            article: rectangle(".md-content__inner"),
            header: rectangle(".md-header"),
            railControls: rectangle(".vh-doc-rail-controls"),
            productControl: rectangle(".vh-doc-rail-controls > .md-header__title"),
            searchControl: rectangle(".vh-doc-rail-controls > .md-search"),
            utilityControl: rectangle(".vh-doc-rail-utility"),
            versionControl: rectangle("[data-vh-version-control] > summary"),
            languageControl: rectangle("[data-vh-language-select]"),
            themeControl: rectangle("[data-vh-theme-toggle]:not([hidden])"),
            sourceControl: rectangle(".vh-source-link"),
            sourceIcon: rectangle(".vh-source-link .md-source__icon"),
            sourceFacts: rectangle(".vh-source-link .md-source__facts"),
            primary: rectangle(".md-sidebar--primary"),
            secondary: rectangle(".md-sidebar--secondary"),
            overflow: document.documentElement.scrollWidth -
              document.documentElement.clientWidth,
            drawerChecked: document.querySelector("#__drawer")?.checked,
            title: document.querySelector("h1")?.textContent?.trim()
              .replace(/¶$/, "").trim(),
            active: Array.from(primary?.querySelectorAll(
              "a.md-nav__link--active"
            ) || []).map(link => link.textContent.trim()),
            visibleActiveLabels: Array.from(primary?.querySelectorAll(
              ".md-nav__item--active > .md-nav__link--active"
            ) || []).filter(visible).map(item => item.textContent.trim()),
            checkedBranches,
            roots: Array.from(primary?.querySelectorAll(
              "nav.md-nav--primary > ul.md-nav__list > li.md-nav__item > " +
              "button.md-nav__link[data-vh-nav-toggle]"
            ) || []).map(button => button.textContent.trim()),
            inactiveFocusableCount: inactiveFocusables.length,
            background: getComputedStyle(document.body).backgroundColor,
            text: getComputedStyle(document.body).color,
          };
        }""")


def _active_focus_state(page: Page) -> dict[str, Any]:
    return page.evaluate(
        r"""() => {
          const element = document.activeElement;
          const primary = document.querySelector(".md-sidebar--primary");
          const secondary = document.querySelector(".md-sidebar--secondary");
          const text = element?.textContent?.trim().replace(/\s+/g, " ") || "";
          const tabLabel = element instanceof HTMLInputElement && element.type === "radio"
            ? Array.from(
              element.parentElement?.querySelectorAll(".tabbed-labels > label[for]") || []
            ).find(label => label instanceof HTMLLabelElement && label.htmlFor === element.id)
            : null;
          const focusTarget = tabLabel || element;
          let descriptor = "unknown";
          if (element === document.body) descriptor = "body";
          else if (element?.matches(".md-skip")) descriptor = `skip:${text}`;
          else if (element?.matches("[data-vh-drawer-trigger]")) descriptor = "header:drawer";
          else if (element?.matches("[data-vh-search-trigger]")) {
            descriptor = "header:search-trigger";
          }
          else if (element?.matches(".md-header__button.md-logo")) descriptor = "header:logo";
          else if (element?.matches(".md-search__input")) descriptor = "header:search";
          else if (element?.matches("[data-vh-version-control] > summary")) {
            descriptor = "header:version";
          } else if (element?.matches("[data-vh-language-select]")) {
            descriptor = "header:language";
          } else if (element?.matches("[data-vh-theme-toggle]")) {
            descriptor = "header:theme";
          } else if (element?.matches(".vh-source-link")) descriptor = "header:source";
          else if (element?.matches("a.md-nav__button.md-logo") && primary?.contains(element)) {
            descriptor = "drawer:home";
          }
          else if (element?.matches("button.md-nav__link[data-vh-nav-toggle]") && primary?.contains(element)) {
            descriptor = `branch:${text}`;
          } else if (element?.matches("a.md-nav__link") && primary?.contains(element)) {
            descriptor = `primary:${text}`;
          } else if (element?.matches("a.md-nav__link") && secondary?.contains(element)) {
            descriptor = `toc:${text}`;
          } else if (tabLabel instanceof HTMLLabelElement) {
            descriptor = `tab:${tabLabel.textContent?.trim().replace(/\s+/g, " ") || ""}`;
          } else if (element instanceof HTMLElement) {
            descriptor = `${element.tagName.toLowerCase()}:${
              element.id || element.getAttribute("href") || text || element.className
            }`;
          }

          const inactivePanelContainsFocus = Array.from(
            primary?.querySelectorAll("input.md-nav__toggle:not(:checked)") || []
          ).some(toggle => {
            const panel = Array.from(toggle.parentElement?.children || [])
              .find(child => child.classList.contains("md-nav"));
            return panel?.contains(element) || false;
          });
          const bounds = focusTarget?.getBoundingClientRect();
          const style = focusTarget instanceof Element ? getComputedStyle(focusTarget) : null;
          const viewportTolerance = 1;
          return {
            descriptor,
            inactivePanelContainsFocus,
            visible: Boolean(bounds && bounds.width > 0 && bounds.height > 0 &&
              style?.display !== "none" && style?.visibility !== "hidden"),
            withinViewport: Boolean(bounds && bounds.left >= -viewportTolerance &&
              bounds.right <= innerWidth + viewportTolerance &&
              bounds.top >= -viewportTolerance &&
              bounds.bottom <= innerHeight + viewportTolerance),
            outlineStyle: style?.outlineStyle,
            outlineWidth: style?.outlineWidth,
            outlineOffset: style?.outlineOffset,
          };
        }""")


def _validate_focused_element(
    case: str,
    state: dict[str, Any],
    *,
    require_viewport: bool = False,
) -> None:
    if not state["visible"]:
        raise DocumentationVisualError(
            f"{case}: native Tab focused invisible element {state['descriptor']!r}.")
    if require_viewport and not state["withinViewport"]:
        raise DocumentationVisualError(
            f"{case}: native Tab focused off-canvas element {state['descriptor']!r}.")
    if state["inactivePanelContainsFocus"]:
        raise DocumentationVisualError(
            f"{case}: native Tab entered inactive branch at {state['descriptor']!r}.")
    if state["descriptor"].startswith(("branch:", "drawer:", "tab:")):
        expected_outline = ("solid", "2px", "2px")
        actual_outline = (
            state["outlineStyle"],
            state["outlineWidth"],
            state["outlineOffset"],
        )
        if actual_outline != expected_outline:
            raise DocumentationVisualError(
                f"{case}: {state['descriptor']!r} focus outline is {actual_outline!r}, "
                f"expected {expected_outline!r}.")


def _focus_prefix_for_viewport(viewport: dict[str, Any]) -> tuple[str, ...]:
    viewport_name = viewport["name"]
    if viewport_name == "desktop":
        return DESKTOP_KEYBOARD_FOCUS_PREFIX
    if viewport_name == "tablet":
        return TABLET_KEYBOARD_FOCUS_PREFIX
    if viewport_name == "mobile":
        return MOBILE_KEYBOARD_FOCUS_PREFIX
    raise DocumentationVisualError(f"Unsupported focus-cycle viewport: {viewport_name!r}.")


def _validate_focus_cycle(
    page: Page,
    case: str,
    expected_prefix: tuple[str, ...],
    *,
    require_viewport: bool = False,
) -> int:
    _reset_keyboard_focus(page)
    focus_sequence: list[str] = []
    focusable_count = page.locator(FOCUSABLE_SELECTOR).count()
    maximum_steps = focusable_count + 1
    for _ in range(maximum_steps):
        page.keyboard.press("Tab")
        state = _active_focus_state(page)
        if state["descriptor"] == "body":
            break
        _validate_focused_element(
            case,
            state,
            require_viewport=require_viewport and len(focus_sequence) < len(expected_prefix),
        )
        focus_sequence.append(state["descriptor"])
    else:
        raise DocumentationVisualError(
            f"{case}: focus did not complete one cycle within {maximum_steps} steps "
            f"for {focusable_count} focusable DOM elements.")

    actual_prefix = tuple(focus_sequence[:len(expected_prefix)])
    if actual_prefix != expected_prefix:
        raise DocumentationVisualError(
            f"{case}: focus prefix is {actual_prefix!r}, expected {expected_prefix!r}.")
    if not focus_sequence:
        raise DocumentationVisualError(f"{case}: focus cycle contains no interactive elements.")

    page.keyboard.press("Tab")
    repeated = _active_focus_state(page)
    _validate_focused_element(case, repeated, require_viewport=require_viewport)
    if repeated["descriptor"] != focus_sequence[0]:
        raise DocumentationVisualError(
            f"{case}: focus resumed at {repeated['descriptor']!r}, expected {focus_sequence[0]!r}.")
    return len(focus_sequence)


def _validate_root_branch_activation(
    page: Page,
    case: str,
    branch_label: str,
    activation_method: str,
    palette: str,
    axe: Axe,
) -> str:
    selector = (
        "nav.md-nav--primary > ul.md-nav__list > li.md-nav__item > "
        "button.md-nav__link[data-vh-nav-toggle]")
    buttons = page.locator(selector)
    root_labels = tuple(value.strip() for value in buttons.all_text_contents())
    if root_labels != TOP_LEVEL_NAVIGATION:
        raise DocumentationVisualError(
            f"{case}: root branch order is {root_labels!r}, expected {TOP_LEVEL_NAVIGATION!r}.")
    button = buttons.nth(TOP_LEVEL_NAVIGATION.index(branch_label))

    def branch_state() -> dict[str, Any]:
        return button.evaluate(
            r"""button => {
              const navigation = button.closest(".md-sidebar--primary");
              const toggle = document.getElementById(button.dataset.vhNavToggle || "");
              const panel = document.getElementById(button.getAttribute("aria-controls") || "");
              const bounds = button.getBoundingClientRect();
              const style = getComputedStyle(button);
              return {
                activeLinks: Array.from(
                  navigation?.querySelectorAll("a.md-nav__link--active") || []
                ).map(link => link.textContent.trim()),
                ariaControls: button.getAttribute("aria-controls"),
                checked: toggle?.checked,
                expanded: button.getAttribute("aria-expanded"),
                focused: document.activeElement === button,
                height: bounds.height,
                palette: document.body.dataset.mdColorScheme,
                panelDisplay: panel ? getComputedStyle(panel).display : null,
                panelId: panel?.id,
                panelLabel: panel?.getAttribute("aria-label"),
                path: location.pathname,
                rootLabels: Array.from(navigation?.querySelectorAll(
                  "nav.md-nav--primary > ul.md-nav__list > li.md-nav__item > " +
                  "button.md-nav__link[data-vh-nav-toggle]"
                ) || []).map(item => item.textContent.trim().replace(/\s+/g, " ")),
                tabIndex: button.tabIndex,
                toggleId: toggle?.id,
                visible: bounds.width > 0 && bounds.height > 0 &&
                  style.display !== "none" && style.visibility !== "hidden",
                viewportWidth: innerWidth,
                width: bounds.width,
                withinViewport: bounds.left >= 0 && bounds.right <= innerWidth &&
                  bounds.top >= 0 && bounds.bottom <= innerHeight,
                x: bounds.x,
                y: bounds.y,
              };
            }""")

    initial = branch_state()
    expected_initial_expanded = branch_label == "Get started"
    if initial["activeLinks"] != ["Inference"]:
        raise DocumentationVisualError(
            f"{case}: initial active links are {initial['activeLinks']!r}, expected ['Inference'].")
    if initial["checked"] is not expected_initial_expanded:
        raise DocumentationVisualError(
            f"{case}: initial checked state is {initial['checked']!r}, "
            f"expected {expected_initial_expanded!r}.")
    if initial["expanded"] != str(expected_initial_expanded).lower():
        raise DocumentationVisualError(f"{case}: initial aria-expanded is {initial['expanded']!r}.")
    expected_initial_display = "block" if expected_initial_expanded else "none"
    if initial["panelDisplay"] != expected_initial_display:
        raise DocumentationVisualError(
            f"{case}: initial panel display is {initial['panelDisplay']!r}, "
            f"expected {expected_initial_display!r}.")
    if initial["ariaControls"] != initial["panelId"] or not initial["toggleId"]:
        raise DocumentationVisualError(f"{case}: root branch wiring is invalid: {initial!r}.")
    if not initial["panelLabel"] or not initial["panelLabel"].endswith(f": {branch_label}"):
        raise DocumentationVisualError(
            f"{case}: panel label is {initial['panelLabel']!r}, expected branch suffix {branch_label!r}.")
    if (initial["focused"] or initial["palette"] != palette or initial["path"] != "/guides/inference/" or
            initial["rootLabels"] != list(TOP_LEVEL_NAVIGATION) or initial["tabIndex"] != 0 or
            not initial["visible"] or not initial["withinViewport"]):
        raise DocumentationVisualError(f"{case}: invalid initial root branch state: {initial!r}.")
    expected_geometry = ((("x", 28), ("width", 225), ("height",
                                                      22.03125)) if initial["viewportWidth"] == 1440 else
                         (("x", 12), ("width", 257), ("height", 24)))
    for field, expected in expected_geometry:
        _assert_close(case, f"root branch {field}", initial[field], expected)
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: initial root branch state introduced overflow.")

    def activate(expected_expanded: bool) -> None:
        if activation_method == "pointer":
            button.click()
        elif activation_method == "keyboard":
            button.focus()
            page.keyboard.press("Enter")
        else:
            raise DocumentationVisualError(
                f"{case}: unsupported root branch activation method {activation_method!r}.")
        page.wait_for_function(
            "args => { const button = Array.from(document.querySelectorAll(args.selector))"
            ".find(item => item.textContent.trim().replace(/\\s+/g, ' ') === args.label); "
            "const toggle = document.getElementById(button?.dataset.vhNavToggle || ''); "
            "return toggle?.checked === args.expanded && "
            "button?.getAttribute('aria-expanded') === String(args.expanded); }",
            arg={
                "expanded": expected_expanded,
                "label": branch_label,
                "selector": selector,
            },
        )

    target_expanded = not expected_initial_expanded
    activate(target_expanded)
    target = branch_state()
    expected_target_display = "block" if target_expanded else "none"
    if (target["checked"] is not target_expanded or target["expanded"] != str(target_expanded).lower() or
            target["panelDisplay"] != expected_target_display or not target["focused"] or
            target["activeLinks"] != ["Inference"] or not target["visible"] or not target["withinViewport"] or
            target["path"] != initial["path"] or target["palette"] != initial["palette"]):
        raise DocumentationVisualError(f"{case}: invalid activated root branch state: {target!r}.")
    for field in ("x", "width", "height"):
        _assert_close(case, f"activated root branch {field}", target[field], initial[field])
    focused = _active_focus_state(page)
    expected_descriptor = f"branch:{branch_label}"
    if focused["descriptor"] != expected_descriptor:
        raise DocumentationVisualError(f"{case}: activated root branch focus state is {focused!r}.")
    if activation_method == "keyboard":
        _validate_focused_element(case, focused, require_viewport=True)
        if (focused["outlineStyle"], focused["outlineWidth"], focused["outlineOffset"]) != ("solid", "2px",
                                                                                            "2px"):
            raise DocumentationVisualError(f"{case}: keyboard focus outline is {focused!r}.")
    elif focused["outlineStyle"] != "none":
        raise DocumentationVisualError(
            f"{case}: pointer activation unexpectedly rendered a focus outline: {focused!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: activated root branch introduced overflow.")
    axe_core = "unknown"
    if target_expanded:
        axe_core = _validate_accessibility(axe, page, f"{case} / expanded")

    activate(expected_initial_expanded)
    restored = branch_state()
    if (restored["checked"] is not expected_initial_expanded or
            restored["expanded"] != str(expected_initial_expanded).lower() or
            restored["panelDisplay"] != expected_initial_display or not restored["focused"] or
            restored["activeLinks"] != ["Inference"] or not restored["visible"] or
            not restored["withinViewport"] or restored["path"] != initial["path"] or
            restored["palette"] != initial["palette"]):
        raise DocumentationVisualError(f"{case}: invalid restored root branch state: {restored!r}.")
    for field in ("x", "width", "height"):
        _assert_close(case, f"restored root branch {field}", restored[field], initial[field])
    _assert_close(case, "restored root branch y", restored["y"], initial["y"])
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: restored root branch introduced overflow.")
    if expected_initial_expanded:
        axe_core = _validate_accessibility(axe, page, f"{case} / restored expanded")
    return axe_core


def _validate_nested_branch_activation(
    page: Page,
    case: str,
    branch_path: tuple[str, ...],
    expected_initial_expanded: bool,
    activation_method: str,
    palette: str,
    axe: Axe,
) -> str:
    expected_path = f"/{SPEECHT5_ROUTE.removesuffix('index.html')}"
    selector = "nav.md-nav--primary button.md-nav__link[data-vh-nav-toggle]"
    buttons = page.locator(selector)
    inventory = buttons.evaluate_all(
        r"""buttons => buttons.map((button, index) => {
          const path = [];
          let item = button.closest("li.md-nav__item");
          while (item) {
            const directButton = Array.from(item.children).find(child =>
              child.matches?.("button.md-nav__link[data-vh-nav-toggle]")
            );
            if (directButton) {
              path.unshift(directButton.textContent.trim().replace(/\s+/g, " "));
            }
            item = item.parentElement?.closest("li.md-nav__item") || null;
          }
          const toggle = document.getElementById(button.dataset.vhNavToggle || "");
          return { checked: toggle?.checked, index, path };
        })""")
    expected_paths = tuple(path for path, _ in SPEECHT5_NESTED_BRANCH_STATES)
    model_paths = tuple(
        tuple(item["path"]) for item in inventory if item["path"][:1] == ["Models"] and len(item["path"]) > 1)
    if model_paths != expected_paths:
        raise DocumentationVisualError(
            f"{case}: model nested branch inventory is {model_paths!r}, expected {expected_paths!r}.")
    matching = [item for item in inventory if tuple(item["path"]) == branch_path]
    if len(matching) != 1:
        raise DocumentationVisualError(
            f"{case}: branch path {branch_path!r} matched {len(matching)} controls.")
    if matching[0]["checked"] is not expected_initial_expanded:
        raise DocumentationVisualError(
            f"{case}: inventory checked state is {matching[0]['checked']!r}, "
            f"expected {expected_initial_expanded!r}.")
    button = buttons.nth(matching[0]["index"])
    button.scroll_into_view_if_needed()

    def branch_state() -> dict[str, Any]:
        return button.evaluate(
            r"""button => {
              const path = [];
              let item = button.closest("li.md-nav__item");
              while (item) {
                const directButton = Array.from(item.children).find(child =>
                  child.matches?.("button.md-nav__link[data-vh-nav-toggle]")
                );
                if (directButton) {
                  path.unshift(directButton.textContent.trim().replace(/\s+/g, " "));
                }
                item = item.parentElement?.closest("li.md-nav__item") || null;
              }
              const navigation = button.closest(".md-sidebar--primary");
              const toggle = document.getElementById(button.dataset.vhNavToggle || "");
              const panel = Array.from(toggle?.parentElement?.children || []).find(element =>
                element.classList.contains("md-nav")
              );
              const bounds = button.getBoundingClientRect();
              const navigationBounds = navigation?.getBoundingClientRect();
              const style = getComputedStyle(button);
              return {
                activeLinks: Array.from(
                  navigation?.querySelectorAll("a.md-nav__link--active") || []
                ).filter(link => !link.closest(".md-nav--secondary"))
                  .map(link => link.textContent.trim().replace(/\s+/g, " ")),
                ariaControls: button.getAttribute("aria-controls"),
                buttonPath: path,
                checked: toggle?.checked,
                expanded: button.getAttribute("aria-expanded"),
                focused: document.activeElement === button,
                height: bounds.height,
                navigationBottom: navigationBounds?.bottom,
                navigationHeight: navigationBounds?.height,
                navigationPosition: navigation ? getComputedStyle(navigation).position : null,
                navigationTop: navigationBounds?.top,
                navigationWidth: navigationBounds?.width,
                navigationX: navigationBounds?.x,
                palette: document.body.dataset.mdColorScheme,
                panelDisplay: panel ? getComputedStyle(panel).display : null,
                panelId: panel?.id,
                panelLabel: panel?.getAttribute("aria-label"),
                path: location.pathname,
                scrollY: window.scrollY,
                shellOffset: getComputedStyle(document.documentElement)
                  .getPropertyValue("--vh-shell-scroll-offset").trim(),
                tabIndex: button.tabIndex,
                toggleId: toggle?.id,
                visible: bounds.width > 0 && bounds.height > 0 &&
                  style.display !== "none" && style.visibility !== "hidden",
                viewportHeight: innerHeight,
                width: bounds.width,
                withinViewport: bounds.left >= 0 && bounds.right <= innerWidth &&
                  bounds.top >= 0 && bounds.bottom <= innerHeight,
                x: bounds.x,
                y: bounds.y,
              };
            }""")

    def validate_branch_state(
        state: dict[str, Any],
        *,
        expected_expanded: bool,
        expected_focused: bool,
        expected_offset: str,
        expected_scroll: int,
    ) -> None:
        expected_display = "block" if expected_expanded else "none"
        if (state["activeLinks"] != ["SpeechT5"] or state["buttonPath"] != list(branch_path) or
                state["checked"] is not expected_expanded or
                state["expanded"] != str(expected_expanded).lower() or
                state["panelDisplay"] != expected_display or state["focused"] is not expected_focused or
                state["navigationPosition"] != "sticky" or state["palette"] != palette or
                state["path"] != expected_path or state["shellOffset"] != expected_offset or
                state["tabIndex"] != 0 or not state["visible"] or not state["withinViewport"]):
            raise DocumentationVisualError(f"{case}: invalid nested branch state: {state!r}.")
        if expected_scroll == 0 and state["scrollY"] != 0:
            raise DocumentationVisualError(f"{case}: document scroll is {state['scrollY']!r}, expected 0.")
        if expected_scroll > 0 and state["scrollY"] < expected_scroll:
            raise DocumentationVisualError(
                f"{case}: document scroll is {state['scrollY']!r}, expected at least {expected_scroll}.")
        if (state["ariaControls"] != state["panelId"] or not state["toggleId"] or not state["panelLabel"] or
                not state["panelLabel"].endswith(f": {branch_path[-1]}")):
            raise DocumentationVisualError(f"{case}: nested branch wiring is invalid: {state!r}.")
        _assert_close(case, "nested rail x", state["navigationX"], 0)
        _assert_close(case, "nested rail width", state["navigationWidth"], 270)
        expected_top = 0 if expected_scroll else 65
        expected_height = state["viewportHeight"] if expected_scroll else state["viewportHeight"] - 65
        _assert_close(case, "nested rail top", state["navigationTop"], expected_top)
        _assert_close(case, "nested rail height", state["navigationHeight"], expected_height)
        _assert_close(case, "nested rail bottom", state["navigationBottom"], state["viewportHeight"])
        if _rendered_state(page)["overflow"] != 0:
            raise DocumentationVisualError(f"{case}: nested branch state introduced overflow.")

    def activate(expected_expanded: bool) -> None:
        if activation_method == "pointer":
            button.click()
        elif activation_method == "keyboard":
            button.focus()
            page.keyboard.press("Enter")
        else:
            raise DocumentationVisualError(
                f"{case}: unsupported nested branch activation method {activation_method!r}.")
        page.wait_for_function(
            "args => { const button = Array.from(document.querySelectorAll(args.selector))"
            ".find(item => item.dataset.vhNavToggle === args.toggleId); "
            "const toggle = document.getElementById(args.toggleId); "
            "return toggle?.checked === args.expanded && "
            "button?.getAttribute('aria-expanded') === String(args.expanded); }",
            arg={
                "expanded": expected_expanded,
                "selector": selector,
                "toggleId": toggle_id,
            },
        )

    initial = branch_state()
    toggle_id = initial["toggleId"]
    validate_branch_state(
        initial,
        expected_expanded=expected_initial_expanded,
        expected_focused=False,
        expected_offset="0px",
        expected_scroll=0,
    )

    target_expanded = not expected_initial_expanded
    activate(target_expanded)
    target = branch_state()
    validate_branch_state(
        target,
        expected_expanded=target_expanded,
        expected_focused=True,
        expected_offset="0px",
        expected_scroll=0,
    )
    focused = _active_focus_state(page)
    if focused["descriptor"] != f"branch:{branch_path[-1]}":
        raise DocumentationVisualError(f"{case}: activated nested branch focus state is {focused!r}.")
    if activation_method == "keyboard":
        _validate_focused_element(case, focused, require_viewport=True)
        if (focused["outlineStyle"], focused["outlineWidth"], focused["outlineOffset"]) != ("solid", "2px",
                                                                                            "2px"):
            raise DocumentationVisualError(f"{case}: nested keyboard focus outline is {focused!r}.")
    elif focused["outlineStyle"] != "none":
        raise DocumentationVisualError(
            f"{case}: nested pointer activation unexpectedly rendered a focus outline: {focused!r}.")

    page.evaluate("window.scrollTo(0, 320)")
    page.wait_for_function(
        "() => window.scrollY >= 320 && getComputedStyle(document.documentElement)"
        ".getPropertyValue('--vh-shell-scroll-offset').trim() === '65px'", )
    sticky = branch_state()
    validate_branch_state(
        sticky,
        expected_expanded=target_expanded,
        expected_focused=True,
        expected_offset="65px",
        expected_scroll=320,
    )
    axe_core = _validate_accessibility(axe, page, f"{case} / sticky target")

    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_function(
        "() => window.scrollY === 0 && getComputedStyle(document.documentElement)"
        ".getPropertyValue('--vh-shell-scroll-offset').trim() === '0px'", )
    activate(expected_initial_expanded)
    restored = branch_state()
    validate_branch_state(
        restored,
        expected_expanded=expected_initial_expanded,
        expected_focused=True,
        expected_offset="0px",
        expected_scroll=0,
    )
    return axe_core


def _validate_mobile_drawer_activation(page: Page, key: str, palette: str) -> None:
    case = f"{KEYBOARD_ROUTE} / mobile / {palette} / {key} drawer activation"
    _reset_keyboard_focus(page)
    target_index = MOBILE_KEYBOARD_FOCUS_PREFIX.index("header:drawer") + 1
    for _ in range(target_index):
        page.keyboard.press("Tab")
    focused = _active_focus_state(page)
    _validate_focused_element(case, focused, require_viewport=True)
    if focused["descriptor"] != "header:drawer":
        raise DocumentationVisualError(
            f"{case}: activation target is {focused['descriptor']!r}, expected 'header:drawer'.")

    page.keyboard.press(key)
    page.wait_for_function(
        """() => {
          const drawer = document.querySelector("#__drawer");
          const trigger = document.querySelector("[data-vh-drawer-trigger]");
          const navigation = document.querySelector(".md-sidebar--primary");
          const bounds = document.activeElement?.getBoundingClientRect();
          return drawer?.checked && trigger?.getAttribute("aria-expanded") === "true" &&
            navigation && !navigation.inert && navigation.contains(document.activeElement) &&
            bounds && bounds.left >= 0 && bounds.right <= innerWidth;
        }""")
    focused = _active_focus_state(page)
    _validate_focused_element(case, focused, require_viewport=True)
    if focused["descriptor"] != "drawer:home":
        raise DocumentationVisualError(
            f"{case}: opening focus is {focused['descriptor']!r}, expected 'drawer:home'.")
    expanded = _rendered_state(page)
    if not expanded["drawerChecked"] or expanded["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: opened drawer state is invalid: {expanded!r}.")

    page.keyboard.press("Escape")
    page.wait_for_function(
        """() => {
          const drawer = document.querySelector("#__drawer");
          const trigger = document.querySelector("[data-vh-drawer-trigger]");
          const navigation = document.querySelector(".md-sidebar--primary");
          return !drawer?.checked && trigger?.getAttribute("aria-expanded") === "false" &&
            navigation?.inert && document.activeElement === trigger;
        }""")
    focused = _active_focus_state(page)
    _validate_focused_element(case, focused, require_viewport=True)
    if focused["descriptor"] != "header:drawer":
        raise DocumentationVisualError(
            f"{case}: closing focus is {focused['descriptor']!r}, expected 'header:drawer'.")
    collapsed = _rendered_state(page)
    if collapsed["drawerChecked"] or collapsed["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: closed drawer state is invalid: {collapsed!r}.")


def _assert_close(case: str, field: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, abs_tol=0.75):
        raise DocumentationVisualError(f"{case}: {field} is {actual:.3f}px, expected {expected:.3f}px.")


def _rectangle_contains(parent: dict[str, float], child: dict[str, float]) -> bool:
    """Allow the same subpixel tolerance used by the geometry assertions."""
    tolerance = 0.75
    return (
        child["x"] >= parent["x"] - tolerance and
        child["x"] + child["width"] <= parent["x"] + parent["width"] + tolerance and
        child["y"] >= parent["y"] - tolerance and
        child["y"] + child["height"] <= parent["y"] + parent["height"] + tolerance)


def _validate_accessibility(axe: Axe, page: Page, case: str) -> str:
    response = axe.run(page).response
    violations = response.get("violations", [])
    if violations:
        summaries = []
        for violation in violations:
            nodes = []
            for node in violation.get("nodes", []):
                target = " ".join(str(part) for part in node.get("target", []))
                failure = " ".join(str(node.get("failureSummary", "")).split())
                nodes.append(f"{target}: {failure}")
            summaries.append(
                f"{violation.get('id')} ({violation.get('impact')}): {violation.get('help')}; "
                f"nodes={nodes!r}")
        raise DocumentationVisualError(f"{case}: Axe violations: {' | '.join(summaries)}")

    engine = response.get("testEngine", {})
    return f"{engine.get('name', 'axe-core')} {engine.get('version', 'unknown')}"


def _prepare_interactive_accessibility_state(
    page: Page,
    state: str,
    viewport: dict[str, Any],
) -> None:
    if state.startswith("search-"):
        page.keyboard.press("Control+K")
        page.wait_for_function(
            "() => document.querySelector('#__search')?.checked && "
            "document.activeElement === document.querySelector('.md-search__input')")
        page.wait_for_function(
            "() => { const searchInner = document.querySelector('.md-search__inner'); "
            "return searchInner && "
            "Number.parseFloat(getComputedStyle(searchInner).opacity) >= 0.999; }")
        if state == "search-open":
            return

        query = "pipeline" if state == "search-results" else "zzzzvoicehubnoresultszzzz"
        page.locator(".md-search__input").press_sequentially(query, delay=20)
        if state == "search-results":
            page.wait_for_function(
                "() => /matching documents/i.test("
                "document.querySelector('.md-search-result__meta')?.textContent || '') && "
                "document.querySelectorAll('.md-search-result article').length > 0")
        else:
            page.wait_for_function(
                "() => /no matching documents/i.test("
                "document.querySelector('.md-search-result__meta')?.textContent || '')")
        return

    if state == "version-open":
        page.locator("[data-vh-version-control] > summary").click()
        page.wait_for_function("document.querySelector('[data-vh-version-control]')?.open")
        return

    if state == "branch-open":
        if viewport["name"] == "mobile":
            raise DocumentationVisualError("branch-open is not a desktop or tablet state.")
        button = page.locator("button[data-vh-nav-toggle='__nav_2']")
        if button.get_attribute("aria-expanded") != "true":
            button.click()
        page.wait_for_function(
            "() => document.querySelector(\"button[data-vh-nav-toggle='__nav_2']\")"
            "?.getAttribute('aria-expanded') === 'true'")
        return

    if state == "drawer-open":
        if viewport["name"] != "mobile":
            raise DocumentationVisualError("drawer-open is only a mobile state.")
        page.locator("[data-vh-drawer-trigger]").click()
        page.wait_for_function(
            """() => {
              const drawer = document.querySelector("#__drawer");
              const navigation = document.querySelector(".md-sidebar--primary");
              const bounds = navigation?.getBoundingClientRect();
              return drawer?.checked && navigation && !navigation.inert && bounds &&
                Math.abs(bounds.left) <= 0.75 && bounds.right <= innerWidth;
            }""")
        return

    raise DocumentationVisualError(f"Unsupported interactive accessibility state: {state!r}.")


def _validate_case(
    *,
    case: str,
    state: dict[str, Any],
    route_expectation: Any,
    viewport: dict[str, Any],
    palette: str,
    hide_secondary: bool = False,
) -> None:
    if state["scheme"] != palette:
        raise DocumentationVisualError(f"{case}: palette is {state['scheme']!r}, expected {palette!r}.")
    colors = PALETTES[palette]
    for field in ("background", "text"):
        if state[field] != colors[field]:
            raise DocumentationVisualError(
                f"{case}: {field} is {state[field]!r}, expected {colors[field]!r}.")
    if state["title"] != route_expectation.title:
        raise DocumentationVisualError(
            f"{case}: title is {state['title']!r}, expected {route_expectation.title!r}.")
    if state["active"] != [route_expectation.active_link]:
        raise DocumentationVisualError(
            f"{case}: active links are {state['active']!r}, "
            f"expected {[route_expectation.active_link]!r}.")
    if state["visibleActiveLabels"] != [route_expectation.active_link]:
        raise DocumentationVisualError(
            f"{case}: visible active labels are {state['visibleActiveLabels']!r}, "
            f"expected one {[route_expectation.active_link]!r}.")
    if tuple(state["checkedBranches"]) != route_expectation.expanded_branches:
        raise DocumentationVisualError(
            f"{case}: checked branches are {state['checkedBranches']!r}, "
            f"expected {route_expectation.expanded_branches!r}.")
    if tuple(state["roots"]) != TOP_LEVEL_NAVIGATION:
        raise DocumentationVisualError(
            f"{case}: roots are {state['roots']!r}, expected {TOP_LEVEL_NAVIGATION!r}.")
    if state["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: document overflow is {state['overflow']}px.")
    if state["drawerChecked"]:
        raise DocumentationVisualError(f"{case}: the navigation drawer starts open.")
    if viewport["width"] >= 960 and state["inactiveFocusableCount"] != 0:
        raise DocumentationVisualError(
            f"{case}: inactive branches expose {state['inactiveFocusableCount']} focusables.")

    if viewport["width"] >= 960:
        control_names = (
            "railControls",
            "productControl",
            "searchControl",
            "utilityControl",
            "versionControl",
            "languageControl",
            "themeControl",
            "sourceControl",
            "sourceIcon",
        )
        missing_controls = [name for name in control_names if state[name] is None]
        if missing_controls:
            raise DocumentationVisualError(f"{case}: desktop rail controls are missing {missing_controls!r}.")
        rail = state["railControls"]
        product = state["productControl"]
        search = state["searchControl"]
        utility = state["utilityControl"]
        _assert_close(case, "rail controls height", rail["height"], 145)
        _assert_close(case, "rail title top padding", product["y"] - rail["y"], 13)
        _assert_close(case, "rail title-to-search gap", search["y"] - product["y"] - product["height"], 15)
        _assert_close(case, "rail search-to-utility gap", utility["y"] - search["y"] - search["height"], 15)
        _assert_close(
            case,
            "rail utility bottom padding",
            rail["y"] + rail["height"] - utility["y"] - utility["height"],
            15,
        )
        utility_center = utility["y"] + utility["height"] / 2
        for control_name in ("versionControl", "languageControl", "themeControl", "sourceControl"):
            control = state[control_name]
            _assert_close(
                case,
                f"{control_name} vertical center",
                control["y"] + control["height"] / 2,
                utility_center,
            )
            _assert_close(case, f"{control_name} height", control["height"], 30)
        source = state["sourceControl"]
        source_children = ("sourceIcon", )
        if state["sourceFacts"] is not None:
            source_children += ("sourceFacts", )
        for child_name in source_children:
            child = state[child_name]
            if not _rectangle_contains(source, child):
                raise DocumentationVisualError(
                    f"{case}: {child_name} escapes the repository control: "
                    f"child={child!r}, source={source!r}.")

    article_width = (
        SPEECHT5_ARTICLE_WIDTHS[viewport["name"]] if hide_secondary else viewport["article_width"])
    for selector, fields in (
        ("article", ("x", )),
        ("header", ("height", )),
        ("primary", ("x", "width")),
    ):
        rectangle = state[selector]
        if rectangle is None:
            raise DocumentationVisualError(f"{case}: {selector} is missing.")
        for field in fields:
            expected_key = f"{selector}_{field}"
            _assert_close(case, expected_key, rectangle[field], viewport[expected_key])
    _assert_close(case, "article_width", state["article"]["width"], article_width)
    secondary = state["secondary"]
    if secondary is None:
        raise DocumentationVisualError(f"{case}: secondary is missing.")
    if hide_secondary:
        if secondary["display"] != "none" or secondary["width"] != 0 or secondary["height"] != 0:
            raise DocumentationVisualError(
                f"{case}: hidden secondary navigation still occupies space: {secondary!r}.")
    else:
        for field in ("x", "width"):
            _assert_close(
                case,
                f"secondary_{field}",
                secondary[field],
                viewport[f"secondary_{field}"],
            )


def _validate_home_state(page: Page, case: str) -> None:
    state = page.evaluate(
        r"""() => {
          const content = document.querySelector(".md-content__inner");
          const normalize = value => value?.trim().replace(/¶$/, "").trim() || "";
          const pathWithHash = link => {
            const url = new URL(link.href);
            return `${url.pathname}${url.hash}`;
          };
          const sectionLinks = id => {
            const links = [];
            let sibling = content?.querySelector(`#${id}`)?.nextElementSibling;
            while (sibling && sibling.tagName !== "H2") {
              links.push(...sibling.querySelectorAll("a[href]"));
              sibling = sibling.nextElementSibling;
            }
            return links;
          };
          const cardLinks = Array.from(
            content?.querySelectorAll(".grid.cards > ul > li a[href]") || []
          );
          const badgeLinks = Array.from(content?.querySelectorAll(".vh-badges a[href]") || []);
          const images = Array.from(content?.querySelectorAll("img") || []);
          const modelPanel = content?.querySelector(".vh-home-models");
          const modelLinks = Array.from(modelPanel?.querySelectorAll(".vh-home-models__actions a[href]") || []);
          const modelStats = Array.from(modelPanel?.querySelectorAll(".vh-home-models__stats > li") || []);
          const teaser = content?.querySelector(".vh-doc-teaser");
          const badges = content?.querySelector(".vh-badges");
          const next = document.querySelector(".md-footer__link--next");
          return {
            headings: Array.from(content?.querySelectorAll("h1, h2, h3") || [])
              .map(heading => [heading.tagName, normalize(heading.textContent)]),
            toc: Array.from(document.querySelectorAll(
              ".md-sidebar--secondary a.md-nav__link"
            )).map(link => normalize(link.textContent)),
            featureLabels: sectionLinks("features").map(link => normalize(link.textContent)),
            featureTargets: sectionLinks("features").map(pathWithHash),
            orderedListRows: Array.from(content?.querySelectorAll("ol") || [])
              .map(list => list.querySelectorAll(":scope > li").length),
            tips: content?.querySelectorAll(".admonition.tip").length || 0,
            cardCount: content?.querySelectorAll(".grid.cards > ul > li").length || 0,
            cardTargets: cardLinks.map(pathWithHash),
            badgeTargets: badgeLinks.map(link => link.href),
            modelTargets: modelLinks.map(pathWithHash),
            modelStats: modelStats.map(item =>
              `${normalize(item.querySelector("strong")?.textContent)} ${normalize(item.querySelector("span")?.textContent)}`
            ),
            modelLabelTarget: modelPanel?.getAttribute("aria-labelledby"),
            modelTitleId: modelPanel?.querySelector("h2")?.id,
            modelPanelAfterTeaser: teaser?.nextElementSibling === modelPanel,
            modelPanelBeforeBadges: modelPanel?.nextElementSibling === badges,
            modelPanelRect: modelPanel?.getBoundingClientRect().toJSON(),
            teaserRect: teaser?.getBoundingClientRect().toJSON(),
            badgesRect: badges?.getBoundingClientRect().toJSON(),
            imageCount: images.length,
            decorativeImages: images.filter(image => image.getAttribute("alt") === "").length,
            badgeAlts: badgeLinks.map(link => normalize(link.querySelector("img")?.alt)),
            tables: content?.querySelectorAll("table").length || 0,
            codeBlocks: content?.querySelectorAll("pre").length || 0,
            pageCopyButtons: content?.querySelectorAll("[data-vh-copy-page]").length || 0,
            editTarget: content?.querySelector("a[href*='/edit/main/docs/index.md']")?.href,
            previousCount: document.querySelectorAll(".md-footer__link--prev").length,
            nextTarget: next ? pathWithHash(next) : null,
            nextLabel: next?.getAttribute("aria-label"),
            text: normalize(content?.textContent).replace(/\s+/g, " "),
          };
        }""")
    headings = tuple(tuple(value) for value in state["headings"])
    if headings != HOME_HEADINGS:
        raise DocumentationVisualError(f"{case}: Home headings are {headings!r}, expected {HOME_HEADINGS!r}.")
    if tuple(state["toc"]) != HOME_TOC:
        raise DocumentationVisualError(
            f"{case}: Home table of contents is {state['toc']!r}, expected {HOME_TOC!r}.")
    if tuple(state["modelTargets"]) != HOME_MODEL_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Home model targets are {state['modelTargets']!r}, "
            f"expected {HOME_MODEL_TARGETS!r}.")
    if tuple(state["modelStats"]) != HOME_MODEL_STATS:
        raise DocumentationVisualError(
            f"{case}: Home model stats are {state['modelStats']!r}, "
            f"expected {HOME_MODEL_STATS!r}.")
    if state["modelLabelTarget"] != state["modelTitleId"] or not state["modelTitleId"]:
        raise DocumentationVisualError(
            f"{case}: Home model panel label target is {state['modelLabelTarget']!r} "
            f"for title {state['modelTitleId']!r}.")
    if not state["modelPanelAfterTeaser"] or not state["modelPanelBeforeBadges"]:
        raise DocumentationVisualError(
            f"{case}: Home model panel is not directly between the teaser and badges.")
    model_rect = state["modelPanelRect"]
    teaser_rect = state["teaserRect"]
    badges_rect = state["badgesRect"]
    if not model_rect or not teaser_rect or not badges_rect:
        raise DocumentationVisualError(f"{case}: Home model panel geometry is incomplete.")
    if model_rect["top"] < teaser_rect["bottom"] - 1 or model_rect["bottom"] > badges_rect["top"] + 1:
        raise DocumentationVisualError(
            f"{case}: Home model panel does not remain between the hero and badges: "
            f"teaser={teaser_rect!r}, model={model_rect!r}, badges={badges_rect!r}.")
    if model_rect["width"] < teaser_rect["width"]:
        raise DocumentationVisualError(
            f"{case}: Home model panel is narrower than the hero: "
            f"model={model_rect['width']!r}, teaser={teaser_rect['width']!r}.")
    if tuple(state["featureLabels"]) != ("Inference", "Trainer", "generate"):
        raise DocumentationVisualError(f"{case}: Home feature labels are {state['featureLabels']!r}.")
    if tuple(state["featureTargets"]) != HOME_FEATURE_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Home feature targets are {state['featureTargets']!r}, "
            f"expected {HOME_FEATURE_TARGETS!r}.")
    if tuple(state["orderedListRows"]) != (2, ) or state["tips"] != 1:
        raise DocumentationVisualError(
            f"{case}: Home design inventory is orderedListRows={state['orderedListRows']!r}, "
            f"tips={state['tips']}; expected [2] and 1.")
    if state["cardCount"] != 13 or tuple(state["cardTargets"]) != HOME_CARD_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Home resource cards are count={state['cardCount']}, "
            f"targets={state['cardTargets']!r}.")
    if tuple(state["badgeTargets"]) != HOME_BADGE_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Home badge targets are {state['badgeTargets']!r}, expected {HOME_BADGE_TARGETS!r}.")
    if (state["imageCount"], state["decorativeImages"], len(state["badgeAlts"])) != (6, 2, 4):
        raise DocumentationVisualError(
            f"{case}: Home image inventory is {state['imageCount']}/"
            f"{state['decorativeImages']}/{len(state['badgeAlts'])}, expected 6/2/4.")
    if any(not label for label in state["badgeAlts"]):
        raise DocumentationVisualError(f"{case}: Home badge alternative text is incomplete.")
    if state["tables"] or state["codeBlocks"] or state["pageCopyButtons"] != 1:
        raise DocumentationVisualError(
            f"{case}: Home component inventory is tables={state['tables']}, "
            f"codeBlocks={state['codeBlocks']}, pageCopyButtons={state['pageCopyButtons']}; "
            "expected 0, 0, and 1.")
    if state["editTarget"] != "https://github.com/kadirnar/voicehub/edit/main/docs/index.md":
        raise DocumentationVisualError(f"{case}: Home edit target is {state['editTarget']!r}.")
    if state["previousCount"] or state["nextTarget"] != "/getting-started/installation/":
        raise DocumentationVisualError(
            f"{case}: Home footer is previousCount={state['previousCount']}, "
            f"nextTarget={state['nextTarget']!r}.")
    if state["nextLabel"] != "Next: Installation":
        raise DocumentationVisualError(f"{case}: Home next label is {state['nextLabel']!r}.")
    for marker in (
            "68 integrations",
            "Find a model for your language and task",
            "34 TTS backends",
            "23 ASR providers",
            "11 VAD providers",
            "configuration, model, and processor contract",
            "Checkpoint weights are downloaded lazily",
            "voicehub[training]",
            "may have separate terms",
    ):
        if marker not in state["text"]:
            raise DocumentationVisualError(f"{case}: rendered Home content is missing {marker!r}.")


def _validate_home_page_copy(page: Page, case: str, key: str) -> None:
    _validate_page_copy(page, f"{case} / Home page copy", key)


def _validate_installation_state(page: Page, case: str) -> None:
    state = page.evaluate(
        r"""() => {
          const content = document.querySelector(".md-content__inner");
          const normalize = value => value?.trim().replace(/¶$/, "").trim() || "";
          const contentLinks = Array.from(content?.querySelectorAll("a[href]") || [])
            .filter(link => !link.closest("pre") &&
              !link.closest(".tabbed-labels") &&
              !link.classList.contains("headerlink") &&
              !link.classList.contains("md-content__button") &&
              !link.closest(".md-source-file"));
          return {
            headings: Array.from(content?.querySelectorAll("h1, h2, h3") || [])
              .map(heading => [heading.tagName, normalize(heading.textContent)]),
            toc: Array.from(document.querySelectorAll(
              ".md-sidebar--secondary a.md-nav__link"
            )).map(link => normalize(link.textContent)),
            tabbedSets: content?.querySelectorAll(".tabbed-set").length || 0,
            codeBlocks: content?.querySelectorAll("pre").length || 0,
            codeCopyButtons: content?.querySelectorAll("button.md-clipboard").length || 0,
            pageCopyButtons: content?.querySelectorAll("[data-vh-copy-page]").length || 0,
            externalTargets: contentLinks
              .map(link => new URL(link.getAttribute("href"), location.href))
              .filter(target => target.origin !== location.origin)
              .map(target => target.href),
            internalTargets: contentLinks
              .map(link => new URL(link.getAttribute("href"), location.href))
              .filter(target => target.origin === location.origin)
              .map(target => `${target.pathname}${target.hash}`),
            editTarget: content?.querySelector("a.md-content__button[href]")?.href || null,
            previousTarget: document.querySelector(".md-footer__link--prev")?.pathname || null,
            previousLabel: document.querySelector(".md-footer__link--prev")
              ?.getAttribute("aria-label") || null,
            nextTarget: document.querySelector(".md-footer__link--next")?.pathname || null,
            nextLabel: document.querySelector(".md-footer__link--next")
              ?.getAttribute("aria-label") || null,
            text: normalize(content?.textContent).replace(/\s+/g, " "),
          };
        }""")
    headings = tuple(tuple(value) for value in state["headings"])
    if headings != INSTALLATION_HEADINGS:
        raise DocumentationVisualError(
            f"{case}: Installation headings are {headings!r}, expected {INSTALLATION_HEADINGS!r}.")
    expected_toc = tuple(label for _, label in INSTALLATION_HEADINGS[1:])
    if tuple(state["toc"]) != expected_toc:
        raise DocumentationVisualError(
            f"{case}: Installation table of contents is {state['toc']!r}, "
            f"expected {expected_toc!r}.")
    if state["tabbedSets"] != 2:
        raise DocumentationVisualError(
            f"{case}: Installation exposes {state['tabbedSets']} platform tab set(s); expected 2.")
    if (state["codeBlocks"], state["codeCopyButtons"], state["pageCopyButtons"]) != (12, 12, 1):
        raise DocumentationVisualError(
            f"{case}: Installation copy inventory is codeBlocks={state['codeBlocks']}, "
            f"codeCopyButtons={state['codeCopyButtons']}, "
            f"pageCopyButtons={state['pageCopyButtons']}; expected 12, 12, and 1.")
    if tuple(state["externalTargets"]) != INSTALLATION_EXTERNAL_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Installation external targets are {state['externalTargets']!r}, "
            f"expected {INSTALLATION_EXTERNAL_TARGETS!r}.")
    if tuple(state["internalTargets"]) != INSTALLATION_INTERNAL_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Installation internal targets are {state['internalTargets']!r}, "
            f"expected {INSTALLATION_INTERNAL_TARGETS!r}.")
    if state["editTarget"] != (
            "https://github.com/kadirnar/voicehub/edit/main/docs/getting-started/installation.md"):
        raise DocumentationVisualError(f"{case}: Installation edit target is {state['editTarget']!r}.")
    if (state["previousTarget"], state["previousLabel"]) != ("/", "Previous: Overview"):
        raise DocumentationVisualError(
            f"{case}: Installation previous action is "
            f"{(state['previousTarget'], state['previousLabel'])!r}.")
    if (state["nextTarget"], state["nextLabel"]) != ("/getting-started/quickstart/", "Next: Quickstart"):
        raise DocumentationVisualError(
            f"{case}: Installation next action is "
            f"{(state['nextTarget'], state['nextLabel'])!r}.")
    for marker in (
            "voicehub @ git+https://github.com/kadirnar/voicehub.git@main",
            "voicehub[training] @ git+https://github.com/kadirnar/voicehub.git@main",
            "VOICEHUB_OFFLINE=1",
            "local_files_only=True",
            "Linux",
            "macOS",
            "Windows",
    ):
        if marker not in state["text"]:
            raise DocumentationVisualError(f"{case}: rendered Installation content is missing {marker!r}.")


def _validate_inference_state(page: Page, case: str) -> None:
    state = page.evaluate(
        r"""() => {
          const content = document.querySelector(".md-content__inner");
          const normalize = value => value?.trim().replace(/¶$/, "").trim() || "";
          return {
            headings: Array.from(content?.querySelectorAll("h1, h2, h3") || [])
              .map(heading => [heading.tagName, normalize(heading.textContent)]),
            toc: Array.from(document.querySelectorAll(
              ".md-sidebar--secondary a.md-nav__link"
            )).map(link => normalize(link.textContent)),
            tables: content?.querySelectorAll("table").length || 0,
            codeBlocks: content?.querySelectorAll("pre").length || 0,
            copyButtons: content?.querySelectorAll("button.md-clipboard").length || 0,
            text: normalize(content?.textContent).replace(/\s+/g, " "),
          };
        }""")
    headings = tuple(tuple(value) for value in state["headings"])
    if headings != INFERENCE_HEADINGS:
        raise DocumentationVisualError(
            f"{case}: Inference headings are {headings!r}, expected {INFERENCE_HEADINGS!r}.")
    expected_toc = tuple(label for _, label in INFERENCE_HEADINGS[1:])
    if tuple(state["toc"]) != expected_toc:
        raise DocumentationVisualError(
            f"{case}: Inference table of contents is {state['toc']!r}, "
            f"expected {expected_toc!r}.")
    if state["tables"] != 1 or state["codeBlocks"] != 6 or state["copyButtons"] != 6:
        raise DocumentationVisualError(
            f"{case}: Inference component inventory is tables={state['tables']}, "
            f"codeBlocks={state['codeBlocks']}, copyButtons={state['copyButtons']}; "
            "expected 1 table, 6 code blocks, and 6 copy buttons.")
    for marker in (
            "TTSOutput",
            "ASROutput",
            "VADOutput",
            "duration < 10",
            "does not provide a universal vectorized batch contract",
            "speech_pipeline.load()",
            "list_model_specs(task=...)",
    ):
        if marker not in state["text"]:
            raise DocumentationVisualError(f"{case}: rendered Inference content is missing {marker!r}.")


def _validate_code_copy(page: Page, case: str, key: str, *, wait_for_idle: bool = False) -> None:
    button = page.locator(".md-content__inner button.md-clipboard:visible").first
    if button.count() != 1:
        raise DocumentationVisualError(f"{case}: page has no first code-copy button.")
    expected = button.locator("xpath=preceding-sibling::pre[1]/code").text_content()
    if not expected:
        raise DocumentationVisualError(f"{case}: first code block is empty.")

    origin = page.evaluate("location.origin")
    page.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=origin)
    page.evaluate("navigator.clipboard.writeText('voicehub-copy-sentinel')")
    button.scroll_into_view_if_needed()
    button.focus()
    focused = button.evaluate(
        """element => {
          const bounds = element.getBoundingClientRect();
          const style = getComputedStyle(element);
          return {
            active: document.activeElement === element,
            visible: bounds.width > 0 && bounds.height > 0 &&
              style.display !== "none" && style.visibility !== "hidden",
            withinViewport: bounds.left >= 0 && bounds.right <= innerWidth &&
              bounds.top >= 0 && bounds.bottom <= innerHeight,
            outlineStyle: style.outlineStyle,
            outlineWidth: style.outlineWidth,
            outlineOffset: style.outlineOffset,
          };
        }""")
    if not focused["active"] or not focused["visible"] or not focused["withinViewport"]:
        raise DocumentationVisualError(f"{case}: code-copy focus state is {focused!r}.")
    actual_outline = (
        focused["outlineStyle"],
        focused["outlineWidth"],
        focused["outlineOffset"],
    )
    if actual_outline != ("solid", "2px", "2px"):
        raise DocumentationVisualError(f"{case}: code-copy outline is {actual_outline!r}.")

    page.keyboard.press(key)
    page.wait_for_function(
        "element => element.classList.contains('md-clipboard--active')",
        arg=button.element_handle(),
    )
    copied = page.evaluate("navigator.clipboard.readText()")
    if copied.rstrip() != expected.rstrip():
        raise DocumentationVisualError(f"{case}: copied {copied!r}, expected {expected!r}.")
    if not button.evaluate("element => document.activeElement === element"):
        raise DocumentationVisualError(f"{case}: code-copy activation moved focus.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: code-copy activation introduced overflow.")
    if wait_for_idle:
        page.wait_for_function(
            "element => !element.classList.contains('md-clipboard--active')",
            arg=button.element_handle(),
            timeout=5000,
        )
        if not button.evaluate("element => document.activeElement === element"):
            raise DocumentationVisualError(f"{case}: code-copy idle state moved focus.")


def _validate_inference_code_copy(page: Page, case: str, key: str) -> None:
    _validate_code_copy(page, f"{case} / Inference code copy", key)


def _validate_installation_code_copy(page: Page, case: str, key: str) -> None:
    _validate_code_copy(page, f"{case} / Installation code copy", key, wait_for_idle=True)


def _validate_model_index_state(page: Page, case: str, viewport: dict[str, Any]) -> None:
    state = page.evaluate(
        r"""() => {
          const content = document.querySelector(".md-content__inner");
          const normalize = value => value?.trim().replace(/¶$/, "").trim() || "";
          const explorer = content?.querySelector("[data-vh-model-explorer]");
          const cards = Array.from(explorer?.querySelectorAll("[data-vh-model-card]") || []);
          const providerLinks = cards.map(card =>
            card.querySelector(".vh-model-card__heading h2 a[href]")
          ).filter(Boolean);
          return {
            headings: Array.from(content?.querySelectorAll("h1, h2, h3") || [])
              .filter(heading => !heading.closest(
                "[data-vh-model-card], [data-vh-model-empty]"
              ))
              .map(heading => [heading.tagName, normalize(heading.textContent)]),
            toc: Array.from(document.querySelectorAll(
              ".md-sidebar--secondary a.md-nav__link"
            )).map(link => normalize(link.textContent)),
            enhanced: explorer?.dataset.enhanced,
            cardCount: cards.length,
            visibleCardCount: cards.filter(card => !card.hidden).length,
            intersectingCardCount: cards.filter(card => {
              if (card.hidden) return false;
              const bounds = card.getBoundingClientRect();
              const style = getComputedStyle(card);
              return bounds.width > 0 && bounds.height > 0 &&
                bounds.right > 0 && bounds.left < innerWidth &&
                bounds.bottom > 0 && bounds.top < innerHeight &&
                style.display !== "none" && style.visibility !== "hidden";
            }).length,
            resultCount: explorer?.querySelector("[data-vh-model-result-count]")?.textContent,
            filterSelectNames: Array.from(
              explorer?.querySelectorAll("[data-vh-model-select]") || []
            ).map(select => select.name),
            sortOptionValues: Array.from(
              explorer?.querySelector("[data-vh-model-sort]")?.options || []
            ).map(option => option.value),
            parameterCounts: cards.map(card => card.getAttribute("data-parameter-count")),
            parameterBands: cards.map(card => card.getAttribute("data-parameter-band")),
            featureFilterCount: explorer?.querySelectorAll(
              'input[data-vh-model-checkbox][name="feature"]'
            ).length || 0,
            resourceFilterCount: explorer?.querySelectorAll(
              'input[data-vh-model-checkbox][name="resource"]'
            ).length || 0,
            languageOptionCount: explorer?.querySelector(
              'select[name="language"]'
            )?.options.length || 0,
            codeBlocks: content?.querySelectorAll("pre").length || 0,
            codeCopyButtons: content?.querySelectorAll("button[data-vh-code-copy]").length || 0,
            providerLabels: providerLinks.map(link => normalize(link.textContent)),
            providerHrefs: providerLinks.map(link => link.getAttribute("href")),
            text: normalize(content?.textContent).replace(/\s+/g, " "),
          };
        }""")
    headings = tuple(tuple(value) for value in state["headings"])
    if headings != MODEL_INDEX_HEADINGS:
        raise DocumentationVisualError(
            f"{case}: model-index headings are {headings!r}, expected {MODEL_INDEX_HEADINGS!r}.")
    if tuple(state["toc"]) != MODEL_INDEX_TOC:
        raise DocumentationVisualError(
            f"{case}: model-index table of contents is {state['toc']!r}, "
            f"expected {MODEL_INDEX_TOC!r}.")
    if state["enhanced"] != "true" or state["cardCount"] != 68 or state["visibleCardCount"] != 68:
        raise DocumentationVisualError(
            f"{case}: model explorer state is enhanced={state['enhanced']!r}, "
            f"cards={state['cardCount']!r}, visible={state['visibleCardCount']!r}; "
            "expected 'true', 68, and 68.")
    minimum_intersections = MODEL_INDEX_MINIMUM_INTERSECTING_CARDS[viewport["name"]]
    if state["intersectingCardCount"] < minimum_intersections:
        raise DocumentationVisualError(
            f"{case}: only {state['intersectingCardCount']!r} model cards intersect the "
            f"initial viewport; expected at least {minimum_intersections!r}.")
    if state["resultCount"] != "68":
        raise DocumentationVisualError(
            f"{case}: model explorer result count is {state['resultCount']!r}, expected '68'.")
    expected_selects = (
        "language",
        "task",
        "parameters",
        "training",
        "checkpoint",
        "license",
        "architecture",
    )
    if tuple(state["filterSelectNames"]) != expected_selects:
        raise DocumentationVisualError(
            f"{case}: model explorer selects are {state['filterSelectNames']!r}, "
            f"expected {expected_selects!r}.")
    expected_sorts = (
        "name",
        "task",
        "parameters-desc",
        "parameters-asc",
        "languages",
        "languages-asc",
        "training",
    )
    if tuple(state["sortOptionValues"]) != expected_sorts:
        raise DocumentationVisualError(
            f"{case}: model explorer sort options are {state['sortOptionValues']!r}, "
            f"expected {expected_sorts!r}.")
    if any(value is None for value in state["parameterCounts"]):
        raise DocumentationVisualError(f"{case}: a model card has no parameter-count attribute.")
    if any(not value for value in state["parameterBands"]):
        raise DocumentationVisualError(f"{case}: a model card has no parameter-band value.")
    declared_parameter_counts = [value for value in state["parameterCounts"] if value != ""]
    if not declared_parameter_counts or not any(value == "" for value in state["parameterCounts"]):
        raise DocumentationVisualError(
            f"{case}: parameter metadata must exercise both declared and unknown counts.")
    if any(not value.isdigit() for value in declared_parameter_counts):
        raise DocumentationVisualError(f"{case}: model-card parameter counts are not non-negative integers.")
    if state["featureFilterCount"] != 11 or state["resourceFilterCount"] != 2:
        raise DocumentationVisualError(
            f"{case}: model explorer checkbox counts are "
            f"{state['featureFilterCount']!r} features and {state['resourceFilterCount']!r} resources; "
            "expected 11 and 2.")
    if state["languageOptionCount"] < 700:
        raise DocumentationVisualError(
            f"{case}: model explorer exposes only {state['languageOptionCount']!r} language options.")
    if state["codeBlocks"] != 1 or state["codeCopyButtons"] != 1:
        raise DocumentationVisualError(
            f"{case}: model-index code inventory is codeBlocks={state['codeBlocks']}, "
            f"copyButtons={state['codeCopyButtons']}; expected 1 and 1.")
    labels = state["providerLabels"]
    hrefs = state["providerHrefs"]
    if len(labels) != 68 or len(set(hrefs)) != 68:
        raise DocumentationVisualError(
            f"{case}: model-index provider inventory has {len(labels)} labels and "
            f"{len(set(hrefs))} unique links; expected 68 of each.")
    invalid_labels = [label for label in labels if not label[:1].isupper()]
    if invalid_labels:
        raise DocumentationVisualError(
            f"{case}: model-index labels are not uppercase-first: {invalid_labels!r}.")
    for marker in (
            "Find the right speech model",
            "Search models",
            "Any language",
            "list_model_specs()",
            "training matrix",
            "optimization catalog",
    ):
        if marker not in state["text"]:
            raise DocumentationVisualError(f"{case}: rendered model index is missing {marker!r}.")


def _validate_page_copy(page: Page, case: str, key: str, *, wait_for_idle: bool = False) -> None:
    button = page.locator("[data-vh-copy-page]")
    if button.count() != 1:
        raise DocumentationVisualError(f"{case}: expected one page-copy button, found {button.count()}.")
    article = page.locator(".md-content__inner")
    expected = article.evaluate(
        """element => {
          const readablePage = element.cloneNode(true);
          readablePage.querySelectorAll(
            ".md-content__button, .headerlink, .md-clipboard, .md-source-file"
          ).forEach(item => item.remove());
          return readablePage.innerText.trim();
        }""")
    origin = page.evaluate("location.origin")
    page.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=origin)
    page.evaluate("navigator.clipboard.writeText('voicehub-page-copy-sentinel')")
    button.scroll_into_view_if_needed()
    # Playwright's select/click helpers leave Chromium in pointer modality.
    # Restore keyboard modality before asserting the keyboard-only focus ring.
    page.keyboard.press("Tab")
    button.focus()
    focused = button.evaluate(
        """element => {
          const bounds = element.getBoundingClientRect();
          const style = getComputedStyle(element);
          return {
            active: document.activeElement === element,
            visible: bounds.width > 0 && bounds.height > 0 &&
              style.display !== "none" && style.visibility !== "hidden",
            withinViewport: bounds.left >= 0 && bounds.right <= innerWidth &&
              bounds.top >= 0 && bounds.bottom <= innerHeight,
            outlineStyle: style.outlineStyle,
            outlineWidth: style.outlineWidth,
            outlineOffset: style.outlineOffset,
          };
        }""")
    if not focused["active"] or not focused["visible"] or not focused["withinViewport"]:
        raise DocumentationVisualError(f"{case}: page-copy focus state is {focused!r}.")
    actual_outline = (
        focused["outlineStyle"],
        focused["outlineWidth"],
        focused["outlineOffset"],
    )
    if actual_outline != ("solid", "2px", "2px"):
        raise DocumentationVisualError(f"{case}: page-copy outline is {actual_outline!r}.")
    page.keyboard.press(key)
    page.wait_for_function(
        'element => element.getAttribute("aria-busy") === "false" && '
        'element.querySelector("[data-vh-copy-page-label]")?.textContent === "Copied"',
        arg=button.element_handle(),
    )
    copied = page.evaluate("navigator.clipboard.readText()")
    if copied != expected:
        raise DocumentationVisualError(
            f"{case}: page copy produced {len(copied)} characters, expected {len(expected)}.")
    if not button.evaluate("element => document.activeElement === element"):
        raise DocumentationVisualError(f"{case}: page-copy activation moved focus.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: page-copy activation introduced overflow.")
    if wait_for_idle:
        page.wait_for_function(
            'element => element.getAttribute("aria-busy") === "false" && '
            '!element.classList.contains("md-clipboard--active") && '
            'element.querySelector("[data-vh-copy-page-label]")?.textContent === "Copy page"',
            arg=button.element_handle(),
            timeout=5000,
        )
        if not button.evaluate("element => document.activeElement === element"):
            raise DocumentationVisualError(f"{case}: page-copy idle state moved focus.")


def _validate_installation_page_copy(page: Page, case: str, key: str) -> None:
    _validate_page_copy(page, f"{case} / Installation page copy", key, wait_for_idle=True)


def _validate_model_index_page_copy(page: Page, case: str, key: str) -> None:
    _validate_page_copy(page, f"{case} / model-index page copy", key)


def _validate_model_explorer_filters(page: Page, case: str) -> None:
    explorer = page.locator("[data-vh-model-explorer]")
    count = explorer.locator("[data-vh-model-result-count]")
    language = explorer.locator('select[name="language"]')
    task = explorer.locator('select[name="task"]')
    parameters_select = explorer.locator('select[name="parameters"]')
    query = explorer.locator("[data-vh-model-query]")
    feature = explorer.locator('input[name="feature"][value="voice-cloning"]')
    details = explorer.locator(".vh-model-filters__advanced")
    advanced_count = explorer.locator("[data-vh-model-advanced-count]")

    language.select_option("tr")
    page.wait_for_function("element => element.textContent === '16'", arg=count.element_handle())
    task.select_option("text-to-speech")
    details.locator(":scope > summary").click()
    feature.check()
    page.wait_for_function("element => element.textContent === '6'", arg=count.element_handle())
    if advanced_count.inner_text() != "1" or advanced_count.is_hidden():
        raise DocumentationVisualError(
            f"{case}: advanced-filter count is not visible with one selected filter.")
    expected_models = ("Chatterbox", "FishTTS", "MossTTS", "OmniVoice", "VoxCPM", "XTTS")
    visible_models = tuple(explorer.locator(".vh-model-card:not([hidden]) h2").all_text_contents())
    if visible_models != expected_models:
        raise DocumentationVisualError(
            f"{case}: Turkish TTS voice-cloning models are {visible_models!r}, "
            f"expected {expected_models!r}.")

    query.fill("Turkish voice cloning")
    page.wait_for_function("element => element.textContent === '6'", arg=count.element_handle())
    parameters = page.evaluate("() => Object.fromEntries(new URLSearchParams(location.search).entries())")
    expected_parameters = {
        "model_q": "Turkish voice cloning",
        "model_language": "tr",
        "model_task": "text-to-speech",
        "model_features": "voice-cloning",
    }
    if parameters != expected_parameters:
        raise DocumentationVisualError(
            f"{case}: model explorer URL state is {parameters!r}, expected {expected_parameters!r}.")
    if explorer.locator("[data-vh-model-active-filters] button").count() != 4:
        raise DocumentationVisualError(f"{case}: model explorer did not render four active filters.")

    query.fill("no-such-model-zzzz")
    empty = explorer.locator("[data-vh-model-empty]")
    if not empty.is_visible() or count.inner_text() != "0":
        raise DocumentationVisualError(f"{case}: model explorer empty state is not visible at zero results.")
    empty.locator("[data-vh-model-clear]").click()
    page.wait_for_function("element => element.textContent === '68'", arg=count.element_handle())
    if not advanced_count.is_hidden():
        raise DocumentationVisualError(
            f"{case}: advanced-filter count remains visible after clearing filters.")

    parameter_options = parameters_select.locator("option").evaluate_all(
        "options => options.map(option => option.value).filter(Boolean)")
    if not parameter_options:
        raise DocumentationVisualError(f"{case}: parameter filter has no selectable bands.")
    for selected_parameter_band in parameter_options:
        parameters_select.select_option(selected_parameter_band)
        parameter_filtered_cards = explorer.locator(".vh-model-card:not([hidden])")
        if parameter_filtered_cards.count() == 0:
            raise DocumentationVisualError(
                f"{case}: parameter band {selected_parameter_band!r} returned no models.")
        filtered_bands = parameter_filtered_cards.evaluate_all(
            "cards => cards.map(card => card.dataset.parameterBand)")
        if any(value != selected_parameter_band for value in filtered_bands):
            raise DocumentationVisualError(
                f"{case}: parameter filter {selected_parameter_band!r} returned bands "
                f"{filtered_bands!r}.")
        parameter_url_state = page.evaluate(
            "() => Object.fromEntries(new URLSearchParams(location.search).entries())")
        if parameter_url_state != {"model_parameters": selected_parameter_band}:
            raise DocumentationVisualError(f"{case}: parameter-filter URL state is {parameter_url_state!r}.")
    parameters_select.select_option("")
    page.wait_for_function("element => element.textContent === '68'", arg=count.element_handle())

    sort = explorer.locator("[data-vh-model-sort]")
    sort.select_option("languages")
    first_card = explorer.locator(".vh-model-card:not([hidden])").first
    if (first_card.get_attribute("data-model-type") != "omnivoice" or
            first_card.get_attribute("data-language-count") != "646"):
        raise DocumentationVisualError(f"{case}: language-coverage sorting did not place OmniVoice first.")
    sort.select_option("languages-asc")
    language_counts = explorer.locator(".vh-model-card:not([hidden])").evaluate_all(
        "cards => cards.map(card => Number(card.dataset.languageCount))")
    if language_counts != sorted(language_counts):
        raise DocumentationVisualError(f"{case}: ascending language-coverage order is {language_counts!r}.")

    sort.select_option("task")
    task_order = explorer.locator(".vh-model-card:not([hidden])").evaluate_all(
        "cards => cards.map(card => [card.dataset.task, card.dataset.name])")
    if task_order != sorted(task_order):
        raise DocumentationVisualError(f"{case}: task sorting order is invalid.")

    for sort_value, reverse in (("parameters-desc", True), ("parameters-asc", False)):
        sort.select_option(sort_value)
        parameter_order = explorer.locator(".vh-model-card:not([hidden])").evaluate_all(
            "cards => cards.map(card => ({ modelType: card.dataset.modelType, "
            "count: card.dataset.parameterCount }))")
        unknown_index = next(
            (index for index, model in enumerate(parameter_order) if model["count"] == ""),
            None,
        )
        if unknown_index is None:
            raise DocumentationVisualError(f"{case}: {sort_value} did not expose an unknown parameter count.")
        if any(model["count"] != "" for model in parameter_order[unknown_index:]):
            raise DocumentationVisualError(
                f"{case}: {sort_value} placed a known parameter count after an unknown one.")
        known_counts = [int(model["count"]) for model in parameter_order[:unknown_index]]
        if known_counts != sorted(known_counts, reverse=reverse):
            raise DocumentationVisualError(
                f"{case}: {sort_value} produced parameter counts {known_counts!r}.")
    sort.select_option("name")


def _validate_speecht5_state(page: Page, case: str, viewport: dict[str, Any]) -> None:
    state = page.evaluate(
        r"""() => {
          const content = document.querySelector(".md-content__inner");
          const detail = content?.querySelector("[data-vh-model-detail]");
          const normalize = value => value?.trim().replace(/¶$/, "").trim()
            .replace(/\s+/g, " ") || "";
          const visible = element => {
            if (!element) return false;
            const bounds = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return bounds.width > 0 && bounds.height > 0 && style.display !== "none" &&
              style.visibility !== "hidden";
          };
          const rectangle = element => {
            if (!element) return null;
            const bounds = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return {
              x: bounds.x,
              y: bounds.y,
              width: bounds.width,
              height: bounds.height,
              right: bounds.right,
              bottom: bounds.bottom,
              display: style.display,
              position: style.position,
              overflowX: style.overflowX,
              clientWidth: element.clientWidth,
              scrollWidth: element.scrollWidth,
            };
          };
          const tables = Array.from(content?.querySelectorAll("table") || []);
          const secondary = document.querySelector(".md-sidebar--secondary");
          const facts = detail?.querySelector("[data-vh-model-facts]");
          const factsDisclosure = facts?.querySelector("[data-vh-model-facts-disclosure]");
          const factsSummary = factsDisclosure?.querySelector(":scope > summary");
          const modelContent = detail?.querySelector(".vh-model-detail__content");
          const modelNamespace = detail?.querySelector(".vh-model-detail__namespace");
          const parameterNote = detail?.querySelector(".vh-model-detail__parameter-note");
          return {
            counts: {
              detail: content?.querySelectorAll("[data-vh-model-detail]").length || 0,
              hero: detail?.querySelectorAll("[data-vh-model-hero]").length || 0,
              h1: detail?.querySelectorAll("h1").length || 0,
              tabs: detail?.querySelectorAll(".vh-model-detail__tabs").length || 0,
              layout: detail?.querySelectorAll(".vh-model-detail__layout").length || 0,
              content: detail?.querySelectorAll(".vh-model-detail__content").length || 0,
              facts: detail?.querySelectorAll("[data-vh-model-facts]").length || 0,
              factsDisclosure: detail?.querySelectorAll(
                "[data-vh-model-facts-disclosure]"
              ).length || 0,
            },
            dataset: detail ? {
              modelType: detail.dataset.modelType,
              task: detail.dataset.task,
              training: detail.dataset.training,
              parameterCount: detail.dataset.parameterCount,
            } : null,
            namespace: {
              avatar: normalize(modelNamespace?.querySelector(
                ".vh-model-detail__owner-avatar"
              )?.textContent),
              avatarHidden: modelNamespace?.querySelector(
                ".vh-model-detail__owner-avatar"
              )?.getAttribute("aria-hidden"),
              owner: normalize(modelNamespace?.querySelector("a")?.textContent),
              repository: normalize(modelNamespace?.querySelector("strong")?.textContent),
              href: modelNamespace?.querySelector("a")?.getAttribute("href"),
            },
            tags: Array.from(detail?.querySelectorAll(".vh-model-detail__chip") || [])
              .map(chip => normalize(chip.textContent)),
            actions: Array.from(detail?.querySelectorAll("[data-vh-model-action]") || [])
              .map(link => [link.dataset.vhModelAction, link.getAttribute("href")]),
            tabs: Array.from(detail?.querySelectorAll("[data-vh-model-tab]") || [])
              .map(link => [
                link.dataset.vhModelTab,
                link.getAttribute("href"),
                normalize(link.textContent),
                link.getAttribute("aria-current"),
                link.getAttribute("role"),
              ]),
            tabNavigation: {
              ariaLabel: detail?.querySelector(".vh-model-detail__tabs")?.getAttribute("aria-label"),
              role: detail?.querySelector(".vh-model-detail__tabs")?.getAttribute("role"),
            },
            facts: Array.from(facts?.querySelectorAll(".vh-model-detail__facts > div") || [])
              .map(row => {
                const value = row.querySelector("dd");
                return [
                  normalize(row.querySelector("dt")?.textContent),
                  normalize(value?.textContent),
                  value?.getAttribute("aria-describedby"),
                ];
              }),
            factsDisclosure: factsDisclosure ? {
              open: factsDisclosure.open,
              summaryCount: factsDisclosure.querySelectorAll(":scope > summary").length,
              labelledBy: factsDisclosure.getAttribute("aria-labelledby"),
              asideLabelledBy: facts?.getAttribute("aria-labelledby"),
              heading: normalize(facts?.querySelector(":scope > h2")?.textContent),
              headingId: facts?.querySelector(":scope > h2")?.id,
              toggleLabel: normalize(factsDisclosure.querySelector(
                ":scope > summary > span"
              )?.textContent),
              summaryTabIndex: factsSummary?.tabIndex,
              summaryVisible: visible(factsSummary),
              factsBeforeContent: Boolean(
                facts && modelContent &&
                facts.compareDocumentPosition(modelContent) & Node.DOCUMENT_POSITION_FOLLOWING
              ),
            } : null,
            parameterNote: parameterNote ? {
              id: parameterNote.id,
              text: normalize(parameterNote.textContent),
              visible: visible(parameterNote),
              describedElements: detail?.querySelectorAll(
                `[aria-describedby~="${CSS.escape(parameterNote.id)}"]`
              ).length || 0,
            } : null,
            copy: (() => {
              const button = detail?.querySelector("button[data-vh-copy-model-id]");
              const label = button?.querySelector("[data-vh-copy-model-id-label]");
              const descriptionId = button?.getAttribute("aria-describedby");
              return button ? {
                count: detail.querySelectorAll("button[data-vh-copy-model-id]").length,
                type: button.getAttribute("type"),
                modelId: button.dataset.modelId,
                ariaLabel: button.getAttribute("aria-label"),
                ariaBusy: button.getAttribute("aria-busy"),
                descriptionId,
                description: normalize(document.getElementById(descriptionId)?.textContent),
                label: normalize(label?.textContent),
                live: label?.getAttribute("aria-live"),
                atomic: label?.getAttribute("aria-atomic"),
              } : null;
            })(),
            apiCards: Array.from(detail?.querySelectorAll("[data-vh-model-api-card]") || [])
              .map(card => ({
                kind: card.dataset.vhModelApiCard,
                badge: normalize(card.querySelector(".vh-model-api-card__badge")?.textContent),
                heading: normalize(card.querySelector("h3")?.textContent),
                source: card.querySelector(".vh-model-api-card__source")?.getAttribute("href"),
                signature: (card.querySelector(".vh-model-api-card__signature code")?.innerText || "")
                  .split("\n").map(line => line.trim()).filter(Boolean),
                parameterHeading: normalize(card.querySelector("h4")?.textContent),
                parameters: Array.from(card.querySelectorAll(
                  ".vh-model-api-card__parameters li"
                )).map(item => normalize(item.textContent)),
              })),
            headings: Array.from(content?.querySelectorAll("h1, h2, h3") || [])
              .map(heading => [heading.tagName, normalize(heading.textContent)]),
            toc: visible(secondary) ? Array.from(secondary.querySelectorAll(
              "a.md-nav__link"
            )).map(link => normalize(link.textContent)) : [],
            tableRows: tables.map(table => table.querySelectorAll("tbody tr").length),
            codeBlocks: content?.querySelectorAll("pre").length || 0,
            codeCopyButtons: content?.querySelectorAll("button[data-vh-code-copy]").length || 0,
            text: normalize(content?.textContent),
            geometry: {
              article: rectangle(content),
              detail: rectangle(detail),
              hero: rectangle(detail?.querySelector("[data-vh-model-hero]")),
              tabs: rectangle(detail?.querySelector(".vh-model-detail__tabs")),
              layout: rectangle(detail?.querySelector(".vh-model-detail__layout")),
              content: rectangle(modelContent),
              facts: rectangle(facts),
              factsDisclosure: rectangle(factsDisclosure),
              factsSummary: rectangle(factsSummary),
            },
          };
        }""")
    if state["counts"] != {"detail": 1, "hero": 1, "h1": 1, "tabs": 1, "layout": 1, "content": 1, "facts": 1,
                           "factsDisclosure": 1}:
        raise DocumentationVisualError(f"{case}: SpeechT5 structural inventory is {state['counts']!r}.")
    expected_dataset = {
        "modelType": "speecht5",
        "task": "text-to-speech",
        "training": "native",
        "parameterCount": "",
    }
    if state["dataset"] != expected_dataset:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 model metadata is {state['dataset']!r}, expected {expected_dataset!r}.")
    expected_namespace = {
        "avatar": "MI",
        "avatarHidden": "true",
        "owner": "microsoft",
        "repository": "speecht5_tts",
        "href": "https://huggingface.co/microsoft",
    }
    if state["namespace"] != expected_namespace:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 namespace is {state['namespace']!r}, expected {expected_namespace!r}.")
    expected_tags = (
        "Text to speech",
        "VoiceHub-native",
        "speecht5",
        "Parameters: Not reported",
        "Language: en",
        "Training: native",
        "License: Checkpoint-specific",
    )
    if tuple(state["tags"]) != expected_tags:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 tags are {state['tags']!r}, expected {expected_tags!r}.")
    if tuple(tuple(value) for value in state["actions"]) != SPEECHT5_ACTIONS:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 actions are {state['actions']!r}, expected {SPEECHT5_ACTIONS!r}.")
    tabs = tuple((value, href, label) for value, href, label, _, _ in state["tabs"])
    expected_tabs = tuple((value, href, label) for value, href, label, _ in SPEECHT5_TABS)
    current_tabs = tuple(value for value, _, _, current, _ in state["tabs"] if current == "location")
    if (tabs != expected_tabs or len(current_tabs) != 1 or
            any(current not in (None, "location") for _, _, _, current, _ in state["tabs"]) or
            any(role is not None for *_, role in state["tabs"])):
        raise DocumentationVisualError(
            f"{case}: SpeechT5 section navigation is {state['tabs']!r}, expected ordinary links "
            f"{expected_tabs!r} with exactly one current location.")
    if state["tabNavigation"] != {"ariaLabel": "Model sections", "role": None}:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 section-navigation semantics are {state['tabNavigation']!r}.")
    expected_disclosure = {
        "open": viewport["name"] == "desktop",
        "summaryCount": 1,
        "labelledBy": "vh-model-facts-title-speecht5",
        "asideLabelledBy": "vh-model-facts-title-speecht5",
        "heading": "Model facts",
        "headingId": "vh-model-facts-title-speecht5",
        "toggleLabel": "Toggle model facts",
        "summaryTabIndex": 0,
        "summaryVisible": viewport["name"] != "desktop",
        "factsBeforeContent": True,
    }
    if state["factsDisclosure"] != expected_disclosure:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 facts disclosure is {state['factsDisclosure']!r}, "
            f"expected {expected_disclosure!r}.")
    fact_labels = tuple(row[0] for row in state["facts"])
    if fact_labels != SPEECHT5_FACT_LABELS:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 fact labels are {fact_labels!r}, expected {SPEECHT5_FACT_LABELS!r}.")
    fact_values = {label: value for label, value, _ in state["facts"]}
    expected_fact_values = {
        "Task": "Text to speech",
        "Parameters": "Not reported",
        "Architecture": "speecht5",
        "Runtime": "VoiceHub-native",
        "Languages": "en",
        "Training": "native",
        "License": "Checkpoint-specific",
        "Default checkpoint": "microsoft/speecht5_tts",
    }
    if any(fact_values.get(label) != value for label, value in expected_fact_values.items()):
        raise DocumentationVisualError(
            f"{case}: SpeechT5 fact values are {fact_values!r}, expected {expected_fact_values!r}.")
    note = state["parameterNote"]
    if (not note or note["id"] != "vh-model-parameters-note-speecht5" or not note["visible"] or
            note["describedElements"] != 2 or "Not reported" not in note["text"] or
            next(row[2] for row in state["facts"] if row[0] == "Parameters") != note["id"]):
        raise DocumentationVisualError(f"{case}: SpeechT5 parameter note is invalid: {note!r}.")
    expected_copy = {
        "count": 1,
        "type": "button",
        "modelId": "microsoft/speecht5_tts",
        "ariaLabel": "Copy model ID",
        "ariaBusy": "false",
        "descriptionId": "vh-model-checkpoint-speecht5",
        "description": "microsoft/speecht5_tts",
        "label": "Copy model ID",
        "live": "polite",
        "atomic": "true",
    }
    if state["copy"] != expected_copy:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 model-ID copy semantics are {state['copy']!r}, "
            f"expected {expected_copy!r}.")
    expected_api_cards = (
        {
            "kind":
            "configuration",
            "badge":
            "Configuration",
            "heading":
            "SpeechT5Config",
            "source": (
                "https://github.com/kadirnar/voicehub/blob/main/"
                "voicehub/models/speecht5/configuration_speecht5.py"),
            "signature": ["SpeechT5Config(**config_kwargs)"],
            "parameterHeading":
            "Parameters",
            "parameters": ["**config_kwargs — Configuration fields validated by SpeechT5Config."],
        },
        {
            "kind":
            "model",
            "badge":
            "Model",
            "heading":
            "SpeechT5ForTextToSpeech",
            "source": (
                "https://github.com/kadirnar/voicehub/blob/main/"
                "voicehub/models/speecht5/modeling_speecht5.py"),
            "signature": [
                "AutoModelForTextToSpeech.from_pretrained(",
                "pretrained_model_name_or_path,",
                "*,",
                "model_type='speecht5',",
                "config=None,",
                "**model_kwargs,",
                ")",
            ],
            "parameterHeading":
            "Parameters",
            "parameters": [
                "pretrained_model_name_or_path — Hub ID or compatible local directory.",
                "model_type — Canonical model type; use 'speecht5'.",
                "config — Optional preloaded SpeechT5Config instance.",
                "**model_kwargs — Model-specific loading arguments.",
            ],
        },
    )
    if tuple(state["apiCards"]) != expected_api_cards:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 API cards are {state['apiCards']!r}, expected {expected_api_cards!r}.")
    if len({card["kind"] for card in state["apiCards"]}) != len(state["apiCards"]):
        raise DocumentationVisualError(f"{case}: SpeechT5 API-card identities are not unique.")
    headings = tuple(tuple(value) for value in state["headings"])
    if headings != SPEECHT5_HEADINGS:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 headings are {headings!r}, expected {SPEECHT5_HEADINGS!r}.")
    if tuple(state["toc"]) != SPEECHT5_TOC:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 table of contents is {state['toc']!r}, expected {SPEECHT5_TOC!r}.")
    if tuple(state["tableRows"]) != SPEECHT5_TABLE_ROWS:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 table rows are {state['tableRows']!r}, "
            f"expected {SPEECHT5_TABLE_ROWS!r}.")
    if state["codeBlocks"] != 6 or state["codeCopyButtons"] != 6:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 code inventory is codeBlocks={state['codeBlocks']}, "
            f"copyButtons={state['codeCopyButtons']}; expected 6 and 6.")
    for marker in (
            "AutoModelForTextToSpeech",
            "AutoProcessor",
            "TTSOutput",
            "microsoft/speecht5_tts",
            "available_optimization_passes()",
            "Paper:",
            "Upstream GitHub:",
            "Real-checkpoint evidence",
            "Public optimizations fail closed",
    ):
        if marker not in state["text"]:
            raise DocumentationVisualError(f"{case}: rendered SpeechT5 content is missing {marker!r}.")

    geometry = state["geometry"]
    if any(value is None for value in geometry.values()):
        raise DocumentationVisualError(f"{case}: SpeechT5 geometry is incomplete: {geometry!r}.")
    for name in ("detail", "hero", "tabs", "layout", "content", "facts"):
        rectangle = geometry[name]
        if rectangle["display"] == "none" or rectangle["width"] <= 0 or rectangle["height"] <= 0:
            raise DocumentationVisualError(f"{case}: SpeechT5 {name} is not visibly sized: {rectangle!r}.")
    facts_disclosure = geometry["factsDisclosure"]
    if (facts_disclosure["display"] == "none" or facts_disclosure["width"] <= 0 or
            viewport["name"] == "desktop" and facts_disclosure["height"] <= 0):
        raise DocumentationVisualError(
            f"{case}: SpeechT5 facts disclosure geometry is invalid: {facts_disclosure!r}.")
    facts_summary = geometry["factsSummary"]
    if viewport["name"] == "desktop":
        if (facts_summary["display"] != "none" or facts_summary["width"] != 0 or
                facts_summary["height"] != 0):
            raise DocumentationVisualError(
                f"{case}: SpeechT5 desktop facts summary is not hidden: {facts_summary!r}.")
    elif (facts_summary["display"] == "none" or facts_summary["width"] <= 0 or facts_summary["height"] <= 0):
        raise DocumentationVisualError(
            f"{case}: SpeechT5 responsive facts summary is not visible: {facts_summary!r}.")
    elif not _rectangle_contains(geometry["facts"], facts_summary):
        raise DocumentationVisualError(
            f"{case}: SpeechT5 responsive facts summary escapes the sidebar: {geometry!r}.")
    containment_pairs = [
        ("article", "detail"),
        ("detail", "hero"),
        ("detail", "tabs"),
        ("detail", "layout"),
        ("layout", "content"),
        ("layout", "facts"),
    ]
    if viewport["name"] == "desktop":
        containment_pairs.append(("facts", "factsDisclosure"))
    for parent_name, child_name in containment_pairs:
        if not _rectangle_contains(geometry[parent_name], geometry[child_name]):
            raise DocumentationVisualError(
                f"{case}: SpeechT5 {child_name} escapes {parent_name}: {geometry!r}.")
    for name in ("hero", "tabs", "layout"):
        _assert_close(case, f"SpeechT5 {name} x", geometry[name]["x"], geometry["detail"]["x"])
        _assert_close(
            case,
            f"SpeechT5 {name} width",
            geometry[name]["width"],
            geometry["detail"]["width"],
        )
    if geometry["tabs"]["position"] != "sticky":
        raise DocumentationVisualError(
            f"{case}: SpeechT5 section navigation is not sticky: {geometry['tabs']!r}.")
    if geometry["tabs"]["scrollWidth"] < geometry["tabs"]["clientWidth"]:
        raise DocumentationVisualError(
            f"{case}: SpeechT5 tab scrolling geometry is invalid: {geometry['tabs']!r}.")
    if viewport["name"] == "desktop":
        if (geometry["content"]["right"] > geometry["facts"]["x"] + 0.75 or
                not math.isclose(geometry["content"]["y"], geometry["facts"]["y"], abs_tol=0.75) or
                geometry["facts"]["position"] != "sticky"):
            raise DocumentationVisualError(
                f"{case}: SpeechT5 desktop content/facts layout is invalid: {geometry!r}.")
    elif (geometry["facts"]["bottom"] > geometry["content"]["y"] + 0.75 or
          geometry["facts"]["position"] != "relative"):
        raise DocumentationVisualError(
            f"{case}: SpeechT5 responsive facts must precede the main content: {geometry!r}.")


def _validate_speecht5_section_navigation(page: Page, case: str) -> None:
    for value, href, _, _ in SPEECHT5_TABS:
        link = page.locator(f'[data-vh-model-tab="{value}"]')
        if link.count() != 1 or page.locator(href).count() != 1:
            raise DocumentationVisualError(
                f"{case}: SpeechT5 section {value!r} does not have one link and target.")
        link.focus()
        focus_state = link.evaluate(
            """link => {
              const style = getComputedStyle(link);
              return {
                focused: document.activeElement === link,
                outlineStyle: style.outlineStyle,
                outlineWidth: style.outlineWidth,
              };
            }""")
        if focus_state != {"focused": True, "outlineStyle": "solid", "outlineWidth": "2px"}:
            raise DocumentationVisualError(
                f"{case}: SpeechT5 section {value!r} focus state is {focus_state!r}.")
        page.keyboard.press("Enter")
        page.wait_for_function("hash => window.location.hash === hash", arg=href)
        page.wait_for_function(
            "selector => { const current = document.querySelector(selector); "
            "return current?.getAttribute('aria-current') === 'location' && "
            "document.querySelectorAll('[data-vh-model-tab][aria-current]').length === 1; }",
            arg=f'[data-vh-model-tab="{value}"]',
        )
        page.wait_for_function(
            "hash => { const target = document.querySelector(hash); "
            "const navigation = document.querySelector('.vh-model-detail__tabs'); "
            "if (!target || !navigation) return false; "
            "return target.getBoundingClientRect().top >= "
            "navigation.getBoundingClientRect().bottom; }",
            arg=href,
            timeout=5000,
        )
        activated = page.evaluate(
            r"""({ selector, hash }) => {
              const link = document.querySelector(selector);
              const target = document.querySelector(hash);
              const bounds = target?.getBoundingClientRect();
              const navigation = document.querySelector(".vh-model-detail__tabs");
              const navigationBounds = navigation?.getBoundingClientRect();
              const hitTarget = navigationBounds
                ? document.elementFromPoint(
                    Math.min(innerWidth - 2, Math.max(1, navigationBounds.left + navigationBounds.width / 2)),
                    navigationBounds.top + navigationBounds.height / 2,
                  )
                : null;
              return {
                targeted: target?.matches(":target") || false,
                current: link?.getAttribute("aria-current"),
                currentCount: document.querySelectorAll(
                  "[data-vh-model-tab][aria-current='location']"
                ).length,
                width: bounds?.width || 0,
                height: bounds?.height || 0,
                navigationHit: navigation?.contains(hitTarget) || false,
                targetBelowNavigation: (bounds?.top || 0) >= (navigationBounds?.bottom || 0),
                overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
              };
            }""",
            {
                "selector": f'[data-vh-model-tab="{value}"]',
                "hash": href
            },
        )
        if activated != {"targeted": True, "current": "location", "currentCount": 1, "width":
                         activated["width"], "height": activated["height"], "navigationHit": True,
                         "targetBelowNavigation": True, "overflow": 0
                         } or activated["width"] <= 0 or activated["height"] <= 0:
            raise DocumentationVisualError(
                f"{case}: SpeechT5 section {value!r} activation is invalid: {activated!r}.")


def _validate_speecht5_model_id_copy(page: Page, case: str) -> None:
    page.evaluate(
        """() => {
          window.__vhCopiedModelId = null;
          window.__vhClipboardText = null;
          Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: {
              writeText: async value => {
                window.__vhCopiedModelId = value;
                window.__vhClipboardText = value;
              },
              readText: async () => window.__vhClipboardText,
            },
          });
        }""")
    button = page.locator("button[data-vh-copy-model-id]")
    if button.count() != 1:
        raise DocumentationVisualError(f"{case}: SpeechT5 has {button.count()} model-ID copy controls.")
    button.focus()
    page.keyboard.press("Enter")
    page.wait_for_function(
        "() => window.__vhCopiedModelId === 'microsoft/speecht5_tts' && "
        "document.querySelector('[data-vh-copy-model-id-label]')?.textContent === 'Copied' && "
        "document.querySelector('[data-vh-copy-model-id]')?.getAttribute('aria-busy') === 'false'")
    copied = page.evaluate(
        """() => {
          const button = document.querySelector("button[data-vh-copy-model-id]");
          const outline = getComputedStyle(button);
          return {
            modelId: window.__vhCopiedModelId,
            label: button?.querySelector("[data-vh-copy-model-id-label]")?.textContent,
            ariaLabel: button?.getAttribute("aria-label"),
            ariaBusy: button?.getAttribute("aria-busy"),
            focused: document.activeElement === button,
            outlineStyle: outline.outlineStyle,
            outlineWidth: outline.outlineWidth,
            overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
          };
        }""")
    if copied != {"modelId": "microsoft/speecht5_tts", "label": "Copied", "ariaLabel": "Copied", "ariaBusy":
                  "false", "focused": True, "outlineStyle": "solid", "outlineWidth": "2px", "overflow": 0}:
        raise DocumentationVisualError(f"{case}: SpeechT5 model-ID copy state is {copied!r}.")


def _validate_speecht5_page_copy(page: Page, case: str, key: str) -> None:
    page.wait_for_function(
        'document.querySelector("[data-vh-copy-model-id-label]")?.textContent === '
        '"Copy model ID"',
        timeout=5000,
    )
    _validate_page_copy(page, f"{case} / SpeechT5 page copy", key)


def _validate_trainer_state(page: Page, case: str) -> None:
    state = page.evaluate(
        r"""() => {
          const content = document.querySelector(".md-content__inner");
          const normalize = value => value?.trim().replace(/¶$/, "").trim() || "";
          const nextStepLinks = Array.from(
            content?.querySelectorAll("#next-steps + ul a[href]") || []
          );
          const edit = document.querySelector('.md-content__button[rel="edit"]');
          const next = document.querySelector(".md-footer__link--next");
          return {
            headings: Array.from(content?.querySelectorAll("h1, h2, h3") || [])
              .map(heading => [heading.tagName, normalize(heading.textContent)]),
            toc: Array.from(document.querySelectorAll(
              ".md-sidebar--secondary a.md-nav__link"
            )).map(link => normalize(link.textContent)),
            tables: content?.querySelectorAll("table").length || 0,
            codeBlocks: content?.querySelectorAll("pre").length || 0,
            pageCopyButtons: content?.querySelectorAll("[data-vh-copy-page]").length || 0,
            nextStepPaths: nextStepLinks.map(link => new URL(link.href).pathname),
            editHref: edit?.getAttribute("href") || "",
            nextPath: next ? new URL(next.href).pathname : "",
            nextText: normalize(next?.textContent).replace(/\s+/g, " "),
            text: normalize(content?.textContent).replace(/\s+/g, " "),
          };
        }""")
    headings = tuple(tuple(value) for value in state["headings"])
    if headings != TRAINER_HEADINGS:
        raise DocumentationVisualError(
            f"{case}: Trainer headings are {headings!r}, expected {TRAINER_HEADINGS!r}.")
    expected_toc = tuple(label for _, label in TRAINER_HEADINGS[1:])
    if tuple(state["toc"]) != expected_toc:
        raise DocumentationVisualError(
            f"{case}: Trainer table of contents is {state['toc']!r}, expected {expected_toc!r}.")
    if state["tables"] != 0 or state["codeBlocks"] != 0 or state["pageCopyButtons"] != 1:
        raise DocumentationVisualError(
            f"{case}: Trainer component inventory is tables={state['tables']}, "
            f"codeBlocks={state['codeBlocks']}, pageCopyButtons={state['pageCopyButtons']}; "
            "expected 0, 0, and 1.")
    if tuple(state["nextStepPaths"]) != TRAINER_NEXT_STEP_PATHS:
        raise DocumentationVisualError(
            f"{case}: Trainer next-step paths are {state['nextStepPaths']!r}, "
            f"expected {TRAINER_NEXT_STEP_PATHS!r}.")
    if not state["editHref"].endswith("/docs/guides/trainer.md"):
        raise DocumentationVisualError(f"{case}: Trainer edit target is {state['editHref']!r}.")
    if state["nextPath"] != "/guides/training/" or "Fine-tuning" not in state["nextText"]:
        raise DocumentationVisualError(
            f"{case}: Trainer footer destination is "
            f"path={state['nextPath']!r}, text={state['nextText']!r}.")
    for marker in (
            "Trainer provides a complete training and evaluation loop",
            "TrainingArguments",
            "model-owned objective",
            "gradient accumulation",
            "exact resume state",
            "Speech architectures do not share one safe fallback loss",
            "inference-only or unsupported path fails closed",
    ):
        if marker not in state["text"]:
            raise DocumentationVisualError(f"{case}: rendered Trainer content is missing {marker!r}.")


def _validate_trainer_page_copy(page: Page, case: str, key: str) -> None:
    _validate_page_copy(page, f"{case} / Trainer page copy", key)


def _validate_optimization_state(page: Page, case: str) -> None:
    state = page.evaluate(
        r"""() => {
          const content = document.querySelector(".md-content__inner");
          const normalize = value => value?.trim().replace(/¶$/, "").trim() || "";
          const tables = Array.from(content?.querySelectorAll("table") || []);
          const nextStepLinks = Array.from(
            content?.querySelectorAll("#next-steps + ul a[href]") || []
          );
          const edit = document.querySelector('.md-content__button[rel="edit"]');
          const previous = document.querySelector(".md-footer__link--prev");
          const next = document.querySelector(".md-footer__link--next");
          const target = link => {
            const url = new URL(link.href);
            return url.pathname + url.hash;
          };
          return {
            headings: Array.from(content?.querySelectorAll("h1, h2, h3") || [])
              .map(heading => [heading.tagName, normalize(heading.textContent)]),
            toc: Array.from(document.querySelectorAll(
              ".md-sidebar--secondary a.md-nav__link"
            )).map(link => normalize(link.textContent)),
            tableRows: tables.map(table => table.querySelectorAll("tbody tr").length),
            codeBlocks: content?.querySelectorAll("pre").length || 0,
            codeCopyButtons: content?.querySelectorAll("button[data-vh-code-copy]").length || 0,
            pageCopyButtons: content?.querySelectorAll("[data-vh-copy-page]").length || 0,
            nextStepTargets: nextStepLinks.map(target),
            editHref: edit?.getAttribute("href") || "",
            previousPath: previous ? new URL(previous.href).pathname : "",
            previousText: normalize(previous?.textContent).replace(/\s+/g, " "),
            nextPath: next ? new URL(next.href).pathname : "",
            nextText: normalize(next?.textContent).replace(/\s+/g, " "),
            text: normalize(content?.textContent).replace(/\s+/g, " "),
          };
        }""")
    headings = tuple(tuple(value) for value in state["headings"])
    if headings != OPTIMIZATION_HEADINGS:
        raise DocumentationVisualError(
            f"{case}: Optimization headings are {headings!r}, expected {OPTIMIZATION_HEADINGS!r}.")
    expected_toc = tuple(label for _, label in OPTIMIZATION_HEADINGS[1:])
    if tuple(state["toc"]) != expected_toc:
        raise DocumentationVisualError(
            f"{case}: Optimization table of contents is {state['toc']!r}, "
            f"expected {expected_toc!r}.")
    if tuple(state["tableRows"]) != (6, ):
        raise DocumentationVisualError(
            f"{case}: Optimization table rows are {state['tableRows']!r}, expected (6,).")
    if (state["codeBlocks"], state["codeCopyButtons"], state["pageCopyButtons"]) != (1, 1, 1):
        raise DocumentationVisualError(
            f"{case}: Optimization component inventory is codeBlocks={state['codeBlocks']}, "
            f"codeCopyButtons={state['codeCopyButtons']}, "
            f"pageCopyButtons={state['pageCopyButtons']}; expected 1, 1, and 1.")
    if tuple(state["nextStepTargets"]) != OPTIMIZATION_NEXT_STEP_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Optimization next-step targets are {state['nextStepTargets']!r}, "
            f"expected {OPTIMIZATION_NEXT_STEP_TARGETS!r}.")
    if not state["editHref"].endswith("/docs/guides/optimization-overview.md"):
        raise DocumentationVisualError(f"{case}: Optimization edit target is {state['editHref']!r}.")
    if (state["previousPath"] != "/guides/data-preparation/" or
            "Data preparation" not in state["previousText"]):
        raise DocumentationVisualError(
            f"{case}: Optimization previous footer destination is "
            f"path={state['previousPath']!r}, text={state['previousText']!r}.")
    if state["nextPath"] != "/optimizations/" or "Optimization catalog" not in state["nextText"]:
        raise DocumentationVisualError(
            f"{case}: Optimization next footer destination is "
            f"path={state['nextPath']!r}, text={state['nextText']!r}.")
    for marker in OPTIMIZATION_PASS_NAMES + (
            "available_optimization_passes()",
            "apply_optimization_plan",
            "optimization_manifest",
            "restore_optimization_plan",
            "validation happens before mutation",
            "no registry-wide public quantization pass",
            "Parallelism is a training or serving topology",
            "Continuous batching belongs to a serving scheduler",
    ):
        if marker not in state["text"]:
            raise DocumentationVisualError(f"{case}: rendered Optimization content is missing {marker!r}.")


def _validate_optimization_page_copy(page: Page, case: str, key: str) -> None:
    _validate_page_copy(page, f"{case} / Optimization page copy", key)


def _validate_contribution_state(page: Page, case: str) -> None:
    state = page.evaluate(
        r"""() => {
          const content = document.querySelector(".md-content__inner");
          const normalize = value => value?.trim().replace(/¶$/, "").trim() || "";
          const tables = Array.from(content?.querySelectorAll("table") || []);
          const finalLinks = Array.from(
            content?.querySelectorAll("#completion-evidence ~ p a[href]") || []
          );
          const edit = document.querySelector('.md-content__button[rel="edit"]');
          const previous = document.querySelector(".md-footer__link--prev");
          const next = document.querySelector(".md-footer__link--next");
          const target = link => {
            const url = new URL(link.href);
            return url.pathname + url.hash;
          };
          return {
            headings: Array.from(content?.querySelectorAll("h1, h2, h3") || [])
              .map(heading => [heading.tagName, normalize(heading.textContent)]),
            toc: Array.from(document.querySelectorAll(
              ".md-sidebar--secondary a.md-nav__link"
            )).map(link => normalize(link.textContent)),
            processLabels: Array.from(content?.querySelectorAll(
              "ol.vh-process li strong"
            ) || []).map(label => normalize(label.textContent)),
            tableRows: tables.map(table => table.querySelectorAll("tbody tr").length),
            codeBlocks: content?.querySelectorAll("pre").length || 0,
            codeCopyButtons: content?.querySelectorAll("button[data-vh-code-copy]").length || 0,
            pageCopyButtons: content?.querySelectorAll("[data-vh-copy-page]").length || 0,
            finalTargets: finalLinks.map(target),
            editHref: edit?.getAttribute("href") || "",
            previousPath: previous ? new URL(previous.href).pathname : "",
            previousText: normalize(previous?.textContent).replace(/\s+/g, " "),
            nextPath: next ? new URL(next.href).pathname : "",
            nextText: normalize(next?.textContent).replace(/\s+/g, " "),
            text: normalize(content?.textContent).replace(/\s+/g, " "),
          };
        }""")
    headings = tuple(tuple(value) for value in state["headings"])
    if headings != CONTRIBUTION_HEADINGS:
        raise DocumentationVisualError(
            f"{case}: Contribution headings are {headings!r}, expected {CONTRIBUTION_HEADINGS!r}.")
    expected_toc = tuple(label for _, label in CONTRIBUTION_HEADINGS[1:])
    if tuple(state["toc"]) != expected_toc:
        raise DocumentationVisualError(
            f"{case}: Contribution table of contents is {state['toc']!r}, "
            f"expected {expected_toc!r}.")
    if tuple(state["processLabels"]) != CONTRIBUTION_PROCESS_LABELS:
        raise DocumentationVisualError(
            f"{case}: Contribution process labels are {state['processLabels']!r}, "
            f"expected {CONTRIBUTION_PROCESS_LABELS!r}.")
    if tuple(state["tableRows"]) != (8, 3, 7):
        raise DocumentationVisualError(
            f"{case}: Contribution table rows are {state['tableRows']!r}, expected (8, 3, 7).")
    if (state["codeBlocks"], state["codeCopyButtons"], state["pageCopyButtons"]) != (13, 13, 1):
        raise DocumentationVisualError(
            f"{case}: Contribution component inventory is codeBlocks={state['codeBlocks']}, "
            f"codeCopyButtons={state['codeCopyButtons']}, "
            f"pageCopyButtons={state['pageCopyButtons']}; expected 13, 13, and 1.")
    if tuple(state["finalTargets"]) != CONTRIBUTION_FINAL_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Contribution final targets are {state['finalTargets']!r}, "
            f"expected {CONTRIBUTION_FINAL_TARGETS!r}.")
    if not state["editHref"].endswith("/docs/project/adding-a-model.md"):
        raise DocumentationVisualError(f"{case}: Contribution edit target is {state['editHref']!r}.")
    if state["previousPath"] != "/reference/models/" or "Models API" not in state["previousText"]:
        raise DocumentationVisualError(
            f"{case}: Contribution previous footer destination is "
            f"path={state['previousPath']!r}, text={state['previousText']!r}.")
    if state["nextPath"] != "/guides/trainer/" or "Trainer overview" not in state["nextText"]:
        raise DocumentationVisualError(
            f"{case}: Contribution next footer destination is "
            f"path={state['nextPath']!r}, text={state['nextText']!r}.")
    for marker in (
            "scripts/scaffold_model.py create",
            "model-integration.json",
            "scripts/scaffold_model.py check",
            "PreTrainedTTSModel",
            "PreTrainedASRModel",
            "PreTrainedVADModel",
            "ArchitectureSpec",
            "ModelSpec",
            "ModelTrainingSpec",
            "apply_optimization_plan",
            "restore_optimization_plan",
            "scripts/generate_model_pages.py --check",
            "scripts/check_distribution.py",
            "unverified or hardware-limited",
    ):
        if marker not in state["text"]:
            raise DocumentationVisualError(f"{case}: rendered Contribution content is missing {marker!r}.")


def _validate_contribution_page_copy(page: Page, case: str, key: str) -> None:
    _validate_page_copy(page, f"{case} / Contribution page copy", key)


def _validate_model_api_state(page: Page, case: str) -> None:
    state = page.evaluate(
        r"""() => {
          const content = document.querySelector(".md-content__inner");
          const normalize = value => value?.trim().replace(/¶$/, "").trim() || "";
          const tables = Array.from(content?.querySelectorAll("table") || []);
          const sourceLinks = Array.from(content?.querySelectorAll(
            'a[href*="/voicehub/blob/main/voicehub/"]'
          ) || []);
          const internalLinks = Array.from(content?.querySelectorAll(
            "a[href]:not(.headerlink):not(.md-content__button)"
          ) || []).filter(link => !link.closest("pre") &&
            new URL(link.href).origin === location.origin);
          const edit = document.querySelector('.md-content__button[rel="edit"]');
          const previous = document.querySelector(".md-footer__link--prev");
          const next = document.querySelector(".md-footer__link--next");
          const target = link => {
            const url = new URL(link.href);
            return url.pathname + url.hash;
          };
          return {
            headings: Array.from(content?.querySelectorAll("h1, h2, h3") || [])
              .map(heading => [heading.tagName, normalize(heading.textContent)]),
            toc: Array.from(document.querySelectorAll(
              ".md-sidebar--secondary a.md-nav__link"
            )).map(link => normalize(link.textContent)),
            tableRows: tables.map(table => table.querySelectorAll("tbody tr").length),
            codeBlocks: content?.querySelectorAll("pre").length || 0,
            codeCopyButtons: content?.querySelectorAll("button[data-vh-code-copy]").length || 0,
            pageCopyButtons: content?.querySelectorAll("[data-vh-copy-page]").length || 0,
            sourceTargets: sourceLinks.map(link => link.getAttribute("href")),
            internalTargets: internalLinks.map(target),
            editHref: edit?.getAttribute("href") || "",
            previousPath: previous ? new URL(previous.href).pathname : "",
            previousText: normalize(previous?.textContent).replace(/\s+/g, " "),
            nextPath: next ? new URL(next.href).pathname : "",
            nextText: normalize(next?.textContent).replace(/\s+/g, " "),
            text: normalize(content?.textContent).replace(/\s+/g, " "),
          };
        }""")
    headings = tuple(tuple(value) for value in state["headings"])
    if headings != MODEL_API_HEADINGS:
        raise DocumentationVisualError(
            f"{case}: Models API headings are {headings!r}, expected {MODEL_API_HEADINGS!r}.")
    expected_toc = tuple(label for _, label in MODEL_API_HEADINGS[1:])
    if tuple(state["toc"]) != expected_toc:
        raise DocumentationVisualError(
            f"{case}: Models API table of contents is {state['toc']!r}, "
            f"expected {expected_toc!r}.")
    if tuple(state["tableRows"]) != (6, 3, 3, 5):
        raise DocumentationVisualError(
            f"{case}: Models API table rows are {state['tableRows']!r}, expected (6, 3, 3, 5).")
    if (state["codeBlocks"], state["codeCopyButtons"], state["pageCopyButtons"]) != (2, 2, 1):
        raise DocumentationVisualError(
            f"{case}: Models API component inventory is codeBlocks={state['codeBlocks']}, "
            f"codeCopyButtons={state['codeCopyButtons']}, "
            f"pageCopyButtons={state['pageCopyButtons']}; expected 2, 2, and 1.")
    if tuple(state["sourceTargets"]) != MODEL_API_SOURCE_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Models API source targets are {state['sourceTargets']!r}, "
            f"expected {MODEL_API_SOURCE_TARGETS!r}.")
    if tuple(state["internalTargets"]) != MODEL_API_INTERNAL_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Models API internal targets are {state['internalTargets']!r}, "
            f"expected {MODEL_API_INTERNAL_TARGETS!r}.")
    if not state["editHref"].endswith("/docs/reference/models.md"):
        raise DocumentationVisualError(f"{case}: Models API edit target is {state['editHref']!r}.")
    if (state["previousPath"] != "/models/providers/vad_webrtc/" or "WebRTCVAD" not in state["previousText"]):
        raise DocumentationVisualError(
            f"{case}: Models API previous footer destination is "
            f"path={state['previousPath']!r}, text={state['previousText']!r}.")
    if state["nextPath"] != "/project/adding-a-model/" or "Add a model" not in state["nextText"]:
        raise DocumentationVisualError(
            f"{case}: Models API next footer destination is "
            f"path={state['nextPath']!r}, text={state['nextText']!r}.")
    for marker in (
            "AutoConfig",
            "AutoModelForTextToSpeech",
            "from_config",
            "is_loaded",
            "PreTrainedSpeechModel",
            "PreTrainedTTSModel",
            "PreTrainedAudioModel",
            "PreTrainedASRModel",
            "PreTrainedVADModel",
            "from_pretrained(source, **kwargs)",
            "load_for_training()",
            "validate_training_support()",
            "save_pretrained(directory, include_native_export=True)",
            "TTSGenerationConfig",
            "TTSOutput",
            "ASROutput",
            "VADOutput",
            "SpeechTrainingOutput",
            "model_state.safetensors",
            "does not expose a public push_to_hub() method",
    ):
        if marker not in state["text"]:
            raise DocumentationVisualError(f"{case}: rendered Models API content is missing {marker!r}.")


def _validate_model_api_page_copy(page: Page, case: str, key: str) -> None:
    _validate_page_copy(page, f"{case} / Models API page copy", key)


def _validate_quickstart_state(page: Page, case: str) -> None:
    state = page.evaluate(
        r"""() => {
          const content = document.querySelector(".md-content__inner");
          const normalize = value => value?.trim().replace(/¶$/, "").trim() || "";
          const visible = element => {
            const bounds = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return bounds.width > 0 && bounds.height > 0 &&
              style.display !== "none" && style.visibility !== "hidden";
          };
          const contentLinks = Array.from(content?.querySelectorAll("a[href]") || [])
            .filter(link => !link.closest("pre") &&
              !link.closest(".tabbed-labels") &&
              !link.classList.contains("headerlink") &&
              !link.classList.contains("md-content__button") &&
              !link.closest(".md-source-file"));
          return {
            headings: Array.from(content?.querySelectorAll("h1, h2, h3") || [])
              .map(heading => [heading.tagName, normalize(heading.textContent)]),
            toc: Array.from(document.querySelectorAll(
              ".md-sidebar--secondary a.md-nav__link"
            )).map(link => normalize(link.textContent)),
            tabs: Array.from(content?.querySelectorAll(".tabbed-set") || [])
              .map(tabSet => {
                const inputs = Array.from(tabSet.querySelectorAll("input[type='radio']"));
                const blocks = Array.from(tabSet.querySelectorAll(
                  ".tabbed-content > .tabbed-block"
                ));
                return {
                  labels: Array.from(tabSet.querySelectorAll(".tabbed-labels > label"))
                    .map(label => normalize(label.textContent)),
                  checked: inputs.findIndex(input => input.checked),
                  visibleBlocks: blocks.map(visible),
                };
              }),
            tips: content?.querySelectorAll(".admonition.tip").length || 0,
            tableRows: Array.from(content?.querySelectorAll("table") || [])
              .map(table => table.querySelectorAll("tbody tr").length),
            codeBlocks: content?.querySelectorAll("pre").length || 0,
            codeCopyButtons: content?.querySelectorAll("button.md-clipboard").length || 0,
            pageCopyButtons: content?.querySelectorAll("[data-vh-copy-page]").length || 0,
            externalTargets: contentLinks
              .map(link => new URL(link.getAttribute("href"), location.href))
              .filter(target => target.origin !== location.origin)
              .map(target => target.href),
            internalTargets: contentLinks
              .map(link => new URL(link.getAttribute("href"), location.href))
              .filter(target => target.origin === location.origin)
              .map(target => `${target.pathname}${target.hash}`),
            editTarget: content?.querySelector("a.md-content__button[href]")?.href || null,
            previousTarget: document.querySelector(".md-footer__link--prev")?.pathname || null,
            previousLabel: document.querySelector(".md-footer__link--prev")
              ?.getAttribute("aria-label") || null,
            nextTarget: document.querySelector(".md-footer__link--next")?.pathname || null,
            nextLabel: document.querySelector(".md-footer__link--next")
              ?.getAttribute("aria-label") || null,
            text: normalize(content?.textContent).replace(/\s+/g, " "),
          };
        }""")
    headings = tuple(tuple(value) for value in state["headings"])
    if headings != QUICKSTART_HEADINGS:
        raise DocumentationVisualError(
            f"{case}: Quickstart headings are {headings!r}, expected {QUICKSTART_HEADINGS!r}.")
    expected_toc = tuple(label for _, label in QUICKSTART_HEADINGS[1:])
    if tuple(state["toc"]) != expected_toc:
        raise DocumentationVisualError(
            f"{case}: Quickstart table of contents is {state['toc']!r}, "
            f"expected {expected_toc!r}.")
    labels = tuple(tuple(tab["labels"]) for tab in state["tabs"])
    if labels != QUICKSTART_TAB_LABELS:
        raise DocumentationVisualError(
            f"{case}: Quickstart tab labels are {labels!r}, expected {QUICKSTART_TAB_LABELS!r}.")
    for index, tab in enumerate(state["tabs"], start=1):
        expected_visibility = [item == 0 for item in range(len(tab["labels"]))]
        if tab["checked"] != 0 or tab["visibleBlocks"] != expected_visibility:
            raise DocumentationVisualError(
                f"{case}: Quickstart tab set {index} starts in invalid state {tab!r}.")
    if (state["tips"], tuple(state["tableRows"]), state["codeBlocks"], state["codeCopyButtons"],
            state["pageCopyButtons"]) != (2, (3, ), 11, 11, 1):
        raise DocumentationVisualError(
            f"{case}: Quickstart component inventory is "
            f"tips={state['tips']}, tableRows={state['tableRows']}, "
            f"codeBlocks={state['codeBlocks']}, codeCopyButtons={state['codeCopyButtons']}, "
            f"pageCopyButtons={state['pageCopyButtons']}; expected 2, [3], 11, 11, and 1.")
    if tuple(state["externalTargets"]) != QUICKSTART_EXTERNAL_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Quickstart external targets are {state['externalTargets']!r}, "
            f"expected {QUICKSTART_EXTERNAL_TARGETS!r}.")
    if tuple(state["internalTargets"]) != QUICKSTART_INTERNAL_TARGETS:
        raise DocumentationVisualError(
            f"{case}: Quickstart internal targets are {state['internalTargets']!r}, "
            f"expected {QUICKSTART_INTERNAL_TARGETS!r}.")
    if state["editTarget"] != (
            "https://github.com/kadirnar/voicehub/edit/main/docs/getting-started/quickstart.md"):
        raise DocumentationVisualError(f"{case}: Quickstart edit target is {state['editTarget']!r}.")
    if (state["previousTarget"], state["previousLabel"]) != ("/getting-started/installation/",
                                                             "Previous: Installation"):
        raise DocumentationVisualError(
            f"{case}: Quickstart previous action is "
            f"{(state['previousTarget'], state['previousLabel'])!r}.")
    if (state["nextTarget"], state["nextLabel"]) != ("/guides/inference/", "Next: Inference"):
        raise DocumentationVisualError(
            f"{case}: Quickstart next action is "
            f"{(state['nextTarget'], state['nextLabel'])!r}.")
    for marker in (
            "VoiceHubConfig",
            "PreTrainedSpeechModel",
            "pipeline()",
            "text-to-speech",
            "automatic-speech-recognition",
            "voice-activity-detection",
            "Training is not supported",
    ):
        if marker not in state["text"]:
            raise DocumentationVisualError(f"{case}: rendered Quickstart content is missing {marker!r}.")


def _localized_route_path(relative_path: str, locale: str) -> str:
    route = relative_path.removesuffix("index.html")
    locale_prefix = "" if locale == "en" else f"{locale}/"
    return f"/voicehub/{locale_prefix}{route}"


def _validate_page_actions(
    page: Page,
    case: str,
    base_url: str,
    relative_path: str,
    expectation: dict[str, Any],
    activation_method: str,
    palette: str,
    axe: Axe,
) -> tuple[str, int]:
    route_url = _route_url(base_url, relative_path)
    origin_path = f"/{relative_path.removesuffix('index.html')}"
    selectors = {
        "edit": '.md-content__button[rel="edit"]',
        "previous": ".md-footer__link--prev",
        "next": ".md-footer__link--next",
        "top": 'button[data-md-component="top"]',
    }

    def reset_origin() -> None:
        page.goto(route_url, wait_until="networkidle")
        page.wait_for_function(
            "palette => document.body.dataset.mdColorScheme === palette",
            arg=palette,
        )
        state = page.evaluate(
            "() => ({ language: document.documentElement.lang, "
            "palette: document.body.dataset.mdColorScheme, path: location.pathname })")
        expected_state = {"language": "en", "palette": palette, "path": origin_path}
        if state != expected_state:
            raise DocumentationVisualError(
                f"{case}: page-action origin is {state!r}, expected {expected_state!r}.")
        if _rendered_state(page)["overflow"] != 0:
            raise DocumentationVisualError(f"{case}: page-action origin introduced overflow.")

    def focus_action(selector: str, action_case: str) -> Any:
        action = page.locator(selector)
        if action.count() != 1:
            raise DocumentationVisualError(
                f"{action_case}: expected one action for {selector!r}, found {action.count()}.")
        action.scroll_into_view_if_needed()
        action.focus()
        state = action.evaluate(
            """action => {
              const bounds = action.getBoundingClientRect();
              const style = getComputedStyle(action);
              return {
                active: document.activeElement === action,
                outlineStyle: style.outlineStyle,
                outlineWidth: style.outlineWidth,
                visible: bounds.width > 0 && bounds.height > 0 &&
                  style.display !== "none" && style.visibility !== "hidden",
                withinViewport: bounds.left >= 0 && bounds.right <= innerWidth &&
                  bounds.top >= 0 && bounds.bottom <= innerHeight,
              };
            }""")
        if not state["active"] or not state["visible"] or not state["withinViewport"]:
            raise DocumentationVisualError(f"{action_case}: focused action state is {state!r}.")
        if state["outlineStyle"] == "none" or state["outlineWidth"] == "0px":
            raise DocumentationVisualError(f"{action_case}: focus outline is {state!r}.")
        return action

    def activate(action: Any, action_case: str) -> None:
        if activation_method == "pointer":
            action.click()
        elif activation_method == "keyboard":
            page.keyboard.press("Enter")
        else:
            raise DocumentationVisualError(
                f"{action_case}: unsupported activation method {activation_method!r}.")

    reset_origin()
    inventory = page.evaluate(
        r"""selectors => {
          const inspectLink = selector => {
            const links = Array.from(document.querySelectorAll(selector));
            const link = links[0];
            return {
              count: links.length,
              href: link?.href || null,
              label: link?.getAttribute("aria-label") || null,
              path: link?.pathname || null,
              tag: link?.tagName || null,
            };
          };
          const top = document.querySelector(selectors.top);
          return {
            edit: inspectLink(selectors.edit),
            previous: inspectLink(selectors.previous),
            next: inspectLink(selectors.next),
            top: {
              count: document.querySelectorAll(selectors.top).length,
              hidden: top?.hidden,
              label: top?.textContent?.trim().replace(/\s+/g, " ") || null,
              tag: top?.tagName || null,
              type: top?.getAttribute("type") || null,
            },
          };
        }""",
        selectors,
    )
    expected_inventory = {
        "edit": {
            "count": 1,
            "href": expectation["edit"],
            "label": None,
            "path": "/kadirnar/voicehub/edit/main/" + expectation["edit"].split("/edit/main/", 1)[1],
            "tag": "A",
        },
        "previous": {
            "count": int(expectation["previous"] is not None),
            "href":
            (f"{base_url}{expectation['previous'][0]}" if expectation["previous"] is not None else None),
            "label": expectation["previous"][1] if expectation["previous"] is not None else None,
            "path": expectation["previous"][0] if expectation["previous"] is not None else None,
            "tag": "A" if expectation["previous"] is not None else None,
        },
        "next": {
            "count": int(expectation["next"] is not None),
            "href": f"{base_url}{expectation['next'][0]}" if expectation["next"] is not None else None,
            "label": expectation["next"][1] if expectation["next"] is not None else None,
            "path": expectation["next"][0] if expectation["next"] is not None else None,
            "tag": "A" if expectation["next"] is not None else None,
        },
        "top": {
            "count": 1,
            "hidden": True,
            "label": "Back to top",
            "tag": "BUTTON",
            "type": "button"
        },
    }
    if inventory != expected_inventory:
        raise DocumentationVisualError(
            f"{case}: page-action inventory is {inventory!r}, expected {expected_inventory!r}.")

    edit_case = f"{case} / edit"
    edit = focus_action(selectors["edit"], edit_case)
    edit_target = expectation["edit"]
    page.route(
        edit_target,
        lambda route: route.fulfill(
            status=200,
            content_type="text/html",
            body=(
                '<!doctype html><html lang="en"><head><title>Edit VoiceHub page</title></head>'
                '<body><main data-vh-edit-navigation-target>Edit VoiceHub page</main></body></html>'),
        ),
    )
    try:
        with page.expect_navigation(wait_until="domcontentloaded"):
            activate(edit, edit_case)
        if page.url != edit_target or page.locator("[data-vh-edit-navigation-target]").count() != 1:
            raise DocumentationVisualError(
                f"{edit_case}: edit activation reached {page.url!r}, expected {edit_target!r}.")
    finally:
        page.unroute(edit_target)

    footer_activations = 0
    for direction in ("previous", "next"):
        expected_footer = expectation[direction]
        if expected_footer is None:
            continue
        reset_origin()
        action_case = f"{case} / {direction}"
        footer = focus_action(selectors[direction], action_case)
        with page.expect_navigation(wait_until="networkidle"):
            activate(footer, action_case)
        destination = page.evaluate(
            "() => ({ palette: document.body.dataset.mdColorScheme, path: location.pathname, "
            'title: document.querySelector("h1")?.textContent?.trim().replace(/¶$/, "").trim() })')
        if destination["path"] != expected_footer[0] or destination["palette"] != palette:
            raise DocumentationVisualError(
                f"{action_case}: footer activation state is {destination!r}, "
                f"expected path={expected_footer[0]!r} and palette={palette!r}.")
        if not destination["title"]:
            raise DocumentationVisualError(f"{action_case}: destination has no rendered page title.")
        if _rendered_state(page)["overflow"] != 0:
            raise DocumentationVisualError(f"{action_case}: destination introduced overflow.")
        footer_activations += 1

    reset_origin()
    page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
    page.wait_for_function("window.scrollY > 0")
    page.evaluate("window.scrollTo(0, Math.max(1, window.scrollY - 64))")
    page.wait_for_function(
        "selector => !document.querySelector(selector)?.hidden",
        arg=selectors["top"],
    )
    top_case = f"{case} / back to top"
    top = focus_action(selectors["top"], top_case)
    page.wait_for_timeout(150)
    axe_core = _validate_accessibility(axe, page, f"{top_case} / visible and focused")
    activate(top, top_case)
    page.wait_for_function("window.scrollY === 0")
    final_state = page.evaluate(
        "() => ({ palette: document.body.dataset.mdColorScheme, path: location.pathname, "
        "scrollY: window.scrollY })")
    expected_final_state = {"palette": palette, "path": origin_path, "scrollY": 0}
    if final_state != expected_final_state:
        raise DocumentationVisualError(
            f"{top_case}: final state is {final_state!r}, expected {expected_final_state!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{top_case}: activation introduced overflow.")
    return axe_core, footer_activations


def _validate_source_activation(
    page: Page,
    case: str,
    activation_method: str,
    palette: str,
    axe: Axe,
) -> str:
    link = page.locator('[data-vh-header-control="source"] a[href]')
    if link.count() != 1:
        raise DocumentationVisualError(f"{case}: expected one source repository link, found {link.count()}.")

    source_state = link.evaluate(
        r"""link => {
          const bounds = link.getBoundingClientRect();
          const style = getComputedStyle(link);
          return {
            active: document.activeElement === link,
            ariaLabel: link.getAttribute("aria-label"),
            href: link.href,
            language: document.documentElement.lang,
            palette: document.body.dataset.mdColorScheme,
            path: location.pathname,
            tabIndex: link.tabIndex,
            visible: bounds.width > 0 && bounds.height > 0 &&
              style.display !== "none" && style.visibility !== "hidden",
            withinViewport: bounds.left >= 0 && bounds.right <= innerWidth &&
              bounds.top >= 0 && bounds.bottom <= innerHeight,
            x: bounds.x,
            y: bounds.y,
            width: bounds.width,
            height: bounds.height,
          };
        }""")
    expected_state = {
        "active": False,
        "ariaLabel": "Open VoiceHub source repository",
        "href": SOURCE_REPOSITORY_URL,
        "language": "en",
        "palette": palette,
        "path": source_state["path"],
        "tabIndex": 0,
        "visible": True,
        "withinViewport": True,
        "x": 203,
        "y": 165,
        "width": 48,
        "height": 30,
    }
    if source_state != expected_state:
        raise DocumentationVisualError(
            f"{case}: source link state is {source_state!r}, expected {expected_state!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: source link introduced overflow.")

    link.focus()
    focused = _active_focus_state(page)
    _validate_focused_element(case, focused, require_viewport=True)
    if focused["descriptor"] != "header:source":
        raise DocumentationVisualError(
            f"{case}: source link focus is {focused['descriptor']!r}, expected 'header:source'.")
    if (focused["outlineStyle"], focused["outlineWidth"], focused["outlineOffset"]) != ("solid", "2px",
                                                                                        "2px"):
        raise DocumentationVisualError(f"{case}: source link focus state is {focused!r}.")
    if not link.evaluate("link => document.activeElement === link"):
        raise DocumentationVisualError(f"{case}: source link did not retain focus before activation.")
    axe_core = _validate_accessibility(axe, page, f"{case} / focused source link")

    page.route(
        SOURCE_REPOSITORY_URL,
        lambda route: route.fulfill(
            status=200,
            content_type="text/html",
            body=(
                '<!doctype html><html lang="en"><head><title>VoiceHub source</title></head>'
                '<body><main data-vh-source-navigation-target>VoiceHub source repository</main></body>'
                '</html>'),
        ),
    )
    try:
        with page.expect_navigation(wait_until="domcontentloaded"):
            if activation_method == "pointer":
                link.click()
            elif activation_method == "keyboard":
                page.keyboard.press("Enter")
            else:
                raise DocumentationVisualError(
                    f"{case}: unsupported source activation method {activation_method!r}.")
        if page.url != SOURCE_REPOSITORY_URL:
            raise DocumentationVisualError(
                f"{case}: source activation reached {page.url!r}, expected {SOURCE_REPOSITORY_URL!r}.")
        if page.locator("[data-vh-source-navigation-target]").count() != 1:
            raise DocumentationVisualError(
                f"{case}: source activation did not load the intercepted repository target.")
    finally:
        page.unroute(SOURCE_REPOSITORY_URL)
    return axe_core


def _validate_theme_activation(
    page: Page,
    case: str,
    activation_method: str,
    source_palette: str,
    target_palette: str,
    axe: Axe,
) -> str:
    control = page.locator('[data-vh-header-control="theme"]')
    toggles = control.locator("[data-vh-theme-toggle]")
    toggle = control.locator("[data-vh-theme-toggle]:not([hidden])")
    palette_inputs = control.locator("input[data-md-color-scheme]")
    if control.count() != 1 or toggles.count() != 2 or toggle.count() != 1 or palette_inputs.count() != 2:
        raise DocumentationVisualError(
            f"{case}: expected one theme control, two toggles, one visible toggle, and two "
            f"palette inputs; found {control.count()}, {toggles.count()}, {toggle.count()}, "
            f"and {palette_inputs.count()}.")

    expected_source_label = "Switch to dark mode" if source_palette == "default" else "Switch to light mode"
    expected_target_label = "Switch to light mode" if target_palette == "slate" else "Switch to dark mode"
    source_state = control.evaluate(
        r"""control => {
          const toggles = Array.from(control.querySelectorAll("[data-vh-theme-toggle]"));
          const visibleToggle = control.querySelector("[data-vh-theme-toggle]:not([hidden])");
          const bounds = visibleToggle?.getBoundingClientRect();
          const style = visibleToggle ? getComputedStyle(visibleToggle) : null;
          const target = visibleToggle
            ? document.getElementById(visibleToggle.dataset.vhThemeTarget || "")
            : null;
          return {
            labels: toggles.map(item => item.getAttribute("aria-label")),
            palette: document.body.dataset.mdColorScheme,
            path: location.pathname,
            language: document.documentElement.lang,
            selectedSchemes: Array.from(
              control.querySelectorAll("input[data-md-color-scheme]:checked")
            ).map(input => input.dataset.mdColorScheme),
            tabIndex: visibleToggle?.tabIndex,
            targetScheme: target?.dataset.mdColorScheme,
            title: visibleToggle?.getAttribute("title"),
            visible: Boolean(bounds && bounds.width > 0 && bounds.height > 0 &&
              style?.display !== "none" && style?.visibility !== "hidden"),
            withinViewport: Boolean(bounds && bounds.left >= 0 && bounds.right <= innerWidth &&
              bounds.top >= 0 && bounds.bottom <= innerHeight),
            x: bounds?.x,
            y: bounds?.y,
            width: bounds?.width,
            height: bounds?.height,
          };
        }""")
    if source_state["labels"] != ["Switch to dark mode", "Switch to light mode"]:
        raise DocumentationVisualError(f"{case}: theme labels are {source_state['labels']!r}.")
    if source_state["palette"] != source_palette:
        raise DocumentationVisualError(
            f"{case}: source palette is {source_state['palette']!r}, expected {source_palette!r}.")
    if source_state["language"] != "en":
        raise DocumentationVisualError(
            f"{case}: source language is {source_state['language']!r}, expected 'en'.")
    if source_state["targetScheme"] != target_palette:
        raise DocumentationVisualError(
            f"{case}: source toggle targets {source_state['targetScheme']!r}, "
            f"expected {target_palette!r}.")
    if source_state["title"] != expected_source_label:
        raise DocumentationVisualError(
            f"{case}: source toggle title is {source_state['title']!r}, "
            f"expected {expected_source_label!r}.")
    if source_state["tabIndex"] != 0 or not source_state["visible"] or not source_state["withinViewport"]:
        raise DocumentationVisualError(
            f"{case}: source toggle is not a visible in-viewport native tab stop: {source_state!r}.")
    for field, expected in (("x", 163), ("y", 165), ("width", 34), ("height", 30)):
        _assert_close(case, f"source theme {field}", source_state[field], expected)
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: source theme control introduced overflow.")

    toggle.focus()
    focused = _active_focus_state(page)
    _validate_focused_element(case, focused, require_viewport=True)
    if focused["descriptor"] != "header:theme":
        raise DocumentationVisualError(
            f"{case}: source theme focus is {focused['descriptor']!r}, expected 'header:theme'.")
    if (focused["outlineStyle"], focused["outlineWidth"], focused["outlineOffset"]) != ("solid", "2px",
                                                                                        "2px"):
        raise DocumentationVisualError(
            f"{case}: source theme focus outline is "
            f"{(focused['outlineStyle'], focused['outlineWidth'], focused['outlineOffset'])!r}.")

    if activation_method == "pointer":
        toggle.click()
    elif activation_method == "keyboard":
        page.keyboard.press("Enter")
    else:
        raise DocumentationVisualError(f"{case}: unsupported theme activation method {activation_method!r}.")
    page.wait_for_function(
        "palette => document.body.dataset.mdColorScheme === palette",
        arg=target_palette,
    )
    page.wait_for_function(
        "label => document.querySelector('[data-vh-theme-toggle]:not([hidden])')"
        "?.getAttribute('aria-label') === label",
        arg=expected_target_label,
    )
    page.wait_for_function(
        "() => { const visibleToggle = document.querySelector("
        "'[data-vh-theme-toggle]:not([hidden])'); "
        "return document.activeElement === visibleToggle; }", )

    target_state = control.evaluate(
        r"""control => {
          const visibleToggle = control.querySelector("[data-vh-theme-toggle]:not([hidden])");
          const bounds = visibleToggle?.getBoundingClientRect();
          const style = visibleToggle ? getComputedStyle(visibleToggle) : null;
          const target = visibleToggle
            ? document.getElementById(visibleToggle.dataset.vhThemeTarget || "")
            : null;
          return {
            activeIsVisibleToggle: document.activeElement === visibleToggle,
            background: getComputedStyle(document.body).backgroundColor,
            text: getComputedStyle(document.body).color,
            palette: document.body.dataset.mdColorScheme,
            path: location.pathname,
            language: document.documentElement.lang,
            selectedSchemes: Array.from(
              control.querySelectorAll("input[data-md-color-scheme]:checked")
            ).map(input => input.dataset.mdColorScheme),
            targetScheme: target?.dataset.mdColorScheme,
            title: visibleToggle?.getAttribute("title"),
            visible: Boolean(bounds && bounds.width > 0 && bounds.height > 0 &&
              style?.display !== "none" && style?.visibility !== "hidden"),
            withinViewport: Boolean(bounds && bounds.left >= 0 && bounds.right <= innerWidth &&
              bounds.top >= 0 && bounds.bottom <= innerHeight),
            x: bounds?.x,
            y: bounds?.y,
            width: bounds?.width,
            height: bounds?.height,
          };
        }""")
    expected_target_state = {
        "activeIsVisibleToggle": True,
        "background": PALETTES[target_palette]["background"],
        "text": PALETTES[target_palette]["text"],
        "palette": target_palette,
        "path": source_state["path"],
        "language": source_state["language"],
        "selectedSchemes": [target_palette],
        "targetScheme": source_palette,
        "title": expected_target_label,
        "visible": True,
        "withinViewport": True,
        "x": source_state["x"],
        "y": source_state["y"],
        "width": source_state["width"],
        "height": source_state["height"],
    }
    if target_state != expected_target_state:
        raise DocumentationVisualError(
            f"{case}: target theme state is {target_state!r}, expected {expected_target_state!r}.")
    focused = _active_focus_state(page)
    _validate_focused_element(case, focused, require_viewport=True)
    if focused["descriptor"] != "header:theme" or (focused["outlineStyle"], focused["outlineWidth"],
                                                   focused["outlineOffset"]) != ("solid", "2px", "2px"):
        raise DocumentationVisualError(f"{case}: target theme focus state is {focused!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: activated theme control introduced overflow.")
    return _validate_accessibility(axe, page, f"{case} / activated palette")


def _validate_language_activation(
    page: Page,
    case: str,
    relative_path: str,
    activation_method: str,
    target_locale: str,
    palette: str,
    axe: Axe,
) -> str:
    select = page.locator("[data-vh-language-select]")
    options = select.locator("option")
    if select.count() != 1 or options.count() != len(LANGUAGE_LOCALES):
        raise DocumentationVisualError(
            f"{case}: expected one language control with {len(LANGUAGE_LOCALES)} options, "
            f"found {select.count()} and {options.count()}.")

    source_state = select.evaluate(
        r"""select => {
          const options = Array.from(select.options);
          const bounds = select.getBoundingClientRect();
          const style = getComputedStyle(select);
          return {
            ariaLabel: select.getAttribute("aria-label"),
            disabled: select.disabled,
            labels: options.map(option => option.textContent.trim()),
            locales: options.map(option => option.lang),
            selectedLocales: Array.from(select.selectedOptions).map(option => option.lang),
            tabIndex: select.tabIndex,
            targets: options.map(
              option => new URL(option.value, location.href).pathname
            ),
            visible: bounds.width > 0 && bounds.height > 0 &&
              style.display !== "none" && style.visibility !== "hidden",
            withinViewport: bounds.left >= 0 && bounds.right <= innerWidth &&
              bounds.top >= 0 && bounds.bottom <= innerHeight,
          };
        }""")
    expected_source = {
        "ariaLabel": "Select language",
        "disabled": False,
        "labels": [locale.upper() for locale in LANGUAGE_LOCALES],
        "locales": list(LANGUAGE_LOCALES),
        "selectedLocales": ["en"],
        "tabIndex": 0,
        "targets": [_localized_route_path(relative_path, locale) for locale in LANGUAGE_LOCALES],
        "visible": True,
        "withinViewport": True,
    }
    if source_state != expected_source:
        raise DocumentationVisualError(
            f"{case}: source language state is {source_state!r}, expected {expected_source!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: source language control introduced overflow.")

    select.focus()
    if activation_method == "pointer":
        select.click()
    elif activation_method != "keyboard":
        raise DocumentationVisualError(
            f"{case}: unsupported language activation method {activation_method!r}.")
    focused = _active_focus_state(page)
    _validate_focused_element(case, focused, require_viewport=True)
    if focused["descriptor"] != "header:language":
        raise DocumentationVisualError(
            f"{case}: language activation focus is {focused['descriptor']!r}, "
            "expected 'header:language'.")

    with page.expect_navigation(wait_until="networkidle"):
        if activation_method == "pointer":
            select.select_option(label=target_locale.upper())
        else:
            page.keyboard.press("ArrowDown")

    expected_target = _localized_route_path(relative_path, target_locale)
    expected_direction = "rtl" if target_locale == "ar" else "ltr"
    target_state = page.evaluate(
        r"""() => {
          const select = document.querySelector("[data-vh-language-select]");
          const bounds = select?.getBoundingClientRect();
          const style = select ? getComputedStyle(select) : null;
          return {
            direction: document.body.dir || document.documentElement.dir,
            language: document.documentElement.lang,
            palette: document.body.dataset.mdColorScheme,
            path: location.pathname,
            selectedLocales: select
              ? Array.from(select.selectedOptions).map(option => option.lang)
              : [],
            visible: Boolean(bounds && bounds.width > 0 && bounds.height > 0 &&
              style?.display !== "none" && style?.visibility !== "hidden"),
            withinViewport: Boolean(bounds && bounds.left >= 0 && bounds.right <= innerWidth &&
              bounds.top >= 0 && bounds.bottom <= innerHeight),
          };
        }""")
    expected_target_state = {
        "direction": expected_direction,
        "language": target_locale,
        "palette": palette,
        "path": expected_target,
        "selectedLocales": [target_locale],
        "visible": True,
        "withinViewport": True,
    }
    if target_state != expected_target_state:
        raise DocumentationVisualError(
            f"{case}: target language state is {target_state!r}, "
            f"expected {expected_target_state!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: localized route introduced overflow.")
    return _validate_accessibility(axe, page, f"{case} / localized route")


def _validate_version_activation(
    page: Page,
    case: str,
    activation_method: str,
    axe: Axe,
) -> str:
    control = page.locator("[data-vh-version-control]")
    summary = control.locator("summary")
    if control.count() != 1 or summary.count() != 1:
        raise DocumentationVisualError(
            f"{case}: expected one version control and summary, found "
            f"{control.count()} and {summary.count()}.")
    summary.scroll_into_view_if_needed()
    summary.focus()

    if activation_method == "pointer":
        summary.click()
    elif activation_method == "keyboard":
        page.keyboard.press("Enter")
    else:
        raise DocumentationVisualError(
            f"{case}: unsupported version activation method {activation_method!r}.")

    page.wait_for_function(
        """() => {
          const control = document.querySelector("[data-vh-version-control]");
          const summary = control?.querySelector("summary");
          return control?.open && summary?.getAttribute("aria-expanded") === "true" &&
            document.activeElement === summary;
        }""")
    opened = control.evaluate(
        r"""control => {
          const summary = control.querySelector("summary");
          const menu = control.querySelector(".vh-header-version__menu");
          const links = Array.from(menu?.querySelectorAll(".md-select__link") || []);
          const bounds = menu?.getBoundingClientRect();
          const target = link => {
            const url = new URL(link.href);
            return url.origin === location.origin ? `${url.pathname}${url.hash}` : url.href;
          };
          return {
            activeTargets: links.filter(
              link => link.getAttribute("aria-current") === "page"
            ).map(target),
            expanded: summary?.getAttribute("aria-expanded"),
            focused: document.activeElement === summary,
            labels: links.map(link => link.textContent.trim().replace(/\s+/g, " ")),
            menuDisplay: menu ? getComputedStyle(menu).display : null,
            menuVisible: Boolean(bounds && bounds.width > 0 && bounds.height > 0 &&
              bounds.left >= 0 && bounds.right <= innerWidth &&
              bounds.top >= 0 && bounds.bottom <= innerHeight),
            open: control.open,
            targets: links.map(target),
          };
        }""")
    expected_open = {
        "activeTargets": ["/"],
        "expanded": "true",
        "focused": True,
        "labels": [
            "main · 0.3.0 source",
            "Release candidate status",
            "Published package · 0.1.6",
        ],
        "menuDisplay": "block",
        "menuVisible": True,
        "open": True,
        "targets": [
            "/",
            "/project/release-readiness/",
            "https://pypi.org/project/voicehub/0.1.6/",
        ],
    }
    if opened != expected_open:
        raise DocumentationVisualError(
            f"{case}: opened version state is {opened!r}, expected {expected_open!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: opening version control introduced overflow.")
    axe_core = _validate_accessibility(axe, page, f"{case} / open")

    page.keyboard.press("Escape")
    page.wait_for_function(
        """() => {
          const control = document.querySelector("[data-vh-version-control]");
          const summary = control?.querySelector("summary");
          return !control?.open && summary?.getAttribute("aria-expanded") === "false" &&
            document.activeElement === summary;
        }""")
    closed = control.evaluate(
        """control => {
          const summary = control.querySelector("summary");
          const menu = control.querySelector(".vh-header-version__menu");
          return {
            expanded: summary?.getAttribute("aria-expanded"),
            focused: document.activeElement === summary,
            menuDisplay: menu ? getComputedStyle(menu).display : null,
            open: control.open,
          };
        }""")
    expected_closed = {
        "expanded": "false",
        "focused": True,
        "menuDisplay": "none",
        "open": False,
    }
    if closed != expected_closed:
        raise DocumentationVisualError(
            f"{case}: closed version state is {closed!r}, expected {expected_closed!r}.")
    focused = _active_focus_state(page)
    _validate_focused_element(case, focused, require_viewport=True)
    if focused["descriptor"] != "header:version":
        raise DocumentationVisualError(
            f"{case}: Escape restored focus to {focused['descriptor']!r}, "
            "expected 'header:version'.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: closing version control introduced overflow.")
    return axe_core


def _validate_search_activation(
    page: Page,
    case: str,
    activation_method: str,
    viewport: dict[str, Any],
    axe: Axe,
) -> str:
    trigger = page.locator("[data-vh-search-trigger]")
    input_element = page.locator(".md-search__input")
    if trigger.count() != 1 or input_element.count() != 1:
        raise DocumentationVisualError(
            f"{case}: expected one search trigger and input, found "
            f"{trigger.count()} and {input_element.count()}.")

    if activation_method == "pointer":
        trigger.click()
    elif activation_method == "keyboard":
        page.keyboard.press("Control+K")
    else:
        raise DocumentationVisualError(f"{case}: unsupported search activation method {activation_method!r}.")

    page.wait_for_function(
        """() => {
          const checkbox = document.querySelector("#__search");
          const trigger = document.querySelector("[data-vh-search-trigger]");
          const input = document.querySelector(".md-search__input");
          const inner = document.querySelector(".md-search__inner");
          return checkbox?.checked && trigger?.getAttribute("aria-expanded") === "true" &&
            document.activeElement === input && inner &&
            Number.parseFloat(getComputedStyle(inner).opacity) >= 0.999;
        }""")
    opened = page.evaluate(
        """() => {
          const checkbox = document.querySelector("#__search");
          const trigger = document.querySelector("[data-vh-search-trigger]");
          const input = document.querySelector(".md-search__input");
          const output = document.querySelector(".md-search__output");
          const scrollwrap = document.querySelector(".md-search__scrollwrap");
          const bounds = input?.getBoundingClientRect();
          return {
            bodyOpen: document.body.classList.contains("vh-search-open"),
            checked: checkbox?.checked,
            expanded: trigger?.getAttribute("aria-expanded"),
            focused: document.activeElement === input,
            inputTabIndex: input?.tabIndex,
            inputVisible: Boolean(bounds && bounds.width > 0 && bounds.height > 0 &&
              bounds.left >= 0 && bounds.right <= innerWidth &&
              bounds.top >= 0 && bounds.bottom <= innerHeight),
            outputHidden: output?.getAttribute("aria-hidden"),
            outputInert: output?.inert,
            scrollwrapTabIndex: scrollwrap?.tabIndex,
          };
        }""")
    expected_open = {
        "bodyOpen": True,
        "checked": True,
        "expanded": "true",
        "focused": True,
        "inputTabIndex": 0,
        "inputVisible": True,
        "outputHidden": "false",
        "outputInert": False,
        "scrollwrapTabIndex": 0,
    }
    if opened != expected_open:
        raise DocumentationVisualError(
            f"{case}: opened search state is {opened!r}, expected {expected_open!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: opening search introduced document overflow.")
    axe_core = _validate_accessibility(axe, page, f"{case} / open")

    page.keyboard.press("Escape")
    expected_focus = ("header:search-trigger" if viewport["name"] == "mobile" else "header:search")
    expected_input_tab_index = -1 if viewport["name"] == "mobile" else 0
    page.wait_for_function(
        """expected => {
          const checkbox = document.querySelector("#__search");
          const trigger = document.querySelector("[data-vh-search-trigger]");
          const input = document.querySelector(".md-search__input");
          const focusTarget = expected === "trigger" ? trigger : input;
          return !checkbox?.checked && trigger?.getAttribute("aria-expanded") === "false" &&
            document.activeElement === focusTarget;
        }""",
        arg="trigger" if viewport["name"] == "mobile" else "input",
    )
    closed = page.evaluate(
        """() => {
          const checkbox = document.querySelector("#__search");
          const trigger = document.querySelector("[data-vh-search-trigger]");
          const input = document.querySelector(".md-search__input");
          const output = document.querySelector(".md-search__output");
          const scrollwrap = document.querySelector(".md-search__scrollwrap");
          return {
            bodyOpen: document.body.classList.contains("vh-search-open"),
            checked: checkbox?.checked,
            expanded: trigger?.getAttribute("aria-expanded"),
            inputTabIndex: input?.tabIndex,
            outputHidden: output?.getAttribute("aria-hidden"),
            outputInert: output?.inert,
            scrollwrapTabIndex: scrollwrap?.tabIndex,
          };
        }""")
    expected_closed = {
        "bodyOpen": False,
        "checked": False,
        "expanded": "false",
        "inputTabIndex": expected_input_tab_index,
        "outputHidden": "true",
        "outputInert": True,
        "scrollwrapTabIndex": -1,
    }
    if closed != expected_closed:
        raise DocumentationVisualError(
            f"{case}: closed search state is {closed!r}, expected {expected_closed!r}.")
    focused = _active_focus_state(page)
    _validate_focused_element(case, focused, require_viewport=True)
    if focused["descriptor"] != expected_focus:
        raise DocumentationVisualError(
            f"{case}: Escape restored focus to {focused['descriptor']!r}, "
            f"expected {expected_focus!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: closing search introduced document overflow.")
    return axe_core


def _validate_table_of_contents_activation(
    page: Page,
    case: str,
    activation_method: str,
) -> None:
    links = page.locator(".md-sidebar--secondary a.md-nav__link[href^='#']")
    if links.count() == 0:
        raise DocumentationVisualError(f"{case}: rendered page exposes no table-of-contents links.")
    target_link = links.last
    target_link.scroll_into_view_if_needed()

    if activation_method == "pointer":
        target_link.click()
    elif activation_method == "keyboard":
        target_link.focus()
        focused = _active_focus_state(page)
        _validate_focused_element(case, focused, require_viewport=True)
        if not focused["descriptor"].startswith("toc:"):
            raise DocumentationVisualError(
                f"{case}: keyboard target is {focused['descriptor']!r}, expected a TOC link.")
        page.keyboard.press("Enter")
    else:
        raise DocumentationVisualError(
            f"{case}: unsupported table-of-contents activation method {activation_method!r}.")

    page.wait_for_function(
        """link => {
          const activeLinks = Array.from(document.querySelectorAll(
            ".md-sidebar--secondary a.md-nav__link[href^='#'].md-nav__link--active"
          ));
          return window.location.hash === link.hash && activeLinks.length === 1 &&
            activeLinks[0] === link;
        }""",
        arg=target_link.element_handle(),
        timeout=5000,
    )
    page.wait_for_timeout(600)
    state = target_link.evaluate(
        """link => {
          const hash = link.hash;
          const target = document.getElementById(decodeURIComponent(hash.slice(1)));
          const header = document.querySelector(".md-header");
          const activeLinks = Array.from(document.querySelectorAll(
            ".md-sidebar--secondary a.md-nav__link[href^='#'].md-nav__link--active"
          ));
          const targetBounds = target?.getBoundingClientRect();
          const headerBounds = header?.getBoundingClientRect();
          return {
            activeHashes: activeLinks.map(activeLink => activeLink.hash),
            currentHash: window.location.hash,
            focused: document.activeElement === link,
            hash,
            isTarget: document.querySelector(":target") === target,
            targetBottom: targetBounds?.bottom,
            targetTop: targetBounds?.top,
            visibleHeaderBottom: Math.max(0, headerBounds?.bottom || 0),
          };
        }""")
    if state["currentHash"] != state["hash"] or state["activeHashes"] != [state["hash"]]:
        raise DocumentationVisualError(
            f"{case}: settled TOC state is {state!r}; expected one active link for the current hash.")
    if not state["isTarget"]:
        raise DocumentationVisualError(f"{case}: current hash does not identify the CSS :target heading.")
    if activation_method == "keyboard" and not state["focused"]:
        raise DocumentationVisualError(f"{case}: Enter activation moved focus away from the TOC link.")
    if (state["targetTop"] is None or state["targetBottom"] is None or
            state["targetTop"] < state["visibleHeaderBottom"] - 2 or
            state["targetTop"] >= page.viewport_size["height"] or state["targetBottom"] <= 0):
        raise DocumentationVisualError(
            f"{case}: hash target is not visibly aligned beneath the header: {state!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: TOC activation introduced document overflow.")


def _reset_quickstart_tabs(page: Page) -> None:
    tab_sets = page.locator(".md-content__inner .tabbed-set")
    for set_index in range(tab_sets.count()):
        tab_sets.nth(set_index).locator("input[type='radio']").first.evaluate("input => input.click()")
    page.wait_for_function(
        "() => Array.from("
        "document.querySelectorAll('.md-content__inner .tabbed-set')"
        ").every(tabSet => tabSet.querySelector(\"input[type='radio']\")?.checked)")


def _validate_quickstart_tabs(page: Page, case: str) -> None:
    tab_sets = page.locator(".md-content__inner .tabbed-set")
    if tab_sets.count() != len(QUICKSTART_TAB_LABELS):
        raise DocumentationVisualError(
            f"{case}: Quickstart exposes {tab_sets.count()} tab sets, "
            f"expected {len(QUICKSTART_TAB_LABELS)}.")

    for set_index, expected_labels in enumerate(QUICKSTART_TAB_LABELS):
        tab_set = tab_sets.nth(set_index)
        inputs = tab_set.locator("input[type='radio']")
        target_index = len(expected_labels) - 1
        target = inputs.nth(target_index)
        target_id = target.get_attribute("id")
        if not target_id:
            raise DocumentationVisualError(
                f"{case}: Quickstart tab set {set_index + 1} has an input without an id.")
        inputs.first.evaluate("input => input.focus({preventScroll: true})")
        for step_index in range(1, target_index + 1):
            step_source = inputs.nth(step_index - 1)
            step_source.evaluate(
                """input => input.ownerDocument.addEventListener("keydown", event => {
                  input.dataset.vhArrowDefaultPrevented = String(event.defaultPrevented);
                }, {once: true})""")
            page.keyboard.press("ArrowRight")
            step_source_handle = step_source.element_handle()
            page.wait_for_function(
                "input => input.dataset.vhArrowDefaultPrevented === 'true'",
                arg=step_source_handle,
            )
            step_target = inputs.nth(step_index)
            page.wait_for_function(
                "input => input.checked && document.activeElement === input",
                arg=step_target.element_handle(),
            )
        target_handle = target.element_handle()
        page.wait_for_function(
            """async input => {
              const focusRequest = input.dataset.vhFocusRequest;
              const readState = () => {
                const label = input.parentElement?.querySelector(
                  `.tabbed-labels > label[for='${input.id}']`
                );
                const bounds = label?.getBoundingClientRect();
                return bounds ? {
                  bottom: bounds.bottom,
                  height: bounds.height,
                  left: bounds.left,
                  right: bounds.right,
                  top: bounds.top,
                  width: bounds.width,
                } : null;
              };
              const before = readState();
              await new Promise(resolve => requestAnimationFrame(
                () => requestAnimationFrame(resolve)
              ));
              const after = readState();
              const viewportTolerance = 1;
              const stableTolerance = 0.25;
              return input.checked && document.activeElement === input &&
                focusRequest && input.dataset.vhFocusRequest === focusRequest &&
                input.dataset.vhFocusSettled === focusRequest &&
                before && after && after.width > 0 && after.height > 0 &&
                Math.abs(after.left - before.left) <= stableTolerance &&
                Math.abs(after.right - before.right) <= stableTolerance &&
                Math.abs(after.top - before.top) <= stableTolerance &&
                Math.abs(after.bottom - before.bottom) <= stableTolerance &&
                after.left >= -viewportTolerance &&
                after.right <= innerWidth + viewportTolerance &&
                after.top >= -viewportTolerance &&
                after.bottom <= innerHeight + viewportTolerance;
            }""",
            arg=target_handle,
        )
        focused = _active_focus_state(page)
        _validate_focused_element(
            f"{case} / tab set {set_index + 1}",
            focused,
            require_viewport=True,
        )
        expected_descriptor = f"tab:{expected_labels[target_index]}"
        if focused["descriptor"] != expected_descriptor:
            raise DocumentationVisualError(
                f"{case}: focused tab is {focused['descriptor']!r}, "
                f"expected {expected_descriptor!r}.")
        selected = tab_set.evaluate(
            """tabSet => {
              const visible = element => {
                const bounds = element.getBoundingClientRect();
                const style = getComputedStyle(element);
                return bounds.width > 0 && bounds.height > 0 &&
                  style.display !== "none" && style.visibility !== "hidden";
              };
              const inputs = Array.from(tabSet.querySelectorAll("input[type='radio']"));
              const blocks = Array.from(tabSet.querySelectorAll(
                ".tabbed-content > .tabbed-block"
              ));
              return {
                checked: inputs.findIndex(input => input.checked),
                visibleBlocks: blocks.map(visible),
              };
            }""")
        expected_visibility = [index == target_index for index in range(len(expected_labels))]
        if selected["checked"] != target_index or selected["visibleBlocks"] != expected_visibility:
            raise DocumentationVisualError(
                f"{case}: Quickstart tab set {set_index + 1} did not activate "
                f"{expected_labels[target_index]!r}: {selected!r}.")
    if _rendered_state(page)["overflow"] != 0:
        raise DocumentationVisualError(f"{case}: Quickstart tab activation introduced overflow.")


def _validate_quickstart_page_copy(page: Page, case: str, key: str) -> None:
    _validate_page_copy(page, f"{case} / Quickstart page copy", key, wait_for_idle=True)


def validate_site(
    site_directory: Path,
    *,
    screenshot_baselines_path: Path | None = None,
    update_screenshot_baselines: bool = False,
    viewport_names: tuple[str, ...] | None = None,
    palette_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not site_directory.is_dir():
        raise DocumentationVisualError(
            f"Rendered site directory does not exist: {site_directory}. Run mkdocs build first.")

    requested_viewport_names = set(viewport_names or VIEWPORTS_BY_NAME)
    unknown_viewport_names = requested_viewport_names - VIEWPORTS_BY_NAME.keys()
    if unknown_viewport_names:
        raise DocumentationVisualError(f"Unknown viewport names: {sorted(unknown_viewport_names)!r}.")
    selected_viewports = tuple(
        viewport for viewport in VIEWPORTS if viewport["name"] in requested_viewport_names)
    selected_non_mobile_viewports = tuple(
        viewport for viewport in selected_viewports if viewport["name"] != "mobile")
    selected_viewport_names = {viewport["name"] for viewport in selected_viewports}
    requested_palette_names = set(palette_names or PALETTES)
    unknown_palette_names = requested_palette_names - PALETTES.keys()
    if unknown_palette_names:
        raise DocumentationVisualError(f"Unknown palette names: {sorted(unknown_palette_names)!r}.")
    selected_palette_names = tuple(palette for palette in PALETTES if palette in requested_palette_names)

    handler = partial(_QuietRequestHandler, directory=str(site_directory.resolve()))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    accessibility_cases = 0
    axe_core = "unknown"
    case_count = 0
    focus_cycle_cases = 0
    focus_steps = 0
    home_cases = 0
    home_interaction_cases = 0
    installation_cases = 0
    installation_code_interaction_cases = 0
    installation_page_interaction_cases = 0
    model_index_cases = 0
    model_index_interaction_cases = 0
    inference_cases = 0
    inference_interaction_cases = 0
    quickstart_cases = 0
    quickstart_interaction_cases = 0
    quickstart_page_interaction_cases = 0
    search_activation_cases = 0
    search_pointer_activation_cases = 0
    search_keyboard_activation_cases = 0
    search_interaction_accessibility_cases = 0
    language_activation_cases = 0
    language_pointer_activation_cases = 0
    language_keyboard_activation_cases = 0
    language_interaction_accessibility_cases = 0
    theme_activation_cases = 0
    theme_pointer_activation_cases = 0
    theme_keyboard_activation_cases = 0
    theme_interaction_accessibility_cases = 0
    source_activation_cases = 0
    source_pointer_activation_cases = 0
    source_keyboard_activation_cases = 0
    source_interaction_accessibility_cases = 0
    page_action_cases = 0
    page_action_edit_activations = 0
    page_action_footer_activations = 0
    page_action_back_to_top_activations = 0
    page_action_pointer_cases = 0
    page_action_keyboard_cases = 0
    page_action_interaction_accessibility_cases = 0
    root_branch_activation_cases = 0
    root_branch_pointer_activation_cases = 0
    root_branch_keyboard_activation_cases = 0
    root_branch_interaction_accessibility_cases = 0
    nested_branch_activation_cases = 0
    nested_branch_pointer_activation_cases = 0
    nested_branch_keyboard_activation_cases = 0
    nested_branch_interaction_accessibility_cases = 0
    version_activation_cases = 0
    version_pointer_activation_cases = 0
    version_keyboard_activation_cases = 0
    version_interaction_accessibility_cases = 0
    toc_activation_cases = 0
    toc_pointer_activation_cases = 0
    toc_keyboard_activation_cases = 0
    toc_interaction_accessibility_cases = 0
    speecht5_cases = 0
    speecht5_interaction_cases = 0
    trainer_cases = 0
    trainer_interaction_cases = 0
    optimization_cases = 0
    optimization_interaction_cases = 0
    contribution_cases = 0
    contribution_interaction_cases = 0
    model_api_cases = 0
    model_api_interaction_cases = 0
    keyboard_activation_cases = 0
    interactive_accessibility_cases = 0
    screenshot_cases = 0
    screenshot_signatures: dict[str, dict[str, Any]] = {}
    if screenshot_baselines_path is None and not update_screenshot_baselines:
        screenshot_baselines_path = _platform_screenshot_baselines_path()
    screenshot_baselines = (
        None if update_screenshot_baselines else _load_screenshot_baselines(screenshot_baselines_path))
    chromium_version = "unknown"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            chromium_version = browser.version
            try:
                page = browser.new_page()
                axe = Axe()
                for palette in selected_palette_names:
                    for viewport in selected_viewports:
                        page.set_viewport_size({
                            "width": viewport["width"],
                            "height": viewport["height"],
                        })
                        for relative_path, expectation in REPRESENTATIVE_ROUTES.items():
                            case = f"{relative_path} / {viewport['name']} / {palette}"
                            page.goto(
                                _route_url(base_url, relative_path),
                                wait_until="networkidle",
                            )
                            _set_palette(page, palette)
                            page.wait_for_function("document.fonts.status === 'loaded'")
                            if relative_path == QUICKSTART_ROUTE:
                                _reset_quickstart_tabs(page)
                            _validate_case(
                                case=case,
                                state=_rendered_state(page),
                                route_expectation=expectation,
                                viewport=viewport,
                                palette=palette,
                                hide_secondary=relative_path == SPEECHT5_ROUTE,
                            )
                            if relative_path == HOME_ROUTE:
                                _validate_home_state(page, case)
                                home_cases += 1
                            if relative_path == INSTALLATION_ROUTE:
                                _validate_installation_state(page, case)
                                installation_cases += 1
                            if relative_path == MODEL_INDEX_ROUTE:
                                _validate_model_index_state(page, case, viewport)
                                model_index_cases += 1
                            if relative_path == INFERENCE_ROUTE:
                                _validate_inference_state(page, case)
                                inference_cases += 1
                            if relative_path == QUICKSTART_ROUTE:
                                _validate_quickstart_state(page, case)
                                quickstart_cases += 1
                            if relative_path == SPEECHT5_ROUTE:
                                _validate_speecht5_state(page, case, viewport)
                                speecht5_cases += 1
                            if relative_path == TRAINER_ROUTE:
                                _validate_trainer_state(page, case)
                                trainer_cases += 1
                            if relative_path == OPTIMIZATION_ROUTE:
                                _validate_optimization_state(page, case)
                                optimization_cases += 1
                            if relative_path == CONTRIBUTION_ROUTE:
                                _validate_contribution_state(page, case)
                                contribution_cases += 1
                            if relative_path == MODEL_API_ROUTE:
                                _validate_model_api_state(page, case)
                                model_api_cases += 1
                            case_axe_core = _validate_accessibility(axe, page, case)
                            if axe_core not in ("unknown", case_axe_core):
                                raise DocumentationVisualError(
                                    f"{case}: Axe engine changed from {axe_core!r} to {case_axe_core!r}.")
                            axe_core = case_axe_core
                            accessibility_cases += 1
                            screenshot_key = _screenshot_case_key(
                                relative_path,
                                viewport["name"],
                                palette,
                            )
                            signature = _screenshot_signature(
                                page.screenshot(animations="disabled", scale="css"),
                                viewport,
                            )
                            screenshot_signatures[screenshot_key] = signature
                            if screenshot_baselines is not None:
                                _compare_screenshot_signature(
                                    case,
                                    signature,
                                    screenshot_baselines["cases"].get(screenshot_key),
                                )
                            if not update_screenshot_baselines:
                                focus_steps += _validate_focus_cycle(
                                    page,
                                    f"{case} / native Tab",
                                    _focus_prefix_for_viewport(viewport),
                                    require_viewport=True,
                                )
                                focus_cycle_cases += 1
                                if relative_path == HOME_ROUTE:
                                    key = "Enter" if palette == "default" else "Space"
                                    _validate_home_page_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied page",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    home_interaction_cases += 1
                                if relative_path == INSTALLATION_ROUTE:
                                    key = "Enter" if palette == "default" else "Space"
                                    _validate_installation_code_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied code",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    installation_code_interaction_cases += 1
                                    _validate_installation_page_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied page",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    installation_page_interaction_cases += 1
                                if relative_path == QUICKSTART_ROUTE:
                                    page.reload(wait_until="networkidle")
                                    _set_palette(page, palette)
                                    page.wait_for_function("document.fonts.status === 'loaded'")
                                    _reset_quickstart_tabs(page)
                                    _validate_quickstart_tabs(page, case)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / activated tabs",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    quickstart_interaction_cases += 1
                                    key = "Enter" if palette == "default" else "Space"
                                    _validate_quickstart_page_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied page",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    quickstart_page_interaction_cases += 1
                                if relative_path == INFERENCE_ROUTE:
                                    key = "Enter" if palette == "default" else "Space"
                                    _validate_inference_code_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied code",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    inference_interaction_cases += 1
                                if relative_path == MODEL_INDEX_ROUTE:
                                    _validate_model_explorer_filters(page, case)
                                    key = "Enter" if palette == "default" else "Space"
                                    _validate_model_index_page_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied page",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    model_index_interaction_cases += 1
                                if relative_path == SPEECHT5_ROUTE:
                                    _validate_speecht5_section_navigation(page, case)
                                    page.reload(wait_until="networkidle")
                                    _set_palette(page, palette)
                                    page.wait_for_function("document.fonts.status === 'loaded'")
                                    _validate_speecht5_model_id_copy(page, case)
                                    key = "Enter" if palette == "default" else "Space"
                                    _validate_speecht5_page_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied page",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    speecht5_interaction_cases += 1
                                if relative_path == TRAINER_ROUTE:
                                    key = "Enter" if palette == "default" else "Space"
                                    _validate_trainer_page_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied page",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    trainer_interaction_cases += 1
                                if relative_path == OPTIMIZATION_ROUTE:
                                    key = "Enter" if palette == "default" else "Space"
                                    _validate_optimization_page_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied page",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    optimization_interaction_cases += 1
                                if relative_path == CONTRIBUTION_ROUTE:
                                    key = "Enter" if palette == "default" else "Space"
                                    _validate_contribution_page_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied page",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    contribution_interaction_cases += 1
                                if relative_path == MODEL_API_ROUTE:
                                    key = "Enter" if palette == "default" else "Space"
                                    _validate_model_api_page_copy(page, case, key)
                                    case_axe_core = _validate_accessibility(
                                        axe,
                                        page,
                                        f"{case} / copied page",
                                    )
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    model_api_interaction_cases += 1
                            screenshot_cases += 1
                            case_count += 1

                if not update_screenshot_baselines:
                    for palette in selected_palette_names:
                        activation_method = ROOT_BRANCH_ACTIVATION_METHOD_BY_PALETTE[palette]
                        for viewport in selected_non_mobile_viewports:
                            page.set_viewport_size({
                                "width": viewport["width"],
                                "height": viewport["height"],
                            })
                            for branch_label in TOP_LEVEL_NAVIGATION:
                                case = (
                                    f"{KEYBOARD_ROUTE} / {viewport['name']} / {palette} / "
                                    f"{branch_label} root branch {activation_method} activation")
                                page.goto(_route_url(base_url, KEYBOARD_ROUTE), wait_until="networkidle")
                                _set_keyboard_palette(page, palette)
                                case_axe_core = _validate_root_branch_activation(
                                    page,
                                    case,
                                    branch_label,
                                    activation_method,
                                    palette,
                                    axe,
                                )
                                if axe_core != case_axe_core:
                                    raise DocumentationVisualError(
                                        f"{case}: Axe engine changed from {axe_core!r} "
                                        f"to {case_axe_core!r}.")
                                root_branch_activation_cases += 1
                                root_branch_interaction_accessibility_cases += 1
                                if activation_method == "pointer":
                                    root_branch_pointer_activation_cases += 1
                                else:
                                    root_branch_keyboard_activation_cases += 1

                    for palette in selected_palette_names:
                        activation_method = NESTED_BRANCH_ACTIVATION_METHOD_BY_PALETTE[palette]
                        for viewport in selected_non_mobile_viewports:
                            page.set_viewport_size({
                                "width": viewport["width"],
                                "height": viewport["height"],
                            })
                            for branch_path, expected_initial_expanded in SPEECHT5_NESTED_BRANCH_STATES:
                                case = (
                                    f"{SPEECHT5_ROUTE} / {viewport['name']} / {palette} / "
                                    f"{' > '.join(branch_path)} nested branch "
                                    f"{activation_method} activation")
                                page.goto(_route_url(base_url, SPEECHT5_ROUTE), wait_until="networkidle")
                                _set_keyboard_palette(page, palette)
                                case_axe_core = _validate_nested_branch_activation(
                                    page,
                                    case,
                                    branch_path,
                                    expected_initial_expanded,
                                    activation_method,
                                    palette,
                                    axe,
                                )
                                if axe_core != case_axe_core:
                                    raise DocumentationVisualError(
                                        f"{case}: Axe engine changed from {axe_core!r} "
                                        f"to {case_axe_core!r}.")
                                nested_branch_activation_cases += 1
                                nested_branch_interaction_accessibility_cases += 1
                                if activation_method == "pointer":
                                    nested_branch_pointer_activation_cases += 1
                                else:
                                    nested_branch_keyboard_activation_cases += 1

                    for palette in selected_palette_names:
                        activation_method = PAGE_ACTION_METHOD_BY_PALETTE[palette]
                        for viewport in selected_viewports:
                            page.set_viewport_size({
                                "width": viewport["width"],
                                "height": viewport["height"],
                            })
                            for relative_path in PAGE_ACTION_ROUTES:
                                case = (
                                    f"{relative_path} / {viewport['name']} / {palette} / "
                                    f"page actions {activation_method} activation")
                                page.goto(_route_url(base_url, relative_path), wait_until="networkidle")
                                _set_keyboard_palette(page, palette)
                                case_axe_core, footer_activations = _validate_page_actions(
                                    page,
                                    case,
                                    base_url,
                                    relative_path,
                                    REPRESENTATIVE_PAGE_ACTIONS[relative_path],
                                    activation_method,
                                    palette,
                                    axe,
                                )
                                if axe_core != case_axe_core:
                                    raise DocumentationVisualError(
                                        f"{case}: Axe engine changed from {axe_core!r} "
                                        f"to {case_axe_core!r}.")
                                page_action_cases += 1
                                page_action_edit_activations += 1
                                page_action_footer_activations += footer_activations
                                page_action_back_to_top_activations += 1
                                page_action_interaction_accessibility_cases += 1
                                if activation_method == "pointer":
                                    page_action_pointer_cases += 1
                                else:
                                    page_action_keyboard_cases += 1

                    for palette in selected_palette_names:
                        activation_method = SOURCE_ACTIVATION_METHOD_BY_PALETTE[palette]
                        for viewport in selected_non_mobile_viewports:
                            page.set_viewport_size({
                                "width": viewport["width"],
                                "height": viewport["height"],
                            })
                            for relative_path in REPRESENTATIVE_ROUTES:
                                route_url = _route_url(base_url, relative_path)
                                case = (
                                    f"{relative_path} / {viewport['name']} / {palette} / "
                                    f"source {activation_method} activation")
                                page.goto(route_url, wait_until="networkidle")
                                _set_keyboard_palette(page, palette)
                                case_axe_core = _validate_source_activation(
                                    page,
                                    case,
                                    activation_method,
                                    palette,
                                    axe,
                                )
                                if axe_core != case_axe_core:
                                    raise DocumentationVisualError(
                                        f"{case}: Axe engine changed from {axe_core!r} "
                                        f"to {case_axe_core!r}.")
                                source_activation_cases += 1
                                source_interaction_accessibility_cases += 1
                                if activation_method == "pointer":
                                    source_pointer_activation_cases += 1
                                else:
                                    source_keyboard_activation_cases += 1

                    for palette in selected_palette_names:
                        activation_method = THEME_ACTIVATION_METHOD_BY_PALETTE[palette]
                        target_palette = THEME_TARGET_BY_PALETTE[palette]
                        for viewport in selected_non_mobile_viewports:
                            page.set_viewport_size({
                                "width": viewport["width"],
                                "height": viewport["height"],
                            })
                            for relative_path in REPRESENTATIVE_ROUTES:
                                route_url = _route_url(base_url, relative_path)
                                case = (
                                    f"{relative_path} / {viewport['name']} / {palette} / "
                                    f"theme {activation_method} activation to {target_palette}")
                                page.goto(route_url, wait_until="networkidle")
                                _set_keyboard_palette(page, palette)
                                case_axe_core = _validate_theme_activation(
                                    page,
                                    case,
                                    activation_method,
                                    palette,
                                    target_palette,
                                    axe,
                                )
                                if axe_core != case_axe_core:
                                    raise DocumentationVisualError(
                                        f"{case}: Axe engine changed from {axe_core!r} "
                                        f"to {case_axe_core!r}.")
                                theme_activation_cases += 1
                                theme_interaction_accessibility_cases += 1
                                if activation_method == "pointer":
                                    theme_pointer_activation_cases += 1
                                else:
                                    theme_keyboard_activation_cases += 1

                    for palette in selected_palette_names:
                        activation_method = LANGUAGE_ACTIVATION_METHOD_BY_PALETTE[palette]
                        target_locale = LANGUAGE_TARGET_BY_PALETTE[palette]
                        for viewport in selected_non_mobile_viewports:
                            page.set_viewport_size({
                                "width": viewport["width"],
                                "height": viewport["height"],
                            })
                            for relative_path in REPRESENTATIVE_ROUTES:
                                route_url = f"{base_url}{_localized_route_path(relative_path, 'en')}"
                                case = (
                                    f"{relative_path} / {viewport['name']} / {palette} / "
                                    f"language {activation_method} activation to {target_locale}")
                                page.goto(route_url, wait_until="networkidle")
                                _set_keyboard_palette(page, palette)
                                case_axe_core = _validate_language_activation(
                                    page,
                                    case,
                                    relative_path,
                                    activation_method,
                                    target_locale,
                                    palette,
                                    axe,
                                )
                                if axe_core != case_axe_core:
                                    raise DocumentationVisualError(
                                        f"{case}: Axe engine changed from {axe_core!r} "
                                        f"to {case_axe_core!r}.")
                                language_activation_cases += 1
                                language_interaction_accessibility_cases += 1
                                if activation_method == "pointer":
                                    language_pointer_activation_cases += 1
                                else:
                                    language_keyboard_activation_cases += 1

                    for palette in selected_palette_names:
                        activation_method = VERSION_ACTIVATION_METHOD_BY_PALETTE[palette]
                        for viewport in selected_viewports:
                            page.set_viewport_size({
                                "width": viewport["width"],
                                "height": viewport["height"],
                            })
                            for relative_path in REPRESENTATIVE_ROUTES:
                                route_url = _route_url(base_url, relative_path)
                                case = (
                                    f"{relative_path} / {viewport['name']} / {palette} / "
                                    f"version {activation_method} activation")
                                page.goto(route_url, wait_until="networkidle")
                                _set_keyboard_palette(page, palette)
                                case_axe_core = _validate_version_activation(
                                    page,
                                    case,
                                    activation_method,
                                    axe,
                                )
                                if axe_core != case_axe_core:
                                    raise DocumentationVisualError(
                                        f"{case}: Axe engine changed from {axe_core!r} "
                                        f"to {case_axe_core!r}.")
                                version_activation_cases += 1
                                version_interaction_accessibility_cases += 1
                                if activation_method == "pointer":
                                    version_pointer_activation_cases += 1
                                else:
                                    version_keyboard_activation_cases += 1

                    for palette in selected_palette_names:
                        for viewport in selected_viewports:
                            page.set_viewport_size({
                                "width": viewport["width"],
                                "height": viewport["height"],
                            })
                            activation_method = SEARCH_ACTIVATION_METHOD_BY_VIEWPORT[viewport["name"]]
                            for relative_path in REPRESENTATIVE_ROUTES:
                                route_url = _route_url(base_url, relative_path)
                                case = (
                                    f"{relative_path} / {viewport['name']} / {palette} / "
                                    f"search {activation_method} activation")
                                page.goto(route_url, wait_until="networkidle")
                                _set_keyboard_palette(page, palette)
                                case_axe_core = _validate_search_activation(
                                    page,
                                    case,
                                    activation_method,
                                    viewport,
                                    axe,
                                )
                                if axe_core != case_axe_core:
                                    raise DocumentationVisualError(
                                        f"{case}: Axe engine changed from {axe_core!r} "
                                        f"to {case_axe_core!r}.")
                                search_activation_cases += 1
                                search_interaction_accessibility_cases += 1
                                if activation_method == "pointer":
                                    search_pointer_activation_cases += 1
                                else:
                                    search_keyboard_activation_cases += 1

                    if "desktop" in selected_viewport_names:
                        page.set_viewport_size({"width": 1440, "height": 900})
                        for palette in selected_palette_names:
                            for relative_path in REPRESENTATIVE_ROUTES:
                                if relative_path == SPEECHT5_ROUTE:
                                    continue
                                route_url = _route_url(base_url, relative_path)
                                for activation_method in TOC_ACTIVATION_METHODS:
                                    case = (
                                        f"{relative_path} / desktop / {palette} / "
                                        f"TOC {activation_method} activation")
                                    page.goto(route_url, wait_until="networkidle")
                                    _set_keyboard_palette(page, palette)
                                    _validate_table_of_contents_activation(
                                        page,
                                        case,
                                        activation_method,
                                    )
                                    case_axe_core = _validate_accessibility(axe, page, case)
                                    if axe_core != case_axe_core:
                                        raise DocumentationVisualError(
                                            f"{case}: Axe engine changed from {axe_core!r} "
                                            f"to {case_axe_core!r}.")
                                    toc_activation_cases += 1
                                    toc_interaction_accessibility_cases += 1
                                    if activation_method == "pointer":
                                        toc_pointer_activation_cases += 1
                                    else:
                                        toc_keyboard_activation_cases += 1

                    keyboard_url = _route_url(base_url, KEYBOARD_ROUTE)
                    for palette in selected_palette_names:
                        for viewport in selected_viewports:
                            page.set_viewport_size({
                                "width": viewport["width"],
                                "height": viewport["height"],
                            })
                            for state in INTERACTIVE_ACCESSIBILITY_STATES:
                                if state == "drawer-open" and viewport["name"] != "mobile":
                                    continue
                                if state == "branch-open" and viewport["name"] == "mobile":
                                    continue
                                case = (f"{KEYBOARD_ROUTE} / {viewport['name']} / {palette} / {state}")
                                page.goto(keyboard_url, wait_until="networkidle")
                                _set_keyboard_palette(page, palette)
                                _prepare_interactive_accessibility_state(page, state, viewport)
                                case_axe_core = _validate_accessibility(axe, page, case)
                                if axe_core != case_axe_core:
                                    raise DocumentationVisualError(
                                        f"{case}: Axe engine changed from {axe_core!r} "
                                        f"to {case_axe_core!r}.")
                                interactive_accessibility_cases += 1

                    if "mobile" in selected_viewport_names:
                        page.set_viewport_size({"width": 390, "height": 844})
                        for key, palette in DRAWER_ACTIVATION_CASES:
                            if palette not in selected_palette_names:
                                continue
                            page.goto(keyboard_url, wait_until="networkidle")
                            _set_keyboard_palette(page, palette)
                            _validate_mobile_drawer_activation(page, key, palette)
                            keyboard_activation_cases += 1
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    if update_screenshot_baselines:
        return {
            "cases": dict(sorted(screenshot_signatures.items())),
            "chromium": chromium_version,
            "playwright": "1.62.0",
            "schema_version": SCREENSHOT_SCHEMA_VERSION,
        }

    baseline_keys = {
        key
        for key in screenshot_baselines["cases"] if key.rsplit("|", 2)[1] in selected_viewport_names and
        key.rsplit("|", 2)[2] in selected_palette_names
    }
    actual_keys = set(screenshot_signatures)
    if baseline_keys != actual_keys:
        raise DocumentationVisualError(
            "Screenshot baseline case inventory differs: "
            f"missing={sorted(actual_keys - baseline_keys)!r}, "
            f"orphaned={sorted(baseline_keys - actual_keys)!r}.")

    keyboard_cases = (
        contribution_interaction_cases + focus_cycle_cases + home_interaction_cases +
        installation_code_interaction_cases + installation_page_interaction_cases +
        keyboard_activation_cases + language_keyboard_activation_cases + model_api_interaction_cases +
        model_index_interaction_cases + optimization_interaction_cases + inference_interaction_cases +
        quickstart_interaction_cases + quickstart_page_interaction_cases + search_keyboard_activation_cases +
        speecht5_interaction_cases + theme_keyboard_activation_cases + toc_keyboard_activation_cases +
        trainer_interaction_cases + version_keyboard_activation_cases + source_keyboard_activation_cases)
    keyboard_cases += root_branch_keyboard_activation_cases
    keyboard_cases += nested_branch_keyboard_activation_cases
    keyboard_cases += page_action_keyboard_cases
    return {
        "accessibility_cases": accessibility_cases,
        "axe_core": axe_core,
        "cases": case_count,
        "contribution_cases": contribution_cases,
        "contribution_interaction_cases": contribution_interaction_cases,
        "focus_cycle_cases": focus_cycle_cases,
        "focus_steps": focus_steps,
        "home_cases": home_cases,
        "home_interaction_cases": home_interaction_cases,
        "installation_cases": installation_cases,
        "installation_code_interaction_cases": installation_code_interaction_cases,
        "installation_page_interaction_cases": installation_page_interaction_cases,
        "interactive_accessibility_cases": interactive_accessibility_cases,
        "keyboard_activation_cases": keyboard_activation_cases,
        "keyboard_cases": keyboard_cases,
        "language_activation_cases": language_activation_cases,
        "language_interaction_accessibility_cases": language_interaction_accessibility_cases,
        "language_keyboard_activation_cases": language_keyboard_activation_cases,
        "language_pointer_activation_cases": language_pointer_activation_cases,
        "model_api_cases": model_api_cases,
        "model_api_interaction_cases": model_api_interaction_cases,
        "model_index_cases": model_index_cases,
        "model_index_interaction_cases": model_index_interaction_cases,
        "nested_branch_activation_cases": nested_branch_activation_cases,
        "nested_branch_interaction_accessibility_cases": nested_branch_interaction_accessibility_cases,
        "nested_branch_keyboard_activation_cases": nested_branch_keyboard_activation_cases,
        "nested_branch_pointer_activation_cases": nested_branch_pointer_activation_cases,
        "optimization_cases": optimization_cases,
        "optimization_interaction_cases": optimization_interaction_cases,
        "page_action_back_to_top_activations": page_action_back_to_top_activations,
        "page_action_cases": page_action_cases,
        "page_action_edit_activations": page_action_edit_activations,
        "page_action_footer_activations": page_action_footer_activations,
        "page_action_interaction_accessibility_cases": page_action_interaction_accessibility_cases,
        "page_action_keyboard_cases": page_action_keyboard_cases,
        "page_action_pointer_cases": page_action_pointer_cases,
        "palettes": len(selected_palette_names),
        "inference_cases": inference_cases,
        "inference_interaction_cases": inference_interaction_cases,
        "quickstart_cases": quickstart_cases,
        "quickstart_interaction_cases": quickstart_interaction_cases,
        "quickstart_page_interaction_cases": quickstart_page_interaction_cases,
        "representative_routes": len(REPRESENTATIVE_ROUTES),
        "root_branch_activation_cases": root_branch_activation_cases,
        "root_branch_interaction_accessibility_cases": root_branch_interaction_accessibility_cases,
        "root_branch_keyboard_activation_cases": root_branch_keyboard_activation_cases,
        "root_branch_pointer_activation_cases": root_branch_pointer_activation_cases,
        "search_activation_cases": search_activation_cases,
        "search_interaction_accessibility_cases": search_interaction_accessibility_cases,
        "search_keyboard_activation_cases": search_keyboard_activation_cases,
        "search_pointer_activation_cases": search_pointer_activation_cases,
        "screenshot_cases": screenshot_cases,
        "speecht5_cases": speecht5_cases,
        "speecht5_interaction_cases": speecht5_interaction_cases,
        "source_activation_cases": source_activation_cases,
        "source_interaction_accessibility_cases": source_interaction_accessibility_cases,
        "source_keyboard_activation_cases": source_keyboard_activation_cases,
        "source_pointer_activation_cases": source_pointer_activation_cases,
        "theme_activation_cases": theme_activation_cases,
        "theme_interaction_accessibility_cases": theme_interaction_accessibility_cases,
        "theme_keyboard_activation_cases": theme_keyboard_activation_cases,
        "theme_pointer_activation_cases": theme_pointer_activation_cases,
        "toc_activation_cases": toc_activation_cases,
        "toc_interaction_accessibility_cases": toc_interaction_accessibility_cases,
        "toc_keyboard_activation_cases": toc_keyboard_activation_cases,
        "toc_pointer_activation_cases": toc_pointer_activation_cases,
        "trainer_cases": trainer_cases,
        "trainer_interaction_cases": trainer_interaction_cases,
        "version_activation_cases": version_activation_cases,
        "version_interaction_accessibility_cases": version_interaction_accessibility_cases,
        "version_keyboard_activation_cases": version_keyboard_activation_cases,
        "version_pointer_activation_cases": version_pointer_activation_cases,
        "viewports": len(selected_viewports),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "site_directory",
        nargs="?",
        type=Path,
        default=Path("site"),
        help="MkDocs output directory (default: site)",
    )
    parser.add_argument(
        "--screenshot-baselines",
        type=Path,
        help="Screenshot signature manifest (default: reviewed fixture for this platform)",
    )
    parser.add_argument(
        "--update-screenshot-baselines",
        action="store_true",
        help="Print a reviewed replacement screenshot signature manifest",
    )
    parser.add_argument(
        "--viewport",
        action="append",
        choices=tuple(VIEWPORTS_BY_NAME),
        dest="viewport_names",
        help="Validate only this viewport; repeat to select multiple viewports",
    )
    parser.add_argument(
        "--palette",
        action="append",
        choices=tuple(PALETTES),
        dest="palette_names",
        help="Validate only this palette; repeat to select multiple palettes",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_site(
                args.site_directory,
                screenshot_baselines_path=args.screenshot_baselines,
                update_screenshot_baselines=args.update_screenshot_baselines,
                viewport_names=tuple(args.viewport_names) if args.viewport_names else None,
                palette_names=tuple(args.palette_names) if args.palette_names else None,
            ),
            indent=2,
            sort_keys=True,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
