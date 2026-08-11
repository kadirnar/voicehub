#!/usr/bin/env python3
"""Generate one concise Colab notebook for every Hub-backed model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from model_documentation import TASK_LABELS, TASK_ORDER, checkpoint_documentation, inference_profile  # noqa: E402

from voicehub import list_model_specs  # noqa: E402

MODEL_NOTEBOOK_DIR = REPOSITORY_ROOT / "notebooks" / "models"
GENERATOR_PATH = "scripts/generate_model_notebooks.py"


def hub_model_specs():
    """Return registry entries whose default checkpoint is a Hub model ID."""
    return tuple(
        spec for spec in list_model_specs(task=None) if checkpoint_documentation(spec).is_hugging_face)


def _markdown(cell_id: str, source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.rstrip() + "\n",
    }


def _code(
        cell_id: str,
        source: str,
        *,
        tags: tuple[str, ...] = (),
) -> dict[str, object]:
    metadata = {"tags": list(tags)} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": metadata,
        "outputs": [],
        "source": source.rstrip() + "\n",
    }


def _indented_lines(lines: tuple[str, ...], *, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in lines)


def _call_arguments(arguments: tuple[str, ...], *, spaces: int = 8) -> str:
    prefix = " " * spaces
    return "".join(f"{prefix}{argument},\n" for argument in arguments)


def _tts_cells(spec) -> tuple[dict[str, object], ...]:
    profile = inference_profile(spec)
    configuration = f'''from pathlib import Path

RUN_INFERENCE = False
MODEL_TYPE = {spec.model_type!r}
CHECKPOINT = {spec.default_model_path!r}
DEVICE = "cuda"
TEXT = {profile.text!r}
OUTPUT_FILE = Path("artifacts/{spec.model_type}.wav")'''
    if profile.high_level_supported:
        setup_import = "    import json\n\n" if any("json." in line for line in profile.setup) else ""
        imports = ", ".join(
            dict.fromkeys((
                "AutoModelForTextToSpeech",
                "TTSGenerationConfig",
                *profile.voicehub_imports,
            )))
        setup = _indented_lines(profile.setup, spaces=4)
        if setup:
            setup += "\n\n"
        load_arguments = _call_arguments(profile.load_arguments)
        arguments = _call_arguments(profile.arguments)
        inference = f'''if RUN_INFERENCE:
{setup_import}    from voicehub import {imports}

{setup}    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForTextToSpeech.from_pretrained(
        CHECKPOINT,
        model_type=MODEL_TYPE,
        device=DEVICE,
        lazy_load=True,
{load_arguments}    )
    output = model.generate(
        TEXT,
        generation_config=TTSGenerationConfig(seed=42, output_file=OUTPUT_FILE),
{arguments}    )
    print(output.file_path, output.sample_rate, output.metadata)'''
    else:
        inference = '''if RUN_INFERENCE:
    from voicehub import AutoModelForTextToSpeech

    model = AutoModelForTextToSpeech.from_pretrained(
        CHECKPOINT,
        model_type=MODEL_TYPE,
        device=DEVICE,
        lazy_load=True,
    )
    model.load()
    required_stages = (
        "forward_lm",
        "forward_tts_lm",
        "sample_speech_latents",
        "decode_speech_latents",
    )
    missing = [name for name in required_stages if not hasattr(model.model, name)]
    if missing:
        raise RuntimeError(f"Missing audited VibeVoice stage(s): {', '.join(missing)}")
    print("High-level synthesis is not verified; available native stages:", required_stages)'''
    return (
        _code("configure", configuration, tags=("smoke-safe", )),
        _markdown(
            "inputs",
            "## Run inference\n\n"
            f"{profile.summary}\n\n{profile.input_note}\n\n"
            "This VoiceHub example is maintained in this repository and is not copied from "
            "an upstream package snippet. Set `RUN_INFERENCE = True` after reviewing inputs.",
        ),
        _code(
            "inference",
            inference,
            tags=("requires-model", "requires-audio-runtime", "writes-data"),
        ),
    )


def _asr_cells(spec) -> tuple[dict[str, object], ...]:
    profile = inference_profile(spec)
    configuration = f'''from pathlib import Path

RUN_INFERENCE = False
MODEL_TYPE = {spec.model_type!r}
CHECKPOINT = {spec.default_model_path!r}
DEVICE = "cuda"
AUDIO_FILE = Path("speech.wav")'''
    arguments = _call_arguments(profile.arguments)
    inference = f'''if RUN_INFERENCE:
    from voicehub import AutoModelForSpeechRecognition

    if not AUDIO_FILE.is_file():
        raise FileNotFoundError(AUDIO_FILE)
    model = AutoModelForSpeechRecognition.from_pretrained(
        CHECKPOINT,
        model_type=MODEL_TYPE,
        device=DEVICE,
        lazy_load=True,
    )
    output = model.transcribe(
        AUDIO_FILE,
{arguments}    )
    print(output.text)
    for segment in output.segments:
        print(segment.start, segment.end, segment.text, segment.confidence)'''
    return (
        _code("configure", configuration, tags=("smoke-safe", )),
        _markdown(
            "inputs",
            "## Run inference\n\n"
            f"{profile.summary}\n\n{profile.input_note}\n\n"
            "Place an authorized recording at `speech.wav`, then set `RUN_INFERENCE = True`.",
        ),
        _code(
            "inference",
            inference,
            tags=("requires-model", "requires-audio-runtime", "requires-data"),
        ),
    )


def _vad_cells(spec) -> tuple[dict[str, object], ...]:
    profile = inference_profile(spec)
    configuration = f'''from pathlib import Path

RUN_INFERENCE = False
MODEL_TYPE = {spec.model_type!r}
CHECKPOINT = {spec.default_model_path!r}
DEVICE = "cpu"
AUDIO_FILE = Path("speech.wav")'''
    arguments = _call_arguments(profile.arguments)
    inference = f'''if RUN_INFERENCE:
    from voicehub import AutoModelForVoiceActivityDetection

    if not AUDIO_FILE.is_file():
        raise FileNotFoundError(AUDIO_FILE)
    model = AutoModelForVoiceActivityDetection.from_pretrained(
        CHECKPOINT,
        model_type=MODEL_TYPE,
        device=DEVICE,
        lazy_load=True,
    )
    output = model.detect(
        AUDIO_FILE,
{arguments}    )
    for segment in output.segments:
        print(segment.start, segment.end, segment.score)'''
    return (
        _code("configure", configuration, tags=("smoke-safe", )),
        _markdown(
            "inputs",
            "## Run inference\n\n"
            f"{profile.summary}\n\n{profile.input_note}\n\n"
            "Place an authorized recording at `speech.wav`, then set `RUN_INFERENCE = True`.",
        ),
        _code(
            "inference",
            inference,
            tags=("requires-model", "requires-audio-runtime", "requires-data"),
        ),
    )


def render_notebook(spec) -> str:
    """Render one deterministic notebook for *spec*."""
    filename = f"{spec.model_type}.ipynb"
    colab_url = (
        "https://colab.research.google.com/github/kadirnar/voicehub/"
        f"blob/main/notebooks/models/{filename}")
    checkpoint = checkpoint_documentation(spec)
    introduction = f'''# `{spec.model_type}` with VoiceHub

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})

- Task: **{TASK_LABELS[spec.task.value]}**
- Hugging Face ID: [`{checkpoint.hugging_face_id}`]({checkpoint.hugging_face_url})

Install VoiceHub using the [installation guide](https://kadirnar.github.io/voicehub/getting-started/installation/)
before opening this model workflow. This notebook contains no package-install cell.

The registry check is safe to run without downloading weights. Inference is disabled by default.'''
    inspection = f'''from voicehub import get_model_spec

model_spec = get_model_spec(MODEL_TYPE)
assert model_spec.task.value == {spec.task.value!r}
assert model_spec.default_model_path == CHECKPOINT
print("task:", model_spec.task.value)
print("checkpoint:", model_spec.default_model_path)
print("capabilities:", ", ".join(model_spec.capabilities))
print("training:", model_spec.training.support.value)'''
    task_cells = {
        "text-to-speech": _tts_cells,
        "automatic-speech-recognition": _asr_cells,
        "voice-activity-detection": _vad_cells,
    }[spec.task.value](spec)
    cells = [
        _markdown("introduction", introduction),
        task_cells[0],
        _markdown("registry-heading", "## Inspect registry support"),
        _code("registry", inspection, tags=("smoke-safe", )),
        *task_cells[1:],
        _markdown(
            "next",
            "## Next\n\n"
            "See the [inference guide](https://kadirnar.github.io/voicehub/guides/inference/) "
            "and [model catalog](https://kadirnar.github.io/voicehub/models/) for the shared "
            "runtime contract and model-specific limitations.",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": filename,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "voicehub": {
                "generated_by": GENERATOR_PATH,
                "model_type": spec.model_type,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def render_gallery(specs) -> str:
    """Render the generated model-notebook index."""
    lines = [
        "# Hugging Face model notebooks",
        "",
        "Each Hub-backed registry entry has a focused inference notebook. Real model",
        "downloads and inference stay disabled until `RUN_INFERENCE` is enabled.",
        "",
        f"Generated by `{GENERATOR_PATH}`. Do not edit the table or notebooks by hand.",
        "",
    ]
    for task in TASK_ORDER:
        task_specs = [spec for spec in specs if spec.task.value == task]
        lines.extend((
            f"## {TASK_LABELS[task]}",
            "",
            "| Model | Hugging Face | Notebook | Colab |",
            "| --- | --- | --- | --- |",
        ))
        for spec in task_specs:
            checkpoint = checkpoint_documentation(spec)
            filename = f"{spec.model_type}.ipynb"
            colab_url = (
                "https://colab.research.google.com/github/kadirnar/voicehub/"
                f"blob/main/notebooks/models/{filename}")
            lines.append(
                f"| `{spec.model_type}` | "
                f"[`{checkpoint.hugging_face_id}`]({checkpoint.hugging_face_url}) | "
                f"[View]({filename}) | [Run]({colab_url}) |")
        lines.append("")
    return "\n".join(lines)


def generated_files() -> dict[Path, str]:
    """Return every expected generated path and its contents."""
    specs = hub_model_specs()
    files = {MODEL_NOTEBOOK_DIR / f"{spec.model_type}.ipynb": render_notebook(spec) for spec in specs}
    files[MODEL_NOTEBOOK_DIR / "README.md"] = render_gallery(specs)
    return files


def generated_notebook_paths() -> tuple[Path, ...]:
    """Return notebooks owned by this generator, ignoring user artifacts."""
    result = []
    if not MODEL_NOTEBOOK_DIR.is_dir():
        return ()
    for path in sorted(MODEL_NOTEBOOK_DIR.glob("*.ipynb")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        except (KeyError, TypeError, ValueError):
            continue
        voicehub = metadata.get("voicehub", {})
        if voicehub.get("generated_by") == GENERATOR_PATH:
            result.append(path)
    return tuple(result)


def check_generated_files(files: dict[Path, str]) -> tuple[Path, ...]:
    """Return generated paths that are missing or stale."""
    stale = [
        path for path, expected in files.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != expected
    ]
    expected = set(files)
    stale.extend(path for path in generated_notebook_paths() if path not in expected)
    return tuple(stale)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when generated notebooks are missing or stale.",
    )
    args = parser.parse_args()
    files = generated_files()
    stale = check_generated_files(files)
    if args.check:
        if stale:
            for path in stale:
                print(f"stale: {path.relative_to(REPOSITORY_ROOT)}", file=sys.stderr)
            return 1
        print(f"OK: {len(files) - 1} model notebooks are current")
        return 0

    MODEL_NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for path in stale:
        if path in files:
            path.write_text(files[path], encoding="utf-8")
            print(f"wrote: {path.relative_to(REPOSITORY_ROOT)}")
        else:
            path.unlink()
            print(f"removed: {path.relative_to(REPOSITORY_ROOT)}")
    print(f"OK: {len(files) - 1} model notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
