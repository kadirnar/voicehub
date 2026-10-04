"""One-time conversion of the official CosyVoice 3 snapshot.

Run ``python -m voicehub.converters.cosyvoice OUTPUT_DIR`` once, then
load ``OUTPUT_DIR`` as a normal native CosyVoice artifact.
"""

from __future__ import annotations

import argparse
import shutil
from importlib.util import find_spec
from pathlib import Path

import torch

from voicehub.architectures.cosyvoice_native.checkpoint import convert_audited_cosyvoice_legacy_checkpoint
from voicehub.architectures.cosyvoice_native.configuration import CosyVoiceArchitectureConfig
from voicehub.architectures.cosyvoice_native.metadata import (
    COSYVOICE3_LEGACY_FILES,
    COSYVOICE3_MODEL_ID,
    COSYVOICE3_MODEL_REVISION,
    COSYVOICE3_SPEECH_TOKENIZER_FILE,
)
from voicehub.architectures.cosyvoice_native.modeling import CosyVoiceNativeModel
from voicehub.architectures.cosyvoice_native.speech_tokenizer import CosyVoiceSpeechTokenizerConfig
from voicehub.converters.cosyvoice_speech_tokenizer import convert_audited_cosyvoice_speech_tokenizer
from voicehub.hub import resolve_pretrained_file, write_json_file

_TEXT_TOKENIZER_SUBFOLDER = "CosyVoice-BlankEN"
_TEXT_TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")


def convert_audited_cosyvoice3_checkpoint(
    destination: str | Path,
    *,
    source: str | Path = COSYVOICE3_MODEL_ID,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    speech_tokenizer: bool = True,
) -> Path:
    """Build a complete native artifact from the audited official snapshot.

    ``source`` is the official Hub ID (pinned to the audited revision by
    default) or a local copy of that snapshot. Every weight file is
    verified by the component converters before it is written.
    """
    target = Path(destination).expanduser()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"CosyVoice conversion destination must be empty: {target}.")
    if speech_tokenizer and find_spec("onnx") is None:
        raise RuntimeError(
            "Converting speech_tokenizer_v3.onnx requires the optional `onnx` "
            "parser: install `voicehub[conversion]`, or skip the optional "
            "speech tokenizer with `--skip-speech-tokenizer`.")
    if revision is None and str(source) == COSYVOICE3_MODEL_ID:
        revision = COSYVOICE3_MODEL_REVISION

    def resolve(filename: str, *, subfolder: str = "") -> Path:
        return resolve_pretrained_file(
            source,
            filename,
            subfolder=subfolder,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )

    target.mkdir(parents=True, exist_ok=True)
    config = CosyVoiceArchitectureConfig()
    with torch.device("meta"):
        model = CosyVoiceNativeModel(config, initialize=False)
    for component, module in (("llm", model.llm), ("flow", model.flow), ("hift", model.hift)):
        convert_audited_cosyvoice_legacy_checkpoint(
            module,
            resolve(COSYVOICE3_LEGACY_FILES[component]["filename"]),
            target / f"{component}.safetensors",
            component=component,
        )
    for filename in _TEXT_TOKENIZER_FILES:
        shutil.copyfile(
            resolve(filename, subfolder=_TEXT_TOKENIZER_SUBFOLDER),
            target / filename,
        )
    if speech_tokenizer:
        convert_audited_cosyvoice_speech_tokenizer(
            resolve(COSYVOICE3_SPEECH_TOKENIZER_FILE["filename"]),
            target / "speech_tokenizer.safetensors",
        )
        write_json_file(
            target / "speech_tokenizer_config.json",
            CosyVoiceSpeechTokenizerConfig().to_dict(),
        )
    # Written last so an interrupted conversion is rejected as incomplete.
    write_json_file(target / "cosyvoice_config.json", config.to_dict())
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voicehub.converters.cosyvoice",
        description="Convert the official CosyVoice 3 snapshot into a native VoiceHub artifact.",
    )
    parser.add_argument("output_dir", type=Path, help="empty directory for the native artifact")
    parser.add_argument(
        "--source",
        default=COSYVOICE3_MODEL_ID,
        help="official Hub ID or a local copy of its snapshot",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-speech-tokenizer",
        action="store_true",
        help="omit the optional speech tokenizer (needed only for prompt audio and raw-audio training)",
    )
    arguments = parser.parse_args(argv)
    output = convert_audited_cosyvoice3_checkpoint(
        arguments.output_dir,
        source=arguments.source,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        local_files_only=arguments.local_files_only,
        speech_tokenizer=not arguments.skip_speech_tokenizer,
    )
    print(output)
    return 0


__all__ = [
    "convert_audited_cosyvoice3_checkpoint",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
