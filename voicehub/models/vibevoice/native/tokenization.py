"""Native Qwen2 byte-BPE tokenization for the VibeVoice family."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from voicehub.models.asr_qwen3.native.tokenization import qwen2_pretokenize
from voicehub.tokenization.assets import DEFAULT_MAX_ASSET_BYTES, TokenizerAssetError, read_bounded_asset
from voicehub.tokenization.base import BatchEncoding, Encoding
from voicehub.tokenization.byte_bpe import ByteBPETokenizer

END_OF_TEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ASR_AUDIO_START = "<|object_ref_start|>"
ASR_AUDIO_END = "<|object_ref_end|>"
ASR_AUDIO = "<|box_start|>"
TTS_SPEECH_START = "<|vision_start|>"
TTS_SPEECH_END = "<|vision_end|>"
TTS_SPEECH_DIFFUSION = "<|vision_pad|>"
MODEL_PADDING = "<|image_pad|>"

VIBEVOICE_TOKEN_IDS = {
    END_OF_TEXT: 151_643,
    IM_START: 151_644,
    IM_END: 151_645,
    ASR_AUDIO_START: 151_646,
    ASR_AUDIO_END: 151_647,
    ASR_AUDIO: 151_648,
    TTS_SPEECH_START: 151_652,
    TTS_SPEECH_END: 151_653,
    TTS_SPEECH_DIFFUSION: 151_654,
    MODEL_PADDING: 151_655,
}


def _read_config(
    path: Path,
    *,
    max_asset_bytes: int,
) -> dict[str, Any]:
    payload = read_bounded_asset(path, max_bytes=max_asset_bytes)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise TokenizerAssetError(f"Invalid VibeVoice tokenizer config {path.name}: {error}.") from error
    if not isinstance(value, dict):
        raise TokenizerAssetError("VibeVoice tokenizer configuration must be an object.")
    return value


class VibeVoiceTokenizer:
    """Immutable tokenizer wrapper with explicit ASR and TTS control IDs."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        tokenizer_path: Path,
        tokenizer_config_path: Path,
        tokenizer_config: Mapping[str, Any],
        vocabulary_limit: int,
    ) -> None:
        self._tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.tokenizer_config_path = tokenizer_config_path
        self.tokenizer_config = dict(tokenizer_config)
        self.vocabulary_limit = vocabulary_limit

    @classmethod
    def from_files(
        cls,
        tokenizer_path: str | Path,
        tokenizer_config_path: str | Path,
        *,
        vocabulary_limit: int,
        max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
    ) -> VibeVoiceTokenizer:
        if (isinstance(vocabulary_limit, bool) or not isinstance(vocabulary_limit, int) or
                vocabulary_limit <= 0):
            raise ValueError("VibeVoice vocabulary limit must be positive.")
        tokenizer_asset = Path(tokenizer_path).expanduser().resolve()
        config_asset = Path(tokenizer_config_path).expanduser().resolve()
        config = _read_config(
            config_asset,
            max_asset_bytes=max_asset_bytes,
        )
        tokenizer = ByteBPETokenizer.from_tokenizer_json(
            tokenizer_asset,
            max_asset_bytes=max_asset_bytes,
            pad_token_id=VIBEVOICE_TOKEN_IDS[END_OF_TEXT],
            pretokenizer=qwen2_pretokenize,
            padding_side="left",
        )
        if tokenizer.token_id_space_size > vocabulary_limit:
            raise TokenizerAssetError(
                "VibeVoice tokenizer ID space exceeds the model vocabulary: "
                f"{tokenizer.token_id_space_size} > {vocabulary_limit}.")
        for spelling, expected_id in VIBEVOICE_TOKEN_IDS.items():
            actual_id = tokenizer.special_tokens.get(spelling)
            if actual_id != expected_id:
                raise TokenizerAssetError(
                    f"VibeVoice token {spelling!r} must use ID "
                    f"{expected_id}; found {actual_id!r}.")
        configured_pad = config.get("pad_token", END_OF_TEXT)
        if configured_pad not in {END_OF_TEXT, None}:
            raise TokenizerAssetError("VibeVoice Qwen tokenizer declares an unsupported pad token.")
        return cls(
            tokenizer,
            tokenizer_path=tokenizer_asset,
            tokenizer_config_path=config_asset,
            tokenizer_config=config,
            vocabulary_limit=vocabulary_limit,
        )

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocabulary_size

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    @property
    def pad_token_id(self) -> int:
        """Text-batch padding used by the ASR processor."""
        return VIBEVOICE_TOKEN_IDS[END_OF_TEXT]

    @property
    def model_padding_id(self) -> int:
        """Pseudo-token padding used by the published TTS processors."""
        return VIBEVOICE_TOKEN_IDS[MODEL_PADDING]

    @property
    def eos_token_id(self) -> int:
        return VIBEVOICE_TOKEN_IDS[END_OF_TEXT]

    @property
    def im_start_id(self) -> int:
        return VIBEVOICE_TOKEN_IDS[IM_START]

    @property
    def im_end_id(self) -> int:
        return VIBEVOICE_TOKEN_IDS[IM_END]

    @property
    def asr_audio_start_id(self) -> int:
        return VIBEVOICE_TOKEN_IDS[ASR_AUDIO_START]

    @property
    def asr_audio_end_id(self) -> int:
        return VIBEVOICE_TOKEN_IDS[ASR_AUDIO_END]

    @property
    def asr_audio_id(self) -> int:
        return VIBEVOICE_TOKEN_IDS[ASR_AUDIO]

    @property
    def speech_start_id(self) -> int:
        return VIBEVOICE_TOKEN_IDS[TTS_SPEECH_START]

    @property
    def speech_end_id(self) -> int:
        return VIBEVOICE_TOKEN_IDS[TTS_SPEECH_END]

    @property
    def speech_diffusion_id(self) -> int:
        return VIBEVOICE_TOKEN_IDS[TTS_SPEECH_DIFFUSION]

    def encode(
        self,
        text: str,
        *,
        max_length: int | None = None,
        truncation: bool | str = False,
    ) -> Encoding:
        return self._tokenizer.encode(
            text,
            allowed_special="all",
            disallowed_special=(),
            max_length=max_length,
            truncation=truncation,
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        *,
        padding: bool | str = True,
        max_length: int | None = None,
        truncation: bool | str = False,
        pad_to_multiple_of: int | None = None,
    ) -> BatchEncoding:
        return self._tokenizer.encode_batch(
            texts,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            allowed_special="all",
            disallowed_special=(),
        )

    def decode(
        self,
        token_ids: Iterable[int] | Encoding,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        tolist = getattr(token_ids, "tolist", None)
        if callable(tolist):
            token_ids = tolist()
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            errors=str(self.tokenizer_config.get("errors", "replace")),
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        for source, filename in (
            (self.tokenizer_path, "tokenizer.json"),
            (self.tokenizer_config_path, "tokenizer_config.json"),
        ):
            target = destination / filename
            if source != target.resolve():
                shutil.copy2(source, target)
        return destination.resolve()


__all__ = [
    "ASR_AUDIO",
    "ASR_AUDIO_END",
    "ASR_AUDIO_START",
    "END_OF_TEXT",
    "IM_END",
    "IM_START",
    "MODEL_PADDING",
    "TTS_SPEECH_DIFFUSION",
    "TTS_SPEECH_END",
    "TTS_SPEECH_START",
    "VIBEVOICE_TOKEN_IDS",
    "VibeVoiceTokenizer",
]
