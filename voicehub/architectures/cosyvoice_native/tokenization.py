"""Dependency-free Qwen2 byte-BPE tokenizer for CosyVoice 3."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor

from voicehub.architectures.qwen3_asr.tokenization import _load_merges, _load_vocabulary, qwen2_pretokenize
from voicehub.tokenization import ByteBPETokenizer, Encoding
from voicehub.tokenization.assets import (
    DEFAULT_MAX_ASSET_BYTES,
    DEFAULT_MAX_MERGES,
    DEFAULT_MAX_TOKENS,
    TokenizerAssetError,
    read_bounded_asset,
)

END_OF_TEXT = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_PROMPT = "<|endofprompt|>"
PUBLISHED_SPECIAL_IDS = {
    END_OF_TEXT: 151_643,
    IM_START: 151_644,
    IM_END: 151_645,
    END_OF_PROMPT: 151_646,
}
# Source ``CosyVoice3Tokenizer`` registers these with Hugging Face
# ``add_special_tokens`` at load time; the published ``CosyVoice-BlankEN``
# assets do not contain them. Missing spellings receive consecutive IDs after
# the asset's last ID, in this order, exactly as the source tokenizer does.
# yapf: disable
COSYVOICE3_SPECIAL_TOKENS = (
    "<|im_start|>", "<|im_end|>", "<|endofprompt|>", "[breath]", "<strong>", "</strong>", "[noise]", "[laughter]",
    "[cough]", "[clucking]", "[accent]", "[quick_breath]", "<laughter>", "</laughter>", "[hissing]", "[sigh]",
    "[vocalized-noise]", "[lipsmack]", "[mn]", "<|endofsystem|>", "[AA]", "[AA0]", "[AA1]", "[AA2]",
    "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
    "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]",
    "[AY]", "[AY0]", "[AY1]", "[AY2]", "[B]", "[CH]", "[D]", "[DH]",
    "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]",
    "[EY]", "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]",
    "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]", "[IY2]", "[JH]",
    "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]",
    "[OW2]", "[OY]", "[OY0]", "[OY1]", "[OY2]", "[P]", "[R]", "[S]",
    "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
    "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
    "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]",
    "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]", "[g]", "[h]",
    "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]",
    "[iào]", "[iá]", "[ián]", "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]",
    "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]", "[iē]",
    "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]",
    "[iǔ]", "[j]", "[k]", "[l]", "[m]", "[n]", "[o]", "[ong]",
    "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]",
    "[uang]", "[ue]", "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]",
    "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]", "[uí]",
    "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]",
    "[uī]", "[uō]", "[uǎ]", "[uǎi]", "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]",
    "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]",
    "[àn]", "[àng]", "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]",
    "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
    "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]",
    "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]", "[ù]", "[ùn]",
    "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]",
    "[ēi]", "[ēn]", "[ēng]", "[ě]", "[ěi]", "[ěn]", "[ěng]", "[ěr]",
    "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]",
    "[ǎ]", "[ǎi]", "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]",
    "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]",
)
# yapf: enable


def _tokenizer_metadata(path: Path, *, max_bytes: int) -> dict:
    payload = read_bounded_asset(path, max_bytes=max_bytes)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenizerAssetError(f"Invalid tokenizer metadata {path}: {error}.") from error
    if not isinstance(value, dict):
        raise TokenizerAssetError("Tokenizer metadata must be a JSON object.")
    return value


class CosyVoiceTextTokenizer:
    """Qwen byte BPE plus CosyVoice control/event tokens."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        vocab_path: Path,
        merges_path: Path,
        tokenizer_config_path: Path,
        metadata: dict,
    ) -> None:
        self._tokenizer = tokenizer
        self.vocab_path = vocab_path
        self.merges_path = merges_path
        self.tokenizer_config_path = tokenizer_config_path
        self.metadata = dict(metadata)

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        tokenizer_config_path: str | Path,
        *,
        validate_published_ids: bool = True,
        register_source_special_tokens: bool | None = None,
        max_asset_bytes: int = DEFAULT_MAX_ASSET_BYTES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_merges: int = DEFAULT_MAX_MERGES,
    ) -> CosyVoiceTextTokenizer:
        vocab_path = Path(vocab_path).expanduser().resolve()
        merges_path = Path(merges_path).expanduser().resolve()
        tokenizer_config_path = Path(tokenizer_config_path).expanduser().resolve()
        vocabulary = _load_vocabulary(
            vocab_path,
            max_bytes=max_asset_bytes,
            max_tokens=max_tokens,
        )
        merges = _load_merges(
            merges_path,
            max_bytes=max_asset_bytes,
            max_merges=max_merges,
        )
        metadata = _tokenizer_metadata(
            tokenizer_config_path,
            max_bytes=max_asset_bytes,
        )
        # Only explicitly registered tokens are atomic. Ordinary BPE entries
        # such as "[" or "[i" must keep merging like any other text.
        special: dict[str, int] = {}
        records = metadata.get("added_tokens_decoder", {})
        if isinstance(records, dict):
            for raw_id, record in records.items():
                if isinstance(record, dict) and isinstance(record.get("content"), str):
                    special[record["content"]] = int(raw_id)
        if register_source_special_tokens is None:
            register_source_special_tokens = validate_published_ids
        if register_source_special_tokens:
            used_ids = set(vocabulary.values()) | set(special.values())
            next_id = max(used_ids) + 1 if used_ids else 0
            for spelling in COSYVOICE3_SPECIAL_TOKENS:
                if spelling in special:
                    continue
                existing = vocabulary.get(spelling.encode("utf-8"))
                if existing is not None:
                    special[spelling] = existing
                    continue
                special[spelling] = next_id
                next_id += 1
        if validate_published_ids:
            for spelling, expected in PUBLISHED_SPECIAL_IDS.items():
                actual = special.get(spelling)
                if actual != expected:
                    raise TokenizerAssetError(
                        f"CosyVoice token {spelling!r} must use ID "
                        f"{expected}; found {actual!r}.")
        tokenizer = ByteBPETokenizer(
            vocabulary,
            merges=merges,
            special_tokens=special,
            pad_token_id=special.get(END_OF_TEXT),
            add_prefix_space=bool(metadata.get("add_prefix_space", False)),
            use_regex=True,
            pretokenizer=qwen2_pretokenize,
            padding_side="left",
        )
        return cls(
            tokenizer,
            vocab_path=vocab_path,
            merges_path=merges_path,
            tokenizer_config_path=tokenizer_config_path,
            metadata=metadata,
        )

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

    def encode(self, text: str) -> Encoding:
        if not isinstance(text, str) or not text:
            raise ValueError("CosyVoice text must be a non-empty string.")
        return self._tokenizer.encode(
            text,
            allowed_special="all",
            disallowed_special=(),
        )

    def encode_tensor(
        self,
        text: str,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        return torch.tensor(
            [self.encode(text).input_ids],
            dtype=torch.long,
            device=device,
        )

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            errors=str(self.metadata.get("errors", "replace")),
        )

    def instruction_tokens(
        self,
        instruction: str | None,
        *,
        device: str | torch.device | None = None,
    ) -> Tensor:
        if instruction is None:
            prompt = f"You are a helpful assistant.{END_OF_PROMPT}"
        else:
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("CosyVoice instruction must be non-empty or None.")
            prompt = (
                instruction if END_OF_PROMPT in instruction else
                f"You are a helpful assistant. {instruction.strip()}{END_OF_PROMPT}")
        return self.encode_tensor(prompt, device=device)

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        for source, filename in (
            (self.vocab_path, "vocab.json"),
            (self.merges_path, "merges.txt"),
            (self.tokenizer_config_path, "tokenizer_config.json"),
        ):
            destination = target / filename
            if source != destination.resolve():
                shutil.copy2(source, destination)
        return target


__all__ = [
    "COSYVOICE3_SPECIAL_TOKENS",
    "CosyVoiceTextTokenizer",
    "END_OF_PROMPT",
    "END_OF_TEXT",
    "IM_END",
    "IM_START",
    "PUBLISHED_SPECIAL_IDS",
]
