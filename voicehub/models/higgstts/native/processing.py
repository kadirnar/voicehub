"""Native Llama-3 text and delayed-audio processing for Higgs Audio v2."""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from voicehub.models.higgstts.native.configuration import HiggsAudioV2Config
from voicehub.models.higgstts.native.tokenizer import HiggsAudioV2TokenizerModel
from voicehub.tokenization import ByteBPETokenizer, Encoding
from voicehub.tokenization.llama3 import llama3_pretokenize

BEGIN_OF_TEXT = "<|begin_of_text|>"
END_OF_TEXT = "<|end_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
END_OF_TURN = "<|eot_id|>"
AUDIO_BOS = "<|audio_out_bos|>"
AUDIO_EOS = "<|audio_eos|>"
AUDIO_DELAY = "<|reserved_special_token_6|>"
AUDIO_OUTPUT = "<|AUDIO_OUT|>"
SCENE_START = "<|scene_desc_start|>"
SCENE_END = "<|scene_desc_end|>"

HIGGS_SPECIAL_TOKEN_IDS = {
    BEGIN_OF_TEXT: 128_000,
    END_OF_TEXT: 128_001,
    START_HEADER: 128_006,
    END_HEADER: 128_007,
    END_OF_TURN: 128_009,
    AUDIO_EOS: 128_012,
    AUDIO_BOS: 128_013,
    AUDIO_DELAY: 128_014,
    AUDIO_OUTPUT: 128_016,
    SCENE_START: 128_018,
    SCENE_END: 128_019,
}

DEFAULT_SYSTEM_PROMPT = "Generate audio following instruction."
DEFAULT_SCENE_PROMPT = "Audio is recorded from a quiet room."


@dataclass(frozen=True)
class HiggsAudioV2Batch:
    """Aligned text/audio tensors accepted by the native model."""

    input_ids: Tensor
    attention_mask: Tensor
    audio_input_ids: Tensor | None = None
    audio_input_ids_mask: Tensor | None = None
    labels: Tensor | None = None
    audio_labels: Tensor | None = None

    def model_inputs(self) -> dict[str, Tensor]:
        result = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }
        for name in (
                "audio_input_ids",
                "audio_input_ids_mask",
                "labels",
                "audio_labels",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


class HiggsAudioV2TextTokenizer:
    """Checkpoint-bound Llama-3 byte BPE with Higgs control tokens."""

    def __init__(
        self,
        tokenizer: ByteBPETokenizer,
        *,
        tokenizer_path: Path,
        tokenizer_config_path: Path | None = None,
        special_tokens_map_path: Path | None = None,
        chat_template_path: Path | None = None,
    ) -> None:
        if not isinstance(tokenizer, ByteBPETokenizer):
            raise TypeError("`tokenizer` must be ByteBPETokenizer.")
        self._tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self.tokenizer_config_path = tokenizer_config_path
        self.special_tokens_map_path = special_tokens_map_path
        self.chat_template_path = chat_template_path
        declared = {
            **dict(tokenizer.special_tokens),
            **dict(tokenizer.added_tokens),
        }
        for token, expected in HIGGS_SPECIAL_TOKEN_IDS.items():
            actual = declared.get(token)
            if actual != expected:
                raise ValueError(f"Higgs token {token!r} must use ID {expected}; "
                                 f"found {actual!r}.")
        if tokenizer.token_id_space_size > 128_256:
            raise ValueError("Higgs tokenizer ID space exceeds the model vocabulary.")

    @classmethod
    def from_files(
        cls,
        tokenizer_json: str | Path,
        *,
        tokenizer_config: str | Path | None = None,
        special_tokens_map: str | Path | None = None,
        chat_template: str | Path | None = None,
    ) -> HiggsAudioV2TextTokenizer:
        tokenizer_path = Path(tokenizer_json).expanduser().resolve()
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Higgs tokenizer was not found: {tokenizer_path}.")
        tokenizer = ByteBPETokenizer.from_tokenizer_json(
            tokenizer_path,
            pad_token_id=HIGGS_SPECIAL_TOKEN_IDS[END_OF_TEXT],
            padding_side="left",
            pretokenizer=llama3_pretokenize,
        )
        return cls(
            tokenizer,
            tokenizer_path=tokenizer_path,
            tokenizer_config_path=(
                Path(tokenizer_config).expanduser().resolve() if tokenizer_config is not None else None),
            special_tokens_map_path=(
                Path(special_tokens_map).expanduser().resolve() if special_tokens_map is not None else None),
            chat_template_path=(
                Path(chat_template).expanduser().resolve() if chat_template is not None else None),
        )

    @property
    def pad_token_id(self) -> int:
        return HIGGS_SPECIAL_TOKEN_IDS[END_OF_TEXT]

    @property
    def bos_token_id(self) -> int:
        return HIGGS_SPECIAL_TOKEN_IDS[BEGIN_OF_TEXT]

    @property
    def eos_token_id(self) -> int:
        return HIGGS_SPECIAL_TOKEN_IDS[END_OF_TEXT]

    @property
    def token_id_space_size(self) -> int:
        return self._tokenizer.token_id_space_size

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
        )

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        for source, filename in (
            (self.tokenizer_path, "tokenizer.json"),
            (self.tokenizer_config_path, "tokenizer_config.json"),
            (self.special_tokens_map_path, "special_tokens_map.json"),
            (self.chat_template_path, "chat_template.jinja"),
        ):
            if source is None:
                continue
            if not source.is_file():
                raise FileNotFoundError(f"Higgs tokenizer asset was not found: {source}.")
            target = destination / filename
            if source != target.resolve():
                shutil.copy2(source, target)
        return destination.resolve()


def _nonempty_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty string.")
    return value.strip()


class HiggsAudioV2Processor:
    """Exact chat protocol and audio delay-pattern preparation."""

    def __init__(
        self,
        tokenizer: HiggsAudioV2TextTokenizer,
        audio_tokenizer: HiggsAudioV2TokenizerModel,
        model_config: HiggsAudioV2Config | Mapping[str, Any],
    ) -> None:
        if not isinstance(tokenizer, HiggsAudioV2TextTokenizer):
            raise TypeError("`tokenizer` must be HiggsAudioV2TextTokenizer.")
        if not isinstance(audio_tokenizer, HiggsAudioV2TokenizerModel):
            raise TypeError("`audio_tokenizer` must be HiggsAudioV2TokenizerModel.")
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.model_config = HiggsAudioV2Config.coerce(model_config)
        if (audio_tokenizer.config.num_quantizers != self.model_config.num_codebooks):
            raise ValueError("Higgs language-model codebooks and tokenizer quantizers "
                             "must agree.")
        for token, model_id in (
            (AUDIO_BOS, self.model_config.audio_bos_token_id),
            (AUDIO_DELAY, self.model_config.audio_delay_token_id),
            (AUDIO_OUTPUT, self.model_config.audio_token_id),
        ):
            if HIGGS_SPECIAL_TOKEN_IDS[token] != model_id:
                raise ValueError(f"Higgs model ID for {token!r} disagrees with tokenizer.")

    @property
    def sample_rate(self) -> int:
        return self.audio_tokenizer.config.sample_rate

    @staticmethod
    def _header(role: str) -> str:
        if not isinstance(role, str) or not role:
            raise ValueError("Higgs message roles must be non-empty strings.")
        return f"{START_HEADER}{role}{END_HEADER}\n\n"

    def render_generation_prompt(
        self,
        text: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        scene_prompt: str | None = DEFAULT_SCENE_PROMPT,
        reference_text: str | None = None,
        include_reference_audio: bool = False,
    ) -> str:
        """Render the audited chat template without a Jinja dependency."""
        text = _nonempty_text("text", text)
        system_prompt = _nonempty_text("system_prompt", system_prompt)
        if scene_prompt is not None:
            scene_prompt = _nonempty_text("scene_prompt", scene_prompt)
        if include_reference_audio:
            reference_text = _nonempty_text(
                "reference_text",
                reference_text,
            )
        elif reference_text is not None:
            raise ValueError("`reference_text` requires a reference audio input.")
        prompt = (BEGIN_OF_TEXT + self._header("system") + system_prompt)
        if scene_prompt is not None:
            prompt += (f"\n\n{SCENE_START}\n{scene_prompt}\n{SCENE_END}")
        prompt += END_OF_TURN
        if include_reference_audio:
            prompt += (
                self._header("user") + reference_text + END_OF_TURN + self._header("assistant") + AUDIO_BOS +
                AUDIO_OUTPUT + AUDIO_EOS + END_OF_TURN)
        prompt += (self._header("user") + text + END_OF_TURN + self._header("assistant") + AUDIO_BOS)
        return prompt

    def render_training_prompt(
        self,
        text: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        scene_prompt: str | None = DEFAULT_SCENE_PROMPT,
        reference_text: str | None = None,
        include_reference_audio: bool = False,
    ) -> str:
        prompt = self.render_generation_prompt(
            text,
            system_prompt=system_prompt,
            scene_prompt=scene_prompt,
            reference_text=reference_text,
            include_reference_audio=include_reference_audio,
        )
        return prompt + AUDIO_OUTPUT + AUDIO_EOS + END_OF_TURN

    def build_delay_pattern(self, audio_codes: Tensor) -> Tensor:
        """Shift each codebook by its index and add BOS/EOS diagonals."""
        if (not isinstance(audio_codes, Tensor) or audio_codes.ndim != 3 or
                audio_codes.shape[1] != self.model_config.num_codebooks):
            raise ValueError("Higgs codes must have shape [batch, codebooks, frames].")
        batch_size, codebooks, sequence_length = audio_codes.shape
        delayed = audio_codes.new_empty((
            batch_size,
            codebooks,
            sequence_length + codebooks - 1,
        ))
        for index in range(codebooks):
            delayed[:, index, :index] = (self.model_config.audio_stream_bos_id)
            delayed[:, index, index:index + sequence_length] = (audio_codes[:, index])
            delayed[:, index, index + sequence_length:] = (self.model_config.audio_stream_eos_id)
        return delayed

    def revert_delay_pattern(self, delayed_codes: Tensor) -> Tensor:
        """Recover aligned codebooks from ``[frames, codebooks]``."""
        if (not isinstance(delayed_codes, Tensor) or delayed_codes.ndim != 2 or
                delayed_codes.shape[1] != self.model_config.num_codebooks):
            raise ValueError("Delayed Higgs codes must have shape [frames, codebooks].")
        frames, codebooks = delayed_codes.shape
        aligned_length = frames - codebooks + 1
        if aligned_length < 1:
            raise ValueError("Delayed Higgs codes are too short to contain one frame.")
        return torch.cat(
            [delayed_codes[
                index:index + aligned_length,
                index:index + 1,
            ] for index in range(codebooks)],
            dim=1,
        )

    def add_audio_boundaries(self, audio_codes: Tensor) -> Tensor:
        bos = audio_codes.new_full(
            (*audio_codes.shape[:2], 1),
            self.model_config.audio_stream_bos_id,
        )
        eos = audio_codes.new_full(
            (*audio_codes.shape[:2], 1),
            self.model_config.audio_stream_eos_id,
        )
        return torch.cat((bos, audio_codes, eos), dim=-1)

    def encode_audio(self, input_values: Tensor) -> Tensor:
        self.audio_tokenizer.eval()
        with torch.no_grad():
            return self.audio_tokenizer.encode(input_values).audio_codes

    def _inject_audio(
        self,
        prompt: str,
        audio_codes: Sequence[Tensor],
    ) -> tuple[str, Tensor | None, tuple[Tensor, ...]]:
        prompt_parts = prompt.split(AUDIO_OUTPUT)
        placeholder_count = len(prompt_parts) - 1
        if placeholder_count != len(audio_codes):
            raise ValueError(
                "Higgs chat audio placeholders and supplied audio values "
                f"differ: {placeholder_count} != {len(audio_codes)}.")
        delayed_blocks = []
        rendered = [prompt_parts[0]]
        for index, codes in enumerate(audio_codes):
            if codes.ndim == 2:
                codes = codes.unsqueeze(0)
            bounded = self.add_audio_boundaries(codes)
            delayed = self.build_delay_pattern(bounded).transpose(1, 2)
            delayed_length = delayed.shape[1]
            replacement = (
                AUDIO_OUTPUT * (delayed_length - self.model_config.num_codebooks + 1) + AUDIO_DELAY *
                (self.model_config.num_codebooks - 1))
            rendered.extend((replacement, prompt_parts[index + 1]))
            delayed_blocks.append(delayed)
        combined = (None if not delayed_blocks else torch.cat(delayed_blocks, dim=1))
        return "".join(rendered), combined, tuple(delayed_blocks)

    def _target_audio_labels(self, audio_codes: Tensor) -> Tensor:
        """Build the source target: masked BOS/padding, supervised EOS."""
        if audio_codes.ndim == 2:
            audio_codes = audio_codes.unsqueeze(0)
        if (audio_codes.ndim != 3 or audio_codes.shape[1] != self.model_config.num_codebooks):
            raise ValueError("Higgs target codes must have shape "
                             "[batch, codebooks, frames].")
        batch_size, codebooks, sequence_length = audio_codes.shape
        bounded = audio_codes.new_full(
            (batch_size, codebooks, sequence_length + 2),
            -100,
        )
        bounded[:, :, 1:-1] = audio_codes
        bounded[:, :, -1] = self.model_config.audio_stream_eos_id
        delayed = audio_codes.new_full(
            (
                batch_size,
                codebooks,
                bounded.shape[-1] + codebooks - 1,
            ),
            -100,
        )
        for index in range(codebooks):
            delayed[
                :,
                index,
                index:index + bounded.shape[-1],
            ] = bounded[:, index]
        return delayed.transpose(1, 2)

    def _single_batch(
        self,
        prompt: str,
        *,
        audio_codes: Sequence[Tensor] = (),
        output_labels: bool = False,
        device: torch.device | str | None = None,
    ) -> HiggsAudioV2Batch:
        prompt, delayed, delayed_blocks = self._inject_audio(
            prompt,
            audio_codes,
        )
        encoded = self.tokenizer.encode(prompt)
        target_device = (torch.device("cpu") if device is None else torch.device(device))
        input_ids = torch.tensor(
            [encoded.input_ids],
            dtype=torch.long,
            device=target_device,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        audio_input_ids = (None if delayed is None else delayed.to(target_device))
        audio_mask = (
            None if audio_input_ids is None else torch.ones(
                audio_input_ids.shape[:2],
                dtype=torch.bool,
                device=target_device,
            ))
        labels = None
        audio_labels = None
        if output_labels:
            labels = torch.full_like(input_ids, -100)
            # The target is the final assistant audio turn. System/user text,
            # role headers, and an optional reference turn are conditioning
            # only. Boson's source recipe supervises the assistant audio BOS,
            # audio EOS, and end-of-turn tokens while masking the expanded
            # audio placeholders from the text objective.
            for token_id in (
                    self.model_config.audio_bos_token_id,
                    HIGGS_SPECIAL_TOKEN_IDS[AUDIO_EOS],
                    HIGGS_SPECIAL_TOKEN_IDS[END_OF_TURN],
            ):
                positions = (input_ids[0] == token_id).nonzero()
                if not len(positions):
                    raise RuntimeError(
                        "Higgs training prompt is missing required target "
                        f"token ID {token_id}.")
                position = int(positions[-1, 0])
                labels[0, position] = input_ids[0, position]
            if audio_input_ids is not None:
                if not audio_codes or not delayed_blocks:
                    raise RuntimeError("Higgs output labels require target audio codes.")
                audio_labels = torch.full_like(audio_input_ids, -100)
                target_start = sum(block.shape[1] for block in delayed_blocks[:-1])
                target_labels = self._target_audio_labels(audio_codes[-1]).to(target_device)
                target_end = target_start + target_labels.shape[1]
                audio_labels[:, target_start:target_end] = target_labels
        return HiggsAudioV2Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_mask,
            labels=labels,
            audio_labels=audio_labels,
        )

    def generation_batch(
        self,
        text: str,
        *,
        reference_codes: Tensor | None = None,
        reference_text: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        scene_prompt: str | None = DEFAULT_SCENE_PROMPT,
        device: torch.device | str | None = None,
    ) -> HiggsAudioV2Batch:
        prompt = self.render_generation_prompt(
            text,
            system_prompt=system_prompt,
            scene_prompt=scene_prompt,
            reference_text=reference_text,
            include_reference_audio=reference_codes is not None,
        )
        references = () if reference_codes is None else (reference_codes, )
        return self._single_batch(
            prompt,
            audio_codes=references,
            device=device,
        )

    def training_example(
        self,
        text: str,
        target_codes: Tensor,
        *,
        reference_codes: Tensor | None = None,
        reference_text: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        scene_prompt: str | None = DEFAULT_SCENE_PROMPT,
        device: torch.device | str | None = None,
    ) -> HiggsAudioV2Batch:
        prompt = self.render_training_prompt(
            text,
            system_prompt=system_prompt,
            scene_prompt=scene_prompt,
            reference_text=reference_text,
            include_reference_audio=reference_codes is not None,
        )
        codes = ((target_codes, ) if reference_codes is None else (reference_codes, target_codes))
        return self._single_batch(
            prompt,
            audio_codes=codes,
            output_labels=True,
            device=device,
        )

    def collate(
        self,
        examples: Sequence[HiggsAudioV2Batch],
        *,
        pad_to_multiple_of: int | None = None,
    ) -> HiggsAudioV2Batch:
        """Left-pad text and right-pad delayed audio without data leakage."""
        if not examples:
            raise ValueError("At least one Higgs example is required.")
        if any(example.input_ids.shape[0] != 1 for example in examples):
            raise ValueError("Higgs collation expects unbatched examples.")
        device = examples[0].input_ids.device
        if any(example.input_ids.device != device for example in examples):
            raise ValueError("All Higgs examples must share one device.")
        text_length = max(example.input_ids.shape[1] for example in examples)
        if pad_to_multiple_of is not None:
            if (isinstance(pad_to_multiple_of, bool) or not isinstance(pad_to_multiple_of, int) or
                    pad_to_multiple_of <= 0):
                raise ValueError("`pad_to_multiple_of` must be a positive integer.")
            text_length = ((text_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of)
        ids = []
        masks = []
        labels = []
        has_labels = all(example.labels is not None for example in examples)
        for example in examples:
            padding = text_length - example.input_ids.shape[1]
            ids.append(functional.pad(
                example.input_ids,
                (padding, 0),
                value=self.tokenizer.pad_token_id,
            ))
            masks.append(functional.pad(
                example.attention_mask,
                (padding, 0),
                value=False,
            ))
            if has_labels:
                assert example.labels is not None
                labels.append(functional.pad(
                    example.labels,
                    (padding, 0),
                    value=-100,
                ))
        any_audio = any(example.audio_input_ids is not None for example in examples)
        audio_ids = None
        audio_masks = None
        audio_labels = None
        if any_audio:
            if any(example.audio_input_ids is None for example in examples):
                raise ValueError("A Higgs batch cannot mix examples with and without "
                                 "audio placeholders.")
            audio_length = max(
                example.audio_input_ids.shape[1] for example in examples
                if example.audio_input_ids is not None)
            audio_rows = []
            audio_mask_rows = []
            audio_label_rows = []
            has_audio_labels = all(example.audio_labels is not None for example in examples)
            for example in examples:
                assert example.audio_input_ids is not None
                assert example.audio_input_ids_mask is not None
                amount = audio_length - example.audio_input_ids.shape[1]
                audio_rows.append(
                    functional.pad(
                        example.audio_input_ids,
                        (0, 0, 0, amount),
                        value=self.model_config.audio_stream_eos_id,
                    ))
                audio_mask_rows.append(
                    functional.pad(
                        example.audio_input_ids_mask,
                        (0, amount),
                        value=False,
                    ))
                if has_audio_labels:
                    assert example.audio_labels is not None
                    audio_label_rows.append(
                        functional.pad(
                            example.audio_labels,
                            (0, 0, 0, amount),
                            value=-100,
                        ))
            audio_ids = torch.cat(audio_rows)
            audio_masks = torch.cat(audio_mask_rows)
            if has_audio_labels:
                audio_labels = torch.cat(audio_label_rows)
        return HiggsAudioV2Batch(
            input_ids=torch.cat(ids),
            attention_mask=torch.cat(masks),
            audio_input_ids=audio_ids,
            audio_input_ids_mask=audio_masks,
            labels=torch.cat(labels) if has_labels else None,
            audio_labels=audio_labels,
        )


__all__ = [
    "AUDIO_BOS",
    "AUDIO_DELAY",
    "AUDIO_EOS",
    "AUDIO_OUTPUT",
    "DEFAULT_SCENE_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "HIGGS_SPECIAL_TOKEN_IDS",
    "HiggsAudioV2Batch",
    "HiggsAudioV2Processor",
    "HiggsAudioV2TextTokenizer",
]
