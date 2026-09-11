"""Native VoxCPM2 tokenization and source-layout batch preparation."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.models.voxcpm.native.configuration import VoxCPM2ArchitectureConfig
from voicehub.tokenization import SentencePieceBPETokenizer, load_sentencepiece_bpe

AUDIO_START_TOKEN_ID = 101
AUDIO_END_TOKEN_ID = 102
REFERENCE_AUDIO_START_TOKEN_ID = 103
REFERENCE_AUDIO_END_TOKEN_ID = 104


def _is_cjk(character: str) -> bool:
    return (
        "\u3400" <= character <= "\u4dbf" or "\u4e00" <= character <= "\u9fff" or
        "\uf900" <= character <= "\ufaff" or "\U00020000" <= character <= "\U0002a6df")


class VoxCPM2Tokenizer:
    """Dependency-free SentencePiece BPE with the published CJK split."""

    def __init__(
        self,
        tokenizer: SentencePieceBPETokenizer,
        *,
        split_map: Mapping[int, tuple[int, ...]],
        source_path: str | Path,
    ) -> None:
        self.tokenizer = tokenizer
        self.split_map = dict(split_map)
        self.source_path = Path(source_path).expanduser().resolve()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        config: VoxCPM2ArchitectureConfig,
    ) -> VoxCPM2Tokenizer:
        source = Path(path).expanduser().resolve()
        assets = load_sentencepiece_bpe(source)
        tokenizer = SentencePieceBPETokenizer(
            assets,
            pad_token_id=config.lm_config.eos_token_id,
            bos_token_id=config.lm_config.bos_token_id,
            eos_token_id=config.lm_config.eos_token_id,
        )
        if tokenizer.token_id_space_size != config.lm_config.vocab_size:
            raise ValueError(
                "VoxCPM tokenizer ID space does not match its text embedding rows: "
                f"{tokenizer.token_id_space_size} != {config.lm_config.vocab_size}.")
        vocabulary = dict(assets.vocabulary)
        vocabulary.update(assets.special_tokens)
        vocabulary.update(assets.added_tokens)
        split_map: dict[int, tuple[int, ...]] = {}
        for spelling, token_id in vocabulary.items():
            clean = spelling.replace("\u2581", "")
            if len(clean) < 2 or not all(_is_cjk(character) for character in clean):
                continue
            character_ids = tuple(vocabulary.get(character, assets.unk_token_id) for character in clean)
            if all(value != assets.unk_token_id for value in character_ids):
                split_map[token_id] = character_ids
        return cls(
            tokenizer,
            split_map=split_map,
            source_path=source,
        )

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("VoxCPM text must be a string.")
        if not text.strip():
            raise ValueError("VoxCPM text cannot be empty.")
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        ).input_ids
        result: list[int] = []
        for token_id in encoded:
            result.extend(self.split_map.get(token_id, (token_id, )))
        return tuple(result)

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        destination = (target / "tokenizer.json").resolve()
        if destination != self.source_path:
            shutil.copyfile(self.source_path, destination)
        return destination


def _waveform(value: Any, *, expected_sample_rate: int) -> Tensor:
    sample_rate = None
    if isinstance(value, Mapping):
        sample_rate = value.get("sampling_rate", value.get("sample_rate"))
        value = value.get("array", value.get("waveform"))
    if value is None:
        raise ValueError("VoxCPM raw-audio records require a waveform.")
    if sample_rate is not None and int(sample_rate) != expected_sample_rate:
        raise ValueError(
            "VoxCPM training waveforms must be resampled to "
            f"{expected_sample_rate} Hz before collation; received {sample_rate} Hz.")
    tensor = (value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)).float().squeeze()
    if tensor.ndim != 1 or not tensor.numel():
        raise ValueError("VoxCPM waveforms must be non-empty mono tensors.")
    if not torch.isfinite(tensor).all():
        raise ValueError("VoxCPM waveform contains NaN or infinite samples.")
    return tensor


class VoxCPM2Processor:
    """Prepare inference prefixes and the exact published SFT batch."""

    def __init__(
        self,
        tokenizer: VoxCPM2Tokenizer,
        config: VoxCPM2ArchitectureConfig,
        *,
        codec=None,
    ) -> None:
        if not isinstance(tokenizer, VoxCPM2Tokenizer):
            raise TypeError("`tokenizer` must be a VoxCPM2Tokenizer.")
        if not isinstance(config, VoxCPM2ArchitectureConfig):
            raise TypeError("`config` must be a VoxCPM2ArchitectureConfig.")
        self.tokenizer = tokenizer
        self.config = config
        self.codec = codec

    @property
    def sample_rate(self) -> int:
        return self.config.audio_vae_config.sample_rate

    def _encode_audio(self, value: Any, *, device=None) -> Tensor:
        if isinstance(value, Mapping) and "audio_features" in value:
            value = value["audio_features"]
        if isinstance(value, Tensor) and value.ndim in (2, 3):
            features = value.detach()
            if (features.ndim == 3 and
                    tuple(features.shape[1:]) == (self.config.patch_size, self.config.feat_dim)):
                return features.to(device=device, dtype=torch.float32)
            if features.ndim == 3:
                if features.shape[0] != 1:
                    raise ValueError("Per-record VoxCPM features require batch size one.")
                features = features[0]
            if (features.shape[-1] == self.config.feat_dim and
                    features.shape[0] % self.config.patch_size == 0):
                features = features.to(device=device, dtype=torch.float32)
                return features.unflatten(0, (-1, self.config.patch_size))
        if self.codec is None:
            raise RuntimeError(
                "Raw-audio VoxCPM preparation requires the frozen native "
                "AudioVAE; supply source-layout `audio_features` otherwise.")
        waveform = _waveform(
            value,
            expected_sample_rate=self.sample_rate,
        ).to(device=device)
        patch_samples = (self.codec.hop_length * self.config.patch_size)
        remainder = waveform.numel() % patch_samples
        if remainder:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, patch_samples - remainder),
            )
        with torch.no_grad():
            features = self.codec.encode(
                waveform[None, None],
                self.sample_rate,
            )
        return (
            features.transpose(1, 2)[0].unflatten(0, (-1, self.config.patch_size)).to(dtype=torch.float32))

    def _ordinary_sequence(
        self,
        text_ids: Tensor,
        target: Tensor,
        *,
        is_prompt: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = text_ids.device
        start = (REFERENCE_AUDIO_START_TOKEN_ID if is_prompt else AUDIO_START_TOKEN_ID)
        end = (REFERENCE_AUDIO_END_TOKEN_ID if is_prompt else AUDIO_END_TOKEN_ID)
        target_length = target.shape[0]
        text_track = torch.cat((
            text_ids,
            torch.tensor([start], device=device),
            torch.zeros(target_length, dtype=torch.long, device=device),
            torch.tensor([end], device=device),
        ))
        text_length = text_ids.numel() + 1
        zeros = torch.zeros(
            text_length,
            self.config.patch_size,
            self.config.feat_dim,
            dtype=target.dtype,
            device=device,
        )
        features = torch.cat((zeros, target, zeros[:1]), dim=0)
        text_mask = torch.cat((
            torch.ones(text_length, device=device),
            torch.zeros(target_length, device=device),
            torch.ones(1, device=device),
        )).long()
        audio_mask = 1 - text_mask
        loss_mask = torch.cat((
            torch.zeros(text_length, device=device),
            torch.zeros(target_length, device=device) if is_prompt else torch.ones(
                target_length, device=device),
            torch.zeros(1, device=device),
        )).long()
        labels = torch.zeros_like(text_track)
        labels[-2] = 1
        return (
            text_track,
            features,
            text_mask,
            audio_mask,
            loss_mask,
            labels,
        )

    def _reference_sequence(
        self,
        text_ids: Tensor,
        reference: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = text_ids.device
        reference_length = reference.shape[0]
        target_length = target.shape[0]
        text_length = text_ids.numel()

        def token(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        text_track = torch.cat((
            token([REFERENCE_AUDIO_START_TOKEN_ID]),
            torch.zeros(reference_length, dtype=torch.long, device=device),
            token([REFERENCE_AUDIO_END_TOKEN_ID]),
            text_ids,
            token([AUDIO_START_TOKEN_ID]),
            torch.zeros(target_length, dtype=torch.long, device=device),
            token([AUDIO_END_TOKEN_ID]),
        ))
        zero = torch.zeros(
            1,
            self.config.patch_size,
            self.config.feat_dim,
            dtype=target.dtype,
            device=device,
        )
        text_zero = zero.expand(text_length, -1, -1)
        features = torch.cat(
            (
                zero,
                reference,
                zero,
                text_zero,
                zero,
                target,
                zero,
            ),
            dim=0,
        )
        text_mask = torch.cat((
            torch.ones(1, device=device),
            torch.zeros(reference_length, device=device),
            torch.ones(text_length + 2, device=device),
            torch.zeros(target_length, device=device),
            torch.ones(1, device=device),
        )).long()
        audio_mask = 1 - text_mask
        loss_mask = torch.cat((
            torch.zeros(
                reference_length + text_length + 3,
                device=device,
            ),
            torch.ones(target_length, device=device),
            torch.zeros(1, device=device),
        )).long()
        labels = torch.zeros_like(text_track)
        labels[-2] = 1
        return (
            text_track,
            features,
            text_mask,
            audio_mask,
            loss_mask,
            labels,
        )

    def training_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        device=None,
    ) -> dict[str, Tensor]:
        """Collate the published `[text, 101, audio, 102]` SFT layout."""
        if (isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records):
            raise ValueError("VoxCPM training requires at least one record.")
        sequences = []
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("Every VoxCPM training record must be a mapping.")
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every VoxCPM record requires non-empty `text`.")
            text_ids = torch.tensor(
                self.tokenizer.encode(text),
                dtype=torch.long,
                device=device,
            )
            target_value = record.get(
                "audio_features",
                record.get("audio", record.get("waveform")),
            )
            target = self._encode_audio(target_value, device=device)
            reference_value = record.get(
                "reference_audio_features",
                record.get("reference_audio", record.get("ref_audio")),
            )
            if reference_value is not None:
                reference = self._encode_audio(
                    reference_value,
                    device=device,
                )
                sequence = self._reference_sequence(
                    text_ids,
                    reference,
                    target,
                )
            else:
                sequence = self._ordinary_sequence(
                    text_ids,
                    target,
                    is_prompt=bool(record.get("is_prompt", False)),
                )
            if sequence[0].shape[0] > self.config.max_length:
                raise ValueError(
                    "VoxCPM packed record exceeds the configured context; "
                    "filter or segment it before training.")
            sequences.append(sequence)
        maximum = max(sequence[0].shape[0] for sequence in sequences)

        def pad_1d(value: Tensor) -> Tensor:
            return torch.nn.functional.pad(
                value,
                (0, maximum - value.shape[0]),
            )

        def pad_features(value: Tensor) -> Tensor:
            return torch.nn.functional.pad(
                value,
                (0, 0, 0, 0, 0, maximum - value.shape[0]),
            )

        return {
            "text_tokens": torch.stack([pad_1d(sequence[0]) for sequence in sequences]),
            "audio_feats": torch.stack([pad_features(sequence[1]) for sequence in sequences]),
            "text_mask": torch.stack([pad_1d(sequence[2]) for sequence in sequences]),
            "audio_mask": torch.stack([pad_1d(sequence[3]) for sequence in sequences]),
            "loss_mask": torch.stack([pad_1d(sequence[4]) for sequence in sequences]),
            "labels": torch.stack([pad_1d(sequence[5]) for sequence in sequences]),
            "position_ids": torch.arange(
                maximum,
                device=device,
            ).expand(len(sequences), -1),
        }

    def generation_prefix(
        self,
        text: str,
        *,
        prompt_features: Tensor | None = None,
        prompt_text: str = "",
        reference_features: Tensor | None = None,
        device=None,
    ) -> dict[str, Tensor]:
        """Build zero-shot, continuation, reference, or combined prefix."""
        if prompt_features is not None and not isinstance(prompt_text, str):
            raise TypeError("VoxCPM prompt text must be a string.")
        target_text = prompt_text + text if prompt_features is not None else text
        text_ids = torch.tensor(
            (
                *self.tokenizer.encode(target_text),
                AUDIO_START_TOKEN_ID,
            ),
            dtype=torch.long,
            device=device,
        )
        text_length = text_ids.numel()
        zero_text_features = torch.zeros(
            text_length,
            self.config.patch_size,
            self.config.feat_dim,
            device=device,
        )
        text_mask = torch.ones(text_length, dtype=torch.long, device=device)
        audio_mask = torch.zeros_like(text_mask)
        if reference_features is not None:
            reference = self._encode_audio(reference_features, device=device)
            reference_length = reference.shape[0]
            zero = torch.zeros_like(zero_text_features[:1])
            text_ids = torch.cat((
                torch.tensor(
                    [REFERENCE_AUDIO_START_TOKEN_ID],
                    device=device,
                ),
                torch.zeros(
                    reference_length,
                    dtype=torch.long,
                    device=device,
                ),
                torch.tensor(
                    [REFERENCE_AUDIO_END_TOKEN_ID],
                    device=device,
                ),
                text_ids,
            ))
            zero_text_features = torch.cat(
                (zero, reference, zero, zero_text_features),
                dim=0,
            )
            text_mask = torch.cat((
                torch.ones(1, device=device),
                torch.zeros(reference_length, device=device),
                torch.ones(text_length + 1, device=device),
            )).long()
            audio_mask = 1 - text_mask
        if prompt_features is not None:
            prompt = self._encode_audio(prompt_features, device=device)
            text_ids = torch.cat((
                text_ids,
                torch.zeros(
                    prompt.shape[0],
                    dtype=torch.long,
                    device=device,
                ),
            ))
            zero_text_features = torch.cat(
                (zero_text_features, prompt),
                dim=0,
            )
            text_mask = torch.cat((
                text_mask,
                torch.zeros(prompt.shape[0], device=device),
            )).long()
            audio_mask = 1 - text_mask
        return {
            "text_tokens": text_ids.unsqueeze(0),
            "text_mask": text_mask.unsqueeze(0),
            "audio_feats": zero_text_features.unsqueeze(0),
            "audio_mask": audio_mask.unsqueeze(0),
        }


__all__ = [
    "AUDIO_END_TOKEN_ID",
    "AUDIO_START_TOKEN_ID",
    "REFERENCE_AUDIO_END_TOKEN_ID",
    "REFERENCE_AUDIO_START_TOKEN_ID",
    "VoxCPM2Processor",
    "VoxCPM2Tokenizer",
]
