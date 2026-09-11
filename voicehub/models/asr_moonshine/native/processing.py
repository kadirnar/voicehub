"""Native raw-waveform and text processing for Moonshine."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_moonshine.native.configuration import MoonshineConfig
from voicehub.tokenization import SentencePieceBPETokenizer


class MoonshineProcessor:
    """Combine exact Moonshine waveform padding and SentencePiece BPE."""

    def __init__(
        self,
        *,
        tokenizer: SentencePieceBPETokenizer,
        preprocessor_config: Mapping[str, Any],
        tokenizer_path: Path | None = None,
    ) -> None:
        if not isinstance(tokenizer, SentencePieceBPETokenizer):
            raise TypeError("`tokenizer` must be a SentencePieceBPETokenizer.")
        if not isinstance(preprocessor_config, Mapping):
            raise TypeError("`preprocessor_config` must be a mapping.")
        values = copy.deepcopy(dict(preprocessor_config))
        expected = {
            "do_normalize": False,
            "feature_size": 1,
            "padding_side": "right",
            "padding_value": 0.0,
            "return_attention_mask": True,
            "sampling_rate": 16_000,
        }
        for name, expected_value in expected.items():
            if values.get(name, expected_value) != expected_value:
                raise ValueError("Native Moonshine preprocessing requires "
                                 f"`{name}={expected_value!r}`.")
        feature_extractor_type = values.get(
            "feature_extractor_type",
            "Wav2Vec2FeatureExtractor",
        )
        if feature_extractor_type != "Wav2Vec2FeatureExtractor":
            raise ValueError(
                "Native Moonshine requires the raw-waveform "
                "Wav2Vec2FeatureExtractor contract.")
        self.tokenizer = tokenizer
        self.sampling_rate = 16_000
        self.padding_value = 0.0
        self.do_normalize = False
        self._preprocessor_config = values
        self._tokenizer_path = tokenizer_path
        # Compatibility with the former Transformers processor surface.
        self.feature_extractor = self

    @classmethod
    def from_artifacts(
        cls,
        *,
        tokenizer_path: str | Path,
        preprocessor_config_path: str | Path,
        config: MoonshineConfig,
    ) -> MoonshineProcessor:
        tokenizer_path = Path(tokenizer_path)
        tokenizer = SentencePieceBPETokenizer.from_tokenizer_json(
            tokenizer_path,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )
        return cls(
            tokenizer=tokenizer,
            preprocessor_config=read_json_file(preprocessor_config_path),
            tokenizer_path=tokenizer_path,
        )

    def prepare_audio_batch(
        self,
        waveforms: Sequence[Any],
    ) -> dict[str, torch.Tensor]:
        if isinstance(waveforms, (str, bytes)) or not isinstance(
                waveforms,
                Sequence,
        ):
            raise TypeError("`waveforms` must be a sequence.")
        if not waveforms:
            raise ValueError("`waveforms` cannot be empty.")
        tensors: list[torch.Tensor] = []
        for index, waveform in enumerate(waveforms):
            try:
                tensor = torch.as_tensor(waveform)
            except (TypeError, ValueError, RuntimeError) as error:
                raise TypeError(f"Waveform {index} cannot be converted to a tensor.") from error
            if tensor.ndim != 1 or tensor.numel() == 0:
                raise ValueError(f"Waveform {index} must be one non-empty mono sequence.")
            if tensor.dtype == torch.bool or tensor.is_complex():
                raise TypeError(f"Waveform {index} must contain real numeric samples.")
            tensor = tensor.to(dtype=torch.float32)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Waveform {index} contains NaN or infinite samples.")
            tensors.append(tensor)
        maximum = max(tensor.numel() for tensor in tensors)
        values = torch.full(
            (len(tensors), maximum),
            self.padding_value,
            dtype=torch.float32,
        )
        attention_mask = torch.zeros(
            (len(tensors), maximum),
            dtype=torch.long,
        )
        for index, tensor in enumerate(tensors):
            values[index, :tensor.numel()] = tensor
            attention_mask[index, :tensor.numel()] = 1
        return {
            "input_values": values,
            "attention_mask": attention_mask,
        }

    def encode_labels(
        self,
        texts: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer.encode_batch(
            texts,
            add_special_tokens=False,
            pad=False,
        )
        targets = tuple((*row, self.tokenizer.eos_token_id) for row in encoded.input_ids)
        maximum = max((len(row) for row in targets), default=0)
        labels = torch.full(
            (len(targets), maximum),
            -100,
            dtype=torch.long,
        )
        label_mask = torch.zeros(
            (len(targets), maximum),
            dtype=torch.long,
        )
        for index, row in enumerate(targets):
            length = len(row)
            labels[index, :length] = torch.tensor(row, dtype=torch.long)
            label_mask[index, :length] = 1
        return {
            "labels": labels,
            "decoder_attention_mask": label_mask,
        }

    def decode(
        self,
        token_ids: Any,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def batch_decode(
        self,
        sequences: Any,
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.detach().cpu().tolist()
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
        )

    def __call__(
        self,
        audio: Any | None = None,
        text: str | Sequence[str] | None = None,
        *,
        sampling_rate: int | None = None,
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("Native Moonshine returns PyTorch tensors only.")
        if padding is not True:
            raise ValueError("Native Moonshine batches require right padding.")
        if sampling_rate not in (None, self.sampling_rate):
            raise ValueError("Resample audio to 16000 Hz at the VoiceHub boundary.")
        if audio is None and text is None:
            raise ValueError("Supply `audio`, `text`, or both.")
        output: dict[str, torch.Tensor] = {}
        if audio is not None:
            if isinstance(audio, torch.Tensor) and audio.ndim == 1:
                audio_values = (audio, )
            elif (isinstance(audio, Sequence) and not isinstance(audio, (str, bytes)) and
                  (not audio or not isinstance(audio[0], (int, float)))):
                audio_values = tuple(audio)
            else:
                audio_values = (audio, )
            output.update(self.prepare_audio_batch(audio_values))
        if text is not None:
            texts = (text, ) if isinstance(text, str) else tuple(text)
            output.update(self.encode_labels(texts))
        return output

    def save_pretrained(self, directory: str | Path) -> None:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(destination)
        values = copy.deepcopy(self._preprocessor_config)
        values.update({
            "do_normalize": False,
            "feature_extractor_type": "Wav2Vec2FeatureExtractor",
            "feature_size": 1,
            "padding_side": "right",
            "padding_value": 0.0,
            "return_attention_mask": True,
            "sampling_rate": self.sampling_rate,
        })
        write_json_file(
            destination / "preprocessor_config.json",
            values,
        )


__all__ = ["MoonshineProcessor"]
