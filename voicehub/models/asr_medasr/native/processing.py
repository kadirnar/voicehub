"""Native MedASR audio/text processor."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.models.asr_medasr.native.configuration import MedASRConfig
from voicehub.models.asr_medasr.native.frontend import MedASRFeatureExtractor
from voicehub.models.asr_medasr.native.tokenization import MedASRTokenizer


class MedASRProcessor:
    """Prepare raw mono waveforms and CTC labels without provider SDKs."""

    def __init__(
        self,
        config: MedASRConfig,
        tokenizer: MedASRTokenizer,
        *,
        preprocessor_config_path: Path | None = None,
        processor_config_path: Path | None = None,
    ) -> None:
        if not isinstance(config, MedASRConfig):
            raise TypeError("`config` must be a MedASRConfig.")
        if not isinstance(tokenizer, MedASRTokenizer):
            raise TypeError("`tokenizer` must be a MedASRTokenizer.")
        if tokenizer.vocabulary_size != config.vocab_size:
            raise ValueError("MedASR tokenizer/model vocabulary mismatch.")
        if tokenizer.pad_token_id != config.pad_token_id:
            raise ValueError("MedASR tokenizer/model CTC blank mismatch.")
        self.config = config
        self.tokenizer = tokenizer
        self.feature_extractor = MedASRFeatureExtractor(config)
        self.preprocessor_config_path = preprocessor_config_path
        self.processor_config_path = processor_config_path

    @classmethod
    def from_artifacts(
        cls,
        *,
        config: MedASRConfig,
        tokenizer_json: str | Path,
        tokenizer_config: str | Path,
        preprocessor_config: str | Path,
        processor_config: str | Path | None = None,
    ) -> MedASRProcessor:
        from voicehub.hub import read_json_file

        preprocessor_path = Path(preprocessor_config, ).expanduser().resolve()
        values = read_json_file(preprocessor_path)
        expected = {
            "feature_size": config.num_mel_bins,
            "hop_length": config.feature_hop_length,
            "n_fft": config.feature_fft_size,
            "padding_side": "right",
            "padding_value": 0.0,
            "return_attention_mask": True,
            "sampling_rate": config.sampling_rate,
            "win_length": config.feature_window_length,
        }
        for name, expected_value in expected.items():
            actual = values.get(name)
            if actual != expected_value:
                raise ValueError(
                    f"MedASR preprocessor `{name}` is {actual!r}; expected "
                    f"{expected_value!r}.")
        tokenizer = MedASRTokenizer.from_files(
            tokenizer_json,
            tokenizer_config=tokenizer_config,
            expected_vocabulary_size=config.vocab_size,
        )
        return cls(
            config,
            tokenizer,
            preprocessor_config_path=preprocessor_path,
            processor_config_path=(
                Path(processor_config).expanduser().resolve() if processor_config is not None else None),
        )

    def prepare_audio_batch(
        self,
        waveforms: Tensor | Sequence[Any],
        *,
        waveform_lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self.feature_extractor(
            waveforms,
            waveform_lengths=waveform_lengths,
        )

    def encode_labels(
        self,
        texts: Sequence[str],
    ) -> Tensor:
        if isinstance(texts, (str, bytes)) or not isinstance(
                texts,
                Sequence,
        ):
            raise TypeError("`texts` must be a sequence of strings.")
        encoded = tuple(self.tokenizer.encode(text) for text in texts)
        if not encoded:
            raise ValueError("A MedASR label batch cannot be empty.")
        maximum = max(len(row) for row in encoded)
        labels = torch.full(
            (len(encoded), maximum),
            self.config.pad_token_id,
            dtype=torch.long,
        )
        for index, token_ids in enumerate(encoded):
            labels[index, :len(token_ids)] = torch.tensor(
                token_ids,
                dtype=torch.long,
            )
        return labels

    def __call__(
        self,
        waveforms: Tensor | Sequence[Any],
        *,
        text: str | Sequence[str] | None = None,
        waveform_lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        prepared = self.prepare_audio_batch(
            waveforms,
            waveform_lengths=waveform_lengths,
        )
        if text is not None:
            texts = (text, ) if isinstance(text, str) else tuple(text)
            prepared["labels"] = self.encode_labels(texts)
            if prepared["labels"].shape[0] != prepared["input_features"].shape[0]:
                raise ValueError("MedASR requires one transcript per waveform.")
        return prepared

    def save_pretrained(self, directory: str | Path) -> Path:
        from voicehub.hub import write_json_file

        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(destination)
        if self.preprocessor_config_path is not None:
            target = destination / "preprocessor_config.json"
            if target.resolve() != self.preprocessor_config_path:
                shutil.copy2(self.preprocessor_config_path, target)
        else:
            write_json_file(
                destination / "preprocessor_config.json",
                {
                    "feature_extractor_type": "VoiceHubMedASRFeatureExtractor",
                    "feature_size": self.config.num_mel_bins,
                    "hop_length": self.config.feature_hop_length,
                    "n_fft": self.config.feature_fft_size,
                    "padding_side": "right",
                    "padding_value": 0.0,
                    "processor_class": "VoiceHubMedASRProcessor",
                    "return_attention_mask": True,
                    "sampling_rate": self.config.sampling_rate,
                    "win_length": self.config.feature_window_length,
                },
            )
        if self.processor_config_path is not None:
            target = destination / "processor_config.json"
            if target.resolve() != self.processor_config_path:
                shutil.copy2(self.processor_config_path, target)
        else:
            write_json_file(
                destination / "processor_config.json",
                {
                    "processor_class": "VoiceHubMedASRProcessor",
                    "sampling_rate": self.config.sampling_rate,
                },
            )
        return destination


__all__ = ["MedASRProcessor"]
