"""VoiceHub-native ESPnet LibriSpeech Transformer inference and training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_native.configuration import ESPnetASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.outputs import ASROutput

_ENGLISH_ALIASES = frozenset({"en", "eng", "english"})


class ESPnetASRForSpeechRecognition(PreTrainedASRModel):
    """Run the exact 2020 ESPnet Transformer release without ESPnet.

    VoiceHub owns the steady-state STFT/log-mel frontend, global CMVN,
    ``conv2d6`` Transformer encoder, Transformer decoder, CTC prefix
    scorer, recurrent language model, tokenizer remapping, hybrid fine-
    tuning loss, and Safetensors export.  Original pickle artifacts are
    accepted only by a hash-pinned, opt-in conversion boundary.
    """

    config_class = ESPnetASRConfig
    default_model_name_or_path = (
        "espnet/"
        "shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_"
        "bpe_sp_valid.acc.best")
    legacy_model_name_or_path = (
        "espnet/"
        "kan-bayashi_librispeech_asr_train_asr_transformer_e18_raw_bpe_"
        "sp_valid.acc.best")
    architecture_family = "speech-seq2seq"
    native_checkpoint_format = ("voicehub-espnet-librispeech-transformer-e18-v1")
    training_support = "native"
    supports_generic_finetuning = True

    def __init__(
        self,
        config: ESPnetASRConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        trust_pickle_checkpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("String `token` values must be non-empty.")
        if not isinstance(trust_pickle_checkpoint, bool):
            raise TypeError("`trust_pickle_checkpoint` must be a boolean.")
        self._hub_token = token
        self._trust_pickle_checkpoint = trust_pickle_checkpoint
        self.artifacts: Any | None = None
        self.native_config: Any | None = None
        self.tokenizer: Any | None = None
        self.language_model: Any | None = None
        self.decoder: Any | None = None
        self.checkpoint_adapter: str | None = None
        resolved = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(
            resolved,
            device=device,
            lazy_load=lazy_load,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(
            device,
            provider="native ESPnet ASR",
        )

    @staticmethod
    def _validate_architecture(values: Mapping[str, Any]) -> None:
        model_type = str(values.get("model_type", "")).strip().lower()
        if model_type not in {
                "asr_espnet",
                "espnet-librispeech-transformer-e18",
        }:
            raise ValueError(
                "Native ESPnet requires a VoiceHub Transformer e18 artifact; "
                f"received model type {model_type or '<missing>'!r}.")
        architecture = str(values.get(
            "architecture",
            "espnet-librispeech-transformer-e18",
        )).strip().lower()
        if architecture != "espnet-librispeech-transformer-e18":
            raise ValueError(
                "Native ESPnet supports only the audited LibriSpeech "
                f"Transformer e18 graph, not {architecture!r}.")
        architectures = values.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        supported = {
            "ESPnetASRForSpeechRecognition",
            "ESPnetLibriSpeechTransformerForASR",
        }
        if architectures and not any(str(name) in supported for name in architectures):
            raise ValueError(
                "Native ESPnet does not support the declared architecture(s): " +
                ", ".join(str(name) for name in architectures) + ".")

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.models.asr_espnet.native.artifacts import resolve_espnet_artifacts
        from voicehub.models.asr_espnet.native.checkpoint import (
            ESPnetASRSafeTensorsCheckpointAdapter,
            load_native_espnet_models,
        )
        from voicehub.models.asr_espnet.native.configuration import ESPnetLibriSpeechTransformerConfig
        from voicehub.models.asr_espnet.native.decoding import ESPnetJointBeamSearch
        from voicehub.models.asr_espnet.native.tokenization import ESPnetLibriSpeechTokenizer

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_espnet_artifacts(
            source,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            trust_pickle_checkpoint=self._trust_pickle_checkpoint,
        )
        values = read_json_file(artifacts.config)
        self._validate_architecture(values)
        native_config = ESPnetLibriSpeechTransformerConfig.from_dict(values)
        if native_config.sampling_rate != self.config.sample_rate:
            raise ValueError(
                "ESPnet provider/model sample-rate mismatch: provider uses "
                f"{self.config.sample_rate}, model expects "
                f"{native_config.sampling_rate}.")
        tokenizer = ESPnetLibriSpeechTokenizer.from_files(
            artifacts.tokenizer,
            artifacts.tokens,
            strict_release=(native_config.variant == "librispeech-transformer-e18"),
        )
        if tokenizer.vocabulary_size != native_config.vocabulary_size:
            raise ValueError("ESPnet tokenizer/model vocabulary-size mismatch.")
        model, language_model = load_native_espnet_models(
            checkpoint=artifacts.checkpoint,
            language_model_checkpoint=artifacts.language_model_checkpoint,
            config=native_config,
            device=self.device,
            dtype=torch.float32,
        )
        model.eval()
        language_model.eval()
        decoding_config = replace(
            native_config,
            variant="custom",
            beam_size=self.config.beam_size,
            ctc_weight=self.config.ctc_weight,
            language_model_weight=self.config.language_model_weight,
        )
        self.artifacts = artifacts
        self.native_config = native_config
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.decoder = ESPnetJointBeamSearch(
            model,
            decoding_config,
            language_model=language_model,
        )
        self.checkpoint_adapter = (ESPnetASRSafeTensorsCheckpointAdapter().qualified_id)
        self.model = model

    @staticmethod
    def _validate_inference_request(
        *,
        language: str | None,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: Any,
        batch_size: int | None,
        num_beams: int | None,
        max_new_tokens: int | None,
        hotwords: Any,
    ) -> tuple[str, int | None]:
        if task != "transcribe":
            raise ValueError("The audited ESPnet checkpoint supports `task='transcribe'` only.")
        if language is None or (isinstance(language, str) and language.strip().lower() in _ENGLISH_ALIASES):
            resolved_language = "en"
        else:
            raise ValueError("The audited ESPnet LibriSpeech checkpoint is English-only.")
        if return_timestamps is not False:
            raise ValueError(
                "This ESPnet release has no calibrated timestamp head; "
                "`return_timestamps` must be False.")
        unsupported = {
            "chunk_length_s": chunk_length_s,
            "stride_length_s": stride_length_s,
            "max_new_tokens": max_new_tokens,
            "hotwords": hotwords,
        }
        active = [name for name, value in unsupported.items() if value is not None]
        if active:
            raise ValueError("Native ESPnet does not support inference option(s): " + ", ".join(active) + ".")
        if batch_size not in (None, 1):
            raise ValueError("One ESPnet request requires `batch_size=1`.")
        if num_beams is not None and (isinstance(num_beams, bool) or not isinstance(num_beams, Integral) or
                                      num_beams < 1):
            raise ValueError("`num_beams` must be a positive integer or None.")
        return resolved_language, (None if num_beams is None else int(num_beams))

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s: Any = None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords: Any = None,
    ) -> ASROutput:
        import torch
        from torch.nn import functional

        from voicehub.processing.waveform import load_native_audio

        resolved_language, beam_size = self._validate_inference_request(
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        if (self.model is None or self.native_config is None or self.tokenizer is None or
                self.decoder is None):
            raise RuntimeError("Native ESPnet runtime is not loaded.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.native_config.sampling_rate,
        )
        waveform = materialized.waveform
        if waveform.numel() < self.native_config.minimum_waveform_samples:
            waveform = functional.pad(
                waveform,
                (
                    0,
                    self.native_config.minimum_waveform_samples - waveform.numel(),
                ),
            )
        parameter = next(self.model.parameters())
        waveforms = waveform.unsqueeze(0).to(
            device=parameter.device,
            dtype=torch.float32,
        )
        waveform_lengths = torch.tensor(
            [waveform.numel()],
            dtype=torch.long,
            device=parameter.device,
        )
        with torch.inference_mode():
            encoder_states, encoder_lengths = self.model.encode(
                waveforms,
                waveform_lengths,
                apply_augmentation=False,
            )
            decoded = self.decoder(
                encoder_states,
                encoder_lengths,
                beam_size=beam_size,
            )
        token_ids = decoded.token_ids[0]
        return ASROutput(
            text=self.tokenizer.decode_ids(token_ids),
            segments=(),
            language=resolved_language,
            duration=materialized.duration,
            metadata={
                "architecture":
                "espnet-librispeech-transformer-e18",
                "architecture_family":
                self.architecture_family,
                "backend":
                "voicehub-native",
                "beam_score":
                decoded.scores[0],
                "beam_size": (self.config.beam_size if beam_size is None else beam_size),
                "checkpoint_adapter":
                self.checkpoint_adapter,
                "checkpoint_format":
                self.native_checkpoint_format,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "converted_from_pickle":
                (False if self.artifacts is None else self.artifacts.converted_from_pickle),
                "ctc_weight":
                self.config.ctc_weight,
                "generated_tokens":
                len(token_ids),
                "language_model_weight":
                self.config.language_model_weight,
            },
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        from voicehub.models.asr_espnet.native.training import prepare_espnet_training_batch

        if self.model is None:
            self.load_for_training()
        return prepare_espnet_training_batch(
            self,
            inputs,
            phase=phase,
        )

    def _validate_training_runtime(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors
        from voicehub.models.asr_espnet.native.checkpoint import (
            NATIVE_ESPNET_FILENAME,
            NATIVE_ESPNET_FORMAT,
            NATIVE_ESPNET_LM_FILENAME,
        )
        from voicehub.models.asr_espnet.native.metadata import ESPNET_REVISION, ESPNET_SOURCE_REVISION

        if (self.model is None or self.language_model is None or self.native_config is None or
                self.tokenizer is None):
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / NATIVE_ESPNET_FILENAME,
            metadata={
                "architecture": "espnet-librispeech-transformer-e18",
                "format": NATIVE_ESPNET_FORMAT,
                "sample_rate": str(self.sample_rate),
            },
        )
        save_safetensors(
            self.language_model.state_dict(),
            save_directory / NATIVE_ESPNET_LM_FILENAME,
            metadata={
                "architecture": "espnet-sequential-rnn-lm",
                "format": NATIVE_ESPNET_FORMAT,
            },
        )
        self.tokenizer.save_pretrained(save_directory)
        values = self.native_config.to_dict()
        values.update({
            'architectures': [
                "ESPnetASRForSpeechRecognition",
                "ESPnetLibriSpeechTransformerForASR",
            ],
            "checkpoint_format":
            NATIVE_ESPNET_FORMAT,
            "model_type":
            self.config.model_type,
            "name_or_path":
            str(save_directory),
            "source_artifact_revision":
            ESPNET_REVISION,
            "source_revision":
            ESPNET_SOURCE_REVISION,
            "voicehub_provider":
            self.config.model_type,
        })
        write_json_file(save_directory / "config.json", values)

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["ESPnetASRForSpeechRecognition"]
