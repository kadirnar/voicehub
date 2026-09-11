"""VoiceHub-native SpeechBrain CRDNN ASR inference and fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_native.configuration import SpeechBrainASRConfig
from voicehub.models.audio import PreTrainedASRModel
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.outputs import ASROutput

_ENGLISH_ALIASES = frozenset({"en", "eng", "english"})


class SpeechBrainASRForSpeechRecognition(PreTrainedASRModel):
    """Run the published CRDNN, attention decoder, and RNNLM natively.

    VoiceHub owns the complete steady-state runtime: waveform features, global
    normalization, CRDNN encoder, location-aware decoder, shallow-fusion beam
    search, SentencePiece unigram tokenization, training objectives, and
    Safetensors export.  Loading an original upstream checkpoint is a one-time,
    explicit, hash-pinned conversion; inference never imports SpeechBrain,
    HyperPyYAML, SentencePiece, protobuf, torchaudio, or Transformers.
    """

    config_class = SpeechBrainASRConfig
    default_model_name_or_path = "speechbrain/asr-crdnn-rnnlm-librispeech"
    architecture_family = "speech-seq2seq"
    native_checkpoint_format = "voicehub-speechbrain-crdnn-asr-v1"
    training_support = "native"
    supports_generic_finetuning = True

    def __init__(
        self,
        config: SpeechBrainASRConfig | str | Path | None = None,
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
            provider="native SpeechBrain ASR",
        )

    @staticmethod
    def _validate_architecture(values: Mapping[str, Any]) -> None:
        model_type = str(values.get("model_type", "")).strip().lower()
        if model_type not in {
                "asr_speechbrain",
                "speechbrain-crdnn-asr",
        }:
            raise ValueError(
                "Native SpeechBrain ASR requires a VoiceHub CRDNN ASR "
                f"artifact; received model type {model_type or '<missing>'!r}.")
        architecture = str(values.get("architecture", "speechbrain-crdnn-asr"), ).strip().lower()
        if architecture != "speechbrain-crdnn-asr":
            raise ValueError(
                "SpeechBrain ASR artifact architecture mismatch: expected "
                f"'speechbrain-crdnn-asr', found {architecture!r}.")
        architectures = values.get('architectures', ())
        if isinstance(architectures, str):
            architectures = (architectures, )
        if architectures and not any(str(name) in {
                "SpeechBrainASRForSpeechRecognition",
                "SpeechBrainCRDNNForASR",
        } for name in architectures):
            names = ", ".join(str(name) for name in architectures)
            raise ValueError(
                "Native SpeechBrain ASR does not support the declared "
                f"architecture(s): {names}.")

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.checkpointing import SafeTensorReader
        from voicehub.models.asr_speechbrain.native.artifacts import resolve_speechbrain_asr_artifacts
        from voicehub.models.asr_speechbrain.native.checkpoint import (
            NATIVE_SPEECHBRAIN_ASR_FORMAT,
            SpeechBrainASRSafeTensorsCheckpointAdapter,
        )
        from voicehub.models.asr_speechbrain.native.configuration import SpeechBrainCRDNNASRConfig
        from voicehub.models.asr_speechbrain.native.decoding import SpeechBrainRNNLMBeamSearch
        from voicehub.models.asr_speechbrain.native.modeling import SpeechBrainCRDNNForASR
        from voicehub.tokenization import SentencePieceUnigramTokenizer

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_speechbrain_asr_artifacts(
            source,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            trust_pickle_checkpoint=self._trust_pickle_checkpoint,
        )
        values = read_json_file(artifacts.config)
        self._validate_architecture(values)
        native_config = SpeechBrainCRDNNASRConfig.from_dict(values)
        if native_config.sampling_rate != self.config.sample_rate:
            raise ValueError(
                "SpeechBrain ASR provider/model sample-rate mismatch: "
                f"provider uses {self.config.sample_rate}, model expects "
                f"{native_config.sampling_rate}.")
        tokenizer = SentencePieceUnigramTokenizer.from_model_file(artifacts.tokenizer, )
        if tokenizer.vocabulary_size != native_config.output_neurons:
            raise ValueError(
                "SpeechBrain tokenizer/model vocabulary mismatch: tokenizer "
                f"contains {tokenizer.vocabulary_size} pieces, model expects "
                f"{native_config.output_neurons}.")

        model = SpeechBrainCRDNNForASR(native_config)
        adapter = SpeechBrainASRSafeTensorsCheckpointAdapter()
        with SafeTensorReader(artifacts.checkpoint) as reader:
            declared_format = reader.metadata.get("format")
            if (declared_format is not None and declared_format != NATIVE_SPEECHBRAIN_ASR_FORMAT):
                raise ValueError(
                    "SpeechBrain ASR Safetensors declares unsupported format "
                    f"{declared_format!r}.")
            adapter.load_streaming(
                model,
                reader,
                values,
                strict=True,
            )
        model.to(device=self.device, dtype=torch.float32)
        self.artifacts = artifacts
        self.native_config = native_config
        self.tokenizer = tokenizer
        self.decoder = SpeechBrainRNNLMBeamSearch(model, native_config)
        self.checkpoint_adapter = adapter.qualified_id
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
            raise ValueError(
                "The published SpeechBrain CRDNN checkpoint supports "
                "`task='transcribe'` only.")
        if language is None or (isinstance(language, str) and language.strip().lower() in _ENGLISH_ALIASES):
            resolved_language = "en"
        else:
            raise ValueError("The audited SpeechBrain LibriSpeech checkpoint is "
                             "English-only.")
        if return_timestamps is not False:
            raise ValueError(
                "The attention decoder does not expose calibrated token "
                "timestamps; `return_timestamps` must be False.")
        unsupported = {
            "chunk_length_s": chunk_length_s,
            "stride_length_s": stride_length_s,
            "max_new_tokens": max_new_tokens,
            "hotwords": hotwords,
        }
        active = [name for name, value in unsupported.items() if value is not None]
        if active:
            raise ValueError(
                "Native SpeechBrain ASR does not support inference option(s): "
                f"{', '.join(active)}.")
        if batch_size not in (None, 1):
            raise ValueError("One SpeechBrain ASR request requires `batch_size=1`.")
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
            raise RuntimeError("Native SpeechBrain ASR runtime is not loaded.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.native_config.sampling_rate,
        )
        waveform = materialized.waveform
        minimum_samples = max(
            self.native_config.win_length,
            self.native_config.hop_length * (self.native_config.time_pooling_size - 1),
        )
        if waveform.numel() < minimum_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, minimum_samples - waveform.numel()),
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
            encoder_states, _, relative_lengths = self.model.encode(
                waveforms,
                waveform_lengths,
                update_normalization=False,
            )
            decoded = self.decoder(
                encoder_states,
                relative_lengths,
                beam_size=beam_size,
            )
        token_ids = decoded.token_ids[0]
        text = self.tokenizer.decode_ids(token_ids)
        return ASROutput(
            text=text,
            segments=(),
            language=resolved_language,
            duration=materialized.duration,
            metadata={
                "architecture":
                "speechbrain-crdnn-asr",
                "architecture_family":
                self.architecture_family,
                "backend":
                "voicehub-native",
                "beam_score":
                decoded.scores[0],
                "beam_size": (self.native_config.beam_size if beam_size is None else beam_size),
                "checkpoint_adapter":
                self.checkpoint_adapter,
                "checkpoint_format":
                self.native_checkpoint_format,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "converted_from_pickle":
                (False if self.artifacts is None else self.artifacts.converted_from_pickle),
                "generated_tokens":
                len(token_ids),
            },
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        from voicehub.models.asr_native.speechbrain_training import prepare_speechbrain_asr_training_batch

        if self.model is None:
            self.load_for_training()
        return prepare_speechbrain_asr_training_batch(
            self,
            inputs,
            phase=phase,
        )

    def _validate_training_runtime(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors
        from voicehub.models.asr_speechbrain.native.checkpoint import (
            NATIVE_SPEECHBRAIN_ASR_FILENAME,
            NATIVE_SPEECHBRAIN_ASR_FORMAT,
        )
        from voicehub.models.asr_speechbrain.native.metadata import (
            SPEECHBRAIN_ASR_REVISION,
            SPEECHBRAIN_ASR_SOURCE_REVISION,
        )

        if (self.model is None or self.native_config is None or self.tokenizer is None):
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / NATIVE_SPEECHBRAIN_ASR_FILENAME,
            metadata={
                "architecture": "speechbrain-crdnn-asr",
                "format": NATIVE_SPEECHBRAIN_ASR_FORMAT,
                "sample_rate": str(self.sample_rate),
            },
        )
        self.tokenizer.save_pretrained(save_directory)
        values = self.native_config.to_dict()
        values.update({
            'architectures': [
                "SpeechBrainASRForSpeechRecognition",
                "SpeechBrainCRDNNForASR",
            ],
            "checkpoint_format": NATIVE_SPEECHBRAIN_ASR_FORMAT,
            "model_type": self.config.model_type,
            "name_or_path": str(save_directory),
            "source_artifact_revision": SPEECHBRAIN_ASR_REVISION,
            "source_training_revision": SPEECHBRAIN_ASR_SOURCE_REVISION,
            "voicehub_provider": self.config.model_type,
        })
        write_json_file(save_directory / "config.json", values)

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["SpeechBrainASRForSpeechRecognition"]
