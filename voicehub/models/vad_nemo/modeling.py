"""VoiceHub-native multilingual MarbleNet voice activity detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.inference_configuration import VADInferenceConfig
from voicehub.models.audio import PreTrainedVADModel
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.models.vad_nemo.configuration import NeMoVADConfig
from voicehub.outputs import VADOutput
from voicehub.vad_utils import frame_probabilities_to_segments


class NeMoVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run and fine-tune released MarbleNet weights without importing NeMo."""

    config_class = NeMoVADConfig
    default_model_name_or_path = ("nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0")
    architecture_family = "frame"
    training_support = "native"
    supports_generic_finetuning = True

    def __init__(
        self,
        config: NeMoVADConfig | str | Path | None = None,
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
        self.checkpoint_adapter: str | None = None
        config = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="native MarbleNet VAD")

    def _load_pretrained_model(self) -> None:
        import torch

        from voicehub.checkpointing import SafeTensorReader
        from voicehub.models.vad_nemo.artifacts import resolve_marblenet_vad_artifacts
        from voicehub.models.vad_nemo.native.checkpoint import MarbleNetVADSafeTensorsCheckpointAdapter
        from voicehub.models.vad_nemo.native.configuration import MarbleNetVADConfig
        from voicehub.models.vad_nemo.native.modeling import MarbleNetVADModel

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_marblenet_vad_artifacts(
            source,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
            trust_pickle_checkpoint=self._trust_pickle_checkpoint,
        )
        values = read_json_file(artifacts.config)
        values["dither"] = self.config.training_dither
        native_config = MarbleNetVADConfig.from_dict(values)
        model = MarbleNetVADModel(native_config)
        adapter = MarbleNetVADSafeTensorsCheckpointAdapter()
        with SafeTensorReader(artifacts.checkpoint) as reader:
            adapter.load_streaming(
                model,
                reader,
                values,
                strict=True,
            )
        model.to(device=self.device, dtype=torch.float32)
        self.config.sample_rate = native_config.sampling_rate
        self.artifacts = artifacts
        self.native_config = native_config
        self.checkpoint_adapter = adapter.qualified_id
        self.model = model

    def _detect(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        threshold: float = 0.5,
        onset: float | None = None,
        offset: float | None = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_speech_duration_s: float | None = None,
        window_size_samples: int | None = None,
        return_frames: bool = False,
    ) -> VADOutput:
        import torch

        from voicehub.processing.waveform import load_native_audio

        if window_size_samples is not None:
            raise ValueError(
                "Native MarbleNet frame geometry is fixed by the released "
                "checkpoint; `window_size_samples` is not supported.")
        if self.model is None or self.native_config is None:
            raise RuntimeError("Native MarbleNet VAD runtime is not loaded.")
        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        parameter = next(self.model.parameters())
        waveform = materialized.waveform.unsqueeze(0).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        lengths = torch.tensor(
            [materialized.waveform.numel()],
            dtype=torch.long,
            device=parameter.device,
        )
        with torch.inference_mode():
            output = self.model(
                waveforms=waveform,
                waveform_lengths=lengths,
            )
        valid_frames = int(output.frame_lengths[0].item())
        scores = output.speech_probabilities[0, :valid_frames].detach().float().cpu()
        postprocessing = VADInferenceConfig(
            threshold=threshold,
            onset=onset,
            offset=offset,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
        frame_hop = self.native_config.output_frame_hop_samples
        segments = frame_probabilities_to_segments(
            scores.tolist(),
            sampling_rate=self.sample_rate,
            frame_hop_samples=frame_hop,
            frame_length_samples=frame_hop,
            duration_samples=materialized.waveform.numel(),
            config=postprocessing,
        )
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=self.sample_rate,
            probabilities=(tuple(float(item) for item in scores.tolist()) if return_frames else None),
            metadata={
                "backend":
                "voicehub-native",
                "architecture":
                "marblenet-vad",
                "source":
                self.config.name_or_path,
                "checkpoint_format":
                "voicehub-marblenet-vad-v1",
                "checkpoint_adapter":
                self.checkpoint_adapter,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "converted_from_pickle":
                (False if self.artifacts is None else self.artifacts.converted_from_pickle),
                "frame_scores_available":
                True,
                "frame_hop_samples":
                frame_hop,
                "frame_length_samples":
                frame_hop,
            },
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        from voicehub.models.vad_nemo.training_vad_nemo import prepare_marblenet_vad_training_batch

        del phase
        if self.model is None:
            self.load_for_training()
        return prepare_marblenet_vad_training_batch(self, inputs)

    def _validate_training_runtime(self) -> None:
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors
        from voicehub.models.vad_nemo.native.checkpoint import (
            NATIVE_MARBLENET_VAD_FILENAME,
            NATIVE_MARBLENET_VAD_FORMAT,
        )

        if self.model is None or self.native_config is None:
            self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / NATIVE_MARBLENET_VAD_FILENAME,
            metadata={
                "architecture": "marblenet-vad",
                "format": NATIVE_MARBLENET_VAD_FORMAT,
                "sample_rate": str(self.sample_rate),
            },
        )
        values = self.native_config.to_dict()
        values.update({
            "model_type": self.config.model_type,
            'architectures': [self.__class__.__name__],
            "name_or_path": str(save_directory),
            "checkpoint_format": NATIVE_MARBLENET_VAD_FORMAT,
        })
        write_json_file(save_directory / "config.json", values)

    def export_native_pretrained(self, save_directory: str | Path) -> Path:
        destination = Path(save_directory).expanduser()
        self._save_pretrained(destination)
        return destination


__all__ = ["NeMoVADForVoiceActivityDetection"]
