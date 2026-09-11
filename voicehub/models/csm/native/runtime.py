"""Native CSM inference, preprocessing, and portable artifact runtime."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from voicehub.audio import load_audio
from voicehub.hub import read_json_file, write_json_file
from voicehub.models.csm.native.artifacts import CSMArtifacts, resolve_csm_artifacts
from voicehub.models.csm.native.checkpoint import export_csm_checkpoint, load_csm_checkpoint
from voicehub.models.csm.native.configuration import CSMArchitectureConfig
from voicehub.models.csm.native.mimi import load_mimi
from voicehub.models.csm.native.modeling import CSMModel
from voicehub.models.csm.native.processing import CSMCodeSegment, CSMProcessor, CSMTextTokenizer


@runtime_checkable
class CSMCodec(Protocol):
    """Minimal codec contract required by CSM raw-audio workflows."""

    sample_rate: int

    def encode(self, waveform: Tensor) -> Tensor:
        ...

    def decode(self, codes: Tensor) -> Tensor:
        ...


def _torch_dtype(
    name: str | torch.dtype,
    *,
    device: str | torch.device,
) -> torch.dtype:
    if isinstance(name, torch.dtype):
        dtype = name
    elif isinstance(name, str):
        aliases = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp32": "float32",
        }
        normalized = aliases.get(name.lower(), name.lower())
        dtype = getattr(torch, normalized, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported CSM dtype {name!r}.")
    else:
        raise TypeError("CSM dtype must be a string or `torch.dtype`.")
    if torch.device(device).type == "cpu" and dtype in (
            torch.float16,
            torch.bfloat16,
    ):
        return torch.float32
    return dtype


class CSMRuntime:
    """A complete native CSM language-model and optional Mimi runtime."""

    def __init__(
        self,
        model: CSMModel,
        processor: CSMProcessor,
        *,
        codec: CSMCodec | None,
        artifacts: CSMArtifacts | None = None,
        audio_postprocessor: Any | None = None,
    ) -> None:
        if not isinstance(model, CSMModel):
            raise TypeError("`model` must be a native `CSMModel`.")
        if not isinstance(processor, CSMProcessor):
            raise TypeError("`processor` must be a native `CSMProcessor`.")
        if codec is not None and not isinstance(codec, CSMCodec):
            raise TypeError("CSM codec must implement encode/decode/sample_rate.")
        if (audio_postprocessor is not None and not callable(audio_postprocessor)):
            raise TypeError("`audio_postprocessor` must be callable or None.")
        watermark_declaration = getattr(
            audio_postprocessor,
            "watermarks_audio",
            False,
        )
        if not isinstance(watermark_declaration, bool):
            raise TypeError("CSM postprocessor `watermarks_audio` must be a boolean.")
        self.model = model
        self.processor = processor
        self.codec = codec
        self.artifacts = artifacts
        self.audio_postprocessor = audio_postprocessor
        self.audio_postprocessor_watermarks = watermark_declaration
        self.sample_rate = (processor.sample_rate if codec is None else int(codec.sample_rate))
        self.device = next(model.parameters()).device
        self.codec_device = self._module_device(codec, fallback=self.device)

    @staticmethod
    def _module_device(
        module: Any | None,
        *,
        fallback: torch.device,
    ) -> torch.device:
        if module is None:
            return fallback
        for accessor_name in ("parameters", "buffers"):
            accessor = getattr(module, accessor_name, None)
            if not callable(accessor):
                continue
            try:
                value = next(iter(accessor()))
            except StopIteration:
                continue
            return value.device
        return fallback

    def _require_codec(self) -> CSMCodec:
        if self.codec is None:
            raise RuntimeError(
                "Raw-audio CSM inference requires the native Mimi artifact "
                "or an injected codec. Token-level fine-tuning remains "
                "available with pre-encoded `audio_codes`.")
        return self.codec

    @torch.no_grad()
    def encode_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
    ) -> Tensor:
        codec = self._require_codec()
        loaded = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        waveform = loaded.waveform.to(
            device=self.codec_device,
            dtype=torch.float32,
        )
        codes = codec.encode(waveform.unsqueeze(0).unsqueeze(0))
        if (codes.ndim != 3 or codes.shape[0] != 1 or
                codes.shape[1] != self.model.config.num_audio_codebooks):
            raise RuntimeError("Mimi returned an incompatible CSM code tensor.")
        return codes[0].to(
            device=self.device,
            dtype=torch.long,
        )

    def context_segment(
        self,
        *,
        speaker: int,
        text: str,
        audio: Any,
        sampling_rate: int | None = None,
    ) -> CSMCodeSegment:
        return CSMCodeSegment(
            speaker=speaker,
            text=text,
            audio_codes=self.encode_audio(
                audio,
                sampling_rate=sampling_rate,
            ),
        )

    @torch.no_grad()
    def generate(
        self,
        text: str,
        *,
        speaker: int = 0,
        context: Sequence[CSMCodeSegment] = (),
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> tuple[Tensor, dict[str, Any]]:
        codec = self._require_codec()
        frames = int(float(max_audio_length_ms) * self.model.config.frame_rate / 1_000.0)
        if frames <= 0:
            raise ValueError("`max_audio_length_ms` must cover at least one CSM frame.")
        prompt_tokens, prompt_mask = self.processor.prompt(
            text,
            speaker=speaker,
            context=context,
            device=self.device,
        )
        codes = self.model.generate_audio_codes(
            prompt_tokens,
            prompt_mask,
            max_new_frames=frames,
            temperature=temperature,
            top_k=top_k,
        )
        audio = codec.decode(codes.to(device=self.codec_device), ).squeeze(0).squeeze(0).float()
        postprocessed = False
        watermarked = False
        if self.audio_postprocessor is not None:
            result = self.audio_postprocessor(audio, self.sample_rate)
            if isinstance(result, tuple):
                audio, sample_rate = result
                if int(sample_rate) != self.sample_rate:
                    raise ValueError("CSM postprocessors must return 24 kHz audio.")
            else:
                audio = result
            if not isinstance(audio, Tensor):
                raise TypeError("CSM audio postprocessors must return a Tensor.")
            postprocessed = True
            watermarked = self.audio_postprocessor_watermarks
        return audio, {
            "audio_frames": codes.shape[-1],
            "context_segments": len(context),
            "native_runtime": True,
            "audio_postprocessed": postprocessed,
            # SilentCipher is a separately trained runtime. VoiceHub does not
            # claim watermarking merely because a generic postprocessor ran.
            "watermarked": watermarked,
        }

    @torch.no_grad()
    def encode_training_records(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert raw waveforms to frozen Mimi codes before collation."""
        output = []
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("Every CSM training record must be a mapping.")
            item = dict(record)
            if "segments" in item:
                segments = item["segments"]
                if (isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence) or not segments):
                    raise ValueError("CSM `segments` must be a non-empty sequence.")
                item["segments"] = self.encode_training_records(segments)
                output.append(item)
                continue
            if "audio_codes" not in item:
                if "audio" not in item:
                    raise ValueError("CSM records require `audio_codes` or raw `audio`.")
                rate = item.pop(
                    "sampling_rate",
                    item.pop("sample_rate", None),
                )
                item["audio_codes"] = self.encode_audio(
                    item.pop("audio"),
                    sampling_rate=rate,
                )
            output.append(item)
        return output

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        depth_decoder_labels_ratio: float = 1.0,
    ) -> dict[str, Tensor]:
        if isinstance(inputs, Mapping) and {
                "tokens",
                "tokens_mask",
                "labels",
        }.issubset(inputs):
            return dict(inputs)
        if isinstance(inputs, Mapping):
            records = inputs.get("records")
            if records is None:
                records = (inputs, )
        else:
            records = inputs
        if (isinstance(records, (str, bytes)) or not isinstance(records, Sequence)):
            raise TypeError("CSM training inputs must contain record mappings.")
        prepared = self.encode_training_records(records)
        return self.processor.training_batch(
            prepared,
            depth_decoder_labels_ratio=depth_decoder_labels_ratio,
            device=self.device,
        )

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        include_codec: bool = True,
    ) -> Path:
        """Export a complete native graph and, when available, frozen Mimi."""
        output = Path(directory).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        export_csm_checkpoint(
            self.model,
            output / "model.safetensors",
        )
        write_json_file(
            output / "config.json",
            self.model.config.to_dict(),
        )
        self.processor.save_pretrained(output)
        if include_codec:
            source = (None if self.artifacts is None else self.artifacts.codec_checkpoint)
            if source is None:
                raise RuntimeError(
                    "Cannot export a complete raw-audio CSM runtime because "
                    "the injected codec has no immutable source artifact.")
            destination = output / "mimi.safetensors"
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
        return output.resolve()


def load_csm_runtime(
    source: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = "bfloat16",
    codec: CSMCodec | None = None,
    codec_path: str | Path | None = None,
    include_codec: bool = True,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_integrity: bool = False,
    verify_checkpoint_integrity: bool = False,
    audio_postprocessor: Any | None = None,
) -> CSMRuntime:
    """Load an official or VoiceHub-exported native CSM runtime."""
    if codec is not None and codec_path is not None:
        raise ValueError("Pass an injected CSM `codec` or `codec_path`, not both.")
    artifacts = resolve_csm_artifacts(
        source,
        revision=revision,
        codec_path=codec_path,
        include_codec=include_codec and codec is None,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        verify_integrity=verify_integrity,
        verify_checkpoint_integrity=verify_checkpoint_integrity,
    )
    config = (
        CSMArchitectureConfig.from_dict(read_json_file(artifacts.config))
        if artifacts.config is not None else CSMArchitectureConfig())
    resolved_dtype = _torch_dtype(dtype, device=device)
    model = CSMModel(
        config,
        device="meta",
        dtype=resolved_dtype,
    )
    load_csm_checkpoint(
        model,
        artifacts.checkpoint,
        device=device,
        dtype=resolved_dtype,
        require_official_inventory=artifacts.official_model,
    )
    tokenizer = CSMTextTokenizer.from_file(artifacts.tokenizer)
    processor = CSMProcessor(tokenizer, config)
    if codec is None and artifacts.codec_checkpoint is not None:
        codec = load_mimi(
            artifacts.codec_checkpoint,
            device=device,
            require_official_inventory=artifacts.official_codec,
        )
    model.eval()
    return CSMRuntime(
        model,
        processor,
        codec=codec,
        artifacts=artifacts,
        audio_postprocessor=audio_postprocessor,
    )


__all__ = ["CSMCodec", "CSMRuntime", "load_csm_runtime"]
