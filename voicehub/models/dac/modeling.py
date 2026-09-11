"""Standalone DAC loading, encoding, decoding, and portable saving."""

from pathlib import Path

from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file
from voicehub.models.codec import PreTrainedCodecModel

from .configuration import DacCodecConfig


class DacForAudioCodec(PreTrainedCodecModel):
    config_class = DacCodecConfig
    default_model_name_or_path = "descript/dac_44khz"

    def _load_pretrained_model(self):
        from voicehub.checkpointing import SafeTensorReader

        from .native.checkpoint import HuggingFaceDacCheckpointAdapter
        from .native.configuration import DacConfig
        from .native.modeling import DacModel

        source = self.checkpoint_source
        source_path = Path(source).expanduser()
        direct_checkpoint = source_path if source and source_path.is_file() else None
        config_source = str(direct_checkpoint.parent) if direct_checkpoint is not None else source
        options = {
            name: getattr(self.config, name)
            for name in ("revision", "cache_dir", "local_files_only") if hasattr(self.config, name)
        }
        values = self.config.native_config
        if source:
            values = read_json_file(resolve_pretrained_file(config_source, "config.json", **options))
            values = values.get("native_config") or values
        native_config = DacConfig.from_dict(values)
        model = DacModel(native_config).to(self.device)
        if source:
            checkpoint = direct_checkpoint or resolve_pretrained_file(source, "model.safetensors", **options)
            with SafeTensorReader(checkpoint) as reader:
                if reader.metadata.get("format") == "voicehub-dac":
                    model.load_state_dict(reader.state_dict(), strict=True)
                else:
                    HuggingFaceDacCheckpointAdapter().load(model, reader, native_config.to_dict())
        self.config.native_config = native_config.to_dict()
        self.config.sample_rate = native_config.sampling_rate
        self.model = model

    def _encode(self, waveform, *, n_quantizers=None):
        if waveform.shape[1] != 1:
            raise ValueError("DAC expects one audio channel.")
        waveform = self.model.preprocess(waveform, self.sample_rate)
        return self.model.encode_output(waveform, n_quantizers=n_quantizers).audio_codes

    def _decode(self, codes):
        import torch

        if not isinstance(codes, torch.Tensor) or codes.ndim != 3 or codes.dtype != torch.long:
            raise ValueError("DAC codes must be a long tensor with shape [batch, codebooks, frames].")
        if not 1 <= codes.shape[1] <= self.model.config.n_codebooks:
            raise ValueError("DAC codebook count does not match this model.")
        if codes.numel() == 0 or bool(((codes < 0) | (codes >= self.model.config.codebook_size)).any()):
            raise ValueError("DAC codes are outside the codebook range.")
        latents, _, _ = self.model.quantizer.from_codes(codes.to(self.device))
        return self.model.decode(latents)

    def _save_pretrained(self, save_directory: Path):
        from voicehub.checkpointing import save_safetensors

        self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        write_json_file(save_directory / "config.json", self.model.config.to_dict())
        save_safetensors(
            self.model.state_dict(),
            save_directory / "model.safetensors",
            metadata={"format": "voicehub-dac"})
