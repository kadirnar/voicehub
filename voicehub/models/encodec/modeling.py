"""Standalone Encodec, including segmented scales and stereo audio."""

from pathlib import Path

from voicehub.models.codec import PreTrainedCodecModel

from .configuration import EncodecCodecConfig


class EncodecForAudioCodec(PreTrainedCodecModel):
    config_class = EncodecCodecConfig
    default_model_name_or_path = "encodec_24khz"

    def _load_pretrained_model(self):
        from voicehub.components.audio.codecs.encodec import EncodecConfig, EncodecModel
        from voicehub.components.audio.codecs.encodec.checkpoint import (
            load_encodec_model,
            load_encodec_model_from_safetensors,
        )

        source = self.checkpoint_source
        path = Path(source).expanduser()
        if source and path.is_dir():
            path = path / "model.safetensors"
        if source and path.is_file() and path.suffix == ".safetensors":
            model = load_encodec_model_from_safetensors(path, device=self.device)
        elif not source and self.config.native_config:
            model = EncodecModel.from_config(EncodecConfig.from_dict(self.config.native_config)).to(
                self.device)
        else:
            options = {
                name: getattr(self.config, name)
                for name in ("cache_dir", "local_files_only", "trust_official_pickle")
                if hasattr(self.config, name)
            }
            model = load_encodec_model(
                source or self.default_model_name_or_path, device=self.device, **options)
        if self.config.bandwidth is not None:
            model.set_target_bandwidth(self.config.bandwidth)
        self.config.sample_rate = model.sample_rate
        self.config.native_config = model.config.to_dict()
        self.model = model

    def _encode(self, waveform):
        return self.model.encode(waveform)

    def _decode(self, codes):
        if not isinstance(codes, (list, tuple)) or not codes:
            raise ValueError("Encodec codes must contain at least one (codes, scale) frame.")
        frames = [(values.to(self.device), None if scale is None else scale.to(self.device))
                  for values, scale in codes]
        return self.model.decode(frames)

    def _save_pretrained(self, save_directory: Path):
        from voicehub.components.audio.codecs.encodec.checkpoint import save_encodec_safetensors

        self.load()
        save_directory.mkdir(parents=True, exist_ok=True)
        save_encodec_safetensors(self.model, save_directory / "model.safetensors")
