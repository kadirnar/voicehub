"""Serializable processors for text-input and audio-input speech tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.configuration import reject_serialized_secrets
from voicehub.hub import read_json_file, resolve_pretrained_file, write_json_file

PROCESSOR_NAME = "processor_config.json"


class BatchFeature(dict):
    """Dictionary of processor values with a tensor-like ``to`` helper."""

    def to(self, device: str):
        """Recursively move tensor-like values and return this instance."""

        def move(value):
            if isinstance(value, dict):
                return value.__class__((key, move(item)) for key, item in value.items())
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if hasattr(value, "to"):
                return value.to(device)
            return value

        for key, value in self.items():
            self[key] = move(value)
        return self


class VoiceHubProcessor:
    """Transform raw synthesis inputs into model-ready values."""

    model_input_names = ("text", )

    def __init__(self, **kwargs):
        reject_serialized_secrets(
            kwargs,
            owner=self.__class__.__name__,
        )
        self.init_kwargs = dict(kwargs)

    def __call__(self, text: str, **kwargs) -> BatchFeature:
        """Prepare text and optional conditioning inputs for generation."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` must be a non-empty string.")
        return BatchFeature(text=text, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return processor construction options."""
        reject_serialized_secrets(
            self.init_kwargs,
            owner=self.__class__.__name__,
        )
        return dict(self.init_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str = "",
        **kwargs,
    ):
        """Load optional processor configuration from local or Hub storage."""
        source = Path(pretrained_model_name_or_path).expanduser()
        cache_dir = kwargs.pop("cache_dir", None)
        revision = kwargs.pop("revision", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        try:
            if source.is_file() and source.name == PROCESSOR_NAME:
                processor_path = source
            else:
                processor_path = resolve_pretrained_file(
                    pretrained_model_name_or_path,
                    PROCESSOR_NAME,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                    revision=revision,
                    token=token,
                    local_files_only=local_files_only,
                )
            values = read_json_file(processor_path)
        except FileNotFoundError:
            values = {}
        values.update(kwargs)
        return cls(**values)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Save processor construction options."""
        output_path = Path(save_directory).expanduser() / PROCESSOR_NAME
        values = self.to_dict()
        reject_serialized_secrets(
            values,
            owner=self.__class__.__name__,
        )
        write_json_file(output_path, values)
        return output_path


class AudioProcessor(VoiceHubProcessor):
    """Validate the public audio-input envelope without loading audio.

    Decoding and resampling stay lazy and are performed by
    :func:`voicehub.audio.load_audio` only when inference begins.
    """

    model_input_names = ("audio", "sampling_rate")

    def __call__(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if audio is None:
            raise ValueError("`audio` cannot be None.")
        if isinstance(audio, (str, Path)) and not str(audio).strip():
            raise ValueError("Audio paths must be non-empty.")
        if sampling_rate is not None:
            if (isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int) or sampling_rate <= 0):
                raise ValueError("`sampling_rate` must be a positive integer or None.")
        return BatchFeature(
            audio=audio,
            sampling_rate=sampling_rate,
            **kwargs,
        )
