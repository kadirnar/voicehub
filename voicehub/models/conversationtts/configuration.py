"""ConversationTTS configuration."""

from __future__ import annotations

from voicehub.configuration import VoiceHubConfig
from voicehub.models.conversationtts.native.metadata import (
    CONVERSATIONTTS_CHECKPOINT_FILENAME,
    CONVERSATIONTTS_CHECKPOINT_REVISION,
    CONVERSATIONTTS_MIMI_FILENAME,
    CONVERSATIONTTS_MIMI_REPOSITORY,
    CONVERSATIONTTS_MIMI_REVISION,
)


class ConversationTTSConfig(VoiceHubConfig):
    """Configuration for the CC BY-NC ConversationTTS release."""

    model_type = "conversationtts"

    def __init__(
        self,
        *,
        checkpoint_filename: str = CONVERSATIONTTS_CHECKPOINT_FILENAME,
        checkpoint_revision: str = CONVERSATIONTTS_CHECKPOINT_REVISION,
        text_tokenizer_path: str | None = None,
        audio_tokenizer_path: str | None = None,
        audio_tokenizer_repo_id: str = CONVERSATIONTTS_MIMI_REPOSITORY,
        audio_tokenizer_filename: str = CONVERSATIONTTS_MIMI_FILENAME,
        audio_tokenizer_revision: str = CONVERSATIONTTS_MIMI_REVISION,
        model_args: dict | None = None,
        torch_dtype: str = "bfloat16",
        sample_rate: int = 24000,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_revision = checkpoint_revision
        self.text_tokenizer_path = text_tokenizer_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.audio_tokenizer_repo_id = audio_tokenizer_repo_id
        self.audio_tokenizer_filename = audio_tokenizer_filename
        self.audio_tokenizer_revision = audio_tokenizer_revision
        self.model_args = model_args or {
            "backbone_flavor": "llama-1B",
            "decoder_flavor": "llama-100M",
            "text_vocab_size": 128_256,
            "audio_vocab_size": 2051,
            "audio_num_codebooks": 32,
        }
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.validate()

    def validate(self) -> None:
        """Reject ambiguous runtime and architecture configuration."""
        for name in (
                "checkpoint_filename",
                "checkpoint_revision",
                "audio_tokenizer_repo_id",
                "audio_tokenizer_filename",
                "audio_tokenizer_revision",
                "torch_dtype",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{name}` must be a non-empty string.")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int) or
                self.sample_rate != 24_000):
            raise ValueError("ConversationTTS and its Mimi codec operate at 24,000 Hz.")
        required = {
            "backbone_flavor",
            "decoder_flavor",
            "text_vocab_size",
            "audio_vocab_size",
            "audio_num_codebooks",
        }
        if not isinstance(self.model_args, dict):
            raise TypeError("`model_args` must be a dictionary.")
        missing = sorted(required - set(self.model_args))
        unexpected = sorted(set(self.model_args) - required)
        if missing or unexpected:
            raise ValueError(
                "ConversationTTS `model_args` has an incompatible schema: "
                f"missing={missing!r}, unexpected={unexpected!r}.")
        for name in (
                "text_vocab_size",
                "audio_vocab_size",
                "audio_num_codebooks",
        ):
            value = self.model_args[name]
            if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"ConversationTTS `{name}` must be a positive integer.")
