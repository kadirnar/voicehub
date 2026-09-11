"""Native Tiron inference, fine-tuning, and portable export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.models.asr_tiron.configuration import TironASRConfig
from voicehub.models.asr_tiron.metadata import TIRON_CHECKPOINT_REVISION, TIRON_HARNESS_REVISION
from voicehub.models.asr_whisper_native.modeling import WhisperForSpeechRecognition
from voicehub.outputs import ASROutput, ASRSegment


class _DeclaredTokenSpaceProcessor:
    """Prevent padded embedding rows from becoming generated tokens."""

    def __init__(self, token_count: int) -> None:
        self.token_count = token_count

    def __call__(self, input_ids: Any, logits: Any) -> Any:
        del input_ids
        if self.token_count < logits.shape[-1]:
            logits[:, self.token_count:] = float("-inf")
        return logits


class TironForSpeechRecognition(WhisperForSpeechRecognition):
    """Run Tiron's Whisper-derived graph entirely inside VoiceHub.

    The runtime supports one native window of at most 30 seconds and returns
    window-local speaker labels. VoiceHub ports the public production grammar
    used by Tiron's reference harness, but deliberately does not pretend that
    local labels are meeting-global identities: cross-window voice embedding
    and clustering is a separate architecture.
    """

    config_class = TironASRConfig
    default_model_name_or_path = "Trelis/tiron"
    default_max_new_tokens = 444
    maximum_window_seconds = 30.0
    architecture_family = "speech-seq2seq"

    def __init__(
        self,
        config: TironASRConfig | str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self._speaker_token_ids: tuple[int, ...] = ()
        self._speaker_by_id: dict[int, int] = {}
        self._declared_token_count = 0
        super().__init__(config, **kwargs)

    def _validate_tokenizer_vocabulary(
        self,
        tokenizer: Any,
        native_config: Any,
    ) -> None:
        from voicehub.models.asr_tiron.constraints import validate_public_tiron_token_layout

        native_model_type = str(native_config.extra_config.get("model_type", "")).strip().lower()
        if native_model_type not in {"whisper", self.config_class.model_type}:
            raise ValueError(
                "Tiron requires a Whisper architecture checkpoint; found "
                f"{native_model_type or '<missing>'!r}.")
        if tokenizer.token_id_space_size > native_config.vocab_size:
            raise ValueError(
                "The Tiron tokenizer declares token IDs outside the model "
                f"vocabulary ({tokenizer.token_id_space_size} > "
                f"{native_config.vocab_size}).")

        speaker_ids = []
        for index in range(1, 9):
            token = f"<|speaker{index}|>"
            try:
                speaker_ids.append(int(tokenizer.special_tokens[token]))
            except KeyError:
                raise ValueError(f"The Tiron tokenizer is missing required token {token!r}.") from None
        validate_public_tiron_token_layout(
            eos_token_id=tokenizer.eot,
            no_speech_token_id=tokenizer.no_speech,
            no_timestamps_token_id=tokenizer.no_timestamps,
            timestamp_begin_id=tokenizer.timestamp_begin,
            timestamp_end_id=tokenizer.timestamp_end,
            speaker_token_ids=speaker_ids,
        )
        self._speaker_token_ids = tuple(speaker_ids)
        self._speaker_by_id = {token_id: index for index, token_id in enumerate(speaker_ids, start=1)}
        self._declared_token_count = tokenizer.token_id_space_size

    def _resolved_language(
        self,
        language: str | None,
    ) -> str:
        requested = self.config.default_language if language is None else language
        if not isinstance(requested, str) or not requested.strip():
            raise ValueError("Tiron language must be a non-empty string.")
        normalized = requested.strip().lower()
        if normalized == "auto":
            return normalized
        resolved = self._normalized_language(normalized)
        if resolved is None:
            raise ValueError(f"Tiron could not resolve language {language!r}.")
        return resolved

    def _detect_language_from_encoder(self, encoder_hidden_states: Any) -> str:
        import torch

        if self.tokenizer is None or self.generation_adapter is None:
            raise RuntimeError("Tiron must be loaded before language detection.")
        language_tokens = self.generation_adapter.token_set.language_tokens
        if not language_tokens:
            raise ValueError("The Tiron checkpoint does not declare language tokens.")
        start = torch.tensor(
            [[self.tokenizer.sot]],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )
        decoded = self.model.decode(
            start,
            encoder_hidden_states,
            use_cache=False,
        )
        candidate_ids = torch.tensor(
            tuple(language_tokens.values()),
            dtype=torch.long,
            device=decoded.logits.device,
        )
        selected = decoded.logits[
            0,
            -1,
        ].index_select(0, candidate_ids).argmax()
        token_id = int(candidate_ids[selected].item())
        for code, candidate in language_tokens.items():
            if candidate == token_id:
                return code
        raise RuntimeError("Tiron language detection returned an unknown token.")

    def _constraint_processor(self, *, max_speakers: int | None) -> Any:
        from voicehub.models.asr_tiron.constraints import TironConstraintLogitsProcessor

        if self.tokenizer is None:
            raise RuntimeError("Tiron tokenizer is not loaded.")
        return TironConstraintLogitsProcessor(
            prompt_length=3,
            speaker_token_ids=self._speaker_token_ids,
            timestamp_begin_id=self.tokenizer.timestamp_begin,
            timestamp_end_id=self.tokenizer.timestamp_end,
            no_timestamps_token_id=self.tokenizer.no_timestamps,
            no_speech_token_id=self.tokenizer.no_speech,
            eos_token_id=self.tokenizer.eot,
            declared_token_count=self._declared_token_count,
            max_speakers=max_speakers,
            allow_initial_no_speech=True,
        )

    def _generate_window(
        self,
        input_features: Any,
        *,
        language: str,
        max_new_tokens: int,
        max_speakers: int | None,
        constrained_decoding: bool,
    ) -> tuple[list[int], str]:
        import torch

        from voicehub.generation import (
            AutoregressiveGenerator,
            GenerationConfig,
            GenerationStepInput,
            GenerationStepOutput,
        )

        if self.model is None or self.native_config is None:
            raise RuntimeError("Tiron model is not loaded.")
        if self.tokenizer is None:
            raise RuntimeError("Tiron tokenizer is not loaded.")

        encoded = self.model.encode(input_features)
        resolved_language = (self._detect_language_from_encoder(encoded) if language == "auto" else language)
        language_token_id = self.tokenizer.to_language_token(resolved_language)
        prompt = torch.tensor(
            [[
                self.tokenizer.sot,
                language_token_id,
                self.tokenizer.transcribe,
            ]],
            dtype=torch.long,
            device=encoded.device,
        )
        limit = min(
            max_new_tokens,
            self.native_config.max_target_positions - prompt.shape[1] - 1,
        )
        if limit < 1:
            raise ValueError("`max_new_tokens` leaves no room in Tiron's decoder context.")

        def decoder_step(step: GenerationStepInput) -> GenerationStepOutput:
            output = self.model.decode(
                step.token_ids,
                encoded,
                past_key_values=step.cache,
                use_cache=step.use_cache,
            )
            return GenerationStepOutput(
                logits=output.logits[:, -1],
                cache=output.past_key_values,
            )

        processor = (
            self._constraint_processor(max_speakers=max_speakers)
            if constrained_decoding else _DeclaredTokenSpaceProcessor(self._declared_token_count))
        generated = AutoregressiveGenerator().generate(
            decoder_step,
            prompt,
            GenerationConfig(
                max_new_tokens=limit,
                do_sample=False,
                eos_token_id=self.tokenizer.eot,
                pad_token_id=self.tokenizer.eot,
                use_cache=True,
            ),
            logits_processors=(processor, ),
        )
        return (
            generated.sequences[0, prompt.shape[1]:].tolist(),
            resolved_language,
        )

    def _decode_text_tokens(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        if self.tokenizer is None:
            raise RuntimeError("Tiron tokenizer is not loaded.")
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        ).strip()

    def _parse_generated_tokens(
        self,
        token_ids: Sequence[int],
        *,
        language: str,
        chunk_duration: float,
    ) -> tuple[str, tuple[ASRSegment, ...]]:
        if self.tokenizer is None:
            raise RuntimeError("Tiron tokenizer is not loaded.")

        segments: list[ASRSegment] = []
        text_tokens: list[int] = []
        speaker: str | None = None
        speaker_index: int | None = None
        start: float | None = None

        def flush(end: float | None = None) -> None:
            nonlocal start
            text = self._decode_text_tokens(text_tokens)
            text_tokens.clear()
            if not text:
                start = None
                return
            normalized_start = (None if start is None else min(max(start, 0.0), chunk_duration))
            normalized_end = (None if end is None else min(max(end, 0.0), chunk_duration))
            if (normalized_start is not None and normalized_end is not None and
                    normalized_end < normalized_start):
                normalized_end = normalized_start
            metadata: dict[str, Any] = {"speaker_scope": "window"}
            if speaker_index is not None:
                metadata["local_speaker_index"] = speaker_index
            segments.append(
                ASRSegment(
                    text=text,
                    start=normalized_start,
                    end=normalized_end,
                    language=language,
                    speaker=speaker,
                    metadata=metadata,
                ))
            start = None

        for raw_token_id in token_ids:
            if isinstance(raw_token_id, bool) or not isinstance(
                    raw_token_id,
                    Integral,
            ):
                raise TypeError("Tiron generation returned a non-integer token ID.")
            token_id = int(raw_token_id)
            if token_id == self.tokenizer.eot:
                break
            if token_id == self.tokenizer.no_speech:
                if not segments and not text_tokens:
                    return "", ()
                text_tokens.clear()
                start = None
                continue
            local_index = self._speaker_by_id.get(token_id)
            if local_index is not None:
                flush()
                speaker_index = local_index
                speaker = f"SPEAKER_{local_index - 1:02d}"
                continue
            timestamp = self.tokenizer.timestamp_for_token(token_id)
            if timestamp is not None:
                if text_tokens:
                    flush(end=timestamp.seconds)
                else:
                    start = timestamp.seconds
                continue
            if token_id in {
                    self.tokenizer.sot,
                    self.tokenizer.transcribe,
                    self.tokenizer.translate,
                    self.tokenizer.no_timestamps,
                    *self.tokenizer.all_language_tokens,
            }:
                continue
            if token_id >= self._declared_token_count:
                raise ValueError(f"Tiron emitted undeclared padded token ID {token_id}.")
            text_tokens.append(token_id)
        flush(end=chunk_duration if start is not None else None)

        text = " ".join(segment.text for segment in segments if segment.text).strip()
        return text, tuple(segments)

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s: float | tuple[float, float] | None = None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords: str | tuple[str, ...] | list[str] | None = None,
        max_speakers: int | None = None,
        constrained_decoding: bool | None = None,
    ) -> ASROutput:
        from voicehub.processing.waveform import load_native_audio

        if task != "transcribe":
            raise ValueError("Tiron supports transcription, not translation.")
        if return_timestamps == "word":
            raise ValueError("Tiron exposes speaker-segment timestamps, not word-level "
                             "alignment.")
        if chunk_length_s is not None or stride_length_s is not None:
            raise ValueError(
                "Native Tiron accepts one window at a time; meeting chunking "
                "and cross-window speaker linking require a separate "
                "orchestration architecture.")
        if batch_size not in (None, 1):
            raise ValueError("Native Tiron accepts one audio window at a time.")
        if num_beams not in (None, 1):
            raise ValueError("Tiron's published decoder requires `num_beams=1`.")
        if hotwords is not None:
            raise ValueError("Tiron does not expose hotword decoding.")
        if constrained_decoding is None:
            constrained_decoding = self.config.constrained_decoding
        if not isinstance(constrained_decoding, bool):
            raise TypeError("`constrained_decoding` must be a boolean.")
        if max_speakers is None:
            max_speakers = self.config.max_speakers
        generated_limit = (self.default_max_new_tokens if max_new_tokens is None else max_new_tokens)
        if isinstance(generated_limit, bool) or not isinstance(
                generated_limit,
                Integral,
        ):
            raise TypeError("`max_new_tokens` must be an integer.")
        generated_limit = int(generated_limit)
        if generated_limit < 1:
            raise ValueError("`max_new_tokens` must be greater than zero.")

        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=16_000,
        )
        if materialized.duration > self.maximum_window_seconds:
            raise ValueError("Native Tiron accepts at most 30 seconds of audio per "
                             "request.")
        requested_language = self._resolved_language(language)
        input_features = self._chunk_features(materialized.waveform)
        token_ids, resolved_language = self._generate_window(
            input_features,
            language=requested_language,
            max_new_tokens=generated_limit,
            max_speakers=max_speakers,
            constrained_decoding=constrained_decoding,
        )
        text, segments = self._parse_generated_tokens(
            token_ids,
            language=resolved_language,
            chunk_duration=materialized.duration,
        )
        return ASROutput(
            text=text,
            segments=segments,
            language=resolved_language,
            duration=materialized.duration,
            metadata={
                "architecture": "whisper",
                "provider": "tiron",
                "backend": "voicehub-native",
                "speaker_scope": "window",
                "native_segment_timestamps": True,
                "constrained_decoding": constrained_decoding,
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
                "reference_harness_revision": TIRON_HARNESS_REVISION,
            },
        )

    def _allowed_training_special_tokens(self) -> frozenset[str]:
        if self.tokenizer is None:
            raise RuntimeError("Tiron tokenizer is not loaded.")
        allowed_ids = {
            self.tokenizer.no_speech,
            *self._speaker_token_ids,
            *range(
                self.tokenizer.timestamp_begin,
                self.tokenizer.timestamp_end + 1,
            ),
        }
        return frozenset(
            token for token, token_id in self.tokenizer.special_tokens.items() if token_id in allowed_ids)

    def _validate_training_payload(self, token_ids: Sequence[int]) -> None:
        if self.tokenizer is None:
            raise RuntimeError("Tiron tokenizer is not loaded.")
        values = tuple(int(token_id) for token_id in token_ids)
        if values == (self.tokenizer.no_speech, ):
            return
        if not values or values[0] != self._speaker_token_ids[0]:
            raise ValueError(
                "A Tiron target must start with `<|speaker1|>` or consist "
                "only of `<|nospeech|>`.")

        state = "after_speaker"
        highest_speaker = 0
        for token_id in values[1:]:
            speaker_position = self._speaker_by_id.get(token_id)
            timestamp = self.tokenizer.is_timestamp(token_id)
            is_text = token_id < self.tokenizer.eot
            if state == "after_speaker":
                if not timestamp:
                    raise ValueError(
                        "Every Tiron speaker token must be followed by an "
                        "opening timestamp.")
                state = "after_open"
            elif state == "after_open":
                if not is_text:
                    raise ValueError("A Tiron opening timestamp must be followed by text.")
                state = "in_text"
            elif state == "in_text":
                if timestamp:
                    state = "after_close"
                elif not is_text:
                    raise ValueError("Tiron text may only be followed by text or a closing "
                                     "timestamp.")
            else:
                if timestamp:
                    state = "after_open"
                elif speaker_position is not None:
                    expected = highest_speaker + 2
                    if speaker_position != expected:
                        raise ValueError(
                            "Tiron speaker blocks must introduce contiguous "
                            f"speaker slots; expected speaker{expected}.")
                    highest_speaker += 1
                    state = "after_speaker"
                else:
                    raise ValueError(
                        "A closing Tiron timestamp must be followed by another "
                        "timestamp, the next speaker, or the end of the target.")
        if state != "after_close":
            raise ValueError("A non-silent Tiron target must end with a closing timestamp.")

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        import torch

        del phase
        if "input_features" in inputs and "labels" in inputs:
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.native_config is None or self.tokenizer is None:
            raise RuntimeError("Tiron training runtime is not loaded.")

        audio = inputs.get("audio")
        text = inputs.get(
            "text",
            inputs.get("transcription", inputs.get("transcript")),
        )
        if isinstance(text, str):
            texts = (text, )
            was_batched = False
        elif isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
            texts = tuple(text)
            was_batched = True
        else:
            raise ValueError(
                "Tiron training records require a non-empty inline "
                "speaker/timestamp target.")
        if not texts or any(not isinstance(value, str) or not value.strip() for value in texts):
            raise ValueError("Tiron training targets must contain non-empty strings.")
        if audio is None:
            raise ValueError("Tiron training records require `audio`.")

        audio_values = self._training_audio_batch(
            audio,
            batch_size=len(texts),
            was_batched=was_batched,
            model_name="Tiron",
            target_name="targets",
        )
        if len(audio_values) != len(texts):
            raise ValueError(
                "Tiron training requires one target per waveform "
                f"({len(texts)} targets, {len(audio_values)} waveforms).")
        rates = self._training_batch_values(
            inputs.get(
                "sampling_rate",
                inputs.get("sample_rate"),
            ),
            batch_size=len(texts),
            name="sampling_rate",
        )
        lengths = self._training_batch_values(
            inputs.get("audio_lengths"),
            batch_size=len(texts),
            name="audio_lengths",
        )
        languages = self._training_batch_values(
            inputs.get("language"),
            batch_size=len(texts),
            name="language",
        )
        tasks = self._training_batch_values(
            inputs.get("task"),
            batch_size=len(texts),
            name="task",
        )
        maximum_samples = min(
            self.native_config.expected_input_frames * 160,
            round(self.maximum_window_seconds * 16_000),
        )
        allowed_special = self._allowed_training_special_tokens()
        features: list[Any] = []
        label_rows: list[tuple[int, ...]] = []
        for row_index, (
                audio_value,
                text_value,
                rate,
                length,
                language,
                task,
        ) in enumerate(zip(
                audio_values,
                texts,
                rates,
                lengths,
                languages,
                tasks,
        )):
            resolved_language = self._resolved_language(
                self.config.default_language if language is None else language, )
            if resolved_language == "auto":
                raise ValueError(f"Tiron training row {row_index} requires an explicit "
                                 "language.")
            resolved_task = "transcribe" if task is None else task
            if resolved_task != "transcribe":
                raise ValueError(f"Tiron training row {row_index} supports transcription "
                                 "only.")

            materialized = self._materialize_training_audio(
                audio_value,
                sampling_rate=rate,
                audio_length=length,
                row_index=row_index,
                model_name="Tiron",
            )
            if materialized.waveform.numel() > maximum_samples:
                maximum_seconds = maximum_samples / 16_000
                raise ValueError(
                    f"Tiron training row {row_index} exceeds the model's "
                    f"{maximum_seconds:g}-second audio context.")
            payload = self.tokenizer.encode(
                text_value,
                allowed_special=allowed_special,
                disallowed_special="all",
            ).input_ids
            try:
                self._validate_training_payload(payload)
            except ValueError as error:
                raise ValueError(f"Tiron training row {row_index}: {error}") from error
            label_ids = (
                self.tokenizer.to_language_token(resolved_language),
                self.tokenizer.transcribe,
                *payload,
                self.tokenizer.eot,
            )
            if len(label_ids) > self.native_config.max_target_positions:
                raise ValueError(
                    f"Tiron training row {row_index} target exceeds the "
                    "decoder context after tokenization "
                    f"({len(label_ids)} > "
                    f"{self.native_config.max_target_positions}).")
            features.append(self._chunk_features(materialized.waveform).squeeze(0), )
            label_rows.append(label_ids)

        input_features = torch.stack(features)
        labels = self._pad_training_labels(
            label_rows,
            padding_value=-100,
            device=input_features.device,
        )
        if not was_batched:
            input_features = input_features[0]
            labels = labels[0]
        return {
            "input_features": input_features,
            "labels": labels,
        }

    def _validate_training_runtime(self) -> None:
        source = str(self.config.name_or_path or self.default_model_name_or_path).lower()
        if source.endswith((".gguf", ".bin", ".onnx")):
            raise ValueError(
                "Tiron serving artifacts are inference-only; fine-tuning "
                "requires the published Safetensors checkpoint.")

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        config_path = save_directory / "config.json"
        values = read_json_file(config_path)
        values.update({
            "model_type": self.config_class.model_type,
            "default_language": self.config.default_language,
            "constrained_decoding": self.config.constrained_decoding,
            "max_speakers": self.config.max_speakers,
            "tiron_checkpoint_revision": TIRON_CHECKPOINT_REVISION,
            "tiron_harness_revision": TIRON_HARNESS_REVISION,
            "tiron_token_grammar": "speaker_blocks-v1",
        })
        write_json_file(config_path, values)


__all__ = ["TironForSpeechRecognition"]
