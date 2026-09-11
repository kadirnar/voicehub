"""Native Qwen3-TTS inference, training ownership, and portable export."""

from __future__ import annotations

import base64
import binascii
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import torch
from torch import Tensor

from voicehub.hub import read_json_file
from voicehub.models.qwen3tts.native.artifacts import Qwen3TTSArtifacts, resolve_qwen3_tts_artifacts
from voicehub.models.qwen3tts.native.checkpoint import (
    export_qwen3_tts_decoder,
    export_qwen3_tts_model,
    export_qwen3_tts_speech_tokenizer,
    inspect_qwen3_tts_checkpoint,
    load_qwen3_tts_decoder_checkpoint,
    load_qwen3_tts_encoder_checkpoint,
    load_qwen3_tts_model_checkpoint,
)
from voicehub.models.qwen3tts.native.codec import Qwen3TTSSpeechDecoder
from voicehub.models.qwen3tts.native.configuration import Qwen3TTSArchitectureConfig, Qwen3TTSTokenizerConfig
from voicehub.models.qwen3tts.native.encoder import Qwen3TTSSpeechEncoder
from voicehub.models.qwen3tts.native.metadata import QWEN3_TTS_CHECKPOINTS
from voicehub.models.qwen3tts.native.modeling import Qwen3TTSForConditionalGeneration
from voicehub.models.qwen3tts.native.tokenization import Qwen3TTSTextTokenizer
from voicehub.optimization.protocols import OptimizationCompileTarget, OptimizationModuleRoot
from voicehub.processing import NativeAudio, load_native_audio, mel_filter_bank

_MAX_REFERENCE_AUDIO_BYTES = 64 * 1024 * 1024


def _dtype(value: str | torch.dtype | None, *, device: torch.device) -> torch.dtype:
    if value is None:
        result = torch.float32
    elif isinstance(value, torch.dtype):
        result = value
    elif isinstance(value, str):
        normalized = value.lower().removeprefix("torch.")
        normalized = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp32": "float32",
        }.get(normalized, normalized)
        result = getattr(torch, normalized, None)
        if not isinstance(result, torch.dtype):
            raise ValueError(f"Unknown Qwen3-TTS dtype {value!r}.")
    else:
        raise TypeError("Qwen3-TTS dtype must be a string, torch.dtype, or None.")
    if not result.is_floating_point:
        raise ValueError("Qwen3-TTS compute dtype must be floating-point.")
    if device.type == "cpu" and result in {torch.float16, torch.bfloat16}:
        return torch.float32
    return result


def _device(value: str | torch.device) -> torch.device:
    if isinstance(value, str) and value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def _assistant_text(text: str) -> str:
    return (f"<|im_start|>assistant\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n")


def _instruction_text(instruction: str) -> str:
    return f"<|im_start|>user\n{instruction}<|im_end|>\n"


def _decode_base64_audio(value: str) -> bytes:
    encoded = value
    if value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        if not separator or not header.lower().startswith("data:audio/"):
            raise ValueError("Qwen3-TTS requires an audio data URL.")
        if ";base64" not in header.lower():
            raise ValueError("Qwen3-TTS audio data URLs must use base64 encoding.")
    try:
        payload = base64.b64decode(
            "".join(encoded.split()),
            validate=True,
        )
    except (binascii.Error, ValueError) as error:
        raise ValueError("Qwen3-TTS received invalid base64 reference audio.") from error
    if not payload:
        raise ValueError("Qwen3-TTS reference audio is empty.")
    if len(payload) > _MAX_REFERENCE_AUDIO_BYTES:
        raise ValueError("Qwen3-TTS reference audio exceeds the 64 MiB limit.")
    return payload


def _download_reference_audio(value: str) -> bytes:
    request = Request(
        value,
        headers={"User-Agent": "VoiceHub-Qwen3-TTS/1"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            final_scheme = urlsplit(response.geturl()).scheme.lower()
            if final_scheme not in {"http", "https"}:
                raise ValueError("Qwen3-TTS reference audio redirected outside HTTP(S).")
            declared_size = response.headers.get("Content-Length")
            if (declared_size is not None and declared_size.isdecimal() and
                    int(declared_size) > _MAX_REFERENCE_AUDIO_BYTES):
                raise ValueError("Qwen3-TTS reference audio exceeds the 64 MiB limit.")
            payload = response.read(_MAX_REFERENCE_AUDIO_BYTES + 1)
    except (OSError, URLError) as error:
        raise ValueError(f"Qwen3-TTS could not download reference audio: {error}.") from error
    if not payload:
        raise ValueError("Qwen3-TTS reference audio download was empty.")
    if len(payload) > _MAX_REFERENCE_AUDIO_BYTES:
        raise ValueError("Qwen3-TTS reference audio exceeds the 64 MiB limit.")
    return payload


def _load_reference_audio(reference_audio: Any):
    if not isinstance(reference_audio, str):
        return load_native_audio(
            reference_audio,
            target_sampling_rate=24_000,
        )
    value = reference_audio.strip()
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        payload = _download_reference_audio(value)
    elif value.startswith("data:audio"):
        payload = _decode_base64_audio(value)
    elif "".join(value.split()).startswith("UklGR"):
        payload = _decode_base64_audio(value)
    else:
        return load_native_audio(
            value,
            target_sampling_rate=24_000,
        )
    with tempfile.TemporaryDirectory(prefix="voicehub-qwen3-tts-") as directory:
        path = Path(directory) / "reference.wav"
        path.write_bytes(payload)
        decoded = load_native_audio(
            path,
            target_sampling_rate=24_000,
        )
    return NativeAudio(
        waveform=decoded.waveform,
        sampling_rate=decoded.sampling_rate,
    )


class Qwen3TTSProcessor:
    """Small processor facade retained for the public training contract."""

    def __init__(self, tokenizer: Qwen3TTSTextTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self,
        *,
        text: str | list[str],
        return_tensors: str | None = None,
        padding: bool | str = False,
        **_: Any,
    ) -> dict[str, Any]:
        texts = [text] if isinstance(text, str) else text
        if not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
            raise TypeError("Qwen3-TTS processor text must be a string or list.")
        encoded = self.tokenizer.encode_batch(texts, padding=padding)
        if return_tensors is None:
            return {
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
            }
        if return_tensors != "pt":
            raise ValueError("Native Qwen3-TTS supports `return_tensors='pt'` only.")
        return {
            "input_ids": torch.tensor(encoded.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                encoded.attention_mask,
                dtype=torch.long,
            ),
        }

    def save_pretrained(self, directory: str | Path) -> Path:
        return self.tokenizer.save_pretrained(directory)


class _SpeechDecoderExporter:

    def __init__(self, runtime: NativeQwen3TTSRuntime) -> None:
        self.runtime = runtime

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        safe_serialization: bool = True,
    ) -> Path:
        if safe_serialization is not True:
            raise ValueError("Native Qwen3-TTS speech export is Safetensors-only.")
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        if self.runtime.speech_encoder is None:
            export_qwen3_tts_decoder(
                self.runtime.speech_decoder,
                target / "model.safetensors",
            )
        else:
            export_qwen3_tts_speech_tokenizer(
                self.runtime.speech_encoder,
                self.runtime.speech_decoder,
                target / "model.safetensors",
            )
        (target / "config.json").write_text(
            json.dumps(
                self.runtime.tokenizer_config.to_dict(),
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        return target.resolve()


class _SpeechFeatureExporter:

    def __init__(self, runtime: NativeQwen3TTSRuntime) -> None:
        self.runtime = runtime

    def save_pretrained(self, directory: str | Path) -> Path:
        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        source = self.runtime.artifacts.speech_preprocessor_config
        if source is not None:
            shutil.copy2(source, target / "preprocessor_config.json")
        return target.resolve()


class _SpeechTokenizerExportFacade:

    def __init__(self, runtime: NativeQwen3TTSRuntime) -> None:
        self.model = _SpeechDecoderExporter(runtime)
        self.feature_extractor = _SpeechFeatureExporter(runtime)


def qwen3_tts_speaker_mel(waveform: Tensor) -> Tensor:
    """Exact 24 kHz, 128-bin log-mel frontend used by the speaker encoder."""
    if not isinstance(waveform, Tensor) or waveform.ndim != 1:
        raise ValueError("Speaker waveform must be a rank-one tensor.")
    padding = (1024 - 256) // 2
    if waveform.numel() <= padding:
        raise ValueError("Qwen3-TTS speaker audio must contain more than 384 samples.")
    waveform = waveform.float().unsqueeze(0)
    waveform = torch.nn.functional.pad(
        waveform.unsqueeze(1),
        (padding, padding),
        mode="reflect",
    ).squeeze(1)
    spectrum = torch.stft(
        waveform,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=torch.hann_window(
            1024,
            device=waveform.device,
            dtype=waveform.dtype,
        ),
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(spectrum.abs().square() + 1e-9)
    filters = mel_filter_bank(
        sample_rate=24_000,
        n_fft=1024,
        n_mels=128,
        minimum_frequency=0,
        maximum_frequency=12_000,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    return torch.log(torch.matmul(filters, magnitude).clamp_min(1e-5)).transpose(1, 2)


@dataclass(slots=True)
class NativeQwen3TTSRuntime:
    artifacts: Qwen3TTSArtifacts
    config: Qwen3TTSArchitectureConfig
    tokenizer_config: Qwen3TTSTokenizerConfig
    tokenizer: Qwen3TTSTextTokenizer
    processor: Qwen3TTSProcessor
    model: Qwen3TTSForConditionalGeneration
    speech_decoder: Qwen3TTSSpeechDecoder
    generation_config: dict[str, Any]
    speech_encoder: Qwen3TTSSpeechEncoder | None = None

    @property
    def device(self) -> torch.device:
        return self.model.device

    def parameters(self):
        """Iterate optimizer-owned modules for runtime capability discovery."""
        yield from self.model.parameters()
        if self.speech_encoder is not None:
            yield from self.speech_encoder.parameters()
        yield from self.speech_decoder.parameters()

    def state_dict(self) -> dict[str, Tensor]:
        """Return canonical keys across the runtime-owned native modules."""
        state = {f"model.{name}": value for name, value in self.model.state_dict().items()}
        state.update({
            f"speech_decoder.{name}": value
            for name, value in self.speech_decoder.state_dict().items()
        })
        if self.speech_encoder is not None:
            state.update({
                f"speech_encoder.{name}": value
                for name, value in self.speech_encoder.state_dict().items()
            })
        return state

    def optimization_module_roots(self):
        """Expose the exact native module roots owned by this runtime."""
        roots = [
            OptimizationModuleRoot("model", self.model),
            OptimizationModuleRoot(
                "speech_decoder",
                self.speech_decoder,
            ),
        ]
        if self.speech_encoder is not None:
            roots.append(OptimizationModuleRoot(
                "speech_encoder",
                self.speech_encoder,
            ))
        return tuple(roots)

    def optimization_compile_targets(self, mode: str):
        """Expose graph boundaries that synthesis or training really
        invokes."""
        if mode == "training":
            return (
                OptimizationCompileTarget(
                    "model.talker.forward",
                    self.model.talker,
                    "forward",
                ),
                OptimizationCompileTarget(
                    "model.talker.forward_sub_talker_finetune",
                    self.model.talker,
                    "forward_sub_talker_finetune",
                ),
            )
        if mode != "inference":
            raise ValueError(f"Unsupported optimization mode {mode!r}.")
        targets = [
            OptimizationCompileTarget(
                "model.talker.generate_codes",
                self.model.talker,
                "generate_codes",
            ),
            OptimizationCompileTarget(
                "speech_decoder.chunked_decode",
                self.speech_decoder,
                "chunked_decode",
            ),
        ]
        if self.speech_encoder is not None:
            targets.append(
                OptimizationCompileTarget(
                    "speech_encoder.forward",
                    self.speech_encoder,
                    "forward",
                ))
        return tuple(targets)

    def _speaker_embedding_from_audio(
        self,
        audio: NativeAudio,
    ) -> Tensor:
        if self.model.speaker_encoder is None:
            raise ValueError("Only Qwen3-TTS Base checkpoints expose a speaker encoder.")
        features = qwen3_tts_speaker_mel(audio.waveform.to(self.device)).to(dtype=self.model.dtype)
        return self.model.speaker_encoder(features)[0]

    def _speaker_embedding(self, reference_audio: Any) -> Tensor:
        return self._speaker_embedding_from_audio(_load_reference_audio(reference_audio))

    def _reference_codes(self, audio: NativeAudio) -> Tensor:
        if self.speech_encoder is None:
            raise NotImplementedError(
                "This Qwen3-TTS artifact does not include a native "
                "speech-tokenizer encoder.")
        parameter = next(self.speech_encoder.parameters())
        waveform = audio.waveform.to(
            device=parameter.device,
            dtype=parameter.dtype,
        ).unsqueeze(0)
        return self.speech_encoder.encode(waveform)[0]

    def _icl_prompt(
        self,
        *,
        target_ids: Tensor,
        reference_text: str,
        reference_codes: Tensor,
        tts_pad: Tensor,
        tts_eos: Tensor,
        non_streaming_mode: bool,
    ) -> tuple[Tensor, Tensor]:
        """Build the published Base-model reference-text/code alignment."""
        if not reference_text.strip():
            raise ValueError("Qwen3-TTS ICL cloning requires reference text.")
        talker = self.model.talker
        groups = self.config.talker_config.num_code_groups
        if (reference_codes.ndim != 2 or reference_codes.shape[1] != groups or reference_codes.shape[0] == 0):
            raise ValueError("Qwen3-TTS reference codes must be "
                             "[frames, codebooks].")
        if (reference_codes.dtype == torch.bool or reference_codes.is_floating_point() or
                reference_codes.is_complex()):
            raise TypeError("Qwen3-TTS reference codes must be integers.")
        reference_codes = reference_codes.to(
            device=self.device,
            dtype=torch.long,
        )
        reference_ids = self.tokenizer.encode_tensor(
            f"<|im_start|>assistant\n{reference_text}<|im_end|>\n",
            device=self.device,
        )
        text_ids = torch.cat(
            (
                reference_ids[:, 3:-2],
                target_ids[:, 3:-5],
            ),
            dim=-1,
        )
        text_embeddings = talker.text_projection(talker.get_text_embeddings()(text_ids))
        text_embeddings = torch.cat(
            (text_embeddings, tts_eos),
            dim=1,
        )

        codebook_embeddings = [talker.get_input_embeddings()(reference_codes[:, :1])]
        codebook_embeddings.extend(
            table(reference_codes[:, index:index + 1]) for index, table in enumerate(
                talker.code_predictor.get_input_embeddings(),
                start=1,
            ))
        codec_embeddings = torch.cat(
            codebook_embeddings,
            dim=1,
        ).sum(dim=1).unsqueeze(0)
        codec_bos = talker.get_input_embeddings()(
            torch.tensor(
                [[self.config.talker_config.codec_bos_id]],
                device=self.device,
                dtype=torch.long,
            ))
        codec_embeddings = torch.cat(
            (codec_bos, codec_embeddings),
            dim=1,
        )

        text_length = text_embeddings.shape[1]
        codec_length = codec_embeddings.shape[1]
        if non_streaming_mode:
            codec_padding = talker.get_input_embeddings()(
                torch.full(
                    (1, text_length),
                    self.config.talker_config.codec_pad_id,
                    device=self.device,
                    dtype=torch.long,
                ))
            prompt = torch.cat(
                (
                    text_embeddings + codec_padding,
                    codec_embeddings + tts_pad,
                ),
                dim=1,
            )
            return prompt, tts_pad
        if text_length > codec_length:
            return (
                text_embeddings[:, :codec_length] + codec_embeddings,
                text_embeddings[:, codec_length:],
            )
        padding = tts_pad.expand(
            -1,
            codec_length - text_length,
            -1,
        )
        text_embeddings = torch.cat(
            (text_embeddings, padding),
            dim=1,
        )
        return text_embeddings + codec_embeddings, tts_pad

    def _prompt(
        self,
        text: str,
        *,
        language: str,
        speaker: str | None,
        instruction: str,
        speaker_embedding: Tensor | None,
        non_streaming_mode: bool,
        reference_text: str | None = None,
        reference_codes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Qwen3-TTS text must be non-empty.")
        talker = self.model.talker
        config = self.config
        talker_config = config.talker_config
        input_ids = self.tokenizer.encode_tensor(
            _assistant_text(text),
            device=self.device,
        )
        if input_ids.shape[1] <= 8:
            raise ValueError("Qwen3-TTS assistant prompt tokenization is incomplete.")
        pieces: list[Tensor] = []
        if instruction:
            instruction_ids = self.tokenizer.encode_tensor(
                _instruction_text(instruction),
                device=self.device,
            )
            pieces.append(talker.text_projection(talker.get_text_embeddings()(instruction_ids)))

        language_name = language.strip().lower()
        if language_name == "auto":
            language_id = None
        else:
            language_id = talker_config.codec_language_id.get(language_name)
            if language_id is None:
                supported = ", ".join(sorted(talker_config.codec_language_id))
                raise ValueError(
                    f"Unsupported Qwen3-TTS language {language!r}. "
                    f"Supported: Auto, {supported}.")
        if speaker_embedding is None and speaker:
            speaker_name = speaker.strip().lower()
            if speaker_name not in talker_config.spk_id:
                supported = ", ".join(sorted(talker_config.spk_id))
                raise ValueError(f"Unsupported Qwen3-TTS speaker {speaker!r}. "
                                 f"Supported: {supported}.")
            speaker_id = talker_config.spk_id[speaker_name]
            speaker_embedding = talker.get_input_embeddings()(torch.tensor(speaker_id, device=self.device))
            dialect = talker_config.spk_is_dialect[speaker_name]
            if (language_name in {"auto", "chinese"} and isinstance(dialect, str) and dialect):
                language_id = talker_config.codec_language_id[dialect]

        controls = torch.tensor(
            [[
                config.tts_bos_token_id,
                config.tts_eos_token_id,
                config.tts_pad_token_id,
            ]],
            device=self.device,
        )
        tts_bos, tts_eos, tts_pad = talker.text_projection(talker.get_text_embeddings()(controls)).chunk(
            3, dim=1)
        codec_prefix = [
            talker_config.codec_nothink_id,
            talker_config.codec_think_bos_id,
            talker_config.codec_think_eos_id,
        ]
        if language_id is not None:
            codec_prefix = [
                talker_config.codec_think_id,
                talker_config.codec_think_bos_id,
                language_id,
                talker_config.codec_think_eos_id,
            ]
        codec_ids = codec_prefix
        codec_embeddings = talker.get_input_embeddings()(torch.tensor([codec_ids], device=self.device))
        tail = talker.get_input_embeddings()(
            torch.tensor(
                [[talker_config.codec_pad_id, talker_config.codec_bos_id]],
                device=self.device,
            ))
        if speaker_embedding is None:
            codec_embeddings = torch.cat((codec_embeddings, tail), dim=1)
        else:
            codec_embeddings = torch.cat(
                (
                    codec_embeddings,
                    speaker_embedding.reshape(1, 1, -1),
                    tail,
                ),
                dim=1,
            )
        role = talker.text_projection(talker.get_text_embeddings()(input_ids[:, :3]))
        aligned_prefix = (
            torch.cat(
                (
                    tts_pad.expand(-1, codec_embeddings.shape[1] - 2, -1),
                    tts_bos,
                ),
                dim=1,
            ) + codec_embeddings[:, :-1])
        prompt = torch.cat((role, aligned_prefix), dim=1)
        if reference_codes is not None:
            if reference_text is None:
                raise ValueError("Qwen3-TTS ICL reference codes require reference text.")
            icl_prompt, trailing = self._icl_prompt(
                target_ids=input_ids,
                reference_text=reference_text,
                reference_codes=reference_codes,
                tts_pad=tts_pad,
                tts_eos=tts_eos,
                non_streaming_mode=non_streaming_mode,
            )
            prompt = torch.cat((prompt, icl_prompt), dim=1)
            pieces.append(prompt)
            prompt = torch.cat(pieces, dim=1)
            attention_mask = torch.ones(
                prompt.shape[:2],
                device=self.device,
                dtype=torch.long,
            )
            return prompt, attention_mask, trailing
        first_text = (
            talker.text_projection(talker.get_text_embeddings()(input_ids[:, 3:4])) +
            codec_embeddings[:, -1:])
        prompt = torch.cat((prompt, first_text), dim=1)
        if non_streaming_mode:
            prompt = prompt[:, :-1]
            text_body = torch.cat(
                (
                    talker.text_projection(talker.get_text_embeddings()(input_ids[:, 3:-5])),
                    tts_eos,
                ),
                dim=1,
            )
            codec_pad = talker.get_input_embeddings()(
                torch.full(
                    (1, text_body.shape[1]),
                    talker_config.codec_pad_id,
                    device=self.device,
                    dtype=torch.long,
                ))
            codec_bos = talker.get_input_embeddings()(
                torch.tensor(
                    [[talker_config.codec_bos_id]],
                    device=self.device,
                ))
            prompt = torch.cat(
                (
                    prompt,
                    text_body + codec_pad,
                    tts_pad + codec_bos,
                ),
                dim=1,
            )
            trailing = tts_pad
        else:
            trailing = torch.cat(
                (
                    talker.text_projection(talker.get_text_embeddings()(input_ids[:, 4:-5])),
                    tts_eos,
                ),
                dim=1,
            )
        pieces.append(prompt)
        prompt = torch.cat(pieces, dim=1)
        attention_mask = torch.ones(
            prompt.shape[:2],
            device=self.device,
            dtype=torch.long,
        )
        return prompt, attention_mask, trailing

    def _generation_values(self, options: dict[str, Any]) -> dict[str, Any]:
        defaults = {
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
            "subtalker_dosample": True,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 0.9,
            "max_new_tokens": 2048,
        }
        defaults.update({name: value for name, value in self.generation_config.items() if name in defaults})
        return {name: options.pop(name, value) for name, value in defaults.items()}

    @torch.no_grad()
    def _synthesize(
        self,
        text: str,
        *,
        language: str,
        speaker: str | None,
        instruction: str,
        speaker_embedding: Tensor | None,
        non_streaming_mode: bool,
        seed: int | None,
        reference_text: str | None = None,
        reference_codes: Tensor | None = None,
        **generation_options: Any,
    ) -> tuple[list[Tensor], int]:
        options = dict(generation_options)
        values = self._generation_values(options)
        if options:
            raise ValueError("Unsupported native Qwen3-TTS generation options: " + ", ".join(sorted(options)))
        prompt, mask, trailing = self._prompt(
            text,
            language=language,
            speaker=speaker,
            instruction=instruction,
            speaker_embedding=speaker_embedding,
            non_streaming_mode=non_streaming_mode,
            reference_text=reference_text,
            reference_codes=reference_codes,
        )
        codes = self.model.talker.generate_codes(
            prompt_embeds=prompt,
            attention_mask=mask,
            trailing_text_hidden=trailing,
            seed=seed,
            **values,
        )
        if codes.shape[0] == 0:
            raise RuntimeError("Qwen3-TTS generated no speech codes.")
        decode_codes = codes
        reference_frames = 0
        if reference_codes is not None:
            reference_codes = reference_codes.to(
                device=codes.device,
                dtype=codes.dtype,
            )
            decode_codes = torch.cat(
                (reference_codes, codes),
                dim=0,
            )
            reference_frames = reference_codes.shape[0]
        waveform = self.speech_decoder.chunked_decode(decode_codes.transpose(0, 1).unsqueeze(0), )[0, 0]
        if reference_frames:
            waveform = waveform[reference_frames * self.tokenizer_config.decode_upsample_rate:]
        return [waveform], self.tokenizer_config.output_sample_rate

    def generate_custom_voice(
        self,
        *,
        text: str,
        language: str,
        speaker: str,
        instruct: str | None = None,
        non_streaming_mode: bool = True,
        seed: int | None = None,
        **generation_options: Any,
    ) -> tuple[list[Tensor], int]:
        if self.config.tts_model_type != "custom_voice":
            raise ValueError("Custom voice generation requires a CustomVoice checkpoint.")
        instruction = "" if self.config.tts_model_size == "0b6" else (instruct or "")
        return self._synthesize(
            text,
            language=language,
            speaker=speaker,
            instruction=instruction,
            speaker_embedding=None,
            non_streaming_mode=non_streaming_mode,
            seed=seed,
            **generation_options,
        )

    def generate_voice_design(
        self,
        *,
        text: str,
        language: str,
        instruct: str,
        non_streaming_mode: bool = True,
        seed: int | None = None,
        **generation_options: Any,
    ) -> tuple[list[Tensor], int]:
        if self.config.tts_model_type != "voice_design":
            raise ValueError("Voice design generation requires a VoiceDesign checkpoint.")
        return self._synthesize(
            text,
            language=language,
            speaker=None,
            instruction=instruct,
            speaker_embedding=None,
            non_streaming_mode=non_streaming_mode,
            seed=seed,
            **generation_options,
        )

    def generate_voice_clone(
        self,
        *,
        text: str,
        language: str,
        ref_audio: Any,
        ref_text: str | None,
        x_vector_only_mode: bool,
        non_streaming_mode: bool = False,
        seed: int | None = None,
        **generation_options: Any,
    ) -> tuple[list[Tensor], int]:
        if self.config.tts_model_type != "base":
            raise ValueError("Voice cloning requires a Base checkpoint.")
        audio = _load_reference_audio(ref_audio)
        speaker_embedding = self._speaker_embedding_from_audio(audio)
        reference_codes = None
        reference_text = None
        if not x_vector_only_mode:
            if ref_text is None or not ref_text.strip():
                raise ValueError("Qwen3-TTS ICL cloning requires `ref_text`.")
            reference_codes = self._reference_codes(audio)
            reference_text = ref_text
        return self._synthesize(
            text,
            language=language,
            speaker=None,
            instruction="",
            speaker_embedding=speaker_embedding,
            non_streaming_mode=non_streaming_mode,
            seed=seed,
            reference_text=reference_text,
            reference_codes=reference_codes,
            **generation_options,
        )

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        model_state_override: dict[str, Tensor] | None = None,
    ) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        speech = destination / "speech_tokenizer"
        speech.mkdir(parents=True, exist_ok=True)
        export_model = self.model
        if (model_state_override is not None and set(model_state_override) != set(self.model.state_dict())):
            export_model = Qwen3TTSForConditionalGeneration(
                self.config,
                initialize=False,
            )
        export_qwen3_tts_model(
            export_model,
            destination / "model.safetensors",
            state_override=model_state_override,
        )
        if self.speech_encoder is None:
            export_qwen3_tts_decoder(
                self.speech_decoder,
                speech / "model.safetensors",
            )
        else:
            export_qwen3_tts_speech_tokenizer(
                self.speech_encoder,
                self.speech_decoder,
                speech / "model.safetensors",
            )
        (destination / "config.json").write_text(
            json.dumps(
                self.config.to_dict(),
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        (speech / "config.json").write_text(
            json.dumps(
                self.tokenizer_config.to_dict(),
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        (destination / "generation_config.json").write_text(
            json.dumps(
                self.generation_config,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        self.processor.save_pretrained(destination)
        if self.artifacts.speech_preprocessor_config is not None:
            shutil.copy2(
                self.artifacts.speech_preprocessor_config,
                speech / "preprocessor_config.json",
            )
        (destination / "voicehub_native.json").write_text(
            json.dumps(
                {
                    "format": "voicehub-qwen3-tts-v1",
                    "source": self.artifacts.source,
                    "revision": self.artifacts.revision,
                    "speech_encoder_included": (self.speech_encoder is not None),
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        return destination.resolve()


def load_qwen3_tts_runtime(
    source: str | Path,
    *,
    device: str | torch.device = "auto",
    compute_dtype: str | torch.dtype | None = "bfloat16",
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    for_training: bool = False,
) -> NativeQwen3TTSRuntime:
    artifacts = resolve_qwen3_tts_artifacts(
        source,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config = Qwen3TTSArchitectureConfig.from_dict(read_json_file(artifacts.config))
    tokenizer_config = Qwen3TTSTokenizerConfig.from_dict(read_json_file(artifacts.speech_config))
    if (config.talker_config.num_code_groups != tokenizer_config.decoder_config.num_quantizers):
        raise ValueError("Qwen3-TTS talker and speech decoder declare different "
                         "codebook counts.")
    if config.tts_model_type == "base" and (config.speaker_encoder_config.mel_dim != 128 or
                                            config.speaker_encoder_config.sample_rate != 24_000):
        raise ValueError("Qwen3-TTS Base speaker encoders require 128-bin, 24 kHz "
                         "reference features.")
    resolved_device = _device(device)
    dtype = _dtype(compute_dtype, device=resolved_device)
    model = Qwen3TTSForConditionalGeneration(
        config,
        initialize=False,
        dtype=dtype,
    )
    load_qwen3_tts_model_checkpoint(
        model,
        artifacts.checkpoint,
        device=resolved_device,
        dtype=dtype,
        source=artifacts.source,
        revision=artifacts.revision,
    )
    verify_official_tokenizer = (
        artifacts.source in QWEN3_TTS_CHECKPOINTS and
        artifacts.revision == QWEN3_TTS_CHECKPOINTS[artifacts.source]["revision"])
    encoder = None
    encoder_report = inspect_qwen3_tts_checkpoint(
        artifacts.speech_checkpoint,
        prefix="encoder.",
    )
    if verify_official_tokenizer and tokenizer_config.encoder_config is None:
        raise ValueError("Published Qwen3-TTS tokenizer config is missing its encoder.")
    load_speech_encoder = (
        tokenizer_config.encoder_config is not None and
        bool(encoder_report.tensor_count or verify_official_tokenizer))
    if load_speech_encoder:
        assert tokenizer_config.encoder_config is not None
        encoder = Qwen3TTSSpeechEncoder(
            tokenizer_config.encoder_config,
            valid_num_quantizers=(tokenizer_config.encoder_valid_num_quantizers),
            initialize=False,
            dtype=dtype,
        )
        load_qwen3_tts_encoder_checkpoint(
            encoder,
            artifacts.speech_checkpoint,
            device=resolved_device,
            dtype=dtype,
            verify_official=verify_official_tokenizer,
        )
    decoder = Qwen3TTSSpeechDecoder(
        tokenizer_config.decoder_config,
        initialize=False,
        dtype=dtype,
    )
    load_qwen3_tts_decoder_checkpoint(
        decoder,
        artifacts.speech_checkpoint,
        device=resolved_device,
        dtype=dtype,
        verify_official=verify_official_tokenizer,
    )
    tokenizer = Qwen3TTSTextTokenizer.from_files(
        artifacts.vocab,
        artifacts.merges,
        artifacts.tokenizer_config,
    )
    generation = (
        read_json_file(artifacts.generation_config) if artifacts.generation_config is not None else {})
    runtime = NativeQwen3TTSRuntime(
        artifacts=artifacts,
        config=config,
        tokenizer_config=tokenizer_config,
        tokenizer=tokenizer,
        processor=Qwen3TTSProcessor(tokenizer),
        model=model,
        speech_encoder=encoder,
        speech_decoder=decoder,
        generation_config=dict(generation),
    )
    model._runtime_owner = runtime
    model.speech_tokenizer = _SpeechTokenizerExportFacade(runtime)
    if for_training:
        runtime.model.train()
        runtime.speech_decoder.eval()
        for parameter in runtime.speech_decoder.parameters():
            parameter.requires_grad_(False)
        if runtime.speech_encoder is not None:
            runtime.speech_encoder.eval()
            for parameter in runtime.speech_encoder.parameters():
                parameter.requires_grad_(False)
    else:
        runtime.model.eval()
        runtime.speech_decoder.eval()
        if runtime.speech_encoder is not None:
            runtime.speech_encoder.eval()
    return runtime


__all__ = [
    "NativeQwen3TTSRuntime",
    "Qwen3TTSProcessor",
    "load_qwen3_tts_runtime",
    "qwen3_tts_speaker_mel",
]
