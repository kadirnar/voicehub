"""Shared checkpoint and inference metadata for generated model documentation.

This module stays dependency-light: importing it must not import a model
backend. Every registry entry is represented explicitly so new models
fail documentation generation until their checkpoint source and public
inference boundary are audited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HUGGING_FACE_MODEL_ID = re.compile(r"^[^/\s]+/[^/\s]+$")
TASK_LABELS = {
    "text-to-speech": "Text to speech",
    "automatic-speech-recognition": "Automatic speech recognition",
    "voice-activity-detection": "Voice activity detection",
}
TASK_ORDER = tuple(TASK_LABELS)


@dataclass(frozen=True)
class CheckpointDocumentation:
    """Declarative runtime and Hugging Face checkpoint presentation."""

    identifier: str
    example: str
    provider: str
    url: str | None
    status: str
    note: str | None
    hugging_face_id: str | None
    hugging_face_url: str | None
    hugging_face_status: str

    @property
    def is_hugging_face(self) -> bool:
        """Whether the registry default itself is a Hugging Face repository."""
        return self.provider == "huggingface"

    @property
    def has_hugging_face_id(self) -> bool:
        """Whether an audited upstream Hugging Face repository is
        documented."""
        return self.hugging_face_id is not None


@dataclass(frozen=True)
class InferenceProfile:
    """Model-specific values used to author a VoiceHub inference example."""

    task: str
    summary: str
    input_note: str
    arguments: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()
    load_arguments: tuple[str, ...] = ()
    voicehub_imports: tuple[str, ...] = ()
    text: str | None = None
    high_level_supported: bool = True


_HUGGING_FACE_OVERRIDES = {
    "vui": (
        "fluxions/vui",
        "Official Vui repository, verified available on 2026-08-11; VoiceHub resolves the registered filename and pinned revision from this repo.",
    ),
    "f5tts": (
        "SWivid/F5-TTS",
        "Official F5-TTS repository, verified available on 2026-08-11; the registry alias selects the F5TTS_v1_Base files inside it.",
    ),
    "melotts": (
        "myshell-ai/MeloTTS-English",
        "Official English MeloTTS repository, verified available on 2026-08-11 and used by the registered EN release alias.",
    ),
    "styletts2": (
        "yl4579/StyleTTS2-LibriTTS",
        "Upstream LibriTTS repository, verified available on 2026-08-11. VoiceHub requires a reviewed local artifact because the published layout is not a native VoiceHub directory.",
    ),
}

_NO_HUGGING_FACE_REASON = {
    "asr_nemo": (
        "No canonical Hugging Face repository for the exact audited QuartzNet15x5 release; "
        "VoiceHub resolves the pinned NeMo/NGC artifact instead."
    ),
    "asr_wenet": (
        "No canonical Hugging Face repository for the exact audited GigaSpeech U2++ release; "
        "the page links the verified external archive and conversion boundary."
    ),
    "vad_transformers": (
        "No single repository applies: this generic adapter requires the caller to choose a "
        "compatible frame-classification checkpoint or local artifact."
    ),
    "vad_webrtc": "Not applicable: WebRTC VAD is a weightless signal-processing algorithm.",
    "vad_auditok": "Not applicable: Auditok VAD is an energy-based detector with no model weights.",
}

_EXAMPLE_OVERRIDES = {
    "styletts2": "checkpoints/styletts2/model.safetensors",
    "vad_transformers": "checkpoints/frame-vad",
}

_CHECKPOINT_STATUS_OVERRIDES = {
    "asr_nemo": (
        "Pinned NeMo/NGC QuartzNet15x5 release; VoiceHub converts the exact audited graph into its safe native artifact boundary"
    ),
    "vad_transformers": (
        "No registry default; the caller must provide a compatible reviewed frame-classification artifact"
    ),
    "vad_webrtc": "Weightless algorithm; version the implementation and configuration, not model weights",
    "vad_auditok": "Weightless energy detector; version the implementation and configuration, not model weights",
}


def checkpoint_documentation(spec) -> CheckpointDocumentation:
    """Resolve audited documentation metadata without importing a backend."""
    metadata = spec.native_architecture.metadata if spec.is_voicehub_native else {}
    inferred_provider = (
        "huggingface" if HUGGING_FACE_MODEL_ID.fullmatch(spec.default_model_path) else "local")
    provider = metadata.get("checkpoint_provider", inferred_provider)
    if provider not in {"external-archive", "huggingface", "local"}:
        raise ValueError(
            f"Unsupported checkpoint documentation provider {provider!r} "
            f"for {spec.model_type!r}.")

    identifier = spec.default_model_path
    example = metadata.get(
        "documentation_checkpoint_path",
        _EXAMPLE_OVERRIDES.get(spec.model_type, identifier or "checkpoints/model"),
    )
    url = metadata.get("reference_checkpoint_url")
    if url is None and provider == "huggingface" and identifier:
        url = f"https://huggingface.co/{identifier}"
    status = metadata.get(
        "reference_checkpoint_status",
        _CHECKPOINT_STATUS_OVERRIDES.get(
            spec.model_type,
            (
                "Registry default; pin an immutable revision for production and reproducible evidence"
                if identifier else
                "No registry default; provide the compatible local artifact described on this page"),
        ),
    )
    note = metadata.get("documentation_checkpoint_note")

    if provider == "huggingface":
        hugging_face_id = identifier
        hugging_face_status = (
            "Repository availability verified through the Hugging Face model API on 2026-08-11; "
            "pin a revision before production use."
        )
    elif spec.model_type in _HUGGING_FACE_OVERRIDES:
        hugging_face_id, hugging_face_status = _HUGGING_FACE_OVERRIDES[spec.model_type]
    else:
        hugging_face_id = None
        try:
            hugging_face_status = _NO_HUGGING_FACE_REASON[spec.model_type]
        except KeyError as error:
            raise ValueError(
                f"Model {spec.model_type!r} needs a Hugging Face ID or an explicit "
                "not-applicable reason.") from error
    hugging_face_url = (
        None if hugging_face_id is None else f"https://huggingface.co/{hugging_face_id}")

    for name, value in (
        ("example", example),
        ("provider", provider),
        ("status", status),
        ("hugging_face_status", hugging_face_status),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Checkpoint documentation {name} for {spec.model_type!r} "
                "must be a non-empty string.")
    if url is not None and (not isinstance(url, str) or not url.startswith("https://")):
        raise ValueError(f"Checkpoint documentation URL for {spec.model_type!r} must use HTTPS.")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise ValueError(
            f"Checkpoint documentation note for {spec.model_type!r} must be "
            "a non-empty string or None.")
    if hugging_face_id is not None and HUGGING_FACE_MODEL_ID.fullmatch(hugging_face_id) is None:
        raise ValueError(f"Invalid Hugging Face ID for {spec.model_type!r}: {hugging_face_id!r}.")
    return CheckpointDocumentation(
        identifier=identifier,
        example=example,
        provider=provider,
        url=url,
        status=status,
        note=note,
        hugging_face_id=hugging_face_id,
        hugging_face_url=hugging_face_url,
        hugging_face_status=hugging_face_status,
    )


REFERENCE_AUDIO_SETUP = (
    'REFERENCE_AUDIO = Path("reference.wav")',
    'REFERENCE_TEXT = "The reference transcript must exactly match the authorized audio."',
    "if not REFERENCE_AUDIO.is_file():",
    "    raise FileNotFoundError(REFERENCE_AUDIO)",
)
SPEAKER_EMBEDDING_SETUP = (
    'SPEAKER_EMBEDDING_FILE = Path("speaker_embedding.json")',
    "SPEAKER_EMBEDDING = json.loads(SPEAKER_EMBEDDING_FILE.read_text(encoding=\"utf-8\"))",
    "if len(SPEAKER_EMBEDDING) != 192:",
    '    raise ValueError("CosyVoice expects exactly 192 speaker-embedding values.")',
)


def _tts(
    summary: str,
    input_note: str,
    *,
    arguments: tuple[str, ...] = (),
    setup: tuple[str, ...] = (),
    load_arguments: tuple[str, ...] = (),
    voicehub_imports: tuple[str, ...] = (),
    text: str = "VoiceHub keeps model integrations explicit and reproducible.",
    high_level_supported: bool = True,
) -> InferenceProfile:
    return InferenceProfile(
        task="text-to-speech",
        summary=summary,
        input_note=input_note,
        arguments=arguments,
        setup=setup,
        load_arguments=load_arguments,
        voicehub_imports=voicehub_imports,
        text=text,
        high_level_supported=high_level_supported,
    )


def _asr(
    summary: str,
    input_note: str,
    *,
    arguments: tuple[str, ...] = (),
) -> InferenceProfile:
    return InferenceProfile(
        task="automatic-speech-recognition",
        summary=summary,
        input_note=input_note,
        arguments=arguments,
    )


def _vad(
    summary: str,
    input_note: str,
    *,
    arguments: tuple[str, ...] = (),
) -> InferenceProfile:
    return InferenceProfile(
        task="voice-activity-detection",
        summary=summary,
        input_note=input_note,
        arguments=arguments,
    )


INFERENCE_PROFILES = {
    "orpheustts": _tts(
        "Selects a released Orpheus voice and bounds autoregressive audio-token generation.",
        "`voice` is required by the VoiceHub wrapper; change it only to a voice present in the selected checkpoint.",
        arguments=('voice="tara"', "max_new_tokens=2_048", "temperature=0.7"),
    ),
    "dia": _tts(
        "Uses Dia speaker tags in the text instead of an unrelated generic single-speaker prompt.",
        "Keep `[S1]`/`[S2]` turns in the text when generating dialogue.",
        text="[S1] VoiceHub keeps the dialogue contract explicit. [S2] That makes review easier.",
    ),
    "vui": _tts(
        "Uses Vui's bounded chunk retry and duration controls with the registered pinned artifacts.",
        "The short registry alias resolves the model and Fluac codec from one immutable repository snapshot.",
        arguments=("max_secs=20", "max_chunk_retries=3", "temperature=0.5", "top_k=100"),
    ),
    "chatterbox": _tts(
        "Demonstrates Chatterbox voice prompting through VoiceHub's normalized reference-audio field.",
        "Use only a reference recording you are authorized to process; omit the argument for the checkpoint's default voice.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=("speaker_audio_path=str(REFERENCE_AUDIO)", "max_new_tokens=1_024"),
    ),
    "kokoro": _tts(
        "Selects a Kokoro voice ID and explicit speaking speed.",
        "Voice IDs are checkpoint-specific; `af_heart` belongs to the registered Kokoro release.",
        arguments=('voice="af_heart"', "speed=1.0"),
    ),
    "echo": _tts(
        "Exposes Echo's flow-matching step count and separate text/speaker guidance scales.",
        "Increase `num_steps` only after measuring latency and audio quality on the target device.",
        arguments=("num_steps=20", "cfg_scale_text=3.0", "cfg_scale_speaker=1.0"),
    ),
    "conversationtts": _tts(
        "Assigns an explicit conversation speaker and caps the generated audio duration.",
        "Use stable integer speaker IDs when building multi-turn context.",
        arguments=("speaker=0", "max_audio_length_ms=30_000", "temperature=0.9", "top_k=50"),
    ),
    "llasa": _tts(
        "Pairs LLaSA reference audio with its exact transcript for voice cloning.",
        "Both reference fields are required together; VoiceHub rejects incomplete cloning context.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            "reference_text=REFERENCE_TEXT",
            "max_new_tokens=1_024",
            "top_p=0.9",
        ),
    ),
    "cosyvoice": _tts(
        "Loads the required 192-value speaker embedding from a reviewable JSON file.",
        "The native boundary intentionally does not run an unverified speaker encoder behind the caller's back.",
        setup=SPEAKER_EMBEDDING_SETUP,
        arguments=("speaker_embedding=SPEAKER_EMBEDDING", "instruction=\"Speak clearly.\"", "flow_steps=10"),
    ),
    "f5tts": _tts(
        "Supplies F5-TTS with the mandatory reference waveform and matching transcript.",
        "The transcript must match the reference audio exactly or alignment quality will degrade.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            "reference_text=REFERENCE_TEXT",
            "nfe_steps=32",
            "cfg_strength=2.0",
        ),
    ),
    "gptsovits": _tts(
        "Defines both target and prompt languages for GPT-SoVITS zero-shot voice prompting.",
        "Use the language codes accepted by the selected GPT-SoVITS checkpoint and an exact prompt transcript.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            'text_language="en"',
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            'prompt_language="en"',
            "prompt_text=REFERENCE_TEXT",
            'text_split_method="cut5"',
        ),
    ),
    "melotts": _tts(
        "Opts into the pinned legacy MeloTTS release explicitly and selects its English speaker table.",
        "The official release is a reviewed pickle checkpoint; keep `trust_pickle_checkpoint` false for arbitrary files.",
        arguments=('speaker="EN-US"', "speed=1.0"),
        load_arguments=("config=AutoConfig.for_model(\"melotts\", trust_pickle_checkpoint=True)",),
        voicehub_imports=("AutoConfig",),
    ),
    "openvoice": _tts(
        "Runs OpenVoice tone-color transfer from a source utterance to an authorized target-speaker recording.",
        "`base.wav` must contain the utterance to convert; `reference.wav` supplies only the target tone color.",
        setup=(
            'BASE_AUDIO = Path("base.wav")',
            'REFERENCE_AUDIO = Path("reference.wav")',
            "for audio_file in (BASE_AUDIO, REFERENCE_AUDIO):",
            "    if not audio_file.is_file():",
            "        raise FileNotFoundError(audio_file)",
        ),
        arguments=(
            "base_audio=str(BASE_AUDIO)",
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            "tau=0.3",
        ),
    ),
    "outetts": _tts(
        "Uses the audited OuteTTS V3 regular generation path with an explicit token limit.",
        "Speaker profiles are optional; add one only if it matches the selected V3 checkpoint protocol.",
        arguments=('generation_type="regular"', "max_length=1_024"),
    ),
    "parlertts": _tts(
        "Separates the spoken text from Parler-TTS's acoustic style description.",
        "Describe voice, pace, and recording conditions in `description`, not in the text to be spoken.",
        arguments=(
            'description="A clear, close-mic voice at a steady pace with very little background noise"',
        ),
    ),
    "styletts2": _tts(
        "Uses an explicit local VoiceHub artifact and the native phoneme boundary required by StyleTTS 2.",
        "Convert or review the upstream LibriTTS files first; the HF repository is provenance, not a drop-in VoiceHub directory.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            "text_is_phonemes=True",
            "diffusion_steps=5",
            "embedding_scale=1.0",
        ),
        text="həˈloʊ fɹʌm vɔɪs hʌb",
    ),
    "mosstts": _tts(
        "Combines MOSS-TTS language, instruction, and quality controls without importing upstream demo code.",
        "Keep instructions descriptive and validate the requested language against the selected checkpoint.",
        arguments=('language="en"', 'instruction="Calm, clear studio speech"', 'quality="high"'),
    ),
    "qwen3tts": _tts(
        "Uses the registered Qwen3-TTS CustomVoice role with an explicit language and speaker.",
        "Checkpoint roles are not interchangeable; voice cloning requires a Base checkpoint and paired reference fields.",
        arguments=('mode="custom_voice"', 'language="English"', 'speaker="Vivian"'),
    ),
    "irodoritts": _tts(
        "Exercises Irodori-TTS's explicit no-reference path and flow sampler controls.",
        "Set a speaker reference instead of `no_reference=True` when cloning an authorized voice.",
        arguments=("no_reference=True", "seconds=4.0", "num_steps=16", "cfg_scale_text=3.0"),
    ),
    "zonos": _tts(
        "Conditions Zonos on an eSpeak language code and an authorized speaker reference.",
        "Tune emotion and sampling only after establishing a deterministic seeded baseline.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            'language="en-us"',
            "cfg_scale=2.0",
            "max_new_tokens=2_048",
        ),
    ),
    "zonos2": _tts(
        "Uses ZONOS2's language, speed, accurate-mode, and speaker-conditioning controls.",
        "Do not pass both a speaker waveform and a precomputed speaker embedding.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            'language="en"',
            "speed=1.0",
            "accurate_mode=True",
        ),
    ),
    "voxcpm": _tts(
        "Conditions VoxCPM2 on a reference timbre and exposes its diffusion guidance and step count.",
        "A prompt transcript is required only with `prompt_audio_path`; the timbre-only field used here is separate.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            "cfg_value=2.0",
            "inference_timesteps=10",
        ),
    ),
    "omnivoice": _tts(
        "Pairs OmniVoice speaker audio with its transcript and selects the native iterative decoder controls.",
        "Voice cloning requires both reference fields; external text normalization stays outside the model boundary.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            'language="en"',
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            "reference_text=REFERENCE_TEXT",
            "num_steps=8",
            "guidance_scale=2.0",
        ),
    ),
    "higgstts": _tts(
        "Provides Higgs Audio with paired reference context and a bounded semantic-token budget.",
        "The reference transcript must correspond to the authorized speaker audio.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            "reference_text=REFERENCE_TEXT",
            'system_prompt="Generate natural, clean speech."',
            "max_new_tokens=1_024",
        ),
    ),
    "xtts": _tts(
        "Supplies the mandatory XTTS v2 speaker reference and a supported language code.",
        "XTTS rejects missing reference files and unsupported checkpoint language codes before synthesis.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=("speaker_audio_path=str(REFERENCE_AUDIO)", 'language="en"', "speed=1.0"),
    ),
    "vibevoice": _tts(
        "Loads the audited VibeVoice realtime stages without claiming an unverified text-to-waveform loop.",
        "High-level cached-prompt synthesis intentionally fails closed until cache serialization, chunk boundaries, and waveform parity are verified.",
        high_level_supported=False,
    ),
    "fishtts": _tts(
        "Pairs Fish S2 reference audio and text while keeping semantic sampling bounded.",
        "Use either reference audio or precomputed codes, never both; each requires a matching transcript.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            "reference_text=REFERENCE_TEXT",
            "top_p=0.8",
            "temperature=0.8",
            "iterative_prompt=True",
        ),
    ),
    "csm": _tts(
        "Builds CSM speaker context from a stable speaker index and paired reference recording.",
        "Reference audio and text must be supplied together; speaker IDs must be non-negative.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            "speaker=0",
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            "reference_text=REFERENCE_TEXT",
            "max_audio_length_ms=30_000",
        ),
    ),
    "neutts": _tts(
        "Uses NeuTTS's required one-of speaker source with the matching reference transcript.",
        "Exactly one of reference audio or reference codes is mandatory for this checkpoint.",
        setup=REFERENCE_AUDIO_SETUP,
        arguments=(
            "speaker_audio_path=str(REFERENCE_AUDIO)",
            "reference_text=REFERENCE_TEXT",
            "temperature=0.8",
            "top_k=50",
        ),
    ),
    "supertonic": _tts(
        "Selects a Supertonic style ID, language, diffusion-step count, and speaking speed.",
        "Voice/style IDs and languages are validated against files published by the checkpoint.",
        arguments=('voice="F1"', 'language="en"', "total_steps=5", "speed=1.05"),
    ),
    "inflecttts": _tts(
        "Uses Inflect's normalized-text frontend with explicit speed and variation controls.",
        "Set `input_is_phonemes=True` only when supplying checkpoint-compatible phoneme text.",
        arguments=("speed=1.0", "variation=0.3"),
    ),
    "bark": _tts(
        "Selects a Bark history prompt and bounds semantic token sampling.",
        "History-prompt names are checkpoint assets and can encode voice plus acoustic context.",
        arguments=('voice_preset="v2/en_speaker_6"', "temperature=0.7", "max_new_tokens=768"),
    ),
    "speecht5": _tts(
        "Passes a reviewed speaker-embedding file through SpeechT5's safe public loader.",
        "Use Safetensors or NPY for embeddings; omit the field to use the wrapper's neutral zero embedding.",
        setup=(
            'SPEAKER_EMBEDDING = Path("speaker_embedding.npy")',
            "if not SPEAKER_EMBEDDING.is_file():",
            "    raise FileNotFoundError(SPEAKER_EMBEDDING)",
        ),
        arguments=("speaker_embedding_path=SPEAKER_EMBEDDING", "threshold=0.5"),
    ),
    "vits": _tts(
        "Controls MMS-VITS speaking rate, stochastic duration, and output-frame guardrails.",
        "The registered English checkpoint is single-speaker; choose a different HF ID for another MMS language.",
        arguments=("speaking_rate=1.0", "noise_scale=0.667", "max_output_frames=240_000"),
    ),
    "asr_transformers": _asr(
        "Runs the generic Transformers ASR adapter with Whisper language/task controls and timestamps.",
        "Use this adapter only for checkpoints compatible with the Transformers speech-recognition pipeline.",
        arguments=('language="en"', 'task="transcribe"', "return_timestamps=True", "chunk_length_s=30.0"),
    ),
    "asr_whisper": _asr(
        "Uses VoiceHub's native Whisper graph with explicit transcription language and word timestamps.",
        "Set `language=None` for model-side detection; keep the task as transcription unless translation is intended.",
        arguments=('language="en"', 'task="transcribe"', 'return_timestamps="word"', "num_beams=5"),
    ),
    "asr_tiron": _asr(
        "Enables Tiron constrained decoding and caps diarized speakers for a meeting recording.",
        "Choose `max_speakers` from the recording context rather than treating diarization as unbounded.",
        arguments=("return_timestamps=True", "max_speakers=4", "constrained_decoding=True"),
    ),
    "asr_qwen3": _asr(
        "Provides Qwen3-ASR with a domain prompt and deterministic decoding controls.",
        "Prompts should contain context, not a fabricated transcript of the input audio.",
        arguments=('language="en"', 'prompt="Technical meeting with VoiceHub terminology"', "do_sample=False"),
    ),
    "asr_vibevoice": _asr(
        "Requests VibeVoice-ASR timestamps with a concise transcription prompt.",
        "Keep the prompt task-focused and verify timestamp granularity for the selected checkpoint revision.",
        arguments=("return_timestamps=True", 'prompt="Transcribe every spoken turn."'),
    ),
    "asr_granite_speech": _asr(
        "Uses Granite Speech's instruction prompt with deterministic generation.",
        "Medical or regulated recordings still require domain review; model output is not a verified record.",
        arguments=('prompt="Transcribe the recording faithfully in English."', "do_sample=False"),
    ),
    "asr_parakeet_tdt": _asr(
        "Runs the native Parakeet TDT decoder and returns its calibrated timestamp segments.",
        "The registered multilingual release accepts automatic language handling; inspect the returned language metadata.",
        arguments=("return_timestamps=True",),
    ),
    "asr_nemotron": _asr(
        "Uses Nemotron's cache-aware native decoder and requests word timestamps.",
        "Chunk geometry is owned by the checkpoint runtime; common chunk and stride overrides intentionally fail closed.",
        arguments=('return_timestamps="word"',),
    ),
    "asr_cohere": _asr(
        "Requests Cohere Transcribe punctuation with an explicit language and bounded decoding.",
        "The native checkpoint has no verified timestamp decoder; retain the source audio with the transcript.",
        arguments=('language="en"', "punctuation=True", "max_new_tokens=256"),
    ),
    "asr_medasr": _asr(
        "Selects the audited English MedASR decoding path without pretending it is a clinical decision system.",
        "Treat the transcript as draft output and review protected or clinical recordings under the applicable policy.",
        arguments=('language="en"',),
    ),
    "asr_wav2vec2": _asr(
        "Runs the native Wav2Vec2 CTC path and requests word-level alignment where supported.",
        "CTC decoding is checkpoint-vocabulary specific; do not reuse this ID for arbitrary languages.",
        arguments=('language="en"', 'return_timestamps="word"'),
    ),
    "asr_hubert": _asr(
        "Uses the HuBERT CTC fine-tuned head with an explicit English transcription task.",
        "The base HuBERT family is self-supervised; this exact HF ID includes the ASR head required here.",
        arguments=('language="en"', 'task="transcribe"'),
    ),
    "asr_wavlm": _asr(
        "Runs the registered WavLM CTC checkpoint with greedy single-beam decoding.",
        "This community LibriSpeech checkpoint is English-specific even though the WavLM architecture is general.",
        arguments=('language="en"', "num_beams=1"),
    ),
    "asr_moonshine": _asr(
        "Uses Moonshine's short-form speech path with deterministic decoding.",
        "Split very long recordings deliberately instead of assuming short-form checkpoint behavior will scale unchanged.",
        arguments=('language="en"', "num_beams=1"),
    ),
    "asr_seamless_m4t_v2": _asr(
        "Selects SeamlessM4T v2 transcription rather than speech translation.",
        "The native complete-waveform path is greedy and does not claim timestamp alignment.",
        arguments=('task="transcribe"', "num_beams=1", "max_new_tokens=256"),
    ),
    "asr_faster_whisper": _asr(
        "Uses the faster-whisper backend with language selection, word timestamps, and a bounded beam.",
        "Benchmark the converted runtime on the deployment device; results depend on compute type and batching.",
        arguments=('language="en"', 'return_timestamps="word"', "num_beams=5"),
    ),
    "asr_whisperx": _asr(
        "Requests WhisperX alignment timestamps through VoiceHub's normalized ASR output.",
        "Alignment can require a language-specific auxiliary model; verify access and timing quality separately.",
        arguments=('language="en"', 'return_timestamps="word"'),
    ),
    "asr_openai_whisper": _asr(
        "Runs the original OpenAI Whisper backend with deterministic beam decoding.",
        "This integration is distinct from native Whisper and faster-whisper even when they share an HF checkpoint ID.",
        arguments=('language="en"', 'task="transcribe"', "num_beams=5"),
    ),
    "asr_nemo": _asr(
        "Runs VoiceHub's native QuartzNet15x5 graph from the pinned NeMo/NGC source.",
        "The audited release is English-only and supports CTC word timestamps, not arbitrary NeMo architectures.",
        arguments=('language="en"', 'return_timestamps="word"'),
    ),
    "asr_speechbrain": _asr(
        "Uses the audited SpeechBrain CRDNN/RNNLM decoder with an explicit beam size.",
        "The released LibriSpeech graph is English-only and does not expose calibrated timestamps.",
        arguments=('language="en"', "num_beams=8"),
    ),
    "asr_funasr": _asr(
        "Runs SenseVoiceSmall's native SANM-CTC graph with language detection and word timestamps.",
        "This provider recognizes SenseVoiceSmall only; VAD, punctuation, and speaker models must be composed separately.",
        arguments=('language="auto"', 'return_timestamps="word"'),
    ),
    "asr_espnet": _asr(
        "Uses the audited ESPnet LibriSpeech transformer with an explicit beam size.",
        "This release is English-only and has no calibrated timestamp head.",
        arguments=('language="en"', "num_beams=10"),
    ),
    "asr_wenet": _asr(
        "Loads a reviewed VoiceHub conversion of WeNet GigaSpeech U2++ and requests word timestamps.",
        "The external release is not a drop-in HF model; convert it through the audited artifact boundary first.",
        arguments=('language="en"', 'return_timestamps="word"', "num_beams=10"),
    ),
    "vad_transformers": _vad(
        "Runs a caller-selected Transformers frame classifier with explicit hysteresis and frame output.",
        "The local directory must be compatible with VoiceHub's generic frame-classification adapter.",
        arguments=("onset=0.6", "offset=0.4", "return_frames=True"),
    ),
    "vad_silero": _vad(
        "Uses Silero's probability threshold plus minimum speech/silence duration smoothing.",
        "Calibrate thresholds on labeled audio from the actual microphone and noise conditions.",
        arguments=("threshold=0.5", "min_speech_duration_ms=250", "min_silence_duration_ms=200"),
    ),
    "vad_webrtc": _vad(
        "Runs weightless WebRTC VAD with frame-compatible duration controls.",
        "Input is resampled and framed by VoiceHub; algorithm aggressiveness belongs to the model configuration.",
        arguments=("min_speech_duration_ms=120", "min_silence_duration_ms=240", "speech_pad_ms=30"),
    ),
    "vad_pyannote": _vad(
        "Applies pyannote VAD with separate onset and offset thresholds.",
        "The repository can be gated; authenticate through the normal Hugging Face token flow before loading.",
        arguments=("onset=0.55", "offset=0.45", "min_speech_duration_ms=100"),
    ),
    "vad_speechbrain": _vad(
        "Uses SpeechBrain CRDNN probabilities with explicit hysteresis and silence merging.",
        "Tune onset and offset jointly; independent threshold changes can fragment segments.",
        arguments=("onset=0.6", "offset=0.4", "min_silence_duration_ms=250"),
    ),
    "vad_nemo": _vad(
        "Runs multilingual MarbleNet frame VAD and retains frame scores for inspection.",
        "Frame scores help calibration but are not speaker labels or ASR confidence values.",
        arguments=("threshold=0.5", "min_silence_duration_ms=200", "return_frames=True"),
    ),
    "vad_funasr": _vad(
        "Uses the FSMN endpoint model with speech padding and maximum-segment limits.",
        "Endpoint behavior is model-specific; calibrate padding before composing it with ASR chunks.",
        arguments=("threshold=0.5", "speech_pad_ms=200", "max_speech_duration_s=30.0"),
    ),
    "vad_auditok": _vad(
        "Runs Auditok's weightless energy detector with conservative speech/silence durations.",
        "Energy thresholds are recording-level heuristics and must be recalibrated after gain changes.",
        arguments=("min_speech_duration_ms=200", "min_silence_duration_ms=300", "speech_pad_ms=20"),
    ),
    "vad_sherpa_onnx": _vad(
        "Uses sherpa-onnx streaming Silero state with an explicit threshold and segment padding.",
        "Keep streaming state per audio stream; do not share one detector instance across unrelated calls.",
        arguments=("threshold=0.5", "speech_pad_ms=100", "max_speech_duration_s=30.0"),
    ),
    "vad_pyannote_segmentation": _vad(
        "Runs pyannote segmentation 3.0 as VAD with hysteresis and frame evidence enabled.",
        "Segmentation scores are converted to speech regions; they are not diarization labels by themselves.",
        arguments=("onset=0.6", "offset=0.4", "return_frames=True"),
    ),
    "vad_pyannote_brouhaha": _vad(
        "Uses Brouhaha speech scores while returning frames for downstream SNR or quality review.",
        "Voice activity output does not expose Brouhaha's other heads as normalized VAD segments.",
        arguments=("threshold=0.5", "min_speech_duration_ms=100", "return_frames=True"),
    ),
}


def inference_profile(spec) -> InferenceProfile:
    """Return and validate the explicit inference profile for one registry
    entry."""
    try:
        profile = INFERENCE_PROFILES[spec.model_type]
    except KeyError as error:
        raise ValueError(f"Model {spec.model_type!r} needs an explicit inference profile.") from error
    if profile.task != spec.task.value:
        raise ValueError(
            f"Inference profile task mismatch for {spec.model_type!r}: "
            f"{profile.task!r} != {spec.task.value!r}.")
    return profile


__all__ = [
    "CheckpointDocumentation",
    "INFERENCE_PROFILES",
    "InferenceProfile",
    "TASK_LABELS",
    "TASK_ORDER",
    "checkpoint_documentation",
    "inference_profile",
]
