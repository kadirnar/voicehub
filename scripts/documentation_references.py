"""Verified research and source references used by generated documentation.

Keep this file explicit. A missing paper is better than an unrelated citation,
and exact key coverage is checked against the model and optimization registries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Reference:
    """One primary research or source link."""

    title: str
    url: str


@dataclass(frozen=True, slots=True)
class ModelReferences:
    """Primary upstream repository and any dedicated research papers."""

    github: Reference
    papers: tuple[Reference, ...] = ()


@dataclass(frozen=True, slots=True)
class OptimizationGuide:
    """Documentation contract for one pass or optional source backend."""

    slug: str
    title: str
    group: str
    summary: str
    availability: str
    fidelity: str
    devices: str
    usage: str
    github: tuple[Reference, ...]
    papers: tuple[Reference, ...]
    implementation: str | None = None
    registry_name: str | None = None
    pass_id: str | None = None
    pass_version: str | None = None
    related_guide: str = "../guides/optimization-overview.md"
    source_install: str | None = None


def _github(title: str, repository: str) -> Reference:
    return Reference(title, f"https://github.com/{repository}")


def _paper(title: str, identifier: str) -> Reference:
    return Reference(title, f"https://arxiv.org/abs/{identifier}")


WHISPER_PAPER = _paper("Robust Speech Recognition via Large-Scale Weak Supervision", "2212.04356")
PYANNOTE_PAPER = _paper("pyannote.audio: neural building blocks for speaker diarization", "1911.01255")
SPEECHBRAIN_PAPER = _paper("SpeechBrain: A General-Purpose Speech Toolkit", "2106.04624")
FUNASR_PAPER = _paper("FunASR: A Fundamental End-to-End Speech Recognition Toolkit", "2305.11013")

# Keep the URL-heavy declarative catalog stable and readable. Automated YAPF
# wrapping makes these records harder to review and can produce Flake8 E251.
# yapf: disable

MODEL_REFERENCES = {
    # Text to speech.
    "orpheustts": ModelReferences(_github("Orpheus-TTS", "canopyai/Orpheus-TTS")),
    "dia": ModelReferences(_github("Dia", "nari-labs/dia")),
    "vui": ModelReferences(_github("Vui", "fluxions-ai/vui")),
    "chatterbox": ModelReferences(_github("Chatterbox", "resemble-ai/chatterbox")),
    "kokoro": ModelReferences(_github("Kokoro", "hexgrad/kokoro")),
    "echo": ModelReferences(_github("Echo-TTS", "jordandare/echo-tts")),
    "conversationtts": ModelReferences(
        _github("ConversationTTS", "Audio-Foundation-Models/ConversationTTS")),
    "llasa": ModelReferences(_github("LLaSA training", "zhenye234/LLaSA_training")),
    "cosyvoice": ModelReferences(
        _github("CosyVoice", "FunAudioLLM/CosyVoice"),
        (_paper("CosyVoice: Multi-Lingual Large Voice Generation Model", "2407.05407"), ),
    ),
    "f5tts": ModelReferences(
        _github("F5-TTS", "SWivid/F5-TTS"),
        (_paper("F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech", "2410.06885"), ),
    ),
    "gptsovits": ModelReferences(_github("GPT-SoVITS", "RVC-Boss/GPT-SoVITS")),
    "melotts": ModelReferences(_github("MeloTTS", "myshell-ai/MeloTTS")),
    "openvoice": ModelReferences(
        _github("OpenVoice", "myshell-ai/OpenVoice"),
        (_paper("OpenVoice: Versatile Instant Voice Cloning", "2312.01479"), ),
    ),
    "outetts": ModelReferences(_github("OuteTTS", "edwko/OuteTTS")),
    "parlertts": ModelReferences(
        _github("Parler-TTS", "huggingface/parler-tts"),
        (_paper("Parler-TTS: A Text-to-Speech Dataset and Model Controlled by Natural Language", "2402.01912"), ),
    ),
    "styletts2": ModelReferences(
        _github("StyleTTS 2", "yl4579/StyleTTS2"),
        (_paper("StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion", "2306.07691"), ),
    ),
    "mosstts": ModelReferences(_github("MOSS-TTS", "OpenMOSS/MOSS-TTS")),
    "qwen3tts": ModelReferences(
        _github("Qwen3-TTS", "QwenLM/Qwen3-TTS"),
        (_paper("Qwen3-TTS Technical Report", "2601.15621"), ),
    ),
    "irodoritts": ModelReferences(_github("Irodori-TTS", "Aratako/Irodori-TTS")),
    "zonos": ModelReferences(_github("Zonos", "Zyphra/Zonos")),
    "zonos2": ModelReferences(_github("ZONOS2", "Zyphra/ZONOS2")),
    "voxcpm": ModelReferences(_github("VoxCPM", "OpenBMB/VoxCPM")),
    "omnivoice": ModelReferences(_github("OmniVoice", "k2-fsa/OmniVoice")),
    "higgstts": ModelReferences(_github("Higgs Audio", "boson-ai/higgs-audio")),
    "xtts": ModelReferences(_github("Coqui TTS", "coqui-ai/TTS")),
    "vibevoice": ModelReferences(
        _github("VibeVoice", "microsoft/VibeVoice"),
        (_paper("VibeVoice Technical Report", "2508.19205"), ),
    ),
    "fishtts": ModelReferences(
        _github("Fish Speech", "fishaudio/fish-speech"),
        (_paper("Fish-Speech: Leveraging Large Language Models for Advanced Multilingual TTS", "2411.01156"), ),
    ),
    "csm": ModelReferences(_github("CSM", "SesameAILabs/csm")),
    "neutts": ModelReferences(_github("NeuTTS", "neuphonic/neutts")),
    "supertonic": ModelReferences(_github("Supertonic", "supertone-inc/supertonic")),
    "inflecttts": ModelReferences(_github("Inflect", "owenawsong/Inflect")),
    "bark": ModelReferences(_github("Bark", "suno-ai/bark")),
    "speecht5": ModelReferences(
        _github("SpeechT5", "microsoft/SpeechT5"),
        (_paper("SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing", "2110.07205"), ),
    ),
    "vits": ModelReferences(
        _github("VITS", "jaywalnut310/vits"),
        (_paper("Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS", "2106.06103"), ),
    ),
    # Automatic speech recognition.
    "asr_transformers": ModelReferences(_github("Transformers", "huggingface/transformers")),
    "asr_whisper": ModelReferences(_github("Whisper", "openai/whisper"), (WHISPER_PAPER, )),
    "asr_tiron": ModelReferences(_github("Tiron", "TrelisResearch/tiron")),
    "asr_qwen3": ModelReferences(_github("Qwen3-ASR", "QwenLM/Qwen3-ASR")),
    "asr_vibevoice": ModelReferences(_github("VibeVoice", "microsoft/VibeVoice")),
    "asr_granite_speech": ModelReferences(
        _github("Granite Speech models", "ibm-granite/granite-speech-models")),
    "asr_parakeet_tdt": ModelReferences(_github("NVIDIA NeMo", "NVIDIA/NeMo")),
    "asr_nemotron": ModelReferences(_github("NVIDIA NeMo", "NVIDIA/NeMo")),
    "asr_cohere": ModelReferences(_github("Transformers", "huggingface/transformers")),
    "asr_medasr": ModelReferences(_github("MedASR", "google-health/medasr")),
    "asr_wav2vec2": ModelReferences(
        _github("fairseq", "facebookresearch/fairseq"),
        (_paper("wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", "2006.11477"), ),
    ),
    "asr_hubert": ModelReferences(
        _github("fairseq", "facebookresearch/fairseq"),
        (_paper("HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction", "2106.07447"), ),
    ),
    "asr_wavlm": ModelReferences(
        _github("UniLM / WavLM", "microsoft/unilm"),
        (_paper("WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", "2110.13900"), ),
    ),
    "asr_moonshine": ModelReferences(
        _github("Moonshine", "moonshine-ai/moonshine"),
        (_paper("Moonshine: Speech Recognition for Live Transcription and Voice Commands", "2410.15608"), ),
    ),
    "asr_seamless_m4t_v2": ModelReferences(
        _github("Seamless Communication", "facebookresearch/seamless_communication"),
        (_paper("Seamless: Multilingual Expressive and Streaming Speech Translation", "2312.05187"), ),
    ),
    "asr_faster_whisper": ModelReferences(
        _github("faster-whisper", "SYSTRAN/faster-whisper"), (WHISPER_PAPER, )),
    "asr_whisperx": ModelReferences(
        _github("WhisperX", "m-bain/whisperX"),
        (_paper("WhisperX: Time-Accurate Speech Transcription of Long-Form Audio", "2303.00747"),
         WHISPER_PAPER),
    ),
    "asr_openai_whisper": ModelReferences(
        _github("Whisper", "openai/whisper"), (WHISPER_PAPER, )),
    "asr_nemo": ModelReferences(
        _github("NVIDIA NeMo", "NVIDIA/NeMo"),
        (_paper("NeMo: a toolkit for building AI applications using Neural Modules", "1909.09577"), ),
    ),
    "asr_speechbrain": ModelReferences(
        _github("SpeechBrain", "speechbrain/speechbrain"), (SPEECHBRAIN_PAPER, )),
    "asr_funasr": ModelReferences(_github("FunASR", "modelscope/FunASR"), (FUNASR_PAPER, )),
    "asr_espnet": ModelReferences(
        _github("ESPnet", "espnet/espnet"),
        (_paper("ESPnet: End-to-End Speech Processing Toolkit", "1804.00015"), ),
    ),
    "asr_wenet": ModelReferences(
        _github("WeNet", "wenet-e2e/wenet"),
        (_paper("WeNet: Production Oriented Streaming and Non-Streaming End-to-End Speech Recognition Toolkit", "2102.01547"), ),
    ),
    # Voice activity detection.
    "vad_transformers": ModelReferences(_github("Transformers", "huggingface/transformers")),
    "vad_silero": ModelReferences(_github("Silero VAD", "snakers4/silero-vad")),
    "vad_webrtc": ModelReferences(_github("py-webrtcvad", "wiseman/py-webrtcvad")),
    "vad_pyannote": ModelReferences(
        _github("pyannote.audio", "pyannote/pyannote-audio"), (PYANNOTE_PAPER, )),
    "vad_speechbrain": ModelReferences(
        _github("SpeechBrain", "speechbrain/speechbrain"), (SPEECHBRAIN_PAPER, )),
    "vad_nemo": ModelReferences(
        _github("NVIDIA NeMo", "NVIDIA/NeMo"),
        (_paper("MarbleNet: Deep 1D Time-Channel Separable Convolutional Neural Network for VAD", "2010.13886"), ),
    ),
    "vad_funasr": ModelReferences(_github("FunASR", "modelscope/FunASR"), (FUNASR_PAPER, )),
    "vad_auditok": ModelReferences(_github("auditok", "amsehili/auditok")),
    "vad_sherpa_onnx": ModelReferences(_github("sherpa-onnx", "k2-fsa/sherpa-onnx")),
    "vad_pyannote_segmentation": ModelReferences(
        _github("pyannote.audio", "pyannote/pyannote-audio"), (PYANNOTE_PAPER, )),
    "vad_pyannote_brouhaha": ModelReferences(
        _github("Brouhaha VAD", "marianne-m/brouhaha-vad")),
}


OPTIMIZATION_GUIDES = (
    OptimizationGuide(
        slug="compile",
        title="Torch compile",
        group="Optimization passes",
        summary="Compile model-owned execution methods while preserving checkpoint keys and reversible eager fallbacks.",
        availability="Registered public pass: `compile`",
        fidelity="Exact intent; verify numerical and audio equivalence for the concrete graph",
        devices="CPU or CUDA; float32, float16, or bfloat16",
        usage='''print(model.available_optimization_passes())
result = model.apply_optimization_plan("compile", mode="inference")
print(result.manifest())
model.restore_optimization_plan(mode="inference")''',
        github=(_github("PyTorch", "pytorch/pytorch"), ),
        papers=(Reference("PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation", "https://pytorch.org/assets/pytorch2-2.pdf"), ),
        implementation="voicehub/optimization/torch_compile.py",
        registry_name="compile",
        pass_id="torch.compile",
        pass_version="1",
    ),
    OptimizationGuide(
        slug="flash-attention-4",
        title="FlashAttention-4",
        group="Optimization passes",
        summary="Select FlashAttention-4 only on native attention modules that expose its validated policy surface.",
        availability="Registered public pass: `flash-attention-4`",
        fidelity="Exact attention intent; validate backend tolerances on the target GPU",
        devices="CUDA with float16 or bfloat16 when the backend is required",
        usage='''result = model.apply_optimization_plan(
    "flash-attention-4",
    mode="inference",
)
print(result.manifest())''',
        github=(_github("FlashAttention", "Dao-AILab/flash-attention"), ),
        papers=(_paper("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "2205.14135"), ),
        implementation="voicehub/optimization/accelerators.py",
        registry_name="flash-attention-4",
        pass_id="flash-attention-4",
        pass_version="1",
    ),
    OptimizationGuide(
        slug="custom-kernels",
        title="Custom kernels",
        group="Optimization passes",
        summary="Select registered Triton or CUDA operators on modules that expose the general kernel protocol.",
        availability="Registered public pass: `custom-kernels`",
        fidelity="Operator-equivalent intent; validate dtype-specific tolerances",
        devices="CPU, CUDA, or MPS; accelerated backends require CUDA",
        usage='''result = model.apply_optimization_plan(
    "custom-kernels",
    mode="inference",
)
print(result.manifest())''',
        github=(_github("Triton", "triton-lang/triton"), ),
        papers=(Reference("Triton: an intermediate language and compiler for tiled neural network computations", "https://dl.acm.org/doi/10.1145/3315508.3329973"), ),
        implementation="voicehub/optimization/accelerators.py",
        registry_name="custom-kernels",
        pass_id="custom-kernels",
        pass_version="1",
    ),
    OptimizationGuide(
        slug="codec-kernels",
        title="Codec kernels",
        group="Optimization passes",
        summary="Restrict kernel selection to discovered neural-audio codec operations.",
        availability="Registered public pass: `codec-kernels`",
        fidelity="Operator-equivalent intent; validate reconstructed audio",
        devices="CPU, CUDA, or MPS; Triton, CuTe, and CUDA extensions require CUDA",
        usage='''result = model.apply_optimization_plan(
    "codec-kernels",
    mode="inference",
)
print(result.manifest())''',
        github=(
            _github("Triton", "triton-lang/triton"),
            _github("NVIDIA CUTLASS / CuTe", "NVIDIA/cutlass"),
        ),
        papers=(Reference("Triton: an intermediate language and compiler for tiled neural network computations", "https://dl.acm.org/doi/10.1145/3315508.3329973"), ),
        implementation="voicehub/optimization/codec_accelerators.py",
        registry_name="codec-kernels",
        pass_id="codec-kernels",
        pass_version="1",
        related_guide="../guides/codec-optimization.md",
    ),
    OptimizationGuide(
        slug="diffusion-cache",
        title="Diffusion cache",
        group="Optimization passes",
        summary="Reuse architecture-owned diffusion block residuals within one isolated generation request.",
        availability="Registered public pass: `diffusion-cache`",
        fidelity="Approximate; generated audio may change",
        devices="CPU, CUDA, or MPS; inference only",
        usage='''result = model.apply_optimization_plan(
    "diffusion-cache",
    mode="inference",
)
print(result.manifest())''',
        github=(
            _github("DeepCache", "horseee/DeepCache"),
            _github("TeaCache", "ali-vilab/TeaCache"),
        ),
        papers=(
            _paper("DeepCache: Accelerating Diffusion Models for Free", "2312.00858"),
            _paper("Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model", "2411.19108"),
        ),
        implementation="voicehub/optimization/diffusion_cache.py",
        registry_name="diffusion-cache",
        pass_id="voicehub.diffusion-block-cache",
        pass_version="1",
        related_guide="../guides/diffusion-optimization.md",
    ),
    OptimizationGuide(
        slug="diffusion-sampling",
        title="Diffusion sampling",
        group="Optimization passes",
        summary="Reduce model evaluations through explicit schedule, guidance, cache, or solver policies.",
        availability="Registered public pass: `diffusion-sampling`",
        fidelity="Approximate; generated audio may change",
        devices="CPU, CUDA, or MPS; inference only",
        usage='''result = model.apply_optimization_plan(
    "diffusion-sampling",
    mode="inference",
)
print(result.manifest())''',
        github=(_github("DPM-Solver", "LuChengTHU/dpm-solver"), ),
        papers=(
            _paper("DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling", "2206.00927"),
            _paper("Flow Matching for Generative Modeling", "2210.02747"),
        ),
        implementation="voicehub/optimization/diffusion_sampling.py",
        registry_name="diffusion-sampling",
        pass_id="voicehub.diffusion-sampling",
        pass_version="1",
        related_guide="../guides/diffusion-optimization.md",
    ),
    OptimizationGuide(
        slug="hqq",
        title="HQQ",
        group="Optional source backends",
        summary="Quantize eligible linear weights without calibration data; this is not yet a VoiceHub public pass.",
        availability="Optional library; no registered VoiceHub pass",
        fidelity="Quantized; measure task quality and generated audio",
        devices="Backend-dependent",
        usage='''python -m pip install \\
  "hqq @ git+https://github.com/dropbox/hqq.git@d88a488ec8aa2d58362ef2038a52bca862db2e74"''',
        github=(_github("HQQ", "dropbox/hqq"), ),
        papers=(Reference("Half-Quadratic Quantization of Large Machine Learning Models", "https://dropbox.github.io/hqq_blog/"), ),
        related_guide="../guides/optional-backends.md",
    ),
    OptimizationGuide(
        slug="gemlite",
        title="GemLite",
        group="Optional source backends",
        summary="Run compatible low-bit matrix kernels; this is not yet a VoiceHub public pass.",
        availability="Optional library; no registered VoiceHub pass",
        fidelity="Kernel- and quantization-dependent",
        devices="Supported CUDA GPUs",
        usage='''python -m pip install \\
  "gemlite @ git+https://github.com/dropbox/gemlite.git@3dc52c3115fee49a09d00fd9e470ef6396885949"''',
        github=(_github("GemLite", "dropbox/gemlite"), ),
        papers=(Reference("GemLite: Towards Building Custom Low-Bit Fused CUDA Kernels", "https://dropbox.github.io/gemlite_blogpost/"), ),
        related_guide="../guides/optional-backends.md",
    ),
    OptimizationGuide(
        slug="audio-cpp",
        title="audio.cpp",
        group="Optional source backends",
        summary="Build a separate C++/GGML audio runtime; this is not a Python optimization pass.",
        availability="External runtime; no registered VoiceHub pass",
        fidelity="Model conversion and runtime dependent",
        devices="CPU and backend-dependent accelerators",
        usage='''git clone https://github.com/0xShug0/audio.cpp.git
cd audio.cpp
git checkout 748c5e28f6a7228b8f38ad7142ca97d29584544b
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target audiocpp_cli -j 8''',
        github=(_github("audio.cpp", "0xShug0/audio.cpp"), ),
        papers=(),
        related_guide="../guides/optional-backends.md",
    ),
    OptimizationGuide(
        slug="vllm",
        title="vLLM",
        group="Serving backends",
        summary="Serve verified LLM-based TTS models through an isolated vLLM or vLLM-Omni HTTP process.",
        availability="Built-in VoiceHub HTTP client; the capability registry controls verified model pairs",
        fidelity="Backend and checkpoint dependent; unsupported pairs fail without native fallback",
        devices="External Linux engine; hardware support follows vLLM and vLLM-Omni",
        usage='''from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    model_type="qwen3tts",
    llm_backend="vllm",
    llm_backend_config={"endpoint": "http://127.0.0.1:8091"},
)
audio = model.generate("Hello from vLLM.")''',
        github=(
            _github("vLLM", "vllm-project/vllm"),
            _github("vLLM-Omni", "vllm-project/vllm-omni"),
        ),
        papers=(
            _paper("Efficient Memory Management for Large Language Model Serving with PagedAttention", "2309.06180"),
            _paper("vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models", "2602.02204"),
        ),
        implementation="voicehub/llm_serving/backends.py",
        related_guide="../guides/llm-serving.md",
        source_install='''git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 568afb3a13806beb53bb2e6bd518269357b237c0
python -m pip install --editable .

cd ..
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
git checkout a4ea67a21b20054dacc6e83952f9bd407e8ee4e7
python -m pip install --editable .''',
    ),
    OptimizationGuide(
        slug="sglang",
        title="SGLang",
        group="Serving backends",
        summary="Serve verified LLM-based TTS models through an isolated SGLang or SGLang-Omni HTTP process.",
        availability="Built-in VoiceHub HTTP client; the capability registry controls verified model pairs",
        fidelity="Backend and checkpoint dependent; unsupported pairs fail without native fallback",
        devices="External Linux engine; hardware support follows SGLang and SGLang-Omni",
        usage='''from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    model_type="qwen3tts",
    llm_backend="sglang",
    llm_backend_config={"endpoint": "http://127.0.0.1:8000"},
)
audio = model.generate("Hello from SGLang.")''',
        github=(
            _github("SGLang", "sgl-project/sglang"),
            _github("SGLang-Omni", "sgl-project/sglang-omni"),
        ),
        papers=(_paper("SGLang: Efficient Execution of Structured Language Model Programs", "2312.07104"), ),
        implementation="voicehub/llm_serving/backends.py",
        related_guide="../guides/llm-serving.md",
        source_install='''git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout d21f3c3a10606ba3c7bf43f981496da0a7d620cd
python -m pip install --editable "python[all]"

cd ..
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
git checkout 76ad450616a696cc4a49777d387c1b22270f2382
python -m pip install --editable .''',
    ),
)
# yapf: enable

__all__ = [
    "MODEL_REFERENCES",
    "OPTIMIZATION_GUIDES",
    "ModelReferences",
    "OptimizationGuide",
    "Reference",
]
