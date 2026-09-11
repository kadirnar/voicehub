"""Built-in native architecture catalogue.

Only lightweight registration modules are imported here. Model graphs,
processors, checkpoint readers, and objectives remain lazy component
references until a runtime explicitly resolves them.
"""

from __future__ import annotations

from importlib import import_module

from voicehub.runtime.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.runtime.specifications import ArchitectureSpec

BUILTIN_ARCHITECTURE_REGISTRARS = (
    (
        "voicehub.models.asr_transformers.native.registration",
        "register_asr_dispatch_architecture",
    ),
    (
        "voicehub.models.asr_whisper_native.native.registration",
        "register_whisper_architecture",
    ),
    (
        "voicehub.models.asr_wav2vec2.native.registration",
        "register_wav2vec2_architecture",
    ),
    (
        "voicehub.models.asr_hubert.native.registration",
        "register_hubert_architecture",
    ),
    (
        "voicehub.models.asr_wavlm.native.registration",
        "register_wavlm_architecture",
    ),
    (
        "voicehub.models.asr_moonshine.native.registration",
        "register_moonshine_architecture",
    ),
    (
        "voicehub.models.asr_qwen3.native.registration",
        "register_qwen3_asr_architecture",
    ),
    (
        "voicehub.models.asr_granite_speech.native.registration",
        "register_granite_speech_architecture",
    ),
    (
        "voicehub.models.asr_parakeet_tdt.native.registration",
        "register_parakeet_tdt_architecture",
    ),
    (
        "voicehub.models.asr_nemotron.native.registration",
        "register_nemotron_asr_architecture",
    ),
    (
        "voicehub.models.asr_cohere.native.registration",
        "register_cohere_asr_architecture",
    ),
    (
        "voicehub.models.asr_seamless_m4t_v2.native.registration",
        "register_seamless_m4t_v2_architecture",
    ),
    (
        "voicehub.models.vibevoice.native.registration",
        "register_vibevoice_asr_architecture",
    ),
    (
        "voicehub.models.asr_medasr.native.registration",
        "register_medasr_architecture",
    ),
    (
        "voicehub.models.asr_funasr.native.registration",
        "register_sensevoice_architecture",
    ),
    (
        "voicehub.models.asr_nemo.native.registration",
        "register_nemo_ctc_architecture",
    ),
    (
        "voicehub.models.asr_wenet.native.registration",
        "register_wenet_u2pp_architecture",
    ),
    (
        "voicehub.models.asr_espnet.native.registration",
        "register_espnet_architecture",
    ),
    (
        "voicehub.models.vits.native.registration",
        "register_vits_architecture",
    ),
    (
        "voicehub.models.vui.native.registration",
        "register_vui_architecture",
    ),
    (
        "voicehub.models.chatterbox.native.registration",
        "register_chatterbox_architecture",
    ),
    (
        "voicehub.models.csm.native.registration",
        "register_csm_architecture",
    ),
    (
        "voicehub.models.conversationtts.native.registration",
        "register_conversationtts_architecture",
    ),
    (
        "voicehub.models.llasa.native.registration",
        "register_llasa_architecture",
    ),
    (
        "voicehub.models.neutts.native.registration",
        "register_neutts_architecture",
    ),
    (
        "voicehub.models.outetts.native.registration",
        "register_outetts_architecture",
    ),
    (
        "voicehub.models.speecht5.native.registration",
        "register_speecht5_architecture",
    ),
    (
        "voicehub.models.kokoro.native.registration",
        "register_kokoro_architecture",
    ),
    (
        "voicehub.models.f5tts.native.registration",
        "register_f5tts_architecture",
    ),
    (
        "voicehub.models.gptsovits.native.registration",
        "register_gptsovits_architecture",
    ),
    (
        "voicehub.models.mosstts.native.registration",
        "register_mosstts_architecture",
    ),
    (
        "voicehub.models.melotts.native.registration",
        "register_melotts_architecture",
    ),
    (
        "voicehub.models.openvoice.native.registration",
        "register_openvoice_architecture",
    ),
    (
        "voicehub.models.parlertts.native.registration",
        "register_parlertts_architecture",
    ),
    (
        "voicehub.models.styletts2.native.registration",
        "register_styletts2_architecture",
    ),
    (
        "voicehub.models.qwen3tts.native.registration",
        "register_qwen3_tts_architecture",
    ),
    (
        "voicehub.models.vibevoice.native.registration",
        "register_vibevoice_tts_architecture",
    ),
    (
        "voicehub.models.voxcpm.native.registration",
        "register_voxcpm2_architecture",
    ),
    (
        "voicehub.models.omnivoice.native.registration",
        "register_omnivoice_architecture",
    ),
    (
        "voicehub.models.higgstts.native.registration",
        "register_higgs_audio_v2_architecture",
    ),
    (
        "voicehub.models.irodoritts.native.registration",
        "register_irodori_architecture",
    ),
    (
        "voicehub.models.cosyvoice.native.registration",
        "register_cosyvoice_architecture",
    ),
    (
        "voicehub.models.xtts.native.registration",
        "register_xtts2_architecture",
    ),
    (
        "voicehub.models.fishtts.native.registration",
        "register_fish_s2_architecture",
    ),
    (
        "voicehub.models.zonos.native.registration",
        "register_zonos_architecture",
    ),
    (
        "voicehub.models.zonos2.native.registration",
        "register_zonos2_architecture",
    ),
    (
        "voicehub.models.supertonic.native.registration",
        "register_supertonic_architecture",
    ),
    (
        "voicehub.models.bark.native.registration",
        "register_bark_architecture",
    ),
    (
        "voicehub.models.inflecttts.native.registration",
        "register_inflect_architecture",
    ),
    (
        "voicehub.models.echo.native.registration",
        "register_echo_architecture",
    ),
    (
        "voicehub.models.vad_sherpa_onnx.native.registration",
        "register_vad_dispatch_architecture",
    ),
    (
        "voicehub.models.vad_silero.native.registration",
        "register_silero_vad_architecture",
    ),
    (
        "voicehub.models.vad_funasr.native.registration",
        "register_fsmn_vad_architecture",
    ),
    (
        "voicehub.models.asr_speechbrain.native.registration",
        "register_speechbrain_asr_architecture",
    ),
    (
        "voicehub.models.vad_speechbrain.native.registration",
        "register_speechbrain_vad_architecture",
    ),
    (
        "voicehub.models.vad_ten.native.registration",
        "register_ten_vad_architecture",
    ),
    (
        "voicehub.models.vad_webrtc.native.registration",
        "register_webrtc_vad_architecture",
    ),
    (
        "voicehub.models.vad_nemo.native.registration",
        "register_marblenet_vad_architecture",
    ),
    (
        "voicehub.models.vad_pyannote.native.registration",
        "register_pyannet_architecture",
    ),
    (
        "voicehub.models.vad_auditok.native.registration",
        "register_energy_vad_architecture",
    ),
    (
        "voicehub.models.causal_lm.native.registration",
        "register_causal_lm_architecture",
    ),
    (
        "voicehub.models.dac.native.registration",
        "register_dac_architecture",
    ),
    (
        "voicehub.models.encodec.native.registration",
        "register_encodec_architecture",
    ),
    (
        "voicehub.models.dia.native.registration",
        "register_dia_architecture",
    ),
)


def register_builtin_architectures(
    *,
    registry: ArchitectureRegistry | None = None,
    exist_ok: bool = True,
) -> tuple[ArchitectureSpec, ...]:
    """Register every built-in declaration without importing model graphs."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    if not isinstance(exist_ok, bool):
        raise TypeError("`exist_ok` must be a boolean.")

    registered = []
    for module_name, function_name in BUILTIN_ARCHITECTURE_REGISTRARS:
        registrar = getattr(import_module(module_name), function_name)
        spec = registrar(
            registry=target,
            exist_ok=exist_ok,
        )
        if not isinstance(spec, ArchitectureSpec):
            raise TypeError(
                f"Built-in registrar {module_name}:{function_name} returned "
                f"{type(spec).__name__}, not ArchitectureSpec.")
        registered.append(spec)
    return tuple(registered)


__all__ = [
    "BUILTIN_ARCHITECTURE_REGISTRARS",
    "register_builtin_architectures",
]
