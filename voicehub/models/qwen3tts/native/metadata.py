"""Pinned source and public checkpoint inventories for native Qwen3-TTS."""

from __future__ import annotations

QWEN3_TTS_SOURCE = {
    "repository": "https://github.com/QwenLM/Qwen3-TTS",
    "revision": "022e286b98fbec7e1e916cb940cdf532cd9f488e",
    "license": "Apache-2.0",
}

QWEN3_TTS_CHECKPOINTS = {
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": {
        "revision": "5d83992436eae1d760afd27aff78a71d676296fc",
        "license": "Apache-2.0",
        "bytes": 1_829_344_272,
        "tensors": 478,
        "parameters": 914_643_008,
        "header_fingerprint": ("3889c5131670e5b82d9cca8f7e14164e69572b3820c26db7c68a486a458f6c4e"),
    },
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": {
        "revision": "85e237c12c027371202489a0ec509ded67b5e4b5",
        "license": "Apache-2.0",
        "bytes": 1_811_626_576,
        "tensors": 402,
        "parameters": 905_788_672,
        "header_fingerprint": ("e19ec7ba43585e1a5992503f4401083394eb97b62c7eb53aff79b45889927b14"),
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": {
        "revision": "fd4b254389122332181a7c3db7f27e918eec64e3",
        "license": "Apache-2.0",
        "bytes": 3_857_413_744,
        "tensors": 480,
        "parameters": 1_928_677_440,
        "header_fingerprint": ("63672482f432aaff94ddb90de517e72da262d1ba4c880903cb4a016cadca33b4"),
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "revision": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
        "license": "Apache-2.0",
        "bytes": 3_833_402_552,
        "tensors": 404,
        "parameters": 1_916_676_352,
        "header_fingerprint": ("80b46c9ea2158e0a7b1f44756f5658859dd157663b004ec607aa1db2be0239d5"),
    },
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
        "revision": "5ecdb67327fd37bb2e042aab12ff7391903235d3",
        "license": "Apache-2.0",
        "bytes": 3_833_402_552,
        "tensors": 404,
        "parameters": 1_916_676_352,
        "header_fingerprint": ("80b46c9ea2158e0a7b1f44756f5658859dd157663b004ec607aa1db2be0239d5"),
    },
}

QWEN3_TTS_SPEECH_TOKENIZER = {
    "repository": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "revision": "5d83992436eae1d760afd27aff78a71d676296fc",
    "license": "Apache-2.0",
    "bytes": 682_293_092,
    "tensors": 496,
    "parameters": 170_557_441,
    "header_fingerprint": ("b0493e243c89bbf06ef06ecf80e2a06152fcf2802add4da2915992db4de3542e"),
    "decoder_tensors": 271,
    "decoder_parameters": 114_323_137,
    "decoder_header_fingerprint": ("804f87cf403a839dd152466b8a53afaef776aacfdf90b6ad0f246534d9c5d5c9"),
    "encoder_tensors": 225,
    "encoder_parameters": 56_234_304,
    "encoder_header_fingerprint": ("5e8d7354d9c30a170d083126d8f73ed15e48395cc12814f1eb9cabe8a320e6e2"),
}

__all__ = [
    "QWEN3_TTS_CHECKPOINTS",
    "QWEN3_TTS_SOURCE",
    "QWEN3_TTS_SPEECH_TOKENIZER",
]
