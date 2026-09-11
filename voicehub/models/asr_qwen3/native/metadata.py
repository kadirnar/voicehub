"""Pinned source and checkpoint metadata for native Qwen3-ASR."""

QWEN3_ASR_SOURCE_REVISION = "7c6daf77a2421100f5fb066495372c00129d39ff"

QWEN3_ASR_CHECKPOINTS = {
    "Qwen/Qwen3-ASR-0.6B": {
        "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "tensors": 612,
        "parameters": 938_008_576,
        "header_fingerprint": ("67bba95c9922ef5ca599b2b0ebd80420ca497e06ad43f9a49f58659a933cfbdb"),
        "files": {
            "model.safetensors": ("79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"),
        },
    },
    "Qwen/Qwen3-ASR-1.7B": {
        "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
        "tensors": 708,
        "parameters": 2_349_217_408,
        "header_fingerprint": ("ecf3f6c30544447679cf6a25390e2a62ddfa2490a1fc01655acfeebe3c803e0e"),
        "files": {
            "model-00001-of-00002.safetensors":
            ("a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6"),
            "model-00002-of-00002.safetensors":
            ("6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc"),
        },
    },
}

__all__ = [
    "QWEN3_ASR_CHECKPOINTS",
    "QWEN3_ASR_SOURCE_REVISION",
]
