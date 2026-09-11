"""Pinned source and checkpoint metadata for native Granite Speech."""

GRANITE_SPEECH_SOURCE_REVISION = ("a08ace4bbd97e721c98751deec37d87b026acadc")
GRANITE_SPEECH_RELEASE_SOURCE_REVISION = ("753d61104116eefc8ffc977327b441ee0c8d599f")

GRANITE_SPEECH_CHECKPOINTS = {
    "ibm-granite/granite-speech-4.1-2b": {
        "revision": "de575db64086f84fdc79da4932d1076e965bc546",
        "tensors": 954,
        "parameters": 2_313_207_148,
        "total_size": 4_626_414_392,
        "header_fingerprint": ("8889064efd770b05c39cc62dba5fac842e006530649b12bcd"
                               "1c3d90c7a474001"),
        "checkpoint": "model.safetensors.index.json",
        "license": "Apache-2.0",
    },
}

__all__ = [
    "GRANITE_SPEECH_CHECKPOINTS",
    "GRANITE_SPEECH_RELEASE_SOURCE_REVISION",
    "GRANITE_SPEECH_SOURCE_REVISION",
]
