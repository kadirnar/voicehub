"""Immutable source and checkpoint identities for native CosyVoice."""

COSYVOICE_SOURCE_REPOSITORY = "https://github.com/FunAudioLLM/CosyVoice"
COSYVOICE_SOURCE_REVISION = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
COSYVOICE3_MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
COSYVOICE3_MODEL_REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
S3TOKENIZER_SOURCE_REPOSITORY = "https://github.com/xingchensong/S3Tokenizer"
S3TOKENIZER_SOURCE_REVISION = "9bf5d845b5e043ffaf4657f4942939091c7697a2"

COSYVOICE3_LEGACY_FILES = {
    "llm": {
        "filename": "llm.pt",
        "sha256": "69f43bd545131c30e98947fb360ea8b4dc9916d8e83dded7757c7ea4f5a24970",
        "size": 2_024_669_519,
        "tensor_count": 293,
        "parameter_count": 642_283_136,
        "header_fingerprint": "705b9f6911d4eb7d45c4229e241e37713616c18fe6e8641301e694cf47bba5da",
    },
    "flow": {
        "filename": "flow.pt",
        "sha256": "a6fab32a7825e5b0bc855ddd948f8db9370b0a786fbc249caa4595e95b608e4b",
        "size": 1_329_116_148,
        "tensor_count": 330,
        "parameter_count": 332_257_120,
        "header_fingerprint": "2821c208171253ef207c93343f6ed05c395e7d73cbe70c6802a8d612cabae0d5",
    },
    "hift": {
        "filename": "hift.pt",
        "sha256": "b279d7641eb97ae55b3b540cfba4f953c26492a2df758328a89a4d007ab87a65",
        "size": 83_202_622,
        "tensor_count": 328,
        "parameter_count": 20_779_887,
        "header_fingerprint": "f8b75ce1db30a46f429aca44b9c8bebce13d853f192bdaa06f1c4a011edf8adf",
    },
}

COSYVOICE3_SPEECH_TOKENIZER_FILE = {
    "filename": "speech_tokenizer_v3.onnx",
    "sha256": "23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d",
    "size": 969_451_503,
    "graph_node_count": 2_810,
    "initializer_count": 198,
    "parameter_count": 242_009_608,
    "onnx_initializer_fingerprint": "1744e0d777657d5d7641952c86d2729066f44a1961d76741b48b4aa6b3854f47",
    "native_header_fingerprint": "d38ce77f1bce44629c4a4befd3bfa2be90fee6655815c9fda764d26a4d838a50",
    "opset": 16,
}

NATIVE_COSYVOICE_FORMAT = "voicehub-native-cosyvoice-v1"

__all__ = [
    "COSYVOICE3_LEGACY_FILES",
    "COSYVOICE3_MODEL_ID",
    "COSYVOICE3_MODEL_REVISION",
    "COSYVOICE3_SPEECH_TOKENIZER_FILE",
    "COSYVOICE_SOURCE_REPOSITORY",
    "COSYVOICE_SOURCE_REVISION",
    "NATIVE_COSYVOICE_FORMAT",
    "S3TOKENIZER_SOURCE_REPOSITORY",
    "S3TOKENIZER_SOURCE_REVISION",
]
