"""Pinned source and checkpoint metadata for native Parakeet TDT."""

PARAKEET_TRANSFORMERS_REVISION = ("af71155683b4d34dd92d8f037392fa6bf334035e")
PARAKEET_NEMO_REVISION = "2381f42f6979449b5b99538f8f80135831009b51"
PARAKEET_TOKENIZERS_REVISION = ("f383101a26663708484cac0727792aad74f78234")
SPM_PRECOMPILED_REVISION = "3795b14343d25782df654b49f5c3e9f2a7db7d6f"

PARAKEET_TDT_CHECKPOINTS = {
    "nvidia/parakeet-tdt-0.6b-v3": {
        "revision": "7c35754d166cca382ad1e53e68b01e7c575f3a1d",
        "license": "CC-BY-4.0",
        "tensors": 723,
        "state_values": 627_057_310,
        "parameters": 627_057_286,
        "header_fingerprint": ("f861cd8d0e811fd9bfbf3b887356fe3f6e226562ec78653dc77d90c1c2757e6b"),
        "files": {
            "config.json": ("e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e"),
            "generation_config.json": ("b141de6ec6d7f982ece13f98f604e3fe1807ea9c0e839185d0ab7064604209d0"),
            "model.safetensors": ("3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11"),
            "processor_config.json": ("8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7"),
            "tokenizer.json": ("bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91"),
            "tokenizer_config.json": ("0b2fe0037599ee335f0b972fa682bf0ece74e4ccfec755cb7daa3405d3d3e874"),
        },
    },
}

__all__ = [
    "PARAKEET_NEMO_REVISION",
    "PARAKEET_TDT_CHECKPOINTS",
    "PARAKEET_TOKENIZERS_REVISION",
    "PARAKEET_TRANSFORMERS_REVISION",
    "SPM_PRECOMPILED_REVISION",
]
