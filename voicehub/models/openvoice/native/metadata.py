"""Immutable source and checkpoint records for OpenVoice V2."""

OPENVOICE_SOURCE_REVISION = "74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
OPENVOICE_CHECKPOINT_REVISION = "f36e7edfe1684461a8343844af60babc2efbb727"
OPENVOICE_MODEL_ID = "myshell-ai/OpenVoiceV2"

OPENVOICE_CONVERTER_CHECKPOINT = {
    "filename": "converter/checkpoint.pth",
    "revision": OPENVOICE_CHECKPOINT_REVISION,
    "license": "MIT",
    "sha256": "9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e",
    "file_bytes": 131_320_490,
    "tensor_data_bytes": 131_168_904,
    "tensors": 486,
    "parameters": 32_792_226,
    "state_values": 32_792_226,
    "dtype": "F32",
    "header_fingerprint": ("fd350d54eb92706417c1c37215a318f80ee914e151d087e9d4bcfbdbd9751216"),
}
OPENVOICE_CONVERTER_CONFIG_SHA256 = ("9dfff60350b8c63f2c664efd92a61b2516efb22671466960f0e5dfebd881fa47")

OPENVOICE_SOURCE_SPEAKER_FILES = {
    "en-au": "5e9782233deef51fc5289d05ad4dd4ce12b196e282eccf6b6db6256bbd02daaa",
    "en-br": "2bf5a88025cfd10473b25d65d5c0e608338ce4533059c5f9a3383e69c812d389",
    "en-default": "e4139de3bc2ea162f45a5a5f9559b710686c9689749b5ab8945ee5e2a082d154",
    "en-india": "ad03d946757e95fe9e13239aa4b11071d98f22316f604f34b1a0b4bdf41cda48",
    "en-newest": "6a3798229b1114f0e9cc137b33211809def7dda5a8a9398d5a112c0b42699177",
    "en-us": "0d092d4af0815a4bfbc6105b65621ab68dc4c61b2f55044d8a66968a34947c32",
    "es": "b8cece8853fb75b9f5217a1f5cda9807bac92a3e4c4547fc651e404d05deff63",
    "fr": "8a01f6d30a73efa368c288a542a522a2bcdd4e2ec5589d8646b307cf8e2ad9ae",
    "jp": "7b645ff428de4a57a22122318968f1e6127ac81fda2e2aa66062deccd3864416",
    "kr": "f501479d6072741a396725bec79144653e9f4a5381b85901e29683aa169795df",
    "zh": "2b353de562700c13faacf096ecfc0adcafd26e6704a9feef572be1279714e031",
}

__all__ = [
    "OPENVOICE_CHECKPOINT_REVISION",
    "OPENVOICE_CONVERTER_CHECKPOINT",
    "OPENVOICE_CONVERTER_CONFIG_SHA256",
    "OPENVOICE_MODEL_ID",
    "OPENVOICE_SOURCE_REVISION",
    "OPENVOICE_SOURCE_SPEAKER_FILES",
]
