import hashlib
import math
from pathlib import Path

import torch

from voicehub.audio import load_audio
from voicehub.processing.waveform import resample_waveform_hann
from voicehub.checkpointing import SafeTensorReader
from voicehub.models.conversationtts.source.conversationtts.tools.tokenizer.MimiCodec.model.models.MimiCodec import MimiCodec
from voicehub.models.conversationtts.source.conversationtts.tools.tokenizer.abs_tokenizer import AbsTokenizer

_MIMI_HEADER_TENSORS = 318
_MIMI_HEADER_VALUES = 96_151_393
_MIMI_HEADER_FINGERPRINT = (
    "4087bffb33f1c565a4a7d49e486c65258f6f750f792cb35dbbc3e98487bb983c"
)
_INFERENCE_UNUSED_PARAMETERS = frozenset({
    "semantic_mapping_layer.ln_layer.bias",
    "semantic_mapping_layer.ln_layer.weight",
})


class MimiTokenizer(AbsTokenizer):
    def __init__(self, ckpt_path, device=torch.device('cpu')):
        super(MimiTokenizer, self).__init__()
        self.device = torch.device(device)
        if ckpt_path is None:
            raise ValueError(
                "`ckpt_path` is required. Resolve Hub artifacts at the "
                "VoiceHub wrapper boundary before constructing MimiTokenizer."
            )
        checkpoint = Path(ckpt_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Mimi tokenizer checkpoint was not found: {checkpoint}."
            )
        if checkpoint.suffix.lower() != ".safetensors":
            raise ValueError("Mimi tokenizer checkpoints must use Safetensors.")
        # This is the exact static configuration published with the
        # ConversationTTS integration. Keeping it typed in executable code
        # removes a YAML parser from the runtime boundary.
        self.model = MimiCodec(
            encoder_rates=[8, 6, 5, 4],
            codebook_size=2048,
            codebook_dim=256,
            rvq_layers=32,
        )
        self._load_checkpoint(checkpoint)
        self.model.eval()
        self.sr = 24000
        self.model = self.model.to(self.device)

    def _load_checkpoint(self, checkpoint: Path) -> None:
        state = self.model.state_dict()
        with SafeTensorReader(checkpoint) as reader:
            available = set(reader.keys())
            expected = set(state)
            missing = sorted(expected - available - _INFERENCE_UNUSED_PARAMETERS)
            unexpected = sorted(available - expected)
            if missing or unexpected:
                raise ValueError(
                    "Mimi tokenizer checkpoint namespace mismatch "
                    f"(missing={missing!r}, unexpected={unexpected!r})."
                )
            number_of_values = sum(
                math.prod(reader.tensor_shape(name))
                for name in reader.keys()
            )
            inventory = "\n".join(
                (
                    f"{name}|{reader.record(name).dtype}|"
                    f"{','.join(str(value) for value in reader.tensor_shape(name))}"
                )
                for name in reader.keys()
            ).encode("utf-8")
            fingerprint = hashlib.sha256(inventory).hexdigest()
            if (
                len(reader) != _MIMI_HEADER_TENSORS
                or number_of_values != _MIMI_HEADER_VALUES
                or fingerprint != _MIMI_HEADER_FINGERPRINT
            ):
                raise ValueError(
                    "Mimi tokenizer checkpoint does not match the audited "
                    "ConversationTTS inventory."
                )
            with torch.no_grad():
                for name in sorted(available):
                    target = state[name]
                    actual_shape = reader.tensor_shape(name)
                    if tuple(actual_shape) != tuple(target.shape):
                        raise ValueError(
                            f"Mimi tensor {name!r} has shape {actual_shape!r}; "
                            f"expected {tuple(target.shape)!r}."
                        )
                    target.copy_(
                        reader.get_tensor(
                            name,
                            device=target.device,
                            dtype=target.dtype,
                        ))

    def encode(self, wav_root):
        if isinstance(wav_root, str):
            decoded = load_audio(wav_root)
            # Upstream: torchaudio.transforms.Resample(sr, 24000).
            wav = resample_waveform_hann(
                decoded.waveform,
                decoded.sampling_rate,
                self.sr,
                match="transform",
            )
            wav = wav.unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            wav = wav_root
        with torch.no_grad():
            codes = self.model.encode(wav)
        codes = codes.squeeze(0) #.detach().cpu() # reduce the save space.
        return codes

    def find_length(self, x):
        return x.shape[1]

    def tokenize2(self, token):
        if isinstance(token, torch.Tensor):
            return token.to(torch.int64).transpose(0, 1)
        else:
            raise NotImplementedError

    def tokenize(self, wav, sample_rate=24000):
        if isinstance(wav, str):
            # if x is the wave path
            return self.encode(wav)
        elif isinstance(wav, torch.Tensor):
            if wav.dim() == 1: # already done offline
                return wav
            if wav.dim() == 2: # transfer to 3 dim
                if wav.numel() == 0:
                    return None
                if sample_rate != self.sr:
                    # Upstream: torchaudio.transforms.Resample(sample_rate, 24000).
                    wav = resample_waveform_hann(
                        load_audio(wav, sampling_rate=sample_rate).waveform,
                        sample_rate,
                        self.sr,
                        match="transform",
                    ).unsqueeze(0)
                wav = wav.unsqueeze(1).to(self.device) # (1,1,len)
            wav = wav.to(self.device)
            with torch.no_grad():
                codes = self.model.encode(wav)
            codes = codes.squeeze(0).detach().cpu().to(torch.int16) # reduce the save space.
            return codes
        else:
            raise NotImplementedError

    def detokenize(self, codes):
        #assert codes.shape[0] == 8
        codes = codes.unsqueeze(0)
        wav = self.model.decode(codes)
        wav = wav.squeeze(1).detach().cpu()
        return wav
