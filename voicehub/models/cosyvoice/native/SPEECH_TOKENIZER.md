# CosyVoice 3 speech-tokenizer boundary

CosyVoice 3 publishes its frozen supervised semantic speech tokenizer as
`speech_tokenizer_v3.onnx`, not as a PyTorch training checkpoint. VoiceHub
does not execute that graph at runtime.

The pinned artifact at model revision
`29e01c4e8d000f4bcd70751be16fa94bf3d85a18` was audited as:

- SHA-256
  `23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d`;
- 969,451,503 bytes, ONNX opset 16, and 2,810 graph nodes;
- 198 float32 initializers containing 242,009,608 parameters;
- two stride-two convolutions, twelve 1,280-dimensional RoPE/FSMN
  transformer blocks, and an eight-dimensional ternary finite-scalar
  quantizer (6,561 codes).

The PyTorch graph is adapted from the Apache-2.0
[S3Tokenizer v3 implementation](https://github.com/xingchensong/S3Tokenizer/tree/9bf5d845b5e043ffaf4657f4942939091c7697a2)
and was checked against the immutable
[CosyVoice 3 artifact](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/blob/29e01c4e8d000f4bcd70751be16fa94bf3d85a18/speech_tokenizer_v3.onnx).
Its full state namespace and one deterministic token sequence match the
published ONNX graph.

## One-time conversion

Install the optional `onnx` parser in the conversion environment, then run:

```python
from voicehub.converters.cosyvoice_speech_tokenizer import (
    convert_audited_cosyvoice_speech_tokenizer,
)

convert_audited_cosyvoice_speech_tokenizer(
    "speech_tokenizer_v3.onnx",
    "speech_tokenizer.safetensors",
)
```

The converter rejects every other filename, byte count, SHA-256, opset,
graph I/O, node count, initializer inventory, parameter count, mapped key, or
shape. It parses initializers but never creates an ONNX Runtime session.

Place `speech_tokenizer.safetensors` beside the normal native CosyVoice
artifact. `speech_tokenizer_config.json` is optional for the official graph
and is written automatically by `save_pretrained`.

## Raw-audio records

Precomputed tokens remain valid:

```python
{"text": "Merhaba", "speech_tokens": [12, 44, 91]}
```

When the converted tokenizer is attached, LM fine-tuning can instead use
16 kHz tensors, audio mappings, or PCM WAVE paths:

```python
{
    "text": "Merhaba",
    "speech_audio": waveform,
    "speech_sampling_rate": 16000,
}
```

Accepted audio keys are `speech_audio`, `audio`, `waveform`, and
`audio_path`. Tensor inputs without an explicit sampling rate are treated as
16 kHz. Other rates are resampled by VoiceHub's native PyTorch frontend.

Inference accepts `prompt_audio` plus optional `prompt_audio_sample_rate`.
This extracts prompt speech tokens only; CosyVoice still requires the
separate 192-dimensional speaker embedding.

## Optimization

`CosyVoiceSpeechTokenizer` exposes its frozen encoder/quantizer roots and a
codec compile target. It therefore works with the shared
`CodecOptimizationConfig` without changing checkpoint keys. Exact mode keeps
the audited attention and FSQ equations; `torch.compile` can optimize the
whole feature-to-token boundary.
