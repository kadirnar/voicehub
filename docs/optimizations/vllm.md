---
description: Usage, support boundaries, paper, and source links for vLLM.
---

# vLLM

Serve verified LLM-based TTS models through an isolated vLLM or vLLM-Omni HTTP process.

## Use

Install the engine from source in a separate environment:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 568afb3a13806beb53bb2e6bd518269357b237c0
python -m pip install --editable .

cd ..
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
git checkout a4ea67a21b20054dacc6e83952f9bd407e8ee4e7
python -m pip install --editable .
```

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    model_type="qwen3tts",
    llm_backend="vllm",
    llm_backend_config={"endpoint": "http://127.0.0.1:8091"},
)
audio = model.generate("Hello from vLLM.")
```

## Support

| Property | Value |
| --- | --- |
| Availability | Built-in VoiceHub HTTP client; the capability registry controls verified model pairs |
| Fidelity | Backend and checkpoint dependent; unsupported pairs fail without native fallback |
| Runtime | External Linux engine; hardware support follows vLLM and vLLM-Omni |
| Registry name | Not registered; do not report this backend as an applied VoiceHub pass |

Unsupported model, backend, transport, or checkpoint combinations fail before a request.
The external server is a serving topology, not an applied VoiceHub optimization pass.

## Paper and GitHub

- **Paper:** [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180); [vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models](https://arxiv.org/abs/2602.02204)
- **Upstream GitHub:** [vLLM](https://github.com/vllm-project/vllm); [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- **VoiceHub source:** [VoiceHub implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/llm_serving/backends.py)

## Verify

Compare native and external serving with the same checkpoint, input, seed, and output
settings. Record latency, memory, audio quality, and both exact source revisions.

See the [related workflow](../guides/llm-serving.md) and
[optimization API](../reference/api.md#optimization).
