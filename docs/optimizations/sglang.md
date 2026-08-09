---
description: Usage, support boundaries, paper, and source links for SGLang.
---

# SGLang

Serve verified LLM-based TTS models through an isolated SGLang or SGLang-Omni HTTP process.

## Use

Install the engine from source in a separate environment:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout d21f3c3a10606ba3c7bf43f981496da0a7d620cd
python -m pip install --editable "python[all]"

cd ..
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
git checkout 76ad450616a696cc4a49777d387c1b22270f2382
python -m pip install --editable .
```

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    model_type="qwen3tts",
    llm_backend="sglang",
    llm_backend_config={"endpoint": "http://127.0.0.1:8000"},
)
audio = model.generate("Hello from SGLang.")
```

## Support

| Property | Value |
| --- | --- |
| Availability | Built-in VoiceHub HTTP client; the capability registry controls verified model pairs |
| Fidelity | Backend and checkpoint dependent; unsupported pairs fail without native fallback |
| Runtime | External Linux engine; hardware support follows SGLang and SGLang-Omni |
| Registry name | Not registered; do not report this backend as an applied VoiceHub pass |

Unsupported model, backend, transport, or checkpoint combinations fail before a request.
The external server is a serving topology, not an applied VoiceHub optimization pass.

## Paper and GitHub

- **Paper:** [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- **Upstream GitHub:** [SGLang](https://github.com/sgl-project/sglang); [SGLang-Omni](https://github.com/sgl-project/sglang-omni)
- **VoiceHub source:** [VoiceHub implementation](https://github.com/kadirnar/voicehub/blob/main/voicehub/llm_serving/backends.py)

## Verify

Compare native and external serving with the same checkpoint, input, seed, and output
settings. Record latency, memory, audio quality, and both exact source revisions.

See the [related workflow](../guides/llm-serving.md) and
[optimization API](../reference/api.md#optimization).
