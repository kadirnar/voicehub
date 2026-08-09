---
description: Serve supported LLM-based TTS models through isolated vLLM, vLLM-Omni, SGLang, or SGLang-Omni processes.
---

# External LLM serving

For the shortest setup, open the dedicated [vLLM](../optimizations/vllm.md)
or [SGLang](../optimizations/sglang.md) page.

VoiceHub can delegate the language-model portion of selected TTS models to
vLLM or SGLang. Backend selection uses the same configuration-first pattern as
the rest of the library:

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    model_type="qwen3tts",
    llm_backend="vllm",
    llm_backend_config={
        "endpoint": "http://127.0.0.1:8091",
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    },
)
```

The integration is HTTP-only. VoiceHub does not import either engine and does
not add one of their CUDA stacks to its environment.

## Keep the engine in a separate environment

VoiceHub pins PyTorch 2.8 for its native runtime. Current engine releases own
tightly coupled PyTorch, CUDA, Triton, FlashAttention, and compiler
dependencies. For example, the pinned SGLang-Omni source stack uses PyTorch
2.11, SGLang 0.5.16, CUDA 13 relay packages, and FlashAttention-4 beta 18 or
newer.
The [SGLang-Omni installation guide](https://sgl-project.github.io/sglang-omni/get_started/installation.html)
therefore recommends its prepared Docker image. vLLM and vLLM-Omni likewise
select platform-specific compiled dependencies.

Run the engine in a container, virtual environment, or another host, then
point VoiceHub at its HTTP endpoint. Do not install vLLM/SGLang into the
VoiceHub environment merely to use this client.

This split has two useful properties:

- the engine controls its CUDA kernels, tensor parallelism, quantization, and
  scheduler without mutating the VoiceHub environment; and
- VoiceHub keeps its model-family input validation, output type, local codec
  work where required, and `AutoModelForTextToSpeech` API.

## Two transport contracts

`transport="auto"` resolves to the only verified transport for a
model/backend pair.

| Transport | Engine route | Work performed by VoiceHub | Work performed by the engine |
| --- | --- | --- | --- |
| `tokens` | vLLM `/v1/completions` or SGLang `/generate` | Prompt tokenization, model-specific token parsing, and codec decoding | Flat causal-LM token generation |
| `speech` | `/v1/audio/speech` | Request normalization and PCM WAVE decoding | The complete tokenizer, model, codec, and vocoder pipeline |

Token transport is appropriate only when the language model is a standard
single-stream causal LM. VoiceHub loads the tokenizer and codec on first use,
but it does not load the local LM checkpoint.

Speech transport is for multi-stage or multi-codebook models. The external
Omni server owns the complete synthesis pipeline, so the VoiceHub wrapper
does not allocate native model weights. The current client requests one
non-streaming WAVE response; engine batch, SSE, raw-streaming, and WebSocket
interfaces are not exposed by this wrapper.

There is no silent fallback. An unverified backend, transport, checkpoint
family, or native-only generation option raises a compatibility error before
synthesis.

## Capability matrix

The registry contains the pairings that have an explicit VoiceHub adapter:

| VoiceHub model type | vLLM | SGLang | Verified checkpoint family |
| --- | --- | --- | --- |
| `orpheustts` | Tokens | Tokens | Dense Llama Orpheus checkpoint |
| `llasa` | Tokens | Tokens | Dense Llama LLaSA checkpoint |
| `qwen3tts` | Speech (vLLM-Omni) | Speech (SGLang-Omni) | Base, CustomVoice, or VoiceDesign |
| `fishtts` | Speech (vLLM-Omni) | Speech (SGLang-Omni) | `fishaudio/s2-pro` |
| `mosstts` | Speech (vLLM-Omni) | Speech (SGLang-Omni) | Engine-specific supported MOSS pipeline |
| `cosyvoice` | Speech (vLLM-Omni) | — | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` |
| `voxcpm` | Speech (vLLM-Omni) | — | `openbmb/VoxCPM2` |
| `omnivoice` | Speech (vLLM-Omni) | — | `k2-fsa/OmniVoice` |
| `higgstts` | Speech (vLLM-Omni) | — | Higgs Audio v2 3B |

SGLang-Omni's Higgs Audio v3 pipeline is not compatible with VoiceHub's
Higgs v2 wrapper. A blank cell means that VoiceHub intentionally rejects the
pairing, even if an upstream engine later adds a model with a similar name.

Query the installed library instead of copying this table into application
code:

```python
from voicehub.llm_serving import list_llm_backend_support

for support in list_llm_backend_support():
    print(
        support.model_type,
        support.backend.value,
        support.default_transport.value,
        support.checkpoint_family,
    )
```

`get_llm_backend_support(model_type, backend, transport="auto")` resolves one
pairing without importing or contacting the engine.

Each support record also owns request-shape differences: default task types,
flat versus `references`-list reference audio, and named non-empty string
options understood by that exact engine pairing. Its `speech_input_options`,
`speech_default_options`, and `speech_native_only_options` properties expose the
resulting request contract. These fields are validated and returned by
`support.to_dict()` as JSON data. The wrapper and direct HTTP client consume the
same schema and do not contain model-name or extension-option branches.

Separately distributed integrations can register the same capability without
editing VoiceHub's client:

```python
from voicehub import (
    LLMBackendSupport,
    register_llm_backend_support,
)

register_llm_backend_support(
    LLMBackendSupport(
        model_type="auroratts",
        backend="vllm",
        transports=("speech",),
        default_transport="speech",
        engine="Aurora vLLM-Omni plugin",
        checkpoint_family="acme/aurora-base",
        task_type_without_reference="Generate",
        task_type_with_reference="Clone",
        task_type_aliases=(("clone", "Clone"),),
        reference_format="references",
        speech_string_options=("emotion_prompt",),
    )
)
```

Register extensions once during process startup and remove temporary records in
tests with `unregister_llm_backend_support()`. A declared
`speech_string_options` entry becomes a recognized wrapper input and a
generation-default key automatically. It must be verified for that engine and
must not redefine a typed or request-owned field such as `temperature` or
`reference_audio`.

When a native architecture has no verified external-engine path, keep the
specific limitation beside that architecture as
`metadata={"external_llm_backend_blocker": "..."}` on its
`ArchitectureSpec`. The shared resolver reads this declaration lazily and uses
a generic fail-closed message when an architecture does not provide one.

## Launch an engine

Install and launch these commands in the engine's environment. The commands
follow the current upstream
[vLLM OpenAI server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/),
[vLLM-Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/),
and [SGLang-Omni TTS](https://sgl-project.github.io/sglang-omni/basic_usage/tts.html)
guides. Check the upstream model recipe when changing engine versions or
checkpoints.

### Flat token servers

Launch a standard vLLM server for Orpheus:

```bash
vllm serve canopylabs/orpheus-3b-0.1-ft \
  --host 0.0.0.0 \
  --port 8000 \
  --generation-config vllm
```

VoiceHub sends token IDs to `/v1/completions`, requests
`return_token_ids=true`, and disables special-token skipping. The current
vLLM completion protocol must return `choices[0].token_ids`.

The equivalent SGLang server is:

```bash
python -m sglang.launch_server \
  --model-path canopylabs/orpheus-3b-0.1-ft \
  --host 0.0.0.0 \
  --port 30000 \
  --skip-tokenizer-init
```

`--skip-tokenizer-init` makes the token-in/token-out boundary explicit.
VoiceHub posts `input_ids` to `/generate` and expects `output_ids`.
SGLang selects the model when the server starts, so VoiceHub does not send a
per-request `model` field on this route.

Replace the model path in both commands with
`HKUSTAudio/Llasa-1B-Multilingual` for LLaSA. The server checkpoint must
exactly match the checkpoint passed to VoiceHub; a different vocabulary or
special-token layout can produce invalid codec tokens.

### Complete speech servers

For Qwen3-TTS CustomVoice with vLLM-Omni:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
  --omni \
  --port 8091 \
  --trust-remote-code \
  --enforce-eager
```

For Fish Speech S2 Pro:

```bash
vllm serve fishaudio/s2-pro --omni --port 8091
```

The explicit Qwen deploy-config path is relative to a vLLM-Omni checkout.
Use the matching upstream recipe or an absolute path when running elsewhere.

SGLang-Omni uses its model-specific pipeline YAML. Run these commands from
the SGLang-Omni checkout, or replace each config with its absolute path:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --config examples/configs/qwen3_tts_0_6b_customvoice.yaml \
  --port 8000
```

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

For MOSS-TTS v1.5:

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config examples/configs/moss_tts.yaml \
  --port 8000
```

When a server fetches reference audio by URL, add only the required domains
with the engine's media-domain allowlist. VoiceHub converts a local reference
file to an inline audio data URL, so the remote process does not need access
to the caller's filesystem.

## Configure the client

Pass a mapping through `llm_backend_config`, or construct a typed runtime
configuration:

```python
import os

from voicehub import AutoModelForTextToSpeech
from voicehub.llm_serving import LLMBackendConfig

backend = LLMBackendConfig(
    backend="vllm",
    endpoint="https://tts.internal.example",
    transport="auto",
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    api_key=os.environ["VOICEHUB_TTS_API_KEY"],
    timeout=300,
    max_response_bytes=512 * 1024 * 1024,
)

model = AutoModelForTextToSpeech.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    model_type="qwen3tts",
    llm_backend="vllm",
    llm_backend_config=backend,
)
```

The configuration fields are:

| Field | Purpose |
| --- | --- |
| `backend` | `native`, `vllm`, or `sglang` |
| `endpoint` | Absolute HTTP(S) server base URL; required for an external backend |
| `transport` | `auto`, `tokens`, or `speech` |
| `model` | Model identifier expected by the server; defaults to the wrapper checkpoint |
| `api_key` | Runtime-only Bearer token |
| `headers` | Additional runtime-only HTTP headers |
| `timeout` | Positive request timeout in seconds |
| `extra_body` | Documented server-specific JSON fields not owned by VoiceHub |
| `max_response_bytes` | Hard upper bound for a response body |

Use the server origin, such as `http://127.0.0.1:8091`, as the endpoint.
VoiceHub appends the required route and also accepts a vLLM-style base ending
in `/v1`.

### Security behavior

- Endpoint credentials, query strings, and fragments are rejected. Use
  `api_key` or one `Authorization` header, never both.
- Header names and values reject newlines and transport-owned headers.
- `api_key` and header values are redacted from `repr()` and `to_dict()`.
  Backend configuration is kept on the live wrapper and is not written to
  `config.json`.
- `extra_body` cannot replace request-owned fields such as `input`, `prompt`,
  `input_ids`, `stream`, or `response_format`. Do not put credentials in it.
- HTTP redirects fail closed. VoiceHub never forwards Bearer credentials or
  custom runtime headers to a redirect target.
- JSON request bodies reject non-finite values. Bounded JSON responses reject
  duplicate object keys, `NaN`, infinities, and numeric overflow before the
  vLLM or SGLang protocol adapter interprets token IDs or usage metadata.
  Diagnostics identify the backend route and offending key or numeric path
  without echoing the discarded value.
- A local reference clip is read by the client, limited to 64 MiB, and sent
  as a data URL. Existing base64 audio data URLs are validated and subject to
  the same decoded-size limit; HTTP(S) reference URLs are also accepted.
- Use HTTPS and server-side authentication whenever traffic leaves a trusted
  host.

### Sampling controls

The client maps sampling controls to each upstream protocol instead of
assuming that the two OpenAI-shaped speech routes have identical schemas:

- vLLM-Omni receives `temperature`, `top_p`, and `top_k` inside
  `extra_params`. Its current speech schema does not expose
  `repetition_penalty`, `duration_tokens`, `token_count`, or `stage_params`,
  so VoiceHub rejects those call options. Its MOSS-SoundEffect adapter does
  accept `ambient_sound`.
- SGLang-Omni receives `temperature`, `top_p`, `top_k`,
  `repetition_penalty`, `duration_tokens`, and `token_count` as top-level
  fields. VoiceHub rejects `stage_params` and `non_streaming_mode`.
- Remote seeds must be integers from `0` through `2**63 - 1`.
- Speech speed must be in the engine schema's `[0.25, 4.0]` interval, and
  mode flags must be actual booleans.

Documented engine-version-specific extensions can be supplied through
`LLMBackendConfig.extra_body`. Explicit VoiceHub call options take precedence
when the client owns the corresponding protocol field.

## Model examples

### Qwen3-TTS through vLLM-Omni

CustomVoice uses a checkpoint speaker and optional style instruction:

```python
from voicehub import AutoModelForTextToSpeech

checkpoint = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
qwen = AutoModelForTextToSpeech.from_pretrained(
    checkpoint,
    model_type="qwen3tts",
    llm_backend="vllm",
    llm_backend_config={
        "endpoint": "http://127.0.0.1:8091",
        "model": checkpoint,
    },
)

output = qwen.generate(
    "The serving engine owns the complete speech pipeline.",
    mode="custom_voice",
    speaker="ryan",
    language="English",
    instruct="Speak warmly and clearly.",
    output_file="qwen-vllm.wav",
)
print(output.sample_rate, output.metadata)
```

Voice cloning requires the Base checkpoint and reference audio:

```python
checkpoint = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
qwen_base = AutoModelForTextToSpeech.from_pretrained(
    checkpoint,
    model_type="qwen3tts",
    llm_backend="sglang",
    llm_backend_config={
        "endpoint": "http://127.0.0.1:8000",
        "model": checkpoint,
    },
)

output = qwen_base.generate(
    "This utterance uses the reference voice.",
    mode="voice_clone",
    speaker_audio_path="reference.wav",
    reference_text="The exact transcript of reference.wav.",
    language="English",
    output_file="qwen-sglang-clone.wav",
)
```

VoiceDesign uses the matching VoiceDesign checkpoint,
`mode="voice_design"`, and a non-empty `instruct`.

### Fish Speech through SGLang-Omni

```python
from voicehub import AutoModelForTextToSpeech

fish = AutoModelForTextToSpeech.from_pretrained(
    "fishaudio/s2-pro",
    model_type="fishtts",
    llm_backend="sglang",
    llm_backend_config={
        "endpoint": "http://127.0.0.1:8000",
        "model": "fishaudio/s2-pro",
    },
)

output = fish.generate(
    "External scheduling, one stable VoiceHub result.",
    speaker_audio_path="reference.wav",
    reference_text="The exact transcript of reference.wav.",
    max_new_tokens=2048,
    temperature=0.8,
    output_file="fish-sglang.wav",
)
```

VoiceHub maps Fish reference audio to SGLang-Omni's `references` request
shape. The same call works with `llm_backend="vllm"` and a vLLM-Omni
endpoint; that adapter uses `ref_audio` and `ref_text`.

### Orpheus through a flat token server

```python
from voicehub import AutoModelForTextToSpeech

checkpoint = "canopylabs/orpheus-3b-0.1-ft"
orpheus = AutoModelForTextToSpeech.from_pretrained(
    checkpoint,
    model_type="orpheustts",
    llm_backend="vllm",
    llm_backend_config={
        "endpoint": "http://127.0.0.1:8000",
        "model": checkpoint,
    },
)

output = orpheus.generate(
    "The language model is remote, while SNAC decoding stays local.",
    voice="tara",
    max_new_tokens=1200,
    temperature=0.6,
    top_p=0.8,
    output_file="orpheus-vllm.wav",
)
```

The first call downloads or opens Orpheus tokenizer/config files and the SNAC
codec, but skips the local Llama weights. Change the backend to `sglang` and
the endpoint to the SGLang server to keep the same VoiceHub generation API.

## Unsupported custom generation

Do not route every model containing a Transformer through the token
transport. Many TTS architectures require semantics that a stock flat-logit
server cannot reproduce:

- OuteTTS needs its exact 64-token repetition window;
- NeuTTS needs checkpoint-specific RoPE behavior and minimum-token EOS
  masking;
- Vui, ConversationTTS, Zonos, Zonos2, CSM, Parler-TTS, and similar models
  generate multiple codebooks or run a hidden-state-conditioned depth
  decoder; and
- Chatterbox, GPT-SoVITS, XTTS, VibeVoice, Bark, and Dia use custom
  conditioning, CFG, semantic heads, or multiple generation stages.

VoiceHub rejects these token-server pairings. Qwen3-TTS, Fish Speech, and
other entries marked `speech` are supported only because the matching Omni
runtime implements their complete architecture-specific pipeline. A future
engine release does not become supported automatically; add and test a
capability record and request adapter first.

## Lifecycle, optimization, and training limits

External backends are an inference mode:

- Select `llm_backend` before calling `load()` or serving a request.
- Only the eager wrapper-side inference strategy is supported.
- An external backend cannot be combined with VoiceHub's in-process
  `TTSOptimizationConfig`. Configure compilation, quantization, attention
  kernels, and parallelism in the engine process.
- `load_for_training()` rejects an external wrapper. Fine-tune with a native
  VoiceHub wrapper, export a server-compatible checkpoint, launch the engine
  against that artifact, and create a new external wrapper.
- `from_pretrained()` cannot attach an external backend while restoring a
  local VoiceHub Trainer state. Point the engine at the exported fine-tuned
  checkpoint instead.
- After a token-backed wrapper has loaded its local tokenizer/codec runtime,
  it cannot detach the remote LM. Create a fresh native wrapper to transition
  back to local inference.
- A token-backed wrapper does not own the LM weights and therefore cannot
  export a complete native pretrained model.
- Requests on one external wrapper may overlap so the server can apply
  continuous batching. Backend configuration cannot be replaced or cleared
  while those requests are active.

These constraints prevent duplicated weights, partially restored training
graphs, and accidental mixing of server-side and in-process optimization
policies.
