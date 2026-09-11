---
description: Optimize native neural audio codecs through typed code batches, structural execution targets, exact custom kernels, torch.compile, and safe fixed-shape CUDA Graphs.
---

# Neural audio codec optimization

VoiceHub optimizes codec *operations and execution boundaries*, not a
hard-coded list of model-specific optimization implementations. A codec may
use dense residual vector
quantization, hierarchical rates, a continuous VAE bottleneck, or a
decoder-only graph embedded in an LLM TTS runtime. The shared layer describes
those differences without changing checkpoint ownership or pretending that
all codec algorithms are interchangeable.

The public building blocks in this guide live in:

```python
from voicehub.components.audio.codecs.base import (
    DenseCodecCodes,
    RaggedCodecCodes,
    separate_audio_codec,
)
from voicehub.components.audio.codecs.catalog import (
    get_codec_entry,
    list_codec_entries,
)
from voicehub.optimization.codecs import (
    CodecOptimizationConfig,
    capture_codec_cuda_graph,
    discover_codec_compile_targets,
)
```

The catalog is graph-free: inspecting it does not import PyTorch or any model
implementation. It is kept separate from runtime model loading while its owner
bindings are tested against the active TTS registry.

## Native codec inventory

VoiceHub's 17 native codec families cover all 23 registered LLM-based TTS
model types. Shared families appear once and list every owning model. “Split”
means waveform reconstruction continues through architecture-owned stages,
such as flow matching plus HiFT, rather than a single `decode()` method.

| Native codec family | LLM-TTS model types | Representation | Encoder / bottleneck / decoder | Available optimization surfaces |
| --- | --- | --- | --- | --- |
| SNAC 24 kHz | `orpheustts` | Hierarchical discrete | Native / native / native | Compile, CUDA Graph, Snake |
| DAC | `dia`, `outetts`, `parlertts`, `zonos`, `zonos2` | Dense discrete | Native / native / native | Compile, CUDA Graph, Snake, CuTe VQ search |
| EnCodec | `bark` | Dense discrete | Native / native / native | Compile, decoder-only CUDA Graph |
| Mimi | `csm`, `conversationtts` | Dense discrete | Native / native / native | Compile, CUDA Graph |
| XCodec2 | `llasa` | Dense discrete | Native / native / native | Compile, CUDA Graph, SnakeBeta |
| NeuCodec | `neutts` | Dense discrete | Native / native / native | Compile, CUDA Graph, SnakeBeta |
| MOSS Audio Tokenizer | `mosstts` | Dense discrete | Native / native / native | Compile |
| Qwen3-TTS 12 Hz tokenizer | `qwen3tts` | Dense discrete | Native / native / native | Compile, decoder-only CUDA Graph, SnakeBeta |
| VibeVoice acoustic tokenizer | `vibevoice` | Continuous VAE | Native / Gaussian / native | Compile, safe CUDA Graph |
| VoxCPM AudioVAE v2 | `voxcpm` | Continuous VAE | Native / Gaussian / native | Compile, safe CUDA Graph, Snake |
| Higgs Audio Tokenizer v2 | `omnivoice`, `higgstts` | Dense discrete | Native / native / native | Compile, Snake |
| Fish ModifiedDAC | `fishtts` | Dense discrete | Native / native / native | Compile, CUDA Graph, Snake |
| Fluac | `vui` | Dense discrete | Native / native / native | Compile, CUDA Graph, Snake |
| Chatterbox S3 | `chatterbox` | Dense discrete | Native / native / split native | Compile, CUDA Graph, Snake |
| GPT-SoVITS S2 | `gptsovits` | Dense discrete | Integrated native pipeline | Compile, CUDA Graph |
| XTTS2 DVAE | `xtts` | Dense discrete | Native / native / native | Compile, CUDA Graph |
| CosyVoice speech tokenizer | `cosyvoice` | Dense discrete | Native / native / split flow + HiFT | Compile, CUDA Graph, Snake |

Query the same source of truth programmatically:

```python
from voicehub.components.audio.codecs import get_codec_entries_for_model, list_codec_entries

for entry in list_codec_entries():
    owners = [owner.model_type for owner in entry.owners]
    print(entry.codec_id, owners, entry.optimization.surfaces)

qwen_codec, = get_codec_entries_for_model("qwen3tts")
assert qwen_codec.stages.is_full_native_codec
```

Each entry also records exact implementation import paths, stage primitives,
fixed-shape graph constraints, aliases, and any remaining artifact-conversion
or split-pipeline boundary. This makes unsupported cases explicit instead of
silently selecting a model-specific fast path.

CUDA Graph support is stage-specific. EnCodec and Qwen3-TTS advertise only
validation-free decoder tensor boundaries; validate code ranges and encoded
frame structure before capture. Their encoders retain host-side length or
bandwidth preparation outside capture. MOSS and Higgs retain
`torch.compile` support, but do not advertise CUDA Graphs because their active
public boundaries extract CUDA tensor values on the host.

## Represent dense and multirate codes

`DenseCodecCodes` represents the common
`[batch, codebook, frame]` layout. The optional `lengths` tensor records the
valid frame count before padding.

```python
import torch

codes = DenseCodecCodes(
    values=torch.randint(
        0,
        1024,
        (2, 8, 150),
        dtype=torch.int64,
        device="cuda",
    ),
    lengths=torch.tensor([150, 121], device="cuda"),
)

print(codes.batch_size, codes.num_codebooks, codes.num_frames)
```

Hierarchical codecs can emit levels at different frame rates.
`RaggedCodecCodes` retains each level rather than padding every level to the
fastest rate. A level may have shape `[batch, frame]` for one codebook or
`[batch, codebook, frame]` for a group of same-rate codebooks.

```python
multirate_codes = RaggedCodecCodes(
    levels=(
        torch.randint(0, 4096, (2, 12), device="cuda"),
        torch.randint(0, 4096, (2, 1, 24), device="cuda"),
        torch.randint(0, 4096, (2, 2, 48), device="cuda"),
    ),
    lengths=(
        torch.tensor([12, 10], device="cuda"),
        torch.tensor([24, 20], device="cuda"),
        torch.tensor([48, 40], device="cuda"),
    ),
    # Temporal stride relative to the fastest level.
    strides=(4, 2, 1),
)

print(multirate_codes.temporal_lengths)
```

The containers validate tensor rank, integer dtype, batch consistency, and
device consistency. Codebook value ranges remain the codec's responsibility
because sizes can differ between levels and checkpoints.

## Separate components without reparenting modules

Assigning an existing encoder or decoder to a second `nn.Module` wrapper can
register it twice and change state-dict paths. `separate_audio_codec()` instead
returns a plain, non-owning view:

```python
before = tuple(codec.state_dict())
parts = separate_audio_codec(codec)

assert parts.encoder is codec.encoder
assert parts.bottleneck is codec.quantizer
assert parts.decoder is codec.decoder
assert tuple(codec.state_dict()) == before
```

The view recognizes conventional `encoder`, `quantizer` or `bottleneck`, and
`decoder` attributes. Explicit attribute names can describe a different
native graph:

```python
parts = separate_audio_codec(
    codec,
    encoder="acoustic_encoder",
    bottleneck="vector_quantizer",
    decoder="acoustic_decoder",
)

features = parts.encode_features(audio)
quantized = parts.apply_bottleneck(features)
waveform = parts.decode_features(quantized)
```

This is a view of the original objects, not a new checkpoint topology. It is
also useful for continuous autoencoders: the encoder, Gaussian bottleneck,
and decoder remain independently addressable, so a diffusion or LLM system
can optimize latent decoding without capturing audio encoding or sampling.

## Discover the executed boundary

`discover_codec_compile_targets()` first honors a codec's explicit
`codec_optimization_compile_targets(mode)` or
`optimization_compile_targets(mode)` declaration. Otherwise it discovers
methods structurally.

- In inference, direct code-to-waveform methods such as `decode_codes`,
  `decode_code`, `from_indices`, and `decode_tokens` take priority over latent
  decoding. A decoder-only module whose operation is `forward` is still a
  valid target.
- In training, `forward` is preferred for both whole-codec and quantizer
  selection so the differentiable encoder, bottleneck, decoder, and losses
  remain connected. In inference, inverse quantizer methods such as
  `from_codes`, `from_indices`, and `embed_codes` take priority.
- Explicit components can select `encode`, `quantizer`, `flow`, `vocoder`,
  `decode`, `forward`, or all available boundaries. `decode` is the serving
  umbrella for declared flow, vocoder, and direct decoder stages.

```python
inference_targets = discover_codec_compile_targets(
    codec,
    mode="inference",
)
training_targets = discover_codec_compile_targets(
    codec,
    mode="training",
)

print([(target.label, target.attribute) for target in inference_targets])
print([(target.label, target.attribute) for target in training_targets])
```

An explicit hook may return `OptimizationCompileTarget` declarations, method
names, or `(label, owner, attribute[, component])` tuples. Architecture-owned
targets set the optional `component` field so automatic and explicit stage
selection resolve the same live callable. Returning an empty iterable is a
deliberate fail-closed declaration: automatic compilation stays eager and
required compilation raises. For a runtime with a declaration, `all` includes
its declared live stages and any structurally separate encoder or quantizer
stages. It does not add an unrelated generic root `forward` method behind the
architecture's back during inference.

## Choose a fidelity policy

Every codec configuration records one of three fidelity tiers:

| Policy | Meaning | Suitable techniques |
| --- | --- | --- |
| `exact` | Preserve the same algorithm and explicit randomness; compare within dtype-appropriate numerical tolerances | PyTorch reference operations, method compilation, fixed-shape graph replay, static conditioning |
| `relaxed` | Permit declared numerical changes while retaining the same high-level algorithm | Triton/CUDA periodic activations, FP16/BF16, fast transcendental math after parity measurement |
| `approximate` | Permit a semantic or quality trade-off that must be benchmarked per checkpoint | Polynomial activations, quantization, altered stochastic behavior |

The policy is a declaration, not an automatic quality downgrade.
`policy="approximate"` does not itself replace an activation or freeze VAE
noise. Conversely, an automatic `exact` plan never selects an approximate
transformation.

## Resolve and apply a decoder-only plan

For LLM TTS, token-to-waveform decoding is normally the latency-critical codec
path. Compile that boundary without compiling an unused encoder:

```python
from voicehub.optimization.capabilities import OptimizationContext

context = OptimizationContext(
    mode="inference",
    device="cuda",
    dtype="float16",
)
config = CodecOptimizationConfig(
    policy="exact",
    kernel_backend="auto",
    compile="auto",
    compile_components="decode",
    compile_config={
        "backend": "inductor",
        "mode": "max-autotune-no-cudagraphs",
        "dynamic": True,
    },
)

plan = config.resolve(codec, context=context)
print(plan.to_json_string())

application = plan.apply(codec)
optimized_codec = application.model
decode_method = (
    plan.compile_targets[0].attribute
    if plan.compile_targets
    else "decode"
)
waveform = getattr(optimized_codec, decode_method)(codes.values)

# Restore method bindings and kernel selectors in reverse order.
application.restore()
```

`compile="auto"` may retain eager execution if no compatible target or
compiler is available. Use `compile="required"` only when fallback would be an
error.

## Resolve an encoder-only plan

Dataset preparation and codec fine-tuning may need the encoder without the
waveform decoder:

```python
encoder_config = CodecOptimizationConfig(
    policy="exact",
    kernel_backend="auto",
    compile="auto",
    compile_components="encode",
    compile_config={
        "backend": "inductor",
        "fullgraph": False,
        "dynamic": True,
    },
)

encoder_plan = encoder_config.resolve(
    codec,
    mode="inference",
    context=context,
)
encoder_application = encoder_plan.apply(codec)
encoded = encoder_application.model.encode(audio)
encoder_application.restore()
```

For end-to-end fine-tuning, resolve in `mode="training"` and use
`compile_components="forward"`. Custom training operations must provide a
backward implementation; a fake or shape-inference registration alone is not
autograd support. PyTorch documents the required operator surface in its
[custom operator tutorial](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html).

## Select exact custom kernels before compilation

An architecture-owned codec block can expose:

- `set_codec_kernel_backend(backend)`;
- a current `codec_kernel_backend` value;
- optional `resolve_codec_kernel_backend(backend, device, dtype)` capability
  resolution; and
- `supported_kernel_operations` plus `supported_codec_kernel_backends`.

When this protocol is present, the plan places `CodecKernelPass` before
`TorchCompilePass`. The compiler therefore captures the already resolved
operation. The pass is codec-scoped and cannot configure a diffusion or
language-model block that happens to expose the older generic selector.
Routing is operation-specific: a single DAC plan may select CuTe for vector
quantization while retaining Torch for Snake.

Current codec Snake and independent-frequency/magnitude SnakeBeta
implementations preserve the native periodic formulas and gradients with
Torch, Triton, and CUDA-extension routes. DAC vector quantizers additionally
implement `audio.codec.euclidean_vq_search`. Its CuTe route sends the
similarity GEMM through NVIDIA's CUTLASS Operator API, caches the compiled CuTe
DSL artifact per shape/layout/SM, and performs the norm correction and
nearest-index reduction in PyTorch. Only the discrete lookup is replaced;
embedding lookup, commitment/codebook losses, and the straight-through
gradient path remain native.

CuTe is an explicit optional backend:

```bash
python -m pip install "nvidia-cutlass-operators[torch]"
```

```python
config = CodecOptimizationConfig(
    policy="relaxed",
    kernel_backend="cute",
    compile=False,
)
plan = config.resolve(codec, context=cuda_context)
optimized = plan.apply(codec)
```

VoiceHub validates Linux, CUDA, the CuTe DSL, the Operator API GEMM interfaces,
and the active tensor contract. If CUTLASS exposes no compatible operator for
the GPU architecture, dtype, shape, or layout, the explicit request fails;
there is no silent Torch fallback for the VQ operation. Importing VoiceHub
does not import CUTLASS or compile a kernel. NVIDIA documents the
[`GemmArguments` discovery and compile/run contract](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/000_gemm.html).
CUDA Graph capture must retain at least one warmup call so the CuTe artifact is
compiled before capture; the default `warmup_steps=3` already satisfies this
constraint.

The exact policy pins these periodic activations to the PyTorch reference.
Triton `tl.sin` and CUDA fast-math can differ numerically, so selecting either
accelerator requires `policy="relaxed"` or `policy="approximate"`. With a
relaxed automatic policy, capability resolution still retains Torch when an
accelerator implementation is unavailable. Explicit CuTe also requires a
relaxed or approximate policy because GEMM rounding can change the selected
code only for vectors at a nearest-neighbor boundary.

If a codec has no selector protocol, `kernel_backend="auto"` leaves the native
graph untouched. An explicit backend with no supported codec operation fails
instead of pretending an optimization was applied. PyTorch's official
[user-defined Triton kernel guide](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
explains how custom kernels participate in `torch.compile`.

## Capture a fixed-shape CUDA Graph

CUDA Graphs reduce repeated Python and kernel-launch overhead, but replay uses
the memory addresses and control flow recorded during capture. VoiceHub's
helper owns static input buffers, copies new values into them, replays the
graph, and clones outputs by default:

```python
codec = codec.cuda().eval()
code_tensor = codes.values.to("cuda")

runner = capture_codec_cuda_graph(
    codec,
    (code_tensor,),
    decoder_only=True,
    warmup_steps=3,
)

same_shape_codes = torch.randint_like(code_tensor, high=1024)
waveform = runner(same_shape_codes)
```

The default `target="auto"` uses the same architecture declaration as
`torch.compile`. For Qwen3-TTS and EnCodec, it resolves to the declared
validation-free decoder tensor boundary; callers must validate codes or frame
metadata before capture. A multi-stage codec such as Chatterbox returns an
ambiguity error until the caller selects a declared label or method such as
`flow_inference` or `hift_inference`. MOSS and Higgs are intentionally absent
from the CUDA Graph catalog until their active boundaries no longer perform
host-synchronized scalar extraction.

Every replay input must retain:

- the same nested tuple, list, mapping, or dataclass structure;
- identical tensor shape, stride, dtype, and CUDA device;
- identical non-tensor argument values; and
- one shared CUDA device for all tensors.

At least one tensor input is required. Length, batch, codebook count, and
streaming chunk variations therefore need separate graph buckets. Set
`clone_outputs=False` only when the caller understands that the returned
static output storage will be overwritten by the next replay.

These constraints follow PyTorch's
[CUDA Graphs documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
For variable shapes, use dynamic `torch.compile` or explicit static buckets
rather than padding every request without measuring the extra decoder work.

## Control stochastic VAE inputs

A VAE encode or full forward commonly performs:

```text
latent = mean + epsilon * standard_deviation
```

PyTorch's default CUDA generator is graph-aware, so captured `randn` operations
advance correctly across replay. VoiceHub permits that native path. Passing
`epsilon` explicitly remains useful when the application needs direct,
request-by-request control:

```python
audio = audio.cuda()
epsilon = torch.randn(
    (audio.shape[0], 64, 100),  # Use the checkpoint's posterior shape.
    device=audio.device,
    dtype=audio.dtype,
)

vae_runner = capture_codec_cuda_graph(
    vae,
    (audio,),
    target="forward",
    epsilon=epsilon,
    stochastic_vae=True,
)

new_epsilon = torch.randn_like(epsilon)
reconstruction = vae_runner(audio, epsilon=new_epsilon)
```

The callable must actually accept the `epsilon` keyword. If it does not,
omit explicit epsilon or capture the deterministic decoder after sampling:

```python
decoder_runner = capture_codec_cuda_graph(
    vae,
    (sampled_latents,),
    target="decode",
    decoder_only=True,
)
waveform = decoder_runner(new_latents)
```

Native codecs can declare `is_stochastic_vae`, `stochastic_vae`, or
`uses_stochastic_bottleneck`; structural detection also recognizes common
Gaussian/VAE bottlenecks. A VAE encoder that only returns posterior parameters
can list its method in `deterministic_codec_targets`; VoxCPM and VibeVoice use
this target-level declaration. Explicit non-default `torch.Generator` inputs
are registered with the graph and must live on the capture device. Hidden
custom generators cannot be registered by a generic wrapper and should be
made explicit by the codec.

## What the fast-codec research changed

The shared design was informed by
[`fast-snac` at `e669190`](https://github.com/kadirnar/fast-snac/tree/e6691906f41a30c1363052e42db4bc87b9da4a8f),
the upstream [SNAC implementation](https://github.com/hubertsiuzdak/snac),
[`fast-dacvae` at `406f2e5`](https://github.com/kadirnar/fast-dacvae/tree/406f2e5c803927ef18cc9bbe38d715e5417459b9),
and the upstream [DACVAE implementation](https://github.com/facebookresearch/dacvae).
Those research repositories are valuable measurements, but their fastest
benchmark paths are not all interchangeable with a reusable training and
serving API.

### Techniques adopted as exact system capabilities

| Research observation | VoiceHub treatment |
| --- | --- |
| Decoder-only compilation removes unused encode work from LLM TTS inference | Structural `decode` discovery and independently resolvable encoder/decoder plans |
| Small periodic Snake/SnakeBeta activations are useful custom-kernel boundaries | Architecture-owned same-formula Torch, Triton, and CUDA selectors run before compilation; accelerator transcendental math requires a relaxed policy |
| `torch.compile` benefits a stable decoder method without requiring a new checkpoint wrapper | Reversible method compilation retains the original model and state-dict keys |
| CUDA Graph replay helps fixed, repeated shapes | A reusable runner copies each new input into owned static buffers and validates the complete shape/stride/device contract |
| VAE encoder, bottleneck, and decoder have different optimization and randomness requirements | A non-owning component view, graph-aware default RNG, and optional explicit epsilon control |

See the reviewed
[`fast-snac` optimization code](https://github.com/kadirnar/fast-snac/blob/e6691906f41a30c1363052e42db4bc87b9da4a8f/snac/optimize.py)
and
[`fast-dacvae` optimization code](https://github.com/kadirnar/fast-dacvae/blob/406f2e5c803927ef18cc9bbe38d715e5417459b9/dacvae/optimize.py)
for the experiments behind these boundaries. The general compilation API is
documented by
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

### Techniques kept opt-in or rejected as defaults

| Research technique | Why it is not an automatic exact optimization |
| --- | --- |
| `fast_sinf` or polynomial Snake | Both replace the native transcendental calculation. They belong to a measured `relaxed` or `approximate` profile, not `exact`. |
| Caching SNAC decoder noise | Reusing one random sample changes stochastic behavior. Exact execution must retain explicit, request-specific randomness. |
| Freezing one DACVAE epsilon | It turns stochastic sampling into replay of one realization. VoiceHub uses graph-aware CUDA RNG by default and accepts explicit epsilon when request-level control is needed. |
| Replacing every Conv1d/ConvTranspose1d with channels-last 2D modules | Performance is hardware- and shape-dependent, and structural replacement changes the module/checkpoint topology. It requires an isolated, reversible implementation and parity benchmarks. |
| Removing weight normalization in place | Materialization can be mathematically equivalent for frozen weights, but mutating the training graph and canonical parameter names is unsafe as a universal pass. |
| Omitting or simplifying the watermark path | This changes externally visible model behavior and cannot be called an execution-only optimization. |
| Shape-specialized replay that never copies a new input | It benchmarks one sample rather than providing a correct serving callable. VoiceHub copies and validates every replay input. |
| Custom operations with only a fake kernel and no backward | They are inference-only. Training support requires registered autograd and gradient-parity tests. |
| FP16/BF16 results | Reduced precision can be worthwhile, but it is a `relaxed` numerical policy and must be reported separately from FP32 parity. |

The reviewed `fast-dacvae` commit also has inconsistent license signals between
its root license and package/README metadata. VoiceHub does not copy that
implementation as a new license authority; source-derived work must follow
the upstream DACVAE Apache-2.0 terms and retain applicable notices.

## Validate before enabling a backend

Report compile cold start separately from steady-state latency. For each
backend and shape bucket, compare:

- encoder, quantizer, decoder, and end-to-end latency;
- p50 and p90 latency, real-time factor, and peak memory;
- code-index agreement for an encoder path;
- waveform length, SNR or SI-SDR, and log-mel error;
- offline and streaming-chunk parity, including state reset; and
- eager versus optimized gradients for every training kernel.

An accelerator path should remain disabled when it does not beat PyTorch for
the measured shapes. Exact parity tests must use the same checkpoint, inputs,
codebook depth, and explicit VAE epsilon.
