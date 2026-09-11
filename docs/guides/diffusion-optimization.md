---
description: Discover and optimize the nine active diffusion, flow-matching, and rectified-flow TTS families without changing their model-specific mathematics.
---

# Diffusion and flow TTS optimization

VoiceHub groups diffusion, conditional flow matching, rectified flow, and
diffusion-like denoising heads under one **execution** optimization inventory.
This is an engineering family, not a claim that the formulations are
mathematically identical. [Flow Matching](https://arxiv.org/abs/2210.02747),
for example, can use diffusion probability paths but is not limited to them.

The inventory is derived from registered architecture traits. It includes
only a diffusion or velocity-estimation graph reached by the public VoiceHub
model. A similarly named class in vendored source code is not enough.

## Active families

The current public inventory contains exactly nine model types:

| Model type | Active diffusion/flow boundary | Registered formulation | Sampling operations | Declared execution surface |
| --- | --- | --- | --- | --- |
| `chatterbox` | S3Gen speech-token-to-mel subgraph after the T3 token model | Conditional flow matching | Classifier-free guidance (CFG), Euler | Compile, SDPA, block cache, schedule reduction, prediction cache |
| `cosyvoice` | Flow estimator between the speech-token LM and HiFT vocoder | Conditional flow matching | CFG, Euler | Compile, fused kernels, schedule/guidance/cache, STORK-2 |
| `echo` | EchoDiT continuous Fish-codec latent generator | Rectified flow | Independent text/speaker CFG, Euler, optional blockwise generation | Compile, fused kernels, schedule/guidance/cache |
| `f5tts` | F5 DiT mel generator | Conditional flow matching | CFG, Euler or midpoint | Compile, attention/kernels, schedule/guidance/cache, STORK-2 for Euler |
| `irodoritts` | RF-DiT continuous DACVAE-latent generator | Rectified flow | Multi-condition CFG, Euler | Compile, SDPA/kernels, schedule/guidance/cache |
| `styletts2` | Style-vector generator inside a larger adversarial TTS graph | Style diffusion | CFG, ADPM2 with a Karras schedule | Compile, block cache, schedule reduction with ADPM2 stages preserved |
| `supertonic` | Iterative text-to-latent vector-estimator graph | Flow matching | Released iterative estimator | Compile, latent-residual cache, discrete total-step reduction |
| `vibevoice` | Acoustic diffusion head driven by the causal language model | Denoising diffusion | CFG, DPM-Solver++(2M) | Compile, SDPA/kernels, rebuilt DPM schedule, guidance/cache |
| `voxcpm` | Local DiT inside the outer autoregressive acoustic-frame loop | Conditional flow matching | CFG/CFG-Zero*, Euler | Compile, SDPA, schedule/guidance/cache |

The formulation and solver assignments follow the active VoiceHub graph and
the primary projects: [Chatterbox](https://github.com/resemble-ai/chatterbox),
[CosyVoice](https://github.com/FunAudioLLM/CosyVoice),
[Echo](https://github.com/jordandare/echo-tts),
[F5-TTS](https://github.com/SWivid/F5-TTS),
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS),
[StyleTTS 2](https://github.com/yl4579/StyleTTS2),
[Supertonic](https://github.com/supertone-inc/supertonic),
[VibeVoice](https://arxiv.org/abs/2508.19205), and
[VoxCPM](https://github.com/OpenBMB/VoxCPM). VoiceHub pins reviewed source
revisions in each `ArchitectureSpec`; the links above explain the underlying
model families.

`compile` in the table means architecture-level compatibility. It does not
promise that every inference and training wrapper exposes the same target.
The loaded runtime still supplies the mode-specific callable. An automatic
policy stays eager when a mode deliberately has no target; a required policy
fails instead.

## Query traits instead of checking names

Use the diffusion inventory to build tools, reports, or presets. Do not copy
the nine model names into application code.

```python
from voicehub.optimization import (
    get_diffusion_model_optimization_support,
    list_diffusion_model_optimization_support,
)

for support in list_diffusion_model_optimization_support():
    print(
        support.model_type,
        support.kind.value,
        [operation.value for operation in support.operations],
        support.compile_supported,
        support.optimization_passes,
    )

# Model aliases resolve to the canonical public model type.
f5 = get_diffusion_model_optimization_support("f5-tts")
assert f5.model_type == "f5tts"
```

Every included architecture declares all of the following:

- the `diffusion-family` feature;
- exactly one normalized `diffusion-kind-*` feature and matching
  `metadata.diffusion_architecture_kind`; and
- one or more `diffusion-operation-*` features in the same order as
  `metadata.diffusion_operations`.

The inventory fails on incomplete or contradictory declarations. That makes
an architecture registration the source of truth and prevents a new model
from silently inheriting an unsafe optimization merely because its name
contains `diffusion`, `flow`, or `dit`.

The operation traits describe the sampler that exists; they do not advertise
interchangeability. `euler-solver` on one model does not authorize replacing
another model's ADPM2 or DPM-Solver++ schedule.

## Resolve the shared execution policy

The diffusion inventory answers *what graph is active*. The universal TTS
resolver answers *which implementations that graph declares safe*.

```python
from voicehub.optimization import TTSOptimizationConfig
from voicehub.optimization import OptimizationContext

context = OptimizationContext(
    mode="inference",
    device="cuda",
    dtype="bfloat16",
)
config = TTSOptimizationConfig(
    attn_implementation="auto",
    kernel_backend="auto",
    compile="auto",
    compile_config={
        "backend": "inductor",
        "mode": "max-autotune-no-cudagraphs",
        "dynamic": True,
    },
)

plan = config.resolve("irodoritts", context=context)
print(plan.support.to_dict())
print(plan.manifest()["decisions"])
```

This separation is intentional:

1. family traits identify the active formulation and operations;
2. architecture capabilities constrain attention, kernels, and compilation;
3. the runtime exposes the callable actually used in inference or training;
4. selectors resolve before compilation; and
5. the application manifest records the chosen path and any fallback.

The shared policy does not replace a model's loss, noise path, timestep
distribution, EMA policy, codec boundary, or optimizer recipe. Training
profiles remain source-specific even when the execution machinery is shared.

## Reduce a 50-step sampler before optimizing its kernels

`diffusion_sampling` is a second, independent optimization layer. It acts at
the solver boundary and can reduce neural-network evaluations (NFE);
`diffusion_cache` acts inside a repeated DiT block list. Neither setting
changes codec kernels.

```python
from voicehub.optimization import TTSOptimizationConfig

optimization = TTSOptimizationConfig(
    diffusion_sampling="required",
    diffusion_sampling_config={
        # Rebuild a native 50-step grid into 20 signed integration steps.
        "target_steps": 20,
        "schedule": "native",
        # Direct velocity-field adapters may replace Euler with STORK-2.
        "solver": "stork2",
        "stork_stages": 9,
    },
    diffusion_cache="disabled",
)

plan = optimization.resolve("f5tts", mode="inference")
```

Schedule reduction happens **before** the loop starts. VoiceHub reconstructs
the grid, preserves both endpoints and strict monotonicity, and then runs the
solver over the larger signed deltas. It never implements “50 to 20” by
continuing past 30 already-created Euler or DPM steps. For multistep schedulers,
the model adapter must also rebuild scheduler history.

The supported schedule shapes are:

| `schedule` | Meaning |
| --- | --- |
| `native` | Select a monotone subsequence of the architecture's native grid |
| `uniform` | Rebuild linearly between the native endpoints |
| `quadratic` | Concentrate points near the starting/noise endpoint |
| `trailing` | Concentrate points near the terminal/data endpoint |

The most important latency number is the number of actual model evaluations,
not the nominal solver-stage count. The
[STORK paper](https://arxiv.org/abs/2505.24210) uses stabilized virtual stages
whose velocities are predicted from trajectory history. VoiceHub's initial
adapter implements the dimension-agnostic STORK-2 recurrence with a
first-order Taylor history and FP32 accumulation. Every outer step still
performs exactly one real velocity-model evaluation. Therefore, STORK at 50
outer steps does not reduce NFE; a 50-step baseline is accelerated by running
and validating STORK at a smaller count such as 30 or 20.

STORK-2 is initially declared only by direct deterministic velocity-field
adapters such as F5-TTS and CosyVoice. It fails closed for stochastic
samplers, epsilon/x0 prediction, learned absolute-latent estimators,
step-size-conditioned mean-flow heads, midpoint/ADPM2 stages, and DPM-Solver
history. VoiceHub does not expose the upstream STORK-4 path yet: the reviewed
official scheduler has a first-stage recurrence inconsistency, so copying it
unchanged would be less safe than retaining Euler.

### Guidance reduction

Classifier-free guidance can cost two or more logical denoiser evaluations.
The controller can narrow a model's native guidance decision but can never
enable guidance that the sampler did not request.

```python
limited_cfg = TTSOptimizationConfig(
    diffusion_sampling=True,
    diffusion_sampling_config={
        "target_steps": 24,
        "guidance": "limited_interval",
        "guidance_start": 0.10,
        "guidance_end": 0.70,
    },
)

adaptive_cfg = TTSOptimizationConfig(
    diffusion_sampling=True,
    diffusion_sampling_config={
        "guidance": "adaptive",
        "adaptive_guidance_threshold": 0.015,
        "adaptive_guidance_warmup_steps": 4,
        "adaptive_guidance_patience": 2,
    },
)
```

Limited-interval guidance adapts the image-side observation that CFG is not
equally useful over the entire trajectory; see
[Limited Interval Guidance](https://arxiv.org/abs/2404.07724).
Adaptive mode observes conditional/unconditional convergence and stops only
after the configured patience. Packed CFG, conditional-only, joint, and
alternate-unconditional paths use independent lanes.

### Whole-prediction cache policies

The sampler controller also exposes four explicitly approximate policies.
They are adaptations of image/video diffusion ideas to a final guided
speech-velocity or denoiser output; they are not presented as checkpoint-free
quality guarantees.

| `prediction_cache` | VoiceHub adaptation | Required calibration |
| --- | --- | --- |
| `fora` | Periodically compute, otherwise reuse the last complete output | Interval and maximum consecutive reuse |
| `teacache` | Accumulate a polynomial-rescaled relative input change | Model-specific `teacache_coefficients` |
| `smoothcache` | Follow an explicit full-compute/reuse step mask | `smoothcache_compute_step_mask` matching the prepared grid |
| `taylor` | First- or second-order polynomial extrapolation from real computed outputs | Order, compute interval, and quality validation |

The source techniques are
[FORA](https://arxiv.org/abs/2407.01425),
[TeaCache](https://arxiv.org/abs/2411.19108),
[SmoothCache](https://arxiv.org/abs/2411.10510), and
[TaylorSeer](https://arxiv.org/abs/2503.06923). TeaCache and SmoothCache are
calibration-dependent in their original forms. VoiceHub therefore refuses an
empty TeaCache polynomial or SmoothCache mask instead of silently substituting
a generic threshold.

Example calibrated TeaCache-style configuration:

```python
teacache = TTSOptimizationConfig(
    diffusion_sampling="required",
    diffusion_sampling_config={
        "target_steps": 24,
        "prediction_cache": "teacache",
        # Obtain these for the exact checkpoint and probe boundary.
        "teacache_coefficients": [0.0, 1.0],
        "cache_rel_l1_threshold": 0.08,
        "cache_error_budget": 0.20,
        "cache_warmup_steps": 2,
        "cache_max_consecutive_steps": 2,
    },
)
```

STORK and whole-prediction caching cannot be selected together: both infer
unobserved velocities and would invalidate each other's history. STORK also
rejects adaptive or limited guidance until an adapter declares the exact
history-reset boundary. Block-residual caching is configured separately and
must be evaluated as a composed approximation.

Run the included NFE plumbing benchmark before a checkpoint benchmark:

```bash
python scripts/benchmark_diffusion_sampling.py \
  --method stork2 \
  --native-steps 50 \
  --target-steps 20 \
  --device cuda
```

The script reports synthetic latency and real model-call counts with
`quality_validated: false`. Production acceptance still requires matched
audio A/B evaluation: real-time factor and p50/p95 latency, WER/CER, speaker
similarity, F0/duration error, mel distance, and representative languages and
utterance lengths.

### Model-specific, fail-closed adapter matrix

A sampler-level output may be velocity, denoised x0, or the next absolute
latent. Query the registered techniques instead of checking model names:

```python
from voicehub.optimization import get_diffusion_model_optimization_support

support = get_diffusion_model_optimization_support("vibevoice")
print(support.sampling_techniques)
# ('schedule', 'guidance', 'prediction-cache')
```

| Model | Step reduction | Guidance policy | Prediction cache | STORK-2 |
| --- | --- | --- | --- | --- |
| Chatterbox | Native Euler grid | Fixed native two-branch CFG only | Yes | No |
| CosyVoice | Native/uniform/quadratic/trailing | Limited/adaptive | Yes | Yes |
| Echo | Native/uniform/quadratic/trailing | Limited/adaptive; resets at KV/block boundaries | Yes | No |
| F5-TTS | Native/uniform/quadratic/trailing | Limited/adaptive | Yes | Euler only |
| Irodori-TTS | Native/uniform/quadratic/trailing | Limited/adaptive; semantic multi-CFG lanes | Yes | No |
| StyleTTS 2 | Active Karras sigma-grid compaction | Native hidden CFG | No; ADPM2 main/midpoint stay distinct | No |
| Supertonic | Discrete total-step reduction | Not applicable | No; output is the next absolute latent | No |
| VibeVoice | Rebuilt DPM timestep/sigma grid and history | Limited/adaptive | Yes; every DPM transition still executes | No |
| VoxCPM | Local native grid | Limited/adaptive | Yes | No; mean/delta conditioning and CFG-Zero boundaries |

StyleTTS 2 and Supertonic accept only `target_steps`. StyleTTS 2 counts active
ADPM2 transitions without replacing ADPM2. Supertonic rebuilds
`current_step/total_step` from zero instead of skipping an existing
recurrence. Required unsupported choices fail during resolution; automatic
choices retain the native sampler.

### Research-to-runtime gates

The controller deliberately exposes techniques, not paper names without a
compatible execution boundary. The image, video, and audio diffusion
literature was mapped to VoiceHub as follows:

| Research family | VoiceHub implementation | Compatibility gate |
| --- | --- | --- |
| Fewer model evaluations | `target_steps` plus native, uniform, quadratic, or trailing schedule reconstruction | The architecture must rebuild its complete grid and solver history |
| Stiff flow integration | STORK-2 with Taylor-1 velocity history | Direct deterministic velocity fields and one real evaluation per outer step only |
| Guidance pruning | Limited-interval and adaptive CFG | The sampler must expose separable native conditional/unconditional branches |
| Whole-prediction reuse | FORA-style periodic reuse, calibrated TeaCache, SmoothCache masks, and Taylor prediction | Final output semantics and request/CFG lanes must be architecture-owned |
| Internal DiT reuse | DBCache and first-block cache, with optional Taylor residual prediction | A reviewed repeated-block boundary plus sampler invalidation is required |
| Kernel execution | SDPA/FlashAttention selectors, fused Triton/CUDA operations, `torch.compile`, and fixed-shape CUDA Graphs | Exact mask, dtype, device, shape, and graph-capture contracts remain mandatory |

[DPM-Solver](https://arxiv.org/abs/2206.00927) and
[UniPC](https://arxiv.org/abs/2302.04867) demonstrate that formulation-aware
high-order methods can reach low NFE, but their diffusion parameterization and
history equations are not interchangeable with every flow-matching or
absolute-latent TTS head. VoiceHub therefore preserves VibeVoice's native
DPM-Solver++(2M) implementation and does not relabel a generic Euler
replacement as DPM-Solver or UniPC.

Likewise, [progressive distillation](https://arxiv.org/abs/2202.00512),
[Consistency Models](https://arxiv.org/abs/2303.01469), and
[Latent Consistency Models](https://arxiv.org/abs/2310.04378) require a
different trained model or a checkpoint-specific distillation/fine-tuning
stage. They are valid future training architectures, but not inference flags
that can safely transform an arbitrary released 50-step speech checkpoint.

## Recommended optimization boundaries

Diffusion TTS repeatedly calls a comparatively regular denoiser or velocity
estimator from a Python solver. The highest-value portable boundary is
usually that tensor graph, not text normalization, audio I/O, or a
data-dependent outer loop. This follows the same pattern demonstrated by
PyTorch's [diffusion optimization recipe](https://pytorch.org/blog/accelerating-generative-ai-3/)
and its newer
[`torch.compile` and Diffusers guide](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/):
compile a stable repeated region, use an efficient attention primitive where
semantics permit it, and measure cold-start separately from steady state.

| Family | Preferred repeated compile unit | Important boundary or cache |
| --- | --- | --- |
| Chatterbox | S3Gen conditional decoder/flow estimator | Keep T3, the S3 tokenizer, and HiFT stages separate; bucket mel lengths |
| CosyVoice | Flow `DiTEstimator` | Separate the LM, flow estimator, and HiFT graphs; cache masks and conditioning |
| Echo | `EchoDiT` denoiser | Cache condition keys/values and RoPE; pack the three CFG branches only after parity testing |
| F5-TTS | `F5DiT` / conditional-flow forward | Retain the existing text cache; bucket frame counts; resolve attention and fused activation selectors first |
| Irodori-TTS | `forward_with_encoded_conditions` | Preserve its context-key/value and RoPE caches; bucket patched latent lengths |
| StyleTTS 2 | Style denoiser | The style vector is small and the waveform decoder may dominate; retain relative-bias attention unless an exact replacement is verified |
| Supertonic | Static vector-estimator graph | Its PyTorch ONNX interpreter may graph-break; compile reviewed graph regions rather than pretending the Python interpreter is one tensor graph |
| VibeVoice | Acoustic diffusion head, separately from the LM | The repeated head is an MLP-heavy target; do not claim end-to-end inference while the public high-level path fails closed |
| VoxCPM | Local fixed-patch DiT | Keep the outer autoregressive loop outside capture; reuse the local condition and capture only stable inner shapes |

For model authors, `fullgraph=True` is useful during development because it
exposes graph breaks. For applications, regional compilation can reduce
cold-start cost. PyTorch's FLUX work documents both
[shape-specialized compilation](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
and a more aggressive
[FLUX optimization stack](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/).
Treat those as optimization patterns, not evidence that image-model kernels
can be copied into a speech graph unchanged.

### Approximate block-residual caching

VoiceHub includes opt-in, native DBCache and first-block-cache layouts.
Image DiTs often produce similar intermediate states at neighboring denoising
steps, and the same property can occur in repeated speech DiT or flow blocks.
The implementation does not import Diffusers or Cache-DiT and does not replace
a model's sampler. It applies the block-boundary algorithm described by the
official [Cache-DiT repository](https://github.com/vipshop/cache-dit),
[unified cache API](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/),
and
[DBCache design](https://cache-dit.readthedocs.io/en/latest/user_guide/DBCACHE_DESIGN/)
to architecture-owned VoiceHub block lists:

1. Always run the first `front_blocks` (`Fn`) blocks and compare their output
   with the previous probe using a normalized relative-L1 score.
2. On a required compute step, run the middle blocks and save their aggregate
   residual.
3. On a cache hit, skip those middle blocks and add either the latest residual
   (`predictor="reuse"`) or a first-, second-, or third-order extrapolation
   from fully computed residuals (`predictor="taylor"`). Predictions are never
   added to the Taylor history.
4. Always run the last `back_blocks` (`Bn`) blocks so the approximate state is
   refined before the model's output projection.

The Taylor predictor never chains predictions. A threshold miss always
refreshes it from an actual middle-block evaluation.

Caching is approximate and disabled by default. Setting `diffusion_cache` to
`"auto"` is an explicit request: it enables caching on declared adapters and
retains exact execution when the architecture cannot provide one.
`"required"` fails resolution instead of falling back.

```python
from voicehub.optimization import DiffusionCacheConfig, TTSOptimizationConfig

optimization = TTSOptimizationConfig(
    attn_implementation="auto",
    kernel_backend="auto",
    diffusion_cache="required",
    diffusion_cache_config=DiffusionCacheConfig(
        # "dbcache" or the constrained first-block preset "fbcache".
        method="dbcache",
        front_blocks=1,
        back_blocks=1,
        residual_diff_threshold=0.05,
        warmup_steps=3,
        warmup_interval=1,
        max_cached_steps=-1,
        max_consecutive_cached_steps=2,
        max_accumulated_relative_error=0.12,
        predictor="reuse",
        taylor_order=1,
        # True means "force a full middle-block evaluation" at that step.
        compute_step_mask=(True, True, False, False, True),
        compute_step_policy="dynamic",  # or explicit static-mask reuse
        num_inference_steps=5,
        force_refresh_step_hint=None,
        force_refresh_step_policy="once",  # or repeat
        probe_downsample_factor=1,
        metrics_history_size=256,
    ),
    compile="auto",
)
plan = optimization.resolve("f5tts", mode="inference")
```

`warmup_steps` and `warmup_interval` define the early full-compute cadence.
`max_cached_steps` limits cache hits across one lane (`-1` means unlimited), while
`max_consecutive_cached_steps` prevents long runs of predictions.
`max_accumulated_relative_error` forces refresh after the accepted probe
scores accumulate beyond its budget. `compute_step_mask` is an SCM-like
explicit full-compute schedule: `True` forces computation. With the default
`dynamic` policy, `False` still evaluates the threshold; with `static`, it
reuses a compatible cache entry without calculating the probe score. Static
mode is faster and more approximate. `num_inference_steps` automatically
starts a new segment for transformer-only callers, while the refresh hint can
invalidate once or periodically. `probe_downsample_factor` reduces decision
work without changing the cached residual. Distributed
execution takes the maximum probe score across ranks by default so every rank
makes the same decision.

`DiffusionCacheConfig.from_dict()` also accepts Cache-DiT names such as
`Fn_compute_blocks`, `Bn_compute_blocks`, `max_warmup_steps`,
`max_continuous_cached_steps`, `steps_computation_mask`, and
`taylorseer_order`. Supplying an alias and its VoiceHub name together is an
error.

Each sampler invalidates its state at request boundaries and when a
conditioning cache changes. Packed CFG uses its own lane; separate
conditional, unconditional, and alternate-unconditional calls use independent
lanes. A shape, dtype, device, or middle-block output mismatch invalidates or
bypasses the entry. Training mode, gradient-enabled calls, and unsupported
block layouts always execute every block. The cache owns no parameters,
buffers, or child modules, so enabling and restoring it does not change
checkpoint keys.

Every cache target reports aggregate and optional per-lane telemetry. The
summary includes dynamic/static hits, each miss reason, residual-difference
min/mean/p25/p50/p75/p95/max, Taylor fallback/order counts, executed/skipped
block evaluations, structural block-compute reduction, active/peak cache
bytes, and bounded step histories. Public `generate()` calls automatically
open a request session, including exception-safe cache release.

```python
from voicehub.optimization import diffusion_cache_summary, reset_diffusion_cache_metrics

reset_diffusion_cache_metrics(model.model)
audio = model.generate("A repeatable benchmark sentence.")
metrics = diffusion_cache_summary(model.model, details=True)
```

All nine active diffusion-family models declare an architecture-owned cache
adapter:

| Model type | Cache boundary | Status |
| --- | --- | --- |
| `chatterbox` | Repeated S3Gen `ConditionalDecoder.mid_blocks` with one packed-CFG lane | Supported |
| `cosyvoice` | Native `DiTEstimator` blocks with separate CFG lanes | Supported |
| `f5tts` | `F5DiT.transformer_blocks` | Supported |
| `echo` | `EchoDiT.blocks`, including sampler invalidation | Supported |
| `irodoritts` | RF-DiT blocks with packed and independent CFG lanes | Supported |
| `voxcpm` | Local DiT decoder layers inside the outer acoustic loop | Supported |
| `vibevoice` | Low-level `VibeVoiceDiffusionHead.layers` only | Experimental; the public high-level inference path still fails closed |
| `styletts2` | `Transformer1d`/`StyleTransformer1d` blocks with independent CFG and ADPM2-stage lanes | Supported |
| `supertonic` | Flattened ONNX vector-estimator boundary; caches `next_latent - current_latent` from a timestep-tagged current-latent probe | Supported; avoids reusing the absolute next latent |

Architecture registration, not a model-name allowlist, declares
`diffusion-cache`. Adding another adapter requires sampler-level invalidation,
CFG-lane coverage, exact cold-path parity, approximate quality evaluation,
and restoration/state-dict tests.

Optimization passes are ordered as architecture kernel selection, attention
selection, sampler acceleration, diffusion-cache enablement, and then
`torch.compile`. This lets the
compiler see the final block implementation. The cache decision itself is
request- and step-dependent Python control flow, so prefer regional
compilation of the repeated tensor blocks and `fullgraph=False`. Do not assume
that a CUDA graph can capture threshold decisions or changing cache lanes.
Cache-DiT likewise documents its
[compile integration](https://cache-dit.readthedocs.io/en/latest/user_guide/COMPILE/)
as a separate concern from cache policy.

### Attention

Prefer an architecture's verified
[`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
path. Select FlashAttention-4 only when the architecture declares the
`attention-backend` protocol; native SDPA support alone does not mean that
its masks, relative bias, dropout, grouped-query layout, or cross-attention
can be switched by an external selector. The
[official FlashAttention repository](https://github.com/Dao-AILab/flash-attention)
contains the current FA4 implementation and hardware scope.

Automatic attention selection must preserve a native or SDPA fallback.
An explicit FA4 request should fail before model mutation when its dtype,
device, head layout, or mask semantics are unsupported.

### Custom Triton and CUDA kernels

Use custom kernels for repeated, bandwidth-bound operations with a stable
contract, such as:

- adaptive-normalization modulation and residual gating;
- RMSNorm and SwiGLU/bias-GELU patterns;
- packed CFG combination plus an Euler update; and
- small solver-vector updates that otherwise launch several pointwise
  kernels.

A portable kernel operation needs a Torch reference, forward and backward
coverage where training uses it, fake/meta registration for compilation, and
Triton or CUDA implementations selected before graph capture. It must not add
parameters, buffers, or state-dict keys. `kernel_backend="auto"` retains
Torch; `triton` or `cuda_extension` is a strict, benchmark-driven request.

Do not label all nine architectures `custom-kernels` merely because a generic
kernel exists. An architecture opts in only after its active module implements
the structural selector protocol and parity tests cover its actual broadcast,
mask, dtype, empty-shape, and gradient cases.

### Static shapes and CUDA graphs

Variable text, mel, and codec-latent lengths are normal in TTS. Start with
dynamic compilation while collecting a length histogram. For repeated
serving traffic, pad to a small set of frame or latent buckets, retain masks,
and compile one graph per useful bucket.

CUDA graphs are appropriate only after batch size, sequence bucket, CFG
branch count, dtype, device, and control flow are stable. PyTorch's
[CUDAGraph Trees documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
explains why changing shapes cause re-recording and why static tensor
addresses matter. Use `mode="reduce-overhead"` for a measured, graph-safe
small-batch inference region. Prefer
`mode="max-autotune-no-cudagraphs"` for dynamic training unless the complete
training step has been proven capture-safe.

The official
[`torch.compile` reference](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
documents the mode tradeoffs. Always report compilation/warm-up latency
separately from steady-state latency.

## External diffusion serving

An engine that accelerates image or video diffusion is not automatically a
TTS server. VoiceHub exposes dependency-free capability records and resolves
only complete pipelines that own text/speaker conditioning, the denoising or
flow loop, and waveform or codec/vocoder output:

```python
from voicehub.llm_serving import LLMBackendConfig
from voicehub.diffusion_serving import bridge_vllm_omni_tts_config, list_diffusion_serving_capabilities, resolve_diffusion_tts_backend

for capability in list_diffusion_serving_capabilities():
    print(
        capability.backend.value,
        capability.diffusion_modalities,
        capability.supports_tts_diffusion,
        capability.verified_tts_models,
    )

plan = resolve_diffusion_tts_backend("cosyvoice", "vllm-omni")
plan, speech_config = bridge_vllm_omni_tts_config(
    "cosyvoice",
    LLMBackendConfig(
        backend="vllm",
        endpoint="http://127.0.0.1:8000",
        model="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    ),
)
assert speech_config.transport.value == "speech"
```

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) is a multimodal engine
with heterogeneous stages and a documented
[diffusion runtime](https://docs.vllm.ai/projects/vllm-omni/en/latest/).
VoiceHub verifies its complete speech bridge only for `cosyvoice` and
`voxcpm`; the bridge reuses the existing `/v1/audio/speech` client rather than
pretending that a DiT-only endpoint returns audio.

Verified pairings are discovered from the registered architecture features
`diffusion-serving-native` and `diffusion-serving-vllm-omni`. An extension can
therefore declare an evidenced pairing beside its architecture integration
without adding a provider-name branch to the shared serving resolver.

The native serving capability similarly excludes `vibevoice`: VoiceHub
exposes and optimizes its low-level diffusion head, but the public high-level
TTS generation path intentionally still fails closed pending parity.

Other models may use the experimental `VLLMOmniDiffusionPlugin` contract only
when an installed engine exposes its public out-of-tree
`register_diffusion_model` hook and the plugin explicitly declares a complete
TTS pipeline with post-processing. `detect_vllm_omni_features()` probes that
optional API lazily. Merely importing `voicehub.diffusion_serving` imports
neither vLLM-Omni nor SGLang.

[SGLang Diffusion](https://github.com/sgl-project/sglang) applies image-side
techniques such as dynamic batching, distributed sequence parallelism,
specialized attention, and Cache-DiT integration, as described in the
official
[SGLang Diffusion announcement](https://www.lmsys.org/blog/2026-01-16-sglang-diffusion/).
Its public diffusion request/output contract is for image and video, not
audio, so `resolve_diffusion_tts_backend(..., "sglang-diffusion")` fails
closed. SGLang-Omni is represented separately: its verified LLM-TTS support
belongs to `voicehub.llm_serving` and must not be relabeled as SGLang
Diffusion support.

## Semantic-preserving and approximate changes

VoiceHub's automatic system is conservative. “Exact” below means the same
model, objective, solver, masks, and checkpoint semantics within an
appropriate floating-point tolerance; it does not promise bitwise-identical
GPU results.

| Change | Safety class | Required evidence |
| --- | --- | --- |
| Compile the same denoiser/estimator | Semantic-preserving | Eager/compiled forward parity, backward parity for training, real target coverage, fallback behavior |
| Use a declared SDPA or selectable attention path | Semantic-preserving | Exact mask/bias/causal/dropout semantics and dtype/device fallback |
| Replace a pointwise sequence with a registered fused kernel | Semantic-preserving | Torch reference parity, gradients, non-contiguous/broadcast/empty cases, unchanged state dict |
| Cache fixed conditioning, RoPE, masks, or key/value projections | Semantic-preserving | Cache invalidation and parity when any conditioning input changes |
| Pad into static buckets | Semantic-preserving | Padded values are fully masked and output is trimmed to the original length |
| Pack CFG branches into the batch dimension | Semantic-preserving | Branch ordering, guidance equation, masks, and random inputs match the separate calls |
| Reuse or extrapolate a middle-block residual across denoising steps | Approximate | Per-model latency/quality curve, request and CFG invalidation, forced-compute parity, and cache-hit statistics |
| Use float16/bfloat16, TF32, or fast-math kernels | Numerically relaxed | Per-model audio/latent metrics, overflow tests, and quality evaluation |
| Reduce function evaluations or change timestep spacing | Approximate | Latency-quality curve for the exact checkpoint and conditioning modes |
| Replace Euler, midpoint, ADPM2, or DPM-Solver++ with another solver | Approximate | Source-backed conversion and model-level perceptual/regression evaluation |
| Change CFG scale, guidance rescaling, or CFG-Zero* behavior | Model behavior change | Explicit user choice and quality/speaker-similarity evaluation |
| Quantize, prune, or distill the estimator | Approximate/model change | Calibration or retraining plus task-specific quality evaluation |

[DPM-Solver](https://arxiv.org/abs/2206.00927) and
[DPM-Solver++](https://arxiv.org/abs/2211.01095) show how specialized
high-order solvers can reduce neural-function evaluations. They do not imply
that a solver can be substituted into an arbitrary flow-TTS checkpoint
without validating its prediction parameterization, time schedule, guidance,
and training distribution.

## Why other registered TTS models are excluded

The inventory follows the executed public graph:

- `gptsovits` currently runs its S1 semantic model and classic VITS S2 graph.
  The unsupported V3/V4 flow/vocoder variants fail closed.
- `qwen3tts` uses VoiceHub's active 12 Hz convolution/Transformer codec
  decoder. A vendored 25 Hz DiT implementation is not reached by the public
  runtime.
- `kokoro` does not expose the unreleased style-diffusion graph in its active
  reconstructed runtime.
- `omnivoice` uses “denoise” as a control token or mode, not an iterative
  diffusion sampler.
- `xtts` does not reach the vendored Tortoise diffusion implementation from
  its registered XTTS execution path.
- VITS-family normalizing flows are invertible coupling transforms, not
  diffusion or flow-matching samplers.

This exclusion rule also applies to future vendored code: add
`diffusion-family` only when the registered model executes and tests the
graph. For WaveNet-gate and VITS-specific optimization, use the
[VITS-family optimization guide](vits-optimization.md). For the universal
resolver and lifecycle, use [TTS optimization](tts-optimization.md).
