---
description: Add one optimization pass that is immediately available to every speech model.
---

# Add an optimization

An optimization is one class plus one registry call. Every model that inherits
the VoiceHub speech base classes sees the new pass automatically. The pass
checks the loaded runtime before it changes anything.

## Implement the pass

```python
from voicehub.optimization import (
    OptimizationCapabilities,
    OptimizationMode,
    OptimizationPass,
    PassResult,
    register_optimization_pass,
)


@register_optimization_pass("acme-eval-mode")
class AcmeEvalModePass(OptimizationPass):
    pass_id = "acme.eval-mode"
    pass_version = "1"
    capabilities = OptimizationCapabilities(
        modes=(OptimizationMode.INFERENCE,),
        reversible=True,
    )

    def manifest_configuration(self):
        return {}

    def validate(self, model, context):
        super().validate(model, context)

    def apply(self, model, context):
        if not callable(getattr(model, "eval", None)):
            return self.not_applicable_result(
                model,
                reason=f"{type(model).__name__} has no eval() method",
            )
        was_training = bool(getattr(model, "training", False))
        model.eval()
        return PassResult(
            model=model,
            state={"was_training": was_training},
            metadata={"outcome": "configured"},
        )

    def restore(self, model, state, context):
        if state.get("kind") == "not-applicable":
            return state.get("model", model)
        model.train(state["was_training"])
        return model
```

A class is callable, so the decorator stores it as a lazy factory. A function
that returns a configured pass works too.

## Apply it to any task

```python
from voicehub import AutoModel

model = AutoModel.from_pretrained(checkpoint, model_type=model_type)
result = model.apply_optimization_plan("acme-eval-mode", mode="inference")
print(result.manifest())
```

The same method exists on TTS, ASR, and VAD wrappers. A public pass must have a
tested path for every registered model. If the relevant protocol is absent,
return `not_applicable_result()` so the manifest explicitly records an
unchanged model, `outcome="not-applicable"`, and an actionable reason. Do not
silently skip the model or describe that result as acceleration. A present but
malformed protocol, an explicit backend that cannot run, and unsupported
hardware must still fail before mutation. Earlier reversible passes roll back
if a later pass fails.

## Declare capabilities honestly

`OptimizationCapabilities` describes execution constraints:

- `modes`: inference, training, or both;
- `devices` and `dtypes`: supported runtime values;
- `streaming_safe` and `distributed_safe`: concurrency guarantees;
- `persistent`: whether transformed state may be checkpointed;
- `reversible`: whether `restore()` is implemented;
- topology flags: whether parameter names or structure change.

Set `requires_architecture_support = True` only when the pass relies on a
manually audited architecture contract that runtime inspection cannot prove.
Most extension passes should validate a protocol or module surface directly,
which avoids editing every model when the pass is added.

## Test the full lifecycle

Test that:

1. registration is lazy;
2. every registered model either configures the pass or reports an explicit
   model-preserving `not-applicable` fallback;
3. malformed protocols and required unsupported hardware fail before mutation;
4. application produces deterministic strict-JSON manifest metadata;
5. normalized task outputs and checkpoint keys remain semantically stable; and
6. restoration returns the original runtime and state keys.

Use an isolated `OptimizationPassRegistry` in unit tests when global
registration is unnecessary. See [Library architecture](../concepts/architecture.md)
for transaction and lifecycle details.

## Document the pass

Add one `OptimizationGuide` entry to
`scripts/documentation_references.py`. Include the official GitHub repository,
the dedicated primary paper when one exists, the pass ID and version, and the
VoiceHub implementation path. Use an empty paper tuple when upstream has not
published a paper.

```bash
python scripts/generate_optimization_pages.py
python scripts/generate_optimization_pages.py --check
```

The generator creates the dedicated page and left-sidebar link. Its coverage
check fails when a registered public pass is missing or an unknown pass is
documented as registered.
