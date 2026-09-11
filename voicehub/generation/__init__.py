"""Lazy public imports; importing this package never loads a model."""

from importlib import import_module

_EXPORTS = {
    **dict.fromkeys(
        "GenerationConfig".split(),
        "voicehub.generation.config",
    ),
    **dict.fromkeys(
        "CTCDecodeResult CTCTokenSpan ctc_forced_alignment ctc_greedy_decode ctc_prefix_beam_search".split(),
        "voicehub.generation.ctc",
    ),
    **dict.fromkeys(
        "AutoregressiveGenerator DecoderStep GenerationOutput GenerationStepInput GenerationStepOutput LogitsProcessor".split(
        ),
        "voicehub.generation.engine",
    ),
    **dict.fromkeys(
        "apply_repetition_penalty filter_min_p filter_top_k filter_top_p process_logits".split(),
        "voicehub.generation.logits",
    ),
    **dict.fromkeys(
        "create_generator sample_next_token".split(),
        "voicehub.generation.sampling",
    ),
    **dict.fromkeys(
        "EosStoppingCriterion StoppingCriterion evaluate_stopping_criteria tokens_match_any".split(),
        "voicehub.generation.stopping",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
