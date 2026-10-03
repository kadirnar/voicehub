"""Beam-search parity with ESPnet 0.8.0 (espnet/espnet@75db853).

The reference below is a line-by-line transcription of the upstream
``espnet.nets.beam_search.BeamSearch`` (``forward``/``search``/``beam``/
``post_process``), ``espnet.nets.e2e_asr_common.end_detect`` and the numpy
``espnet.nets.ctc_prefix_score.CTCPrefixScore`` as configured by
``espnet2/bin/asr_inference.py`` (no pre-beam, full-vocabulary CTC). It
drives the same tiny native graph so search semantics can be compared
without ESPnet installed.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from voicehub.architectures.espnet_transformer.configuration import ESPnetLibriSpeechTransformerConfig
from voicehub.architectures.espnet_transformer.decoding import (
    ESPnetCTCPrefixScorer,
    ESPnetJointBeamSearch,
    _end_detect,
    _Hypothesis,
)
from voicehub.architectures.espnet_transformer.modeling import (
    ESPnetLibriSpeechTransformerForASR,
    ESPnetSequentialRNNLanguageModel,
)

_LOGZERO = -10000000000.0


def _config(**overrides) -> ESPnetLibriSpeechTransformerConfig:
    values = dict(
        variant="custom",
        n_fft=16,
        win_length=16,
        hop_length=4,
        n_mels=13,
        vocabulary_size=8,
        blank_token_id=0,
        unknown_token_id=1,
        sos_eos_token_id=7,
        encoder_dimension=8,
        encoder_attention_heads=2,
        encoder_linear_units=16,
        encoder_blocks=1,
        decoder_attention_heads=2,
        decoder_linear_units=16,
        decoder_blocks=1,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        apply_spec_augment=False,
        language_model_layers=1,
        language_model_units=8,
        language_model_dropout=0.0,
        beam_size=3,
        ctc_weight=0.3,
        language_model_weight=0.6,
    )
    values.update(overrides)
    return ESPnetLibriSpeechTransformerConfig(**values)


class _UpstreamCTCPrefixScore:
    """``espnet/nets/ctc_prefix_score.py::CTCPrefixScore`` (numpy)."""

    def __init__(self, x, blank, eos):
        self.logzero = _LOGZERO
        self.blank = blank
        self.eos = eos
        self.input_length = len(x)
        self.x = x

    def initial_state(self):
        r = np.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def __call__(self, y, cs, r_prev):
        output_length = len(y) - 1
        r = np.full((self.input_length, 2, len(cs)), self.logzero, dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            r[output_length - 1] = self.logzero
        r_sum = np.logaddexp(r_prev[:, 0], r_prev[:, 1])
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = np.ndarray((self.input_length, len(cs)), dtype=np.float32)
            for i in range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum
        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in range(start, self.input_length):
            r[t, 0] = np.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = np.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.x[t, self.blank]
            log_psi = np.logaddexp(log_psi, log_phi[t - 1] + xs[t])
        eos_pos = np.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]
        blank_pos = np.where(cs == self.blank)[0]
        if len(blank_pos) > 0:
            log_psi[blank_pos] = self.logzero
        return log_psi, np.rollaxis(r, 2)


def _upstream_end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[0]
    for m in range(M):
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x["yseq"]) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x["score"], reverse=True)[0]
            if best_hyp_same_length["score"] - best_hyp["score"] < D_end:
                count += 1
    return count == M


def _upstream_beam_search(model, language_model, config, x, *, beam_size, maxlenratio=0.0):
    """``BeamSearch.forward`` with decoder, CTC and LM scorers."""
    vocabulary = config.vocabulary_size
    sos = eos = config.sos_eos_token_id
    weights = {
        "decoder": 1.0 - config.ctc_weight,
        "ctc": config.ctc_weight,
        "lm": config.language_model_weight,
    }
    logp = model.ctc.ctc_lo(x).log_softmax(dim=-1).numpy()
    ctc = _UpstreamCTCPrefixScore(logp, config.blank_token_id, eos)
    use = {name for name, weight in weights.items() if weight != 0}

    def full_scores(hyp):
        scores, states = {}, {}
        if "decoder" in use:
            scores["decoder"] = model.decoder.score(hyp["yseq"], x)
            states["decoder"] = None
        if "lm" in use:
            values, state = language_model.score(hyp["yseq"][-1], hyp["states"].get("lm"))
            scores["lm"], states["lm"] = values[0], state
        return scores, states

    running = [{
        "yseq": torch.tensor([sos]),
        "score": torch.tensor(0.0),
        "states": {
            "ctc": (np.float32(0.0), ctc.initial_state())
        },
    }]
    ended = []
    maxlen = x.shape[0] if maxlenratio == 0 else max(1, int(maxlenratio * x.shape[0]))
    for i in range(maxlen):
        best = []
        part_ids = torch.arange(vocabulary)
        for hyp in running:
            scores, states = full_scores(hyp)
            part_scores, part_states = {}, {}
            if "ctc" in use:
                prev_score, state = hyp["states"]["ctc"]
                presub, new_state = ctc(hyp["yseq"].numpy(), part_ids.numpy(), state)
                part_scores["ctc"] = torch.as_tensor(presub - prev_score)
                part_states["ctc"] = (presub, new_state)
            weighted = torch.zeros(vocabulary)
            for name in ("decoder", "lm"):
                if name in scores:
                    weighted += weights[name] * scores[name]
            if "ctc" in part_scores:
                weighted[part_ids] += weights["ctc"] * part_scores["ctc"]
            weighted += hyp["score"]
            top_ids = weighted.topk(beam_size)[1]
            for j in top_ids.tolist():
                new_states = dict(states)
                if "ctc" in part_states:
                    new_states["ctc"] = (part_states["ctc"][0][j], part_states["ctc"][1][j])
                best.append({
                    "yseq": torch.cat((hyp["yseq"], torch.tensor([j]))),
                    "score": weighted[j],
                    "states": new_states,
                })
            best = sorted(best, key=lambda h: h["score"], reverse=True)[:min(len(best), beam_size)]
        if i == maxlen - 1:
            best = [dict(h, yseq=torch.cat((h["yseq"], torch.tensor([eos])))) for h in best]
        running = []
        for hyp in best:
            (ended if int(hyp["yseq"][-1]) == eos else running).append(hyp)
        if maxlenratio == 0.0 and _upstream_end_detect(
            [{
                "yseq": h["yseq"].tolist(),
                "score": float(h["score"])
            } for h in ended],
                i,
        ):
            break
        if not running:
            break
    nbest = sorted(ended, key=lambda h: h["score"], reverse=True)
    hyp = nbest[0]
    tokens = [t for t in hyp["yseq"][1:-1].tolist() if t != 0]
    return tuple(tokens), float(hyp["score"])


class ESPnetUpstreamBeamSearchParityTests(unittest.TestCase):

    def _graph(self, seed, **overrides):
        torch.manual_seed(seed)
        config = _config(**overrides)
        model = ESPnetLibriSpeechTransformerForASR(config).eval()
        language_model = ESPnetSequentialRNNLanguageModel(config).eval()
        with torch.no_grad():
            # Sharpen the random heads so searches produce non-trivial,
            # seed-dependent hypotheses instead of an immediate EOS.
            model.ctc.ctc_lo.weight.mul_(6.0)
            model.decoder.output_layer.weight.mul_(4.0)
            language_model.decoder.weight.mul_(3.0)
        return config, model, language_model

    def _assert_matches_upstream(self, seed, *, beam_size, frames, **overrides):
        config, model, language_model = self._graph(seed, **overrides)
        generator = torch.Generator().manual_seed(seed + 1000)
        memory = torch.randn(frames, config.encoder_dimension, generator=generator)
        with torch.inference_mode():
            expected_tokens, expected_score = _upstream_beam_search(
                model,
                language_model,
                config,
                memory,
                beam_size=beam_size,
                maxlenratio=config.maximum_decode_ratio,
            )
            decoded = ESPnetJointBeamSearch(model, config, language_model=language_model)(
                memory.unsqueeze(0),
                torch.tensor([frames]),
                beam_size=beam_size,
            )
        tokens = tuple(token for token in decoded.token_ids[0] if token != config.blank_token_id)
        self.assertEqual(tokens, expected_tokens, f"seed={seed}")
        self.assertAlmostEqual(decoded.scores[0], expected_score, places=3, msg=f"seed={seed}")

    def test_search_matches_upstream_beam_search_with_end_detection(self):
        for seed in range(24):
            with self.subTest(seed=seed):
                self._assert_matches_upstream(seed, beam_size=3, frames=12 + seed % 7)

    def test_search_matches_upstream_with_fixed_length_ratio(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                self._assert_matches_upstream(seed, beam_size=2, frames=10, maximum_decode_ratio=0.5)

    def test_zero_weight_scorers_are_not_evaluated_like_upstream(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                self._assert_matches_upstream(seed, beam_size=3, frames=11, ctc_weight=0.0)
                self._assert_matches_upstream(seed, beam_size=3, frames=11, language_model_weight=0.0)

    def test_batched_ctc_extension_matches_single_prefix_and_upstream(self):
        generator = torch.Generator().manual_seed(7)
        values = torch.randn(9, 6, generator=generator).log_softmax(dim=-1)
        scorer = ESPnetCTCPrefixScorer(values, blank_token_id=0, eos_token_id=5)
        upstream = _UpstreamCTCPrefixScore(values.numpy(), 0, 5)
        candidates = torch.arange(6)
        first, states = scorer.extend((5, ), candidates, scorer.initial_state)
        up_first, up_states = upstream(np.array([5]), candidates.numpy(), upstream.initial_state())
        np.testing.assert_allclose(first.numpy(), up_first, rtol=1e-5, atol=1e-4)
        prefixes = torch.tensor([[5, 2], [5, 3]])
        previous = torch.stack((states[2], states[3]))
        batch_scores, batch_states = scorer.extend_batch(prefixes, candidates.expand(2, -1), previous)
        for row, token in enumerate((2, 3)):
            single, single_states = scorer.extend((5, token), candidates, states[token])
            torch.testing.assert_close(batch_scores[row], single)
            torch.testing.assert_close(batch_states[row], single_states)
            reference, _ = upstream(np.array([5, token]), candidates.numpy(), up_states[token])
            np.testing.assert_allclose(single.numpy(), reference, rtol=1e-5, atol=1e-3)

    def test_frontend_builds_the_stft_window_on_the_input_device(self):
        # ESPnet's Stft calls torch.hann_window(win_length, dtype, device)
        # per forward; a CPU-built window moved to CUDA differs by one ulp.
        from unittest import mock

        from voicehub.architectures.espnet_transformer.frontend import ESPnetDefaultFrontend

        frontend = ESPnetDefaultFrontend(_config())
        self.assertNotIn("_window", dict(frontend.named_buffers()))
        calls = []
        original = torch.hann_window

        def recording_window(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        with mock.patch.object(torch, "hann_window", recording_window):
            frontend(torch.randn(1, 64, dtype=torch.float64).float())
        self.assertEqual(calls, [{"dtype": torch.float32, "device": torch.device("cpu")}])

    def test_end_detect_matches_upstream(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            ended = []
            for _ in range(int(rng.integers(0, 6))):
                length = int(rng.integers(2, 9))
                ended.append(_Hypothesis(
                    tokens=tuple(range(length)),
                    score=float(rng.normal(-20.0, 8.0)),
                    ctc_state=torch.zeros(1, 2),
                    ctc_score=0.0,
                    lm_state=None,
                ))
            step = int(rng.integers(0, 10))
            expected = _upstream_end_detect([{
                "yseq": list(value.tokens),
                "score": value.score
            } for value in ended], step)
            self.assertEqual(_end_detect(ended, step), expected)


if __name__ == "__main__":
    unittest.main()
