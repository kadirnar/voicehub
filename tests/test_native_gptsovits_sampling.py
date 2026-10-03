"""Regression tests: native GPT-SoVITS S1 sampling matches the released
sampler.

The reference below is a verbatim transcription of upstream
``GPT_SoVITS/AR/models/utils.py`` (``logits_to_probs`` and
``multinomial_sample_one_no_sync``) at RVC-Boss/GPT-SoVITS
d523079fc05d9a8028d6085bffe4a2757c32abb6.
"""

from __future__ import annotations

import unittest

import torch

from voicehub.architectures.gptsovits.semantic import GPTSoVITSSemanticModel, _sample_next_token


def _upstream_sample(logits, previous_tokens, *, top_k, top_p, temperature, repetition_penalty):
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(dim=1, index=previous_tokens, src=score)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


class GPTSoVITSSamplingParityTests(unittest.TestCase):

    def test_sampler_matches_upstream_tokens_rng_and_penalized_logits(self):
        settings = [
            dict(top_k=15, top_p=1.0, temperature=1.0, repetition_penalty=1.35),
            dict(top_k=15, top_p=0.8, temperature=0.7, repetition_penalty=1.35),
            dict(top_k=5, top_p=0.3, temperature=1.6, repetition_penalty=1.0),
            dict(top_k=1, top_p=1.0, temperature=1.0, repetition_penalty=1.35),
            dict(top_k=50, top_p=0.95, temperature=0.0, repetition_penalty=2.0),
        ]
        source = torch.Generator().manual_seed(0)
        for options in settings:
            for trial in range(20):
                logits = torch.randn(1, 1_025, generator=source) * 3
                previous = torch.randint(0, 1_024, (1, 40), generator=source)
                expected_logits, actual_logits = logits.clone(), logits.clone()
                torch.manual_seed(trial)
                expected = _upstream_sample(expected_logits, previous, **options)
                expected_next = torch.rand(1)
                torch.manual_seed(trial)
                actual = _sample_next_token(actual_logits, previous, **options)
                actual_next = torch.rand(1)
                with self.subTest(options=options, trial=trial):
                    self.assertEqual(actual.tolist(), expected.tolist())
                    self.assertEqual(actual.dtype, torch.long)
                    # Same random-number consumption as the released sampler.
                    self.assertTrue(torch.equal(actual_next, expected_next))
                    # The greedy EOS check reads the in-place penalized scores.
                    self.assertTrue(torch.equal(actual_logits, expected_logits))

    def test_top_p_precedes_temperature_and_uses_unshifted_mask(self):
        # Untempered probabilities ~[0.665, 0.245, 0.090]; with top_p=0.6 the
        # released sampler keeps only the first token. A temperature-first or
        # shifted-mask filter would keep more than one candidate.
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        tokens = set()
        for seed in range(64):
            torch.manual_seed(seed)
            tokens.add(
                _sample_next_token(
                    logits.clone(),
                    torch.empty(1, 0, dtype=torch.long),
                    top_k=3,
                    top_p=0.6,
                    temperature=5.0,
                    repetition_penalty=1.0,
                ).item())
        self.assertEqual(tokens, {0})

    def test_generate_accepts_upstream_temperature_and_top_p_bounds(self):
        torch.manual_seed(0)
        model = GPTSoVITSSemanticModel().eval()
        for temperature, top_p in ((0.0, 1.0), (1.0, 0.0)):
            generated = model.generate(
                phoneme_ids=torch.tensor([[1, 2, 3]]),
                phoneme_lengths=torch.tensor([3]),
                bert_features=torch.zeros(1, 1_024, 3),
                prompt_semantic_ids=torch.tensor([[4, 5]]),
                top_k=5,
                top_p=top_p,
                temperature=temperature,
                maximum_new_tokens=2,
            )
            self.assertEqual(generated.shape[0], 1)
            self.assertLessEqual(generated.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
