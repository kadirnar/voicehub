"""Regression tests: native GPT-SoVITS S1 decodes with a key/value cache.

The released ``Text2SemanticDecoder.infer_panel_naive`` encodes the text
and prompt once (``T2STransformer.process_prompt``) and then feeds only the
newest semantic token per step (``decode_next_token``). Re-running the whole
``nn.TransformerEncoder`` over the growing sequence every step gives the
same tokens but costs quadratic time in the generated length.
"""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional

from voicehub.architectures.gptsovits.configuration import GPTSoVITSS1Config
from voicehub.architectures.gptsovits.semantic import GPTSoVITSSemanticModel, _sample_next_token


def _small_model(seed: int) -> GPTSoVITSSemanticModel:
    torch.manual_seed(seed)
    config = GPTSoVITSS1Config(
        embedding_dim=64,
        hidden_dim=64,
        attention_heads=4,
        layers=3,
        bert_feature_dim=16,
    )
    return GPTSoVITSSemanticModel(config).eval()


def _inputs(prompt: bool) -> dict:
    source = torch.Generator().manual_seed(7)
    return {
        "phoneme_ids": torch.randint(0, 732, (1, 9), generator=source),
        "phoneme_lengths": torch.tensor([9]),
        "bert_features": torch.randn(1, 16, 9, generator=source),
        "prompt_semantic_ids": (torch.randint(0, 1_024, (1, 5), generator=source) if prompt else None),
    }


@torch.no_grad()
def _full_recompute_generate(decoder, *, phoneme_ids, phoneme_lengths, bert_features, prompt_semantic_ids, steps):
    """Reference: re-encode the full text + semantic sequence every step."""
    del phoneme_lengths
    generated = (
        torch.empty(1, 0, dtype=torch.long) if prompt_semantic_ids is None else prompt_semantic_ids.clone())
    prefix = generated.shape[1]
    text = decoder._text_hidden(phoneme_ids, bert_features)
    for step in range(steps):
        semantic = decoder.ar_audio_position(decoder.ar_audio_embedding(generated))
        hidden = torch.cat([text, semantic], dim=1)
        text_steps, semantic_steps = text.shape[1], generated.shape[1]
        text_attention = functional.pad(
            torch.zeros(text_steps, text_steps, dtype=torch.bool),
            (0, semantic_steps),
            value=True,
        )
        semantic_attention = functional.pad(
            torch.triu(torch.ones(semantic_steps, semantic_steps, dtype=torch.bool), diagonal=1),
            (text_steps, 0),
        )
        attention = torch.cat([text_attention, semantic_attention])
        logits = decoder.ar_predict_layer(decoder.h(hidden, mask=attention)[:, -1])
        if step < 11:
            logits = logits[:, :-1]
        sample = _sample_next_token(
            logits,
            generated,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.35,
        )
        generated = torch.cat([generated, sample], dim=1)
    return generated[:, prefix:]


class GPTSoVITSS1CacheTests(unittest.TestCase):

    def test_decode_steps_embed_only_the_newest_token(self):
        model = _small_model(0)
        widths = []
        model.model.ar_audio_embedding.register_forward_hook(
            lambda module, args, output: widths.append(args[0].shape[1]))
        generated = model.generate(**_inputs(prompt=True), top_k=1, maximum_new_tokens=8)
        self.assertEqual(generated.shape, (1, 8))
        # The prompt is embedded once; every later step adds one position.
        self.assertEqual(widths[0], 5)
        self.assertEqual(set(widths[1:]), {1})

    def test_cached_decode_matches_full_recompute(self):
        for seed in range(3):
            for prompt in (True, False):
                model = _small_model(seed)
                inputs = _inputs(prompt)
                with self.subTest(seed=seed, prompt=prompt):
                    expected = _full_recompute_generate(model.model, **inputs, steps=10)
                    actual = model.generate(**inputs, top_k=1, maximum_new_tokens=10)
                    self.assertEqual(actual.tolist(), expected.tolist())

    def test_cached_hidden_states_match_encoder(self):
        model = _small_model(1)
        decoder = model.model
        hidden = torch.randn(1, 12, 64)
        attention = torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)
        attention[:4, 4:] = True
        attention[4:, :4] = False
        with torch.no_grad():
            expected = decoder.h(hidden, mask=attention)
            cache = []
            prompt = decoder._decode(hidden[:, :8], attention[:8, :8], cache)
            steps = [decoder._decode(hidden[:, index:index + 1], None, cache) for index in range(8, 12)]
        actual = torch.cat([prompt, *steps], dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual([key.shape[1] for key, _ in cache], [12] * 3)


if __name__ == "__main__":
    unittest.main()
