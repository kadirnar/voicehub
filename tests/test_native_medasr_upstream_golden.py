"""Golden-value regression test against the upstream LASR (MedASR) runtime.

The expected values were produced by ``transformers`` at commit
``65dc261512cbdb1ee72b88ae5b222f2605aad8e5`` (the revision pinned by the
``google/medasr`` model card) with ``LasrFeatureExtractor`` and
``LasrForCTC`` (eager attention) in float64 on CPU, for an RNG-free tiny
graph and a synthetic chirp. The generator is
``/workspace/parity/recipes/asr_medasr_golden.py`` of the upstream-
parity audit; it builds the same parameters as ``_deterministic_state``
below. VoiceHub must reproduce the frontend and the graph exactly
(padding mask, RoPE, Conformer residual weights, BatchNorm, CTC head),
so these checks need no network, GPU or upstream package.
"""

from __future__ import annotations

import math
import unittest

import torch

from voicehub.architectures.medasr.configuration import MedASRConfig
from voicehub.architectures.medasr.frontend import MedASRFeatureExtractor
from voicehub.architectures.medasr.modeling import MedASRForCTC

UPSTREAM_FEATURE_SHAPE = (1, 38, 128)
UPSTREAM_FEATURE_SUM = -43210.43964332342
UPSTREAM_FEATURE_SQUARE_SUM = 505019.02588842076
UPSTREAM_FEATURE_PROBES = {
    (0, 0): -11.512925148010254,
    (5, 17): -3.382408857345581,
    (20, 64): 2.5997061729431152,
    (37, 127): -11.512925148010254,
}
UPSTREAM_LOGITS = (
    (
        0.9344340398309, -0.3658915551504, -1.04679502141, -0.003751174078258, 1.093770322824, 0.4666534622,
        -0.924012766515, -0.8907799958112, 0.5071495885417, 1.061220730824, -0.07533893627613,
        -1.054897144051),
    (
        -0.112645220273, -0.913160303007, -0.2219121895524, 0.8784280303731, 0.6270608232589,
        -0.6050143365005, -0.8924108250703, 0.1937184948631, 0.9158645836532, 0.1426645108148,
        -0.8569967064174, -0.4537015726726),
    (
        -0.5192692451268, -0.6448725185401, 0.2936392767026, 0.8194591487562, 0.08756741933247,
        -0.7650850259648, -0.4179076766118, 0.5464420554674, 0.584570642959, -0.3445676078125,
        -0.7235237585276, 0.08772187049594),
    (
        -0.2615768039059, -0.5812023405727, 0.06179755440825, 0.6616589881117, 0.2553407411855,
        -0.5391672626304, -0.4939561859518, 0.2896478721195, 0.5563582101811, -0.09922794689293,
        -0.5957010307188, -0.1057205203292),
    (
        -0.2844190476533, -0.5162325773633, 0.1110181545206, 0.6166732671352, 0.1878555013671,
        -0.5215812062484, -0.419330835786, 0.3023604328171, 0.4868942835484, -0.1401435369675,
        -0.5428492318402, -0.04334658490943),
    (
        -0.2197850241276, -0.4779923703623, 0.06191002932835, 0.5584946814334, 0.213342554908,
        -0.4530546309128, -0.4169954353745, 0.2347820557041, 0.4571214024109, -0.08465324132684,
        -0.4905467467069, -0.07760156488991),
    (
        -0.1353809824637, -0.5290059868947, -0.04320603572394, 0.5668301509181, 0.3218429013362,
        -0.4173379054553, -0.5109944460471, 0.1609008208318, 0.52112391075, 0.01521363227187,
        -0.5140023292299, -0.1869916362371),
)


def _config() -> MedASRConfig:
    return MedASRConfig(
        variant="custom",
        vocab_size=12,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        conv_kernel_size=4,
        subsampling_conv_channels=8,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=128,
    )


def _deterministic_state(model: MedASRForCTC) -> dict[str, torch.Tensor]:
    state = {}
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    for index, (name, shape) in enumerate(sorted(shapes.items())):
        count = math.prod(shape)
        base = torch.sin(torch.arange(count, dtype=torch.float64) * 0.7 + 1.3 * index).reshape(shape)
        if name.endswith("num_batches_tracked"):
            state[name] = torch.tensor(0)
        elif name.endswith("running_var"):
            state[name] = 1.0 + 0.5 * base.square()
        elif name.endswith("running_mean"):
            state[name] = 0.1 * base
        elif len(shape) == 1 and "norm" in name:
            state[name] = 1.0 + 0.1 * base
        elif len(shape) == 1:
            state[name] = 0.05 * base
        else:
            state[name] = base / math.sqrt(count / shape[0])
    return state


def _waveform() -> torch.Tensor:
    n = torch.arange(6400, dtype=torch.float64)
    value = (0.5 * torch.sin(2 * math.pi * (200 + 1500 * n / 6400) * n / 16000) + 0.05 * torch.sin(0.37 * n))
    return value.to(torch.float32)


class MedASRUpstreamGoldenTest(unittest.TestCase):

    def test_log_mel_frontend_matches_upstream(self):
        prepared = MedASRFeatureExtractor(_config())(_waveform())
        features = prepared["input_features"]
        self.assertEqual(tuple(features.shape), UPSTREAM_FEATURE_SHAPE)
        self.assertTrue(bool(prepared["attention_mask"].all()))
        values = features[0].double()
        self.assertAlmostEqual(values.sum().item(), UPSTREAM_FEATURE_SUM, delta=1e-6)
        self.assertAlmostEqual(
            values.square().sum().item(),
            UPSTREAM_FEATURE_SQUARE_SUM,
            delta=1e-5,
        )
        for (frame, mel), expected in UPSTREAM_FEATURE_PROBES.items():
            self.assertEqual(values[frame, mel].item(), expected)

    def test_ctc_logits_match_upstream(self):
        config = _config()
        model = MedASRForCTC(config)
        model.load_state_dict(_deterministic_state(model), strict=True)
        model = model.double().eval()
        prepared = MedASRFeatureExtractor(config)(_waveform())
        with torch.no_grad():
            logits = model(
                prepared["input_features"].double(),
                attention_mask=prepared["attention_mask"],
            ).logits
        expected = torch.tensor(UPSTREAM_LOGITS, dtype=torch.float64)
        self.assertEqual(tuple(logits.shape), (1, *expected.shape))
        # The model runs in float64, but the log-mel features are float32 like
        # upstream; macOS FFT/log kernels move the logits by up to ~7e-8.
        torch.testing.assert_close(logits[0], expected, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
