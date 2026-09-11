from __future__ import annotations

import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from voicehub.checkpointing import ONNXAttribute, ONNXNode, SafeTensorReader, save_safetensors
from voicehub.models.vad_sherpa_onnx.modeling import SherpaONNXVADForVoiceActivityDetection
from voicehub.models.vad_ten.native.checkpoint import (
    NATIVE_TEN_VAD_FILENAME,
    NATIVE_TEN_VAD_FORMAT,
    TENVADSafeTensorsCheckpointAdapter,
    _graph_fingerprint,
    convert_ten_vad_onnx_checkpoint,
)
from voicehub.models.vad_ten.native.configuration import TENVADConfig
from voicehub.models.vad_ten.native.frontend import TENVADFrontend
from voicehub.models.vad_ten.native.metadata import (
    KALDI_NATIVE_FBANK_VERSION,
    SHERPA_ONNX_REVISION,
    TEN_VAD_GRAPH_FINGERPRINT,
    TEN_VAD_INITIALIZER_INVENTORY_FINGERPRINT,
    TEN_VAD_ONNX_SHA256,
    TEN_VAD_REVISION,
    TEN_VAD_SOURCE_LICENSE,
)
from voicehub.models.vad_ten.native.modeling import TENVADModel
from voicehub.models.vad_ten.native.registration import create_ten_vad_architecture_spec
from voicehub.training.auto import AutoTrainingAdapter

_KALDI_NATIVE_FBANK_REFERENCE = torch.tensor(
    [
        [
            -1.025548935,
            0.3843173981,
            -0.157541275,
            -2.93337059,
            -4.091026306,
            -4.239488602,
            -3.921255112,
            -3.406572342,
            -2.865638733,
            -2.284492493,
            -1.208852768,
            0.8028793335,
            0.8083839417,
            -1.291557312,
            -2.61139679,
            -3.31401062,
            -3.781139374,
            -4.233766556,
            -4.518106461,
            -4.297887802,
            -5.491697311,
            -6.769681931,
            -6.946318626,
            -7.410849571,
            -5.733108521,
            -4.160650253,
            -5.903408051,
            -6.393671036,
            -4.762371063,
            -4.670244217,
            -9.011449814,
            -4.925748825,
            -4.60308075,
            -5.897699356,
            -4.154541016,
            -6.35847187,
            -4.16850853,
            -5.111357689,
            -4.340852737,
            -4.349266052,
        ],
        [
            -2.351758957,
            -0.1347579956,
            -0.2486495972,
            -2.116823196,
            -3.145069122,
            -3.977855682,
            -4.400184631,
            -4.274795532,
            -3.694904327,
            -2.854824066,
            -1.386756897,
            0.7325668335,
            0.8224506378,
            -1.029163361,
            -2.115102768,
            -2.636049271,
            -2.945713043,
            -3.285697937,
            -3.633310318,
            -4.193138123,
            -3.57336998,
            -3.797483444,
            -3.978900909,
            -4.112924576,
            -4.331344604,
            -4.279270172,
            -4.268075943,
            -4.363155365,
            -4.235614777,
            -4.485980988,
            -4.586154938,
            -4.111852646,
            -4.542627335,
            -4.478460312,
            -4.05245018,
            -4.851445198,
            -3.903558731,
            -4.538295746,
            -4.110380173,
            -3.983011246,
        ],
    ],
    dtype=torch.float32,
)


def _write_native_artifact(
    directory: str | Path,
    *,
    config: TENVADConfig | None = None,
) -> tuple[Path, TENVADModel]:
    destination = Path(directory)
    config = TENVADConfig() if config is None else config
    model = TENVADModel(config)
    save_safetensors(
        model.state_dict(),
        destination / NATIVE_TEN_VAD_FILENAME,
        metadata={
            "format": NATIVE_TEN_VAD_FORMAT,
            "architecture": "ten-vad",
        },
    )
    (destination / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    return destination, model


class TENVADFrontendTests(unittest.TestCase):

    def test_frontend_matches_pinned_kaldi_native_fbank_oracle(self):
        frontend = TENVADFrontend(TENVADConfig())
        state = frontend.initial_state(1)
        actual = []
        for frame_index in range(2):
            samples = torch.arange(
                frame_index * 256,
                (frame_index + 1) * 256,
                dtype=torch.float64,
            )
            frame = (
                0.22 * torch.sin(2.0 * math.pi * 173.0 * samples / 16_000.0) +
                0.07 * torch.cos(2.0 * math.pi * 913.0 * samples / 16_000.0) +
                ((samples.remainder(19.0)) - 9.0) * 0.0007).to(dtype=torch.float32)
            output = frontend(frame, state)
            state = output.state
            raw_log_mel = (output.features[0, :40] / frontend.inv_stddev[:40] + frontend.mean[:40])
            actual.append(raw_log_mel)

        actual = torch.stack(actual)
        self.assertLessEqual(
            (actual - _KALDI_NATIVE_FBANK_REFERENCE).abs().max().item(),
            1e-5,
        )
        self.assertEqual(tuple(state.history.shape), (1, 2, 41))

    def test_frontend_and_recurrent_state_are_explicit(self):
        model = TENVADModel(TENVADConfig())
        first, state = model.score_audio_frame(torch.zeros(1, 256))
        second, next_state = model.score_audio_frame(torch.zeros(1, 256), state)

        self.assertEqual(tuple(first.speech_probabilities.shape), (1, ))
        self.assertEqual(tuple(second.speech_probabilities.shape), (1, ))
        self.assertEqual(tuple(next_state.frontend.history.shape), (1, 2, 41))
        self.assertEqual(tuple(next_state.recurrent.hidden_1.shape), (1, 64))
        self.assertEqual(tuple(next_state.recurrent.cell_1.shape), (1, 64))
        self.assertEqual(tuple(next_state.recurrent.hidden_2.shape), (1, 64))
        self.assertEqual(tuple(next_state.recurrent.cell_2.shape), (1, 64))


class TENVADArchitectureTests(unittest.TestCase):

    def test_graph_fingerprint_covers_operator_attributes(self):

        def fingerprint(group: int) -> str:
            node = ONNXNode(
                op_type="Conv",
                domain="",
                inputs=("features", "weight"),
                outputs=("encoded", ),
                attributes={
                    "group": ONNXAttribute(
                        name="group",
                        attribute_type=2,
                        value=group,
                    ),
                },
            )
            return _graph_fingerprint(SimpleNamespace(graph=SimpleNamespace(nodes=(node, ))))

        self.assertNotEqual(fingerprint(1), fingerprint(16))

    def test_architecture_declares_training_streaming_and_provenance(self):
        spec = create_ten_vad_architecture_spec()

        self.assertEqual(spec.architecture_id, "ten-vad")
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.capabilities.streaming)
        self.assertEqual(spec.upstream_revision, TEN_VAD_REVISION)
        self.assertEqual(spec.license_id, TEN_VAD_SOURCE_LICENSE)
        self.assertEqual(KALDI_NATIVE_FBANK_VERSION, "1.22.3")
        self.assertEqual(len(SHERPA_ONNX_REVISION), 40)
        self.assertEqual(len(TEN_VAD_ONNX_SHA256), 64)
        self.assertEqual(len(TEN_VAD_GRAPH_FINGERPRINT), 64)
        self.assertEqual(len(TEN_VAD_INITIALIZER_INVENTORY_FINGERPRINT), 64)

    def test_raw_audio_forward_masks_loss_and_backpropagates(self):
        torch.manual_seed(31)
        model = TENVADModel(TENVADConfig())
        waveforms = torch.randn(2, 1_024)
        labels = torch.tensor([
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        mask = torch.tensor([
            [True, True, True, True],
            [True, True, False, False],
        ])

        output = model(
            waveforms=waveforms,
            waveform_lengths=torch.tensor([1_024, 512]),
            labels=labels,
            label_mask=mask,
            positive_weight=1.5,
        )
        output.loss.backward()

        self.assertEqual(tuple(output.logits.shape), (2, 4))
        self.assertEqual(
            output.frame_mask.tolist(), [
                [True, True, True, True],
                [True, True, False, False],
            ])
        self.assertTrue(torch.isfinite(output.loss))
        self.assertGreater(
            sum(parameter.grad is not None for parameter in model.parameters()),
            10,
        )

    def test_provider_training_export_and_fresh_reload(self):
        with tempfile.TemporaryDirectory() as source_directory:
            source, _ = _write_native_artifact(source_directory)
            wrapper = SherpaONNXVADForVoiceActivityDetection(
                model_path=source,
                model_family="ten",
                model_filename=NATIVE_TEN_VAD_FILENAME,
                lazy_load=True,
            )
            adapter = AutoTrainingAdapter.from_model(wrapper)
            adapter.setup()
            context = adapter.create_training_context({
                "waveforms": torch.zeros(800),
                "sampling_rate": 16_000,
                "segments": [{
                    "start": 0.0,
                    "end": 0.025
                }],
            })
            prepared = adapter.prepare_batch(context.inputs, context)
            output = adapter(**prepared)
            output.loss.backward()

            self.assertEqual(type(adapter).__name__, "NativeTENVADTrainingAdapter")
            self.assertEqual(tuple(prepared["labels"].shape), (1, 4))
            self.assertEqual(prepared["labels"].tolist(), [[1.0, 1.0, 0.0, 0.0]])
            self.assertEqual(output.metadata["source_recipe_published"], False)

            with tempfile.TemporaryDirectory() as export_directory:
                adapter.save_pretrained(export_directory)
                fresh = SherpaONNXVADForVoiceActivityDetection(
                    model_path=export_directory,
                    model_family="ten",
                    model_filename=NATIVE_TEN_VAD_FILENAME,
                    lazy_load=False,
                )
                result = fresh.detect(
                    torch.zeros(1_024),
                    sampling_rate=16_000,
                    return_frames=True,
                )
                self.assertEqual(result.metadata["backend"], "voicehub-native")
                self.assertEqual(len(result.probabilities), 4)

    def test_export_preserves_non_default_ten_window_size(self):
        with tempfile.TemporaryDirectory() as source_directory:
            source, _ = _write_native_artifact(
                source_directory,
                config=TENVADConfig(window_size=384),
            )
            wrapper = SherpaONNXVADForVoiceActivityDetection(
                model_path=source,
                model_family="ten",
                model_filename=NATIVE_TEN_VAD_FILENAME,
                window_size_samples=384,
                lazy_load=False,
            )

            with tempfile.TemporaryDirectory() as export_directory:
                wrapper.export_native_pretrained(export_directory)
                values = json.loads((Path(export_directory) / "config.json").read_text(encoding="utf-8", ))
                fresh = SherpaONNXVADForVoiceActivityDetection.from_pretrained(
                    export_directory,
                    lazy_load=False,
                )

                self.assertEqual(values["window_size"], 384)
                self.assertEqual(values["window_size_samples"], 384)
                self.assertEqual(fresh.native_config.window_size, 384)
                self.assertEqual(fresh._window_shift(), 384)


_ONNX_PATH = os.environ.get("VOICEHUB_TEN_VAD_ONNX")
_ORT_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None


@unittest.skipUnless(
    _ONNX_PATH and Path(_ONNX_PATH).is_file() and _ORT_AVAILABLE,
    "Set VOICEHUB_TEN_VAD_ONNX to run the development ONNX oracle",
)
class TENVADDifferentialOracleTests(unittest.TestCase):

    def test_converted_native_graph_matches_onnxruntime_recurrently(self):
        import numpy as np
        import onnxruntime

        source = Path(_ONNX_PATH)
        with tempfile.TemporaryDirectory() as directory:
            destination = convert_ten_vad_onnx_checkpoint(
                source,
                directory,
                trust_onnx_checkpoint=True,
                expected_source_sha256=TEN_VAD_ONNX_SHA256,
            )
            config_values = json.loads((destination / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                config_values["source_onnx_sha256"],
                TEN_VAD_ONNX_SHA256,
            )
            self.assertTrue(config_values["official_source"])
            model = TENVADModel(TENVADConfig.from_dict(config_values))
            with SafeTensorReader(destination / NATIVE_TEN_VAD_FILENAME) as reader:
                TENVADSafeTensorsCheckpointAdapter().load_streaming(
                    model,
                    reader,
                    config_values,
                    strict=True,
                )
            model.eval()

            session = onnxruntime.InferenceSession(
                str(source),
                providers=["CPUExecutionProvider"],
            )
            generator = torch.Generator().manual_seed(1729)
            state = model.initial_recurrent_state(1)
            maximum_probability_error = 0.0
            maximum_state_error = 0.0
            for _ in range(25):
                features = torch.randn(
                    1,
                    3,
                    41,
                    generator=generator,
                    dtype=torch.float32,
                )
                reference = session.run(
                    None,
                    {
                        "input_1": features.numpy(),
                        "input_2": state.hidden_1.detach().numpy(),
                        "input_3": state.cell_1.detach().numpy(),
                        "input_6": state.hidden_2.detach().numpy(),
                        "input_7": state.cell_2.detach().numpy(),
                    },
                )
                native = model.score_context(features, state)
                maximum_probability_error = max(
                    maximum_probability_error,
                    abs(native.speech_probabilities.item() - float(np.asarray(reference[0]).reshape(-1)[0])),
                )
                for native_state, reference_state in zip(
                    (
                        native.state.hidden_1,
                        native.state.cell_1,
                        native.state.hidden_2,
                        native.state.cell_2,
                    ),
                        reference[1:],
                ):
                    maximum_state_error = max(
                        maximum_state_error,
                        (native_state - torch.from_numpy(np.asarray(reference_state))).abs().max().item(),
                    )
                state = native.state

        self.assertLessEqual(maximum_probability_error, 2e-7)
        self.assertLessEqual(maximum_state_error, 2e-6)


if __name__ == "__main__":
    unittest.main()
