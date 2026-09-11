import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.hub import write_json_file
from voicehub.models.vad_pyannote import PyannoteVADConfig, PyannoteVADForVoiceActivityDetection
from voicehub.models.vad_pyannote.native.checkpoint import (
    PyanNetSafeTensorsCheckpointAdapter,
    convert_pyannote_lightning_checkpoint,
)
from voicehub.models.vad_pyannote.native.configuration import PyanNetConfig
from voicehub.models.vad_pyannote.native.inference import PyanNetFrameInference
from voicehub.models.vad_pyannote.native.modeling import (
    C50_MAX_DB,
    C50_MIN_DB,
    SNR_MAX_DB,
    SNR_MIN_DB,
    ParametricSincFilterbank,
    PyanNet,
)
from voicehub.models.vad_pyannote.native.powerset import Powerset
from voicehub.models.vad_pyannote_brouhaha import (
    PyannoteBrouhahaVADConfig,
    PyannoteBrouhahaVADForVoiceActivityDetection,
)
from voicehub.models.vad_pyannote_segmentation import (
    PyannoteSegmentationVADConfig,
    PyannoteSegmentationVADForVoiceActivityDetection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(variant):
    common = {
        "lstm_hidden_size": 8,
        "lstm_num_layers": 1,
        "lstm_dropout": 0.0,
        "linear_hidden_size": 8,
        "linear_num_layers": 1,
        "chunk_duration_s": 0.1,
        "chunk_step_s": 0.05,
    }
    if variant == "segmentation":
        return PyanNetConfig(**common)
    if variant == "powerset-segmentation":
        return PyanNetConfig(
            variant=variant,
            num_classes=3,
            max_active_classes=2,
            **common,
        )
    return PyanNetConfig(
        variant="brouhaha",
        num_classes=3,
        repeat_final_chunk=True,
        **common,
    )


def _artifact(root, variant):
    config = _tiny_config(variant)
    model = PyanNet(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        if variant == "powerset-segmentation":
            model.classifier.bias[1] = 10.0
        elif variant == "brouhaha":
            model.classifier.linears["vad"].bias.fill_(10.0)
        else:
            model.classifier.bias.fill_(10.0)
    root.mkdir(parents=True, exist_ok=True)
    save_safetensors(
        model.state_dict(),
        root / "model.safetensors",
        metadata={"format": "voicehub-pyannet-v1"},
    )
    write_json_file(root / "config.json", config.to_dict())
    return model, config


class NativePyanNetArchitectureTests(unittest.TestCase):

    def test_provider_imports_do_not_import_external_runtimes(self):
        code = """
import json
import sys
from voicehub.models.vad_pyannote import PyannoteVADForVoiceActivityDetection
from voicehub.models.vad_pyannote_segmentation import PyannoteSegmentationVADForVoiceActivityDetection
from voicehub.models.vad_pyannote_brouhaha import PyannoteBrouhahaVADForVoiceActivityDetection
print(json.dumps({
    name: name in sys.modules
    for name in ("pyannote", "pyannote.audio", "torchaudio")
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(completed.stdout.strip()),
            {
                "pyannote": False,
                "pyannote.audio": False,
                "torchaudio": False,
            },
        )

    def test_sinc_filters_match_the_pinned_upstream_equations(self):
        torch.manual_seed(3)
        native = ParametricSincFilterbank()
        low = native.min_low_hz + native.low_hz_.abs()
        high = torch.clamp(
            low + native.min_band_hz + native.band_hz_.abs(),
            native.min_low_hz,
            native.sample_rate / 2,
        )
        band = (high - low)[:, 0]
        ft_low = low @ native.n_
        ft_high = high @ native.n_
        even_left = ((torch.sin(ft_high) - torch.sin(ft_low)) / (native.n_ / 2)) * native.window_
        odd_left = ((torch.cos(ft_low) - torch.cos(ft_high)) / (native.n_ / 2)) * native.window_
        even = torch.cat(
            (
                even_left,
                2 * band[:, None],
                even_left.flip(1),
            ),
            dim=1,
        ) / (2 * band[:, None])
        odd = torch.cat(
            (
                odd_left,
                torch.zeros_like(band[:, None]),
                -odd_left.flip(1),
            ),
            dim=1,
        ) / (2 * band[:, None])
        reference = torch.cat((even, odd), dim=0).reshape(80, 1, 251)
        torch.testing.assert_close(native.filters(), reference, rtol=0, atol=0)

    def test_powerset_conversion_uses_upstream_subset_order(self):
        powerset = Powerset(3, 2)
        self.assertEqual(
            powerset.mapping.tolist(),
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
        )
        scores = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]]])
        self.assertEqual(powerset(scores).tolist(), [[[1.0, 1.0, 0.0]]])
        self.assertEqual(powerset.to_speech(scores).tolist(), [[1.0]])

    def test_native_graph_loss_backward_for_every_variant(self):
        for variant in (
                "segmentation",
                "powerset-segmentation",
                "brouhaha",
        ):
            with self.subTest(variant=variant):
                model = PyanNet(_tiny_config(variant))
                waveform = torch.randn(2, 1_600)
                frame_count = model.frame_count(1_600)
                if variant == "powerset-segmentation":
                    labels = torch.ones(2, frame_count, dtype=torch.long)
                elif variant == "brouhaha":
                    labels = torch.zeros(2, frame_count, 3)
                    labels[..., 0] = 1
                    labels[..., 1] = 12
                    labels[..., 2] = 4
                else:
                    labels = torch.ones(2, frame_count, 4)
                output = model(waveform, labels=labels)
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_brouhaha_all_silence_batch_has_a_finite_zero_snr_loss(self):
        model = PyanNet(_tiny_config("brouhaha"))
        waveform = torch.randn(2, 1_600)
        labels = torch.zeros(2, model.frame_count(1_600), 3)
        output = model(waveform, labels=labels)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(output.loss_snr.item(), 0.0)
        output.loss.backward()

    def test_sliding_inference_is_deterministic_and_frame_aligned(self):
        model = PyanNet(_tiny_config("segmentation")).eval()
        waveform = torch.randn(3_200)
        inference = PyanNetFrameInference(model, batch_size=2)
        with torch.inference_mode():
            first = inference(waveform)
            second = inference(waveform)
        torch.testing.assert_close(first.scores, second.scores)
        chunk_samples = round(model.config.chunk_duration_s * model.config.sampling_rate)
        self.assertEqual(
            first.frame_hop_samples,
            round(chunk_samples / model.frame_count(chunk_samples)),
        )
        self.assertEqual(first.valid_samples, 3_200)

    def test_overlap_add_uses_upstream_frame_center_indexing(self):
        model = PyanNet(_tiny_config("segmentation")).eval()
        call_index = 0

        def fake_forward(batch):
            nonlocal call_index
            frames = model.frame_count(batch.shape[-1])
            values = []
            for _ in range(batch.shape[0]):
                call_index += 1
                values.append(
                    torch.full(
                        (frames, 4),
                        float(call_index),
                        dtype=batch.dtype,
                        device=batch.device,
                    ))
            return torch.stack(values)

        model.forward = fake_forward
        output = PyanNetFrameInference(model, batch_size=2)(torch.zeros(3_200))
        self.assertEqual(output.scores.shape, (7, 1))
        expected = (1.0 + 2.0 * 0.08) / 1.08
        self.assertAlmostEqual(output.scores[1, 0].item(), expected, places=6)
        self.assertEqual(output.scores[-1, 0].item(), 0.0)

    def test_brouhaha_uses_checkpoint_introspection_frame_step(self):
        model = PyanNet(_tiny_config("brouhaha")).eval()
        output = PyanNetFrameInference(model)(torch.zeros(3_200))
        self.assertEqual(output.frame_hop_samples, 270)
        self.assertEqual(output.frame_length_samples, 270)

    def test_lightning_conversion_is_explicit_and_strict(self):
        model = PyanNet(PyanNetConfig.segmentation_3())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "upstream.ckpt"
            destination = Path(directory) / "native"
            torch.save({"state_dict": model.state_dict()}, source)
            with self.assertRaisesRegex(ValueError, "trust_pickle_checkpoint"):
                convert_pyannote_lightning_checkpoint(
                    source,
                    destination,
                    variant="segmentation-3.0",
                )
            convert_pyannote_lightning_checkpoint(
                source,
                destination,
                variant="segmentation-3.0",
                trust_pickle_checkpoint=True,
            )
            restored = PyanNet(PyanNetConfig.segmentation_3())
            with SafeTensorReader(destination / "model.safetensors") as reader:
                report = PyanNetSafeTensorsCheckpointAdapter().load_streaming(
                    restored,
                    reader,
                    PyanNetConfig.segmentation_3().to_dict(),
                    strict=True,
                )
            self.assertTrue(report.is_compatible)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name])


class NativePyanNetProviderTests(unittest.TestCase):

    def _provider_case(self, variant, model_class, config_class):
        with tempfile.TemporaryDirectory() as directory:
            expected, native_config = _artifact(
                Path(directory),
                variant,
            )
            wrapper = model_class(
                config_class(name_or_path=directory, batch_size=2),
                device="cpu",
            )
            wrapper.load()
            self.assertEqual(wrapper.training_support, "native")
            self.assertTrue(wrapper.supports_generic_finetuning)
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(
                    value,
                    wrapper.model.state_dict()[name],
                )
            output = wrapper.detect(
                torch.zeros(3_200),
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )
            self.assertEqual(output.metadata["backend"], "voicehub-native")
            self.assertEqual(output.metadata["variant"], variant)
            self.assertTrue(output.probabilities)
            self.assertTrue(output.segments)

            frame_count = wrapper.model.frame_count(1_600)
            if variant == "powerset-segmentation":
                labels = torch.ones(1, frame_count, dtype=torch.long)
            elif variant == "brouhaha":
                labels = torch.zeros(1, frame_count, 3)
                labels[..., 0] = 1
                labels[..., 1] = 12
                labels[..., 2] = 4
            else:
                labels = torch.ones(1, frame_count, 4)
            adapter = wrapper.get_training_adapter()
            trained = adapter(
                audio=torch.zeros(1_600),
                sampling_rate=16_000,
                labels=labels,
            )
            self.assertTrue(torch.isfinite(trained.loss))
            trained.loss.backward()

            export = Path(directory) / "export"
            wrapper.export_native_pretrained(export)
            fresh = model_class(
                config_class(name_or_path=export),
                device="cpu",
            ).load()
            for name, value in wrapper.model.state_dict().items():
                torch.testing.assert_close(
                    value,
                    fresh.model.state_dict()[name],
                )
            if variant == "brouhaha":
                snr_values = output.metadata["frame_snr_db"]
                c50_values = output.metadata["frame_c50_db"]
                expected_snr = (SNR_MAX_DB + SNR_MIN_DB) / 2
                expected_c50 = (C50_MAX_DB + C50_MIN_DB) / 2
                self.assertAlmostEqual(snr_values[0], expected_snr, places=5)
                self.assertAlmostEqual(c50_values[0], expected_c50, places=5)
                self.assertEqual(snr_values[-1], 0.0)
                self.assertEqual(c50_values[-1], 0.0)
                self.assertAlmostEqual(
                    output.metadata["mean_snr_db"],
                    sum(snr_values) / len(snr_values),
                    places=5,
                )
                self.assertAlmostEqual(
                    output.metadata["mean_c50_db"],
                    sum(c50_values) / len(c50_values),
                    places=5,
                )
                self.assertEqual(
                    len(output.metadata["frame_snr_db"]),
                    len(output.probabilities),
                )
            self.assertEqual(native_config.variant, variant)

    def test_all_three_public_providers_infer_train_export_and_reload(self):
        cases = (
            (
                "segmentation",
                PyannoteVADForVoiceActivityDetection,
                PyannoteVADConfig,
            ),
            (
                "powerset-segmentation",
                PyannoteSegmentationVADForVoiceActivityDetection,
                PyannoteSegmentationVADConfig,
            ),
            (
                "brouhaha",
                PyannoteBrouhahaVADForVoiceActivityDetection,
                PyannoteBrouhahaVADConfig,
            ),
        )
        for case in cases:
            with self.subTest(variant=case[0]):
                self._provider_case(*case)

    def test_official_pickle_artifacts_require_explicit_acknowledgement(self):
        for model in (
                PyannoteVADForVoiceActivityDetection(PyannoteVADConfig()),
                PyannoteSegmentationVADForVoiceActivityDetection(PyannoteSegmentationVADConfig()),
                PyannoteBrouhahaVADForVoiceActivityDetection(PyannoteBrouhahaVADConfig()),
        ):
            with self.subTest(model=model.config.model_type):
                with self.assertRaisesRegex(
                        ValueError,
                        "trust_pickle_checkpoint",
                ):
                    model._load_pretrained_model()


if __name__ == "__main__":
    unittest.main()
