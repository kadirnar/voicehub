from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voicehub.architectures.ctc_alignment import CTCAlignment, align_ctc_transcript, build_trellis
from voicehub.modeling_outputs import ASROutput, ASRSegment
from voicehub.models.asr_native.configuration import WhisperXConfig
from voicehub.models.asr_native.whisperx import WhisperXForSpeechRecognition
from voicehub.models.asr_whisper_native import NativeWhisperTrainingAdapter, WhisperForSpeechRecognition


def _emission(labels: tuple[int, ...], vocabulary_size: int) -> torch.Tensor:
    logits = torch.full((len(labels), vocabulary_size), -8.0)
    for frame, label in enumerate(labels):
        logits[frame, label] = 8.0
    return logits.log_softmax(dim=-1)


class NativeCTCAlignmentTests(unittest.TestCase):

    def test_trellis_matches_the_reference_stay_change_recurrence(self):
        emission = torch.tensor(
            [
                [-0.1, -2.0, -3.0],
                [-1.0, -0.2, -3.0],
                [-0.3, -2.0, -0.4],
            ],
            dtype=torch.float64,
        )

        actual = build_trellis(emission, (1, 2), blank_id=0)

        expected = emission.new_full((4, 3), -torch.inf)
        expected[0, 0] = 0
        expected[1:, 0] = torch.cumsum(emission[:, 0], dim=0)
        expected[-2:, 0] = torch.inf
        for frame in range(3):
            expected[frame + 1, 1:] = torch.maximum(
                expected[frame, 1:] + emission[frame, 0],
                expected[frame, :-1] + emission[frame, (1, 2)],
            )

        torch.testing.assert_close(actual, expected)

    def test_alignment_returns_typed_word_and_character_intervals(self):
        emission = _emission(
            (0, 1, 1, 2, 3, 4, 4, 0),
            vocabulary_size=5,
        )

        result = align_ctc_transcript(
            emission,
            "hi a",
            {
                "<pad>": 0,
                "h": 1,
                "i": 2,
                "|": 3,
                "a": 4,
            },
            blank_id=0,
            language="en",
            segment_start=2.0,
            segment_end=4.0,
        )

        self.assertIsInstance(result, CTCAlignment)
        self.assertEqual(tuple(word.text for word in result.words), ("hi", "a"))
        self.assertEqual(
            tuple(character.character for character in result.characters),
            ("h", "i", " ", "a"),
        )
        self.assertGreaterEqual(result.words[0].start, 2.0)
        self.assertLessEqual(result.words[-1].end, 4.0)
        self.assertTrue(all(0.0 <= word.confidence <= 1.0 for word in result.words))

    def test_unknown_characters_use_the_reference_wildcard_column(self):
        emission = _emission(
            (0, 1, 2, 1, 0),
            vocabulary_size=3,
        )

        result = align_ctc_transcript(
            emission,
            "h?",
            {
                "<pad>": 0,
                "h": 1,
                "|": 2,
            },
            blank_id=0,
            segment_start=0.0,
            segment_end=1.0,
        )

        self.assertEqual(tuple(word.text for word in result.words), ("h?", ))
        self.assertEqual(len(result.characters), 2)

    def test_characters_are_lower_cased_not_case_folded_like_whisperx(self):
        # "ς".casefold() is "σ" but "ς".lower() stays "ς": WhisperX aligns a
        # final sigma missing from the vocabulary against the wildcard column.
        emission = _emission((0, 2, 0), vocabulary_size=3)

        result = align_ctc_transcript(
            emission,
            "ς",
            {
                "<pad>": 0,
                "σ": 1,
                "|": 2,
            },
            blank_id=0,
            segment_start=0.0,
            segment_end=1.0,
        )

        self.assertEqual(tuple(word.text for word in result.words), ("ς", ))
        self.assertGreater(result.words[0].confidence, 0.99)

    def test_transcript_longer_than_emission_fails_closed(self):
        result = align_ctc_transcript(
            _emission((1, ), vocabulary_size=3),
            "too long",
            {
                "<pad>": 0,
                "t": 1,
                "|": 2,
            },
            blank_id=0,
            segment_start=0.0,
            segment_end=1.0,
        )

        self.assertEqual(result, CTCAlignment(words=(), characters=()))


class NativeWhisperXProviderTests(unittest.TestCase):

    def test_provider_uses_native_whisper_training_adapter(self):
        model = WhisperXForSpeechRecognition(
            WhisperXConfig(name_or_path="small"),
            device="cpu",
        )

        adapter = model.get_training_adapter()

        self.assertIsInstance(model, WhisperForSpeechRecognition)
        self.assertEqual(model.config.name_or_path, "openai/whisper-small")
        self.assertIsInstance(adapter, NativeWhisperTrainingAdapter)
        self.assertTrue(adapter.spec.native_training)

    def test_word_request_composes_native_whisper_and_ctc_alignment(self):
        model = WhisperXForSpeechRecognition(
            WhisperXConfig(
                name_or_path="small",
                alignment_model_path="local/alignment",
            ),
            device="cpu",
        )
        model.generation_adapter = SimpleNamespace(token_set=SimpleNamespace(is_multilingual=True), )
        model.native_config = SimpleNamespace(expected_input_frames=3000)
        tokenizer = SimpleNamespace(
            vocabulary={
                "<pad>": 0,
                "h": 1,
                "i": 2,
                "|": 3,
            },
            pad_token_id=0,
            word_delimiter_token="|",
        )
        runtime = SimpleNamespace(ctc_processor=SimpleNamespace(tokenizer=tokenizer), )
        base_output = ASROutput(
            text="hi",
            segments=(ASRSegment(
                text="hi",
                start=0.0,
                end=1.0,
                language="en",
            ), ),
            language="en",
            duration=1.0,
            metadata={"backend": "voicehub-native"},
        )

        with (
                patch.object(
                    WhisperForSpeechRecognition,
                    "_transcribe",
                    return_value=base_output,
                ) as transcribe,
                patch.object(
                    model,
                    "_load_alignment_model",
                    return_value=runtime,
                ),
                patch.object(
                    model,
                    "_ctc_emission",
                    return_value=_emission(
                        (0, 1, 1, 2, 2, 0),
                        vocabulary_size=4,
                    ),
                ),
        ):
            output = model._transcribe(
                torch.zeros(16_000),
                sampling_rate=16_000,
                return_timestamps="word",
            )

        self.assertTrue(output.metadata["aligned"])
        self.assertEqual(output.metadata["pipeline"], "voicehub-native-whisperx")
        self.assertEqual(output.segments[0].words[0].text, "hi")
        # WhisperX transcribes without timestamp tokens before aligning.
        self.assertEqual(
            transcribe.call_args.kwargs["return_timestamps"],
            False,
        )
        self.assertEqual(output.segments[0].start, output.segments[0].words[0].start)
        self.assertEqual(output.segments[0].end, output.segments[0].words[-1].end)

    def test_word_request_aligns_untimestamped_whisper_windows_like_whisperx(self):
        # WhisperX decodes each <= 30 s chunk without timestamps and aligns
        # the text against the whole chunk, so the transcript is the one a
        # plain request returns.
        model = WhisperXForSpeechRecognition(
            WhisperXConfig(name_or_path="small", alignment_model_path="local/alignment"),
            device="cpu",
        )
        model.generation_adapter = SimpleNamespace(token_set=SimpleNamespace(is_multilingual=True))
        model.native_config = SimpleNamespace(expected_input_frames=3000)
        windows = []

        def transcribe_window(audio, **kwargs):
            windows.append((audio.numel(), kwargs["return_timestamps"]))
            text = " first window." if len(windows) == 1 else " second window."
            return ASROutput(text=text, language="en", duration=audio.numel() / 16_000)

        aligned = []

        def align(output, *, waveform, language):
            aligned.append((output, waveform.numel(), language))
            return output.segments

        with (
                patch.object(WhisperForSpeechRecognition, "_transcribe", side_effect=transcribe_window),
                patch.object(model, "_align_segments", side_effect=align),
        ):
            output = model._transcribe(
                torch.zeros(40 * 16_000),
                sampling_rate=16_000,
                language="en",
                return_timestamps="word",
            )

        self.assertEqual(windows, [(480_000, False), (160_000, False)])
        self.assertEqual(output.text, "first window. second window.")
        self.assertEqual(
            [(segment.text, segment.start, segment.end) for segment in aligned[0][0].segments],
            [("first window.", 0.0, 30.0), ("second window.", 30.0, 40.0)],
        )
        self.assertEqual(aligned[0][1:], (640_000, "en"))

    def test_alignment_emission_uses_raw_samples_like_whisperx(self):
        # WhisperX feeds its alignment models raw audio; the Transformers
        # processor's zero-mean/unit-variance normalization must not apply.
        class RecordingCTC(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))
                self.inputs = []

            def forward(self, input_values, attention_mask=None):
                self.inputs.append((input_values.clone(), attention_mask))
                frames = 2 + input_values.shape[-1] // 400
                logits = torch.arange(frames * 3, dtype=torch.float32).reshape(1, frames, 3)
                return SimpleNamespace(logits=logits, input_lengths=torch.tensor([0]))

        model = RecordingCTC()
        processor = SimpleNamespace(
            prepare_audio_batch=lambda waveforms: self.fail("processor normalization used"))
        runtime = SimpleNamespace(
            model=model,
            native_config=SimpleNamespace(minimum_input_samples=400),
            ctc_processor=processor,
        )
        waveform = torch.linspace(0.1, 0.3, 800)

        emission = WhisperXForSpeechRecognition._ctc_emission(runtime, waveform)

        torch.testing.assert_close(model.inputs[0][0], waveform[None])
        self.assertEqual(emission.shape, (4, 3))
        torch.testing.assert_close(emission.exp().sum(dim=-1), torch.ones(4))

    def test_short_alignment_segments_are_zero_padded_without_failing(self):
        # Like WhisperX, a segment shorter than the receptive field is
        # zero-padded and aligned against every emitted frame.
        class OneFrameCTC(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))
                self.inputs = []

            def forward(self, input_values, attention_mask=None):
                self.inputs.append(input_values.clone())
                return SimpleNamespace(
                    logits=torch.zeros(1, 1, 3),
                    input_lengths=torch.tensor([0]),
                )

        model = OneFrameCTC()
        runtime = SimpleNamespace(
            model=model,
            native_config=SimpleNamespace(minimum_input_samples=400),
            ctc_processor=SimpleNamespace(),
        )

        emission = WhisperXForSpeechRecognition._ctc_emission(runtime, torch.full((320, ), 0.5))

        self.assertEqual(emission.shape, (1, 3))
        self.assertEqual(model.inputs[0].shape, (1, 400))
        torch.testing.assert_close(model.inputs[0][0, :320], torch.full((320, ), 0.5))
        torch.testing.assert_close(model.inputs[0][0, 320:], torch.zeros(80))

    def test_alignment_segments_truncate_times_to_samples_like_whisperx(self):
        # WhisperX slices audio[int(start * 16000):int(end * 16000)].
        model = WhisperXForSpeechRecognition(
            WhisperXConfig(name_or_path="small", alignment_model_path="local/alignment"),
            device="cpu",
        )
        tokenizer = SimpleNamespace(
            vocabulary={
                "<pad>": 0,
                "h": 1,
                "i": 2,
                "|": 3
            },
            pad_token_id=0,
            word_delimiter_token="|",
        )
        runtime = SimpleNamespace(ctc_processor=SimpleNamespace(tokenizer=tokenizer))
        output = ASROutput(
            text="hi",
            segments=(ASRSegment(text="hi", start=0.10004, end=0.90004, language="en"), ),
            language="en",
            duration=1.0,
        )
        slices = []

        def emission(_runtime, waveform):
            slices.append(waveform.clone())
            return _emission((0, 1, 1, 2, 2, 0), vocabulary_size=4)

        with (
                patch.object(model, "_load_alignment_model", return_value=runtime),
                patch.object(model, "_ctc_emission", side_effect=emission),
        ):
            model._align_segments(output, waveform=torch.arange(16_000, dtype=torch.float32), language="en")

        self.assertEqual(slices[0][0].item(), 1600.0)
        self.assertEqual(slices[0][-1].item(), 14399.0)

    def test_alignment_model_defaults_to_float32_like_whisperx(self):
        # WhisperX always runs its alignment models in float32, also on CUDA.
        self.assertEqual(WhisperXConfig().alignment_torch_dtype, "float32")
        self.assertEqual(
            WhisperXConfig(alignment_torch_dtype="float16").alignment_torch_dtype,
            "float16",
        )

    def test_alignment_is_not_loaded_for_plain_transcription(self):
        model = WhisperXForSpeechRecognition(
            WhisperXConfig(name_or_path="small"),
            device="cpu",
        )
        base_output = ASROutput(
            text="plain",
            language="en",
            duration=1.0,
        )
        with (
                patch.object(
                    WhisperForSpeechRecognition,
                    "_transcribe",
                    return_value=base_output,
                ),
                patch.object(model, "_load_alignment_model") as align_loader,
        ):
            output = model._transcribe(
                torch.zeros(16_000),
                sampling_rate=16_000,
            )

        align_loader.assert_not_called()
        self.assertFalse(output.metadata["aligned"])


if __name__ == "__main__":
    unittest.main()
