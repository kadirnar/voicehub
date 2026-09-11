from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voicehub.models.asr_native.configuration import WhisperXConfig
from voicehub.models.asr_native.whisperx import WhisperXForSpeechRecognition
from voicehub.models.asr_whisper_native import NativeWhisperTrainingAdapter, WhisperForSpeechRecognition
from voicehub.neural.ctc_alignment import CTCAlignment, align_ctc_transcript, build_trellis
from voicehub.outputs import ASROutput, ASRSegment


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
        self.assertEqual(
            transcribe.call_args.kwargs["return_timestamps"],
            True,
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
