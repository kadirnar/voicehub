from __future__ import annotations

import collections
import unittest
from unittest import mock

import torch

from voicehub.architectures.causal_lm import LlamaConfig, LlamaForCausalLM
from voicehub.generation import (
    AutoregressiveGenerator,
    GenerationConfig,
    GenerationStepOutput,
    apply_repetition_penalty,
    evaluate_stopping_criteria,
    filter_min_p,
    filter_top_k,
    filter_top_p,
    process_logits,
    sample_next_token,
)

# Tensor methods that copy a device value to the host. On CUDA every call
# blocks until all queued kernels finish, so a decoding step should use at
# most one of them.
_HOST_READS = ("__bool__", "__float__", "__index__", "__int__", "item", "tolist")


def _count_host_reads(function):
    """Run ``function`` and count tensor-to-host reads it performs."""
    counts = collections.Counter()

    def counting(name):
        original = getattr(torch.Tensor, name)

        def wrapper(self, *args, **kwargs):
            counts[name] += 1
            return original(self, *args, **kwargs)

        return wrapper

    with mock.patch.multiple(torch.Tensor, **{name: counting(name) for name in _HOST_READS}):
        result = function()
    return result, sum(counts.values())


_SAMPLING_CONFIG = {
    "do_sample": True,
    "temperature": 0.8,
    "top_k": 20,
    "top_p": 0.9,
    "min_p": 0.01,
    "repetition_penalty": 1.1,
    "seed": 3,
}


class GenerationConfigurationTests(unittest.TestCase):

    def test_configuration_normalizes_terminal_tokens_and_copies_safely(self):
        config = GenerationConfig(
            max_new_tokens=8,
            eos_token_id=[7, 9],
            top_k=0,
            top_p=0.95,
            min_p=0.05,
        )

        self.assertEqual(config.eos_token_ids, (7, 9))
        self.assertEqual(config.effective_pad_token_id, 7)
        updated = config.with_updates(max_new_tokens=4, pad_token_id=0)
        self.assertEqual(config.max_new_tokens, 8)
        self.assertEqual(updated.max_new_tokens, 4)
        self.assertEqual(updated.effective_pad_token_id, 0)

    def test_configuration_rejects_ambiguous_or_unsafe_values(self):
        invalid_cases = (
            ({
                "max_new_tokens": 0
            }, "max_new_tokens"),
            ({
                "temperature": 0.0
            }, "temperature"),
            ({
                "top_p": 1.1
            }, "top_p"),
            ({
                "min_p": float("nan")
            }, "min_p"),
            ({
                "repetition_penalty": 0.0
            }, "repetition_penalty"),
            ({
                "eos_token_id": ()
            }, "eos_token_id"),
            ({
                "eos_token_id": (2, 2)
            }, "duplicates"),
            ({
                "seed": 2**64
            }, "seed"),
        )
        for values, message in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    GenerationConfig(**values)


class LogitTransformTests(unittest.TestCase):

    def test_repetition_penalty_is_sign_aware_and_non_mutating(self):
        logits = torch.tensor([[-2.0, 3.0, 1.0, 0.0]])
        original = logits.clone()
        result = apply_repetition_penalty(
            logits,
            torch.tensor([[0, 1, 1]]),
            penalty=2.0,
        )

        torch.testing.assert_close(logits, original)
        torch.testing.assert_close(
            result,
            torch.tensor([[-4.0, 1.5, 1.0, 0.0]]),
        )

    def test_sampling_filters_retain_their_required_candidates(self):
        probabilities = torch.tensor([[0.60, 0.25, 0.10, 0.05]])
        logits = probabilities.log()

        top_k = filter_top_k(logits, 2)
        self.assertEqual(torch.isfinite(top_k).sum().item(), 2)
        self.assertTrue(torch.isfinite(top_k[0, :2]).all())

        top_p = filter_top_p(logits, 0.70)
        self.assertEqual(torch.isfinite(top_p).sum().item(), 2)
        self.assertTrue(torch.isfinite(top_p[0, :2]).all())

        min_p = filter_min_p(logits, 0.50)
        self.assertEqual(torch.isfinite(min_p).sum().item(), 1)
        self.assertTrue(torch.isfinite(min_p[0, 0]))

    def test_filters_reject_rows_without_a_finite_candidate(self):
        logits = torch.full((1, 3), float("-inf"))
        with self.assertRaisesRegex(ValueError, "finite candidate"):
            filter_top_k(logits, 1)


class AutoregressiveGeneratorTests(unittest.TestCase):

    def test_cache_is_reused_and_eos_stops_each_row_independently(self):
        requests = []

        def decoder_step(request):
            requests.append(request)
            logits = torch.full((2, 4), -20.0)
            if request.step_index == 0:
                logits[0, 2] = 10.0
                logits[1, 1] = 10.0
            else:
                logits[0, 0] = 10.0
                logits[1, 2] = 10.0
            return GenerationStepOutput(
                logits=logits,
                cache=f"cache-{request.step_index}",
            )

        output = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0, 1], [1, 0]]),
            GenerationConfig(
                max_new_tokens=5,
                eos_token_id=2,
                pad_token_id=3,
                use_cache=True,
            ),
        )

        self.assertEqual([tuple(request.token_ids.shape) for request in requests], [(2, 2), (2, 1)])
        self.assertIsNone(requests[0].cache)
        self.assertEqual(requests[1].cache, "cache-0")
        torch.testing.assert_close(
            output.sequences,
            torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2]]),
        )
        torch.testing.assert_close(output.generated_lengths, torch.tensor([1, 2]))
        self.assertTrue(output.finished.all())
        self.assertEqual(output.cache, "cache-1")

    def test_disabled_cache_receives_the_growing_sequence(self):
        sequence_widths = []

        def decoder_step(request):
            sequence_widths.append(request.token_ids.shape[1])
            self.assertIsNone(request.cache)
            self.assertFalse(request.use_cache)
            logits = torch.tensor([[0.0, 1.0, -1.0]])
            return GenerationStepOutput(logits=logits, cache=object())

        output = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0, 2]]),
            GenerationConfig(max_new_tokens=3, use_cache=False),
        )

        self.assertEqual(sequence_widths, [2, 3, 4])
        torch.testing.assert_close(output.sequences, torch.tensor([[0, 2, 1, 1, 1]]))
        self.assertIsNone(output.cache)
        self.assertFalse(output.finished.any())

    def test_seeded_sampling_is_request_local_and_repeatable(self):
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

        def decoder_step(request):
            return GenerationStepOutput(logits=logits, cache=request.step_index)

        config = GenerationConfig(
            max_new_tokens=12,
            do_sample=True,
            temperature=0.8,
            top_k=3,
            seed=1234,
        )
        global_state = torch.random.get_rng_state().clone()
        first = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0]]),
            config,
        )
        torch.testing.assert_close(torch.random.get_rng_state(), global_state)

        torch.rand(17)
        state_after_unrelated_work = torch.random.get_rng_state().clone()
        second = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0]]),
            config,
        )
        torch.testing.assert_close(torch.random.get_rng_state(), state_after_unrelated_work)
        torch.testing.assert_close(first.sequences, second.sequences)

    def test_three_dimensional_decoder_logits_use_the_last_time_step(self):

        def decoder_step(request):
            logits = torch.tensor([[[9.0, 0.0], [0.0, 9.0]]])
            return GenerationStepOutput(logits=logits)

        output = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0]]),
            GenerationConfig(max_new_tokens=1),
        )
        self.assertEqual(output.sequences.tolist(), [[0, 1]])

    def test_logits_processors_receive_complete_history_without_mutating_model_logits(self):
        model_logits = torch.tensor([[0.0, 10.0, 9.0]])
        histories = []

        def decoder_step(request):
            del request
            return GenerationStepOutput(logits=model_logits)

        def force_token_two(input_ids, logits):
            histories.append(input_ids.clone())
            logits[:, 1] = float("-inf")
            return logits

        output = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0]]),
            GenerationConfig(max_new_tokens=2),
            logits_processors=(force_token_two, ),
        )

        self.assertEqual(output.sequences.tolist(), [[0, 2, 2]])
        self.assertEqual(
            [history.tolist() for history in histories],
            [[[0]], [[0, 2]]],
        )
        torch.testing.assert_close(
            model_logits,
            torch.tensor([[0.0, 10.0, 9.0]]),
        )

    def test_logits_processors_must_preserve_tensor_contract(self):

        def decoder_step(request):
            del request
            return GenerationStepOutput(logits=torch.ones(1, 3))

        invalid_processors = (
            (lambda input_ids, logits: None, "PyTorch tensor"),
            (lambda input_ids, logits: logits[:, :2], "shape"),
            (lambda input_ids, logits: logits.long(), "floating-point"),
        )
        for processor, message in invalid_processors:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    AutoregressiveGenerator().generate(
                        decoder_step,
                        torch.tensor([[0]]),
                        GenerationConfig(max_new_tokens=1),
                        logits_processors=(processor, ),
                    )

    def test_stopping_criteria_must_return_boolean_row_decisions(self):

        def invalid_criterion(sequences, next_tokens, step_index):
            del sequences, step_index
            return torch.ones_like(next_tokens)

        with self.assertRaisesRegex(TypeError, "boolean"):
            evaluate_stopping_criteria(
                [invalid_criterion],
                torch.tensor([[0, 1]]),
                torch.tensor([1]),
                0,
            )


class HostSynchronizationTests(unittest.TestCase):

    def test_engine_reads_the_device_once_per_step(self):
        vocabulary = torch.linspace(-2.0, 2.0, 41).unsqueeze(0)

        def decoder_step(request):
            del request
            return GenerationStepOutput(logits=vocabulary.clone(), cache="cache")

        config = GenerationConfig(
            max_new_tokens=10,
            eos_token_id=40,
            pad_token_id=0,
            **_SAMPLING_CONFIG,
        )
        output, reads = _count_host_reads(
            lambda: AutoregressiveGenerator().generate(
                decoder_step,
                torch.tensor([[1, 5, 6]]),
                config,
            ))

        steps = output.sequences.shape[1] - 3
        self.assertGreater(steps, 0)
        self.assertLessEqual(reads, steps)

    def test_causal_lm_generation_reads_the_device_at_most_twice_per_step(self):
        # One read is the engine's combined validity/stop decision; the other
        # is the decoder's input-ID range guard, which prevents an
        # out-of-range embedding lookup from becoming a device assertion.
        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=41,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=40,
        )
        model = LlamaForCausalLM(config).eval()
        generation = GenerationConfig(
            max_new_tokens=10,
            eos_token_id=40,
            pad_token_id=0,
            use_cache=True,
            **_SAMPLING_CONFIG,
        )
        prompt = torch.tensor([[1, 5, 6, 7]])
        with torch.no_grad():
            output, reads = _count_host_reads(lambda: model.generate(prompt, generation_config=generation))

        steps = output.sequences.shape[1] - prompt.shape[1]
        self.assertGreater(steps, 0)
        self.assertLessEqual(reads, 2 * steps)

    def test_deferred_validation_still_raises_the_eager_errors(self):
        nan_logits = torch.tensor([[0.0, float("nan"), 1.0]])
        no_candidate = torch.full((1, 3), float("-inf"))
        cases = (
            (nan_logits, torch.tensor([[0]]), "NaN or positive infinity"),
            (no_candidate, torch.tensor([[0]]), "finite candidate"),
            (torch.zeros(1, 3), torch.tensor([[0, 3]]), "outside the logits vocabulary"),
        )
        for do_sample in (False, True):
            config = GenerationConfig(
                max_new_tokens=2,
                do_sample=do_sample,
                top_k=2,
                top_p=0.9,
                repetition_penalty=1.3,
                seed=0,
            )
            for logits, history, message in cases:
                with self.subTest(do_sample=do_sample, message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        process_logits(
                            logits,
                            history,
                            do_sample=do_sample,
                            top_k=2,
                            top_p=0.9,
                            repetition_penalty=1.3,
                        )
                    with self.assertRaisesRegex(ValueError, message):
                        sample_next_token(logits, history, config)
                    with self.assertRaisesRegex(ValueError, message):
                        AutoregressiveGenerator().generate(
                            lambda request, logits=logits: GenerationStepOutput(logits=logits),
                            history,
                            config,
                        )


if __name__ == "__main__":
    unittest.main()
