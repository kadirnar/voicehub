import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.benchmark_upstream_parity import compare_results, latency_comparison, run_side
from scripts.upstream_parity_worker import json_safe, load_tensor_inputs, tts_inputs


def result(task, **sample):
    return {"status": "ok", "task": task, "samples": [{"id": "sample", "warm_seconds": [1.0] * 10, **sample}]}


def test_failed_runs_are_not_comparable():
    assert compare_results({"status": "error"}, {"status": "not-run"})["status"] == "not-comparable"


def test_different_executed_requests_cannot_establish_parity():
    upstream = result("asr", text="hello", reference="hello")
    candidate = result("asr", text="hello", reference="hello")
    upstream["request_sha256"] = "first-checkpoint-and-settings"
    candidate["request_sha256"] = "different-checkpoint-or-settings"
    assert compare_results(upstream, candidate)["status"] == "not-comparable"


def test_timing_flags_regression_and_keeps_noisy_measurements_inconclusive():
    assert latency_comparison([1.0] * 10, [1.3] * 10, tolerance=0.1)["regression_flag"]
    assert not latency_comparison([1.0] * 10, [0.5, 2.0] * 5, tolerance=0.1)["regression_flag"]
    assert not latency_comparison([1.0] * 10, [1.0] * 10, tolerance=0.1)["regression_flag"]
    for invalid in ([1.0], [float("nan")] * 10, [0.0] * 10):
        with pytest.raises(ValueError):
            latency_comparison(invalid, [1.0] * 10, tolerance=0.1)


def test_too_few_timings_still_compare_outputs(tmp_path):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.array([0.1, 0.2], dtype=np.float32))
    np.save(b, np.array([0.1, 0.2], dtype=np.float32))
    upstream = result("tts", audio=str(a), sample_rate=16000)
    candidate = result("tts", audio=str(b), sample_rate=16000)
    upstream["samples"][0]["warm_seconds"] = [9.0]
    comparison = compare_results(upstream, candidate)
    sample = comparison["samples"][0]
    assert comparison["status"] == "measured"
    assert sample["waveform_exact"]
    assert sample["timing"]["status"] == "insufficient-samples"
    assert sample["timing"]["upstream_samples"] == 1
    assert sample["timing"]["voicehub_samples"] == 10
    assert sample["timing"]["regression_flag"] is None
    assert comparison["timing_regression_flag"] is False
    json.dumps(comparison, allow_nan=False)


def test_enough_timings_keep_the_regression_flag():
    upstream = result("asr", text="hello", reference="hello")
    candidate = result("asr", text="hello", reference="hello")
    candidate["samples"][0]["warm_seconds"] = [1.3] * 5
    comparison = compare_results(upstream, candidate)
    assert comparison["samples"][0]["timing"]["status"] == "measured"
    assert comparison["samples"][0]["timing"]["regression_flag"]
    assert comparison["timing_regression_flag"]


def test_invalid_timings_are_rejected_even_when_insufficient():
    upstream = result("asr", text="hello", reference="hello")
    upstream["samples"][0]["warm_seconds"] = [float("nan")]
    with pytest.raises(ValueError, match="finite and positive"):
        compare_results(upstream, result("asr", text="hello", reference="hello"))


def test_worker_merges_generic_tts_inputs(tmp_path):
    shared, own = tmp_path / "shared.npz", tmp_path / "own.npz"
    np.savez(shared, speaker_embedding=np.arange(3, dtype=np.float32), scale=np.array(0.5))
    np.savez(own, scale=np.array(2.0), tokens=np.array([[1, 2]], dtype=np.int64))
    request = {
        "generation": dict(temperature=0.0, speaker="a"),
        "tensor_inputs": str(shared),
        "matched_phonemes": True,
        "phoneme_argument": "phoneme_text",
    }
    sample = {"text": "hi", "phonemes": "h ai", "generation": {"speaker": "b"}, "tensor_inputs": str(own)}
    text, kwargs = tts_inputs(request, sample)
    assert text == "hi"
    assert kwargs["temperature"] == 0.0 and kwargs["speaker"] == "b"
    assert kwargs["phoneme_text"] == "h ai" and "phonemes" not in kwargs
    assert kwargs["speaker_embedding"].dtype == torch.float32
    assert kwargs["speaker_embedding"].tolist() == [0.0, 1.0, 2.0]
    assert kwargs["scale"].item() == 2.0
    assert kwargs["tokens"].dtype == torch.int64

    kwargs["speaker_embedding"].add_(1)
    assert tts_inputs(request, sample)[1]["speaker_embedding"].tolist() == [0.0, 1.0, 2.0]

    phonemized = {"text": "hi", "phonemes": "h ai"}
    assert tts_inputs({"phonemes_as_text": True}, phonemized) == ("h ai", {})
    assert tts_inputs({"matched_phonemes": True}, phonemized) == ("hi", {"phonemes": "h ai"})
    with pytest.raises(ValueError, match="mutually exclusive"):
        tts_inputs({"matched_phonemes": True, "phonemes_as_text": True}, phonemized)
    assert load_tensor_inputs(None) == {}


def test_worker_tensor_inputs_refuse_pickled_arrays(tmp_path):
    archive = tmp_path / "objects.npz"
    np.savez(archive, value=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError):
        load_tensor_inputs(archive)


def test_worker_metadata_is_json_safe():
    metadata = {
        "durations": torch.ones(2, 3),
        "steps": np.int64(4),
        "scale": torch.tensor(0.5),
        "nested": [np.zeros(2, dtype=np.float32), (1, "a")],
        "path": Path("/tmp/x.wav"),
        "missing": None,
        "bad": float("nan"),
        "other": object,
    }
    safe = json_safe(metadata)
    assert safe["durations"] == {"shape": [2, 3], "dtype": "torch.float32"}
    assert safe["steps"] == 4 and safe["scale"] == 0.5
    assert safe["nested"] == [{"shape": [2], "dtype": "float32"}, [1, "a"]]
    assert safe["path"] == "/tmp/x.wav" and safe["missing"] is None
    json.dumps(safe, allow_nan=False)


def test_transcript_agreement_is_distinct_from_accuracy():
    original = result("asr", text="Hello there!", reference="Hello world")
    candidate = result("asr", text="hello there", reference="Hello world")
    sample = compare_results(original, candidate)["samples"][0]
    assert sample["normalized_text_equal"]
    assert not sample["text_exact"]
    assert sample["upstream_wer"] == sample["voicehub_wer"] == 0.5


def test_vad_agreement_does_not_manufacture_ground_truth():
    sample = compare_results(result("vad", segments=[[0.1, 0.5]]),
                             result("vad", segments=[[0.1, 0.5]]))["samples"][0]
    assert sample["segments_equal"]
    assert sample["ground_truth_error_rate"] is None


@pytest.mark.parametrize("field", ["segments", "probabilities"])
def test_nonfinite_vad_measurements_are_rejected_before_report_serialization(field):
    original = result("vad", segments=[[0.1, 0.5]], probabilities=[0.8])
    candidate = result("vad", segments=[[0.1, 0.5]], probabilities=[0.8])
    candidate["samples"][0][field] = [[0.1, float("nan")]] if field == "segments" else [float("nan")]
    with pytest.raises(ValueError, match="must be finite"):
        compare_results(original, candidate)


def test_waveform_shape_rate_and_nonfinite_values_are_reported(tmp_path):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.array([0.1, 0.2]))
    np.save(b, np.array([0.1, 0.2, 0.3]))
    sample = compare_results(
        result("tts", audio=str(a), sample_rate=16000), result("tts", audio=str(b),
                                                               sample_rate=24000))["samples"][0]
    assert not sample["shape_equal"]
    assert not sample["sample_rate_equal"]
    assert sample["waveform_rmse"] is None
    np.save(b, np.array([float("nan"), 0.2]))
    sample = compare_results(
        result("tts", audio=str(a), sample_rate=16000), result("tts", audio=str(b),
                                                               sample_rate=16000))["samples"][0]
    assert not sample["finite_audio"]
    json.dumps(sample, allow_nan=False)


def test_stale_success_cannot_survive_failed_worker(tmp_path):
    request, output = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text("{}")
    output.write_text('{"status":"ok"}')
    spec = {"cwd": str(tmp_path), "command": [sys.executable, "-c", "raise SystemExit(3)"]}
    record = run_side(spec, request, output, timeout=10)
    assert record["status"] == "error"
    assert record["returncode"] == 3
    assert record["request_sha256"]


def test_worker_executes_in_its_own_directory_without_pythonpath(tmp_path, monkeypatch):
    request, output = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text("{}")
    monkeypatch.setenv("PYTHONPATH", "/not-the-original-repository")
    code = (
        "import json,os,sys; from pathlib import Path; "
        "Path(sys.argv[-1]).write_text(json.dumps(dict(status='ok',cwd=os.getcwd(),"
        "pythonpath=os.environ.get('PYTHONPATH'))))")
    spec = {"cwd": str(tmp_path), "command": [sys.executable, "-c", code]}
    record = run_side(spec, request, output, timeout=10)
    assert record["status"] == "ok"
    assert record["cwd"] == str(tmp_path)
    assert record["pythonpath"] is None
