"""Tests for training job specs."""

import pytest

from onepool.jobs import TrainJob


def test_minimal_yaml(tmp_path):
    f = tmp_path / "job.yaml"
    f.write_text("model: sshleifer/tiny-gpt2\ndataset: data.txt\n")
    job = TrainJob.from_yaml(f)
    assert job.model == "sshleifer/tiny-gpt2"
    assert job.batch_size == 4  # defaults applied


def test_missing_required_keys(tmp_path):
    f = tmp_path / "job.yaml"
    f.write_text("model: foo\n")
    with pytest.raises(ValueError, match="dataset"):
        TrainJob.from_yaml(f)


def test_unknown_keys_preserved_not_fatal(tmp_path):
    f = tmp_path / "job.yaml"
    f.write_text("model: m\ndataset: d\nfuture_knob: 42\n")
    job = TrainJob.from_yaml(f)
    assert job.extra == {"future_knob": 42}


def test_invalid_precision_rejected(tmp_path):
    f = tmp_path / "job.yaml"
    f.write_text("model: m\ndataset: d\nprecision: fp8\n")
    with pytest.raises(ValueError, match="precision"):
        TrainJob.from_yaml(f)


def test_rounds_split():
    job = TrainJob(model="m", dataset="d", steps=250, inner_steps=100)
    assert job.rounds == [100, 100, 50]
    job = TrainJob(model="m", dataset="d", steps=100, inner_steps=100)
    assert job.rounds == [100]
    job = TrainJob(model="m", dataset="d", steps=30, inner_steps=100)
    assert job.rounds == [30]
