"""Unit tests for the TensorBoard + metrics.jsonl observability hooks added to
``Model.__init__`` / ``Model._log_metrics`` / ``Model.close_training_handles``.

Covers the plumbing, not the end-to-end training loop — whether the four
hook sites inside ``train_model`` fire at the right steps is separately
validated by the gradient-accumulation regression suite and by production
AWS runs that inspect the produced artifacts.

Skips cleanly when the training stack isn't installed; openWakeWord's
inference-only install must remain tensorboard-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Training-stack gate — matches the convention used by
# ``test_train_gradient_accumulation.py``. Any of these missing means the
# consumer hasn't ``pip install -e .[train]``'d the fork.
torch = pytest.importorskip("torch")
pytest.importorskip("torchmetrics")
pytest.importorskip("openwakeword.data")
pytest.importorskip("tensorboard")  # required for the TB SummaryWriter path

from openwakeword.train import EventType, Model as TrainModel  # noqa: E402 — gate first


def _make_model(**kwargs) -> TrainModel:
    """Minimal ``Model`` instance — small layer_dim, deterministic seed."""
    torch.manual_seed(0)
    return TrainModel(input_shape=(16, 96), layer_dim=32, n_classes=1, model_type="dnn", **kwargs)


def test_handles_absent_when_config_unset() -> None:
    """Default construction must leave both observability handles as ``None``
    so inference-only callers pay no cost and the file-system stays clean.
    """
    model = _make_model()
    assert model.writer is None
    assert model.metrics_jsonl is None


def test_log_metrics_noop_without_handles(tmp_path: Path) -> None:
    """``_log_metrics`` with both handles absent must not raise and must not
    create any files. Regression guard against a future refactor where one
    path accidentally assumes a handle exists.
    """
    model = _make_model()
    model._log_metrics(EventType.TRAIN, step=0, loss=1.0, recall=0.5)
    # ``tmp_path`` is the test's clean working dir — if the no-op path
    # accidentally wrote anything it'd land here under cwd.
    assert not any(tmp_path.iterdir())


def test_tensorboard_writer_emits_event_file(tmp_path: Path) -> None:
    """With ``tensorboard_log_dir`` set, an event file must appear in the
    directory after at least one ``_log_metrics`` call and ``close``.
    TB event files follow the ``events.out.tfevents.*`` naming convention.
    """
    log_dir = tmp_path / "tb"
    model = _make_model(tensorboard_log_dir=str(log_dir))
    assert model.writer is not None

    model._log_metrics(EventType.TRAIN, step=0, loss=1.2345, recall=0.6789)
    model._log_metrics(EventType.VAL, step=10, accuracy=0.9, recall=0.85, n_fp=3.0)
    model.close_training_handles()

    event_files = list(log_dir.glob("events.out.tfevents.*"))
    assert len(event_files) == 1, f"expected exactly one TB event file in {log_dir}, got {event_files}"
    assert event_files[0].stat().st_size > 0, "TB event file is empty — writer never flushed scalars"


def test_metrics_jsonl_records_one_line_per_event(tmp_path: Path) -> None:
    """Each ``_log_metrics`` call appends exactly one JSON object to the
    JSONL file with the expected shape: ``step``, ``event``, ``ts``, plus
    the metric kwargs.
    """
    jsonl_path = tmp_path / "metrics.jsonl"
    model = _make_model(metrics_jsonl_path=str(jsonl_path))
    assert model.metrics_jsonl is not None

    model._log_metrics(EventType.TRAIN, step=0, loss=1.2345, recall=0.6789)
    model._log_metrics(EventType.VAL_FP, step=10, fp_per_hour=0.25)
    model._log_metrics(EventType.CHECKPOINT, step=10, recall=0.85, fp=12, total_saved=3)
    model.close_training_handles()

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 3, f"expected 3 JSONL lines, got {len(lines)}: {lines}"

    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["event"] == EventType.TRAIN.value
    assert parsed[0]["step"] == 0
    assert parsed[0]["loss"] == pytest.approx(1.2345)
    assert parsed[0]["recall"] == pytest.approx(0.6789)
    assert "ts" in parsed[0]

    assert parsed[1]["event"] == EventType.VAL_FP.value
    assert parsed[1]["fp_per_hour"] == pytest.approx(0.25)

    assert parsed[2]["event"] == EventType.CHECKPOINT.value
    assert parsed[2]["total_saved"] == 3


def test_numpy_and_tensor_scalars_serialize_to_jsonl(tmp_path: Path) -> None:
    """Production call sites feed numpy scalars and 0-d tensors into
    ``_log_metrics``. Both must serialize via ``default=float`` without
    raising — this is the subtle bug that drops an otherwise green run.
    """
    import numpy as np

    jsonl_path = tmp_path / "metrics.jsonl"
    model = _make_model(metrics_jsonl_path=str(jsonl_path))

    numpy_scalar = np.float32(0.123)
    numpy_0d = np.array(0.456)
    torch_scalar = torch.tensor(0.789)
    model._log_metrics(
        EventType.VAL,
        step=5,
        accuracy=numpy_scalar,
        recall=numpy_0d,
        n_fp=torch_scalar,
    )
    model.close_training_handles()

    record = json.loads(jsonl_path.read_text().strip())
    assert record["accuracy"] == pytest.approx(0.123, abs=1e-5)
    assert record["recall"] == pytest.approx(0.456)
    assert record["n_fp"] == pytest.approx(0.789, abs=1e-5)


def test_close_training_handles_is_idempotent(tmp_path: Path) -> None:
    """``close_training_handles`` must tolerate repeat calls (e.g. caller
    closes explicitly and ``auto_train``'s ``finally`` closes again).
    """
    model = _make_model(
        tensorboard_log_dir=str(tmp_path / "tb"),
        metrics_jsonl_path=str(tmp_path / "metrics.jsonl"),
    )
    model.close_training_handles()
    model.close_training_handles()  # must not raise
    assert model.writer is None
    assert model.metrics_jsonl is None
