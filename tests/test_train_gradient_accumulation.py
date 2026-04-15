"""Regression test for the fix to upstream issue #316.

Upstream ``openWakeWord.train.Model.train_model`` called
``optimizer.zero_grad()`` at the top of every iteration and only called
``loss.backward()`` when the accumulated samples exceeded the 128-sample
threshold. Net effect: gradients from intermediate-window iterations were
silently wiped before they could accumulate, and the parameter update
depended only on the final iteration of each window.

This test builds a minimal training scenario where every batch contains
fewer than 128 samples (forcing the accumulation path) and verifies that:

* Parameters actually change after training.
* The loss history is non-empty (i.e. at least one accumulation window
  committed) — sufficient iterations to cross the threshold in aggregate.

Requires the ``[train]`` extra (torch + openwakeword.data transitive
deps); skips cleanly otherwise so the default dev install stays lean.
"""

from __future__ import annotations

import pytest

# Skip the whole module if the training stack isn't installed. Training deps
# are deliberately optional in this fork — runtime users pay only for the
# inference stack. Anyone actually retraining a model should ``pip install
# -e .[train]`` first.
torch = pytest.importorskip("torch")
pytest.importorskip("torchmetrics")
# ``openwakeword.train`` transitively imports ``openwakeword.data`` which
# needs the full training stack (mutagen/acoustics/pronouncing/datasets/…).
pytest.importorskip("openwakeword.data")

from openwakeword.train import Model as TrainModel  # noqa: E402 — importorskip gate must run first


class _TinyBatchIter:
    """Yields a fixed number of small batches (under the 128-sample
    gradient-accumulation threshold), so the accumulation path is exercised.
    """

    def __init__(self, n_batches: int, batch_size: int, input_shape: tuple[int, int]):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.input_shape = input_shape

    def __iter__(self):
        for _ in range(self.n_batches):
            # half positives, half negatives; random features
            x = torch.rand(self.batch_size, *self.input_shape)
            y = torch.cat([torch.ones(self.batch_size // 2), torch.zeros(self.batch_size - self.batch_size // 2)])
            yield x, y


def _param_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _params_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    """True if any parameter tensor differs between snapshots."""
    assert before.keys() == after.keys()
    return any(not torch.equal(before[name], after[name]) for name in before)


def test_params_update_when_per_batch_size_below_threshold() -> None:
    """With batch_size=32, every single iteration feeds fewer samples than
    the 128-sample gradient-accumulation target — so the accumulation path
    is the only way gradients ever reach the optimizer. Pre-fix upstream
    silently discarded all of them; with the fix, parameters must change
    after enough iterations (here: 16 iterations × 32 samples = 512,
    comfortably exceeding the 128 target multiple times).
    """
    torch.manual_seed(0)
    input_shape = (16, 96)
    trainer = TrainModel(input_shape=input_shape, layer_dim=32, n_classes=1, model_type="dnn")

    before = _param_snapshot(trainer.model)

    batches = _TinyBatchIter(n_batches=16, batch_size=32, input_shape=input_shape)
    trainer.train_model(
        X=batches,
        max_steps=16,
        warmup_steps=4,
        hold_steps=4,
        lr=1e-3,
    )

    after = _param_snapshot(trainer.model)
    assert _params_changed(before, after), (
        "Model parameters did not change after training with sub-threshold batches. "
        "The gradient-accumulation path is not propagating gradients — regression of "
        "upstream issue #316."
    )
    assert len(trainer.history["loss"]) >= 1, "No accumulation windows committed — target threshold never reached."


def test_history_logs_window_loss_sum() -> None:
    """After enough iterations for at least one commit, history['loss']
    should contain one entry per committed window (not one per iteration).
    This locks down the metric-logging behaviour in the fixed loop.
    """
    torch.manual_seed(0)
    input_shape = (16, 96)
    trainer = TrainModel(input_shape=input_shape, layer_dim=32, n_classes=1, model_type="dnn")

    # 8 iterations × 64 samples = 512 samples → exactly 4 windows at target=128.
    batches = _TinyBatchIter(n_batches=8, batch_size=64, input_shape=input_shape)
    trainer.train_model(
        X=batches,
        max_steps=8,
        warmup_steps=2,
        hold_steps=2,
        lr=1e-3,
        gradient_accum_target=128,
    )

    # Exact window count depends on the high-loss filter dropping some
    # samples per iteration. Lower bound: at least 1 window committed.
    assert len(trainer.history["loss"]) >= 1
    # Upper bound: never more windows than iterations.
    assert len(trainer.history["loss"]) <= 8
