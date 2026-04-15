"""Focused tests for the Silero VAD v6 integration.

Upstream openWakeWord shipped a v4 VAD. This fork ports v6, which changes
the model I/O contract in two silent-failure-prone ways (single combined
state tensor + mandatory 64-sample rolling context). These tests lock in:

* The ``VAD`` class still exposes the upstream-compatible public API
  (``predict``, ``__call__``, ``reset_states``, ``prediction_buffer``).
* Feeding speech audio of arbitrary lengths produces non-zero probabilities
  — the v6 protocol bug we hit during Marvin's v4→v6 migration silently
  returned ~0 on real speech when context/window were wrong.
* ``reset_states`` fully clears internal state so a fresh run on the same
  audio yields the same result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile

import openwakeword
from openwakeword.vad import VAD

_VAD_MODEL = Path(__file__).parent.parent / "openwakeword" / "resources" / "models" / "silero_vad.onnx"
_SPEECH_WAV = Path(__file__).parent / "data" / "alexa_test.wav"


def _load_speech() -> np.ndarray:
    """Load a known-good 16 kHz speech WAV, return int16 mono samples."""
    openwakeword.utils.download_models(model_names=["alexa_v0.1"])  # ensures resources dir exists
    sr, audio = scipy.io.wavfile.read(_SPEECH_WAV)
    assert sr == 16000, f"fixture must be 16 kHz, got {sr}"
    if audio.ndim == 2:
        audio = audio[:, 0]
    return audio.astype(np.int16)


@pytest.fixture(scope="module")
def vad() -> VAD:
    # ``download_models`` fetches silero_vad.onnx for us.
    openwakeword.utils.download_models()
    return VAD(model_path=str(_VAD_MODEL))


def test_public_api_surface(vad: VAD) -> None:
    """Backward-compat surface from the v4 class must still exist."""
    assert hasattr(vad, "predict")
    assert callable(vad)
    assert hasattr(vad, "reset_states")
    assert hasattr(vad, "prediction_buffer")


def test_detects_speech_non_zero(vad: VAD) -> None:
    """On real speech, v6 must return high probabilities for at least one
    window. Near-zero across the whole clip means the v4→v6 protocol bug
    has regressed (wrong state tensor shape, missing context, or wrong
    window size)."""
    vad.reset_states()
    audio = _load_speech()
    scores: list[float] = []
    # Feed in 1280-sample chunks (what openwakeword.Model does internally).
    for i in range(0, len(audio) - 1280, 1280):
        scores.append(vad.predict(audio[i : i + 1280]))

    assert scores, "no windows completed — audio too short for VAD"
    assert max(scores) > 0.5, (
        f"max VAD probability on speech audio was {max(scores):.3f}, expected > 0.5 — "
        "likely a v6 I/O contract regression (state tensor or rolling context)."
    )


def test_silence_stays_low(vad: VAD) -> None:
    """On digital silence, probabilities should stay near zero across many windows."""
    vad.reset_states()
    silence = np.zeros(16000 * 2, dtype=np.int16)  # 2 s of silence
    scores = []
    for i in range(0, len(silence) - 1280, 1280):
        scores.append(vad.predict(silence[i : i + 1280]))

    assert scores
    assert max(scores) < 0.2, f"silence produced max VAD prob {max(scores):.3f}, expected < 0.2"


def test_reset_restores_initial_state(vad: VAD) -> None:
    """Two passes over the same audio, with reset_states between, must produce
    identical probabilities — otherwise internal state (rolling context,
    LSTM state, or buffer) leaked across passes."""
    audio = _load_speech()
    chunk = audio[: 1280 * 3]  # three 80 ms chunks

    vad.reset_states()
    first_pass = [vad.predict(chunk[i : i + 1280]) for i in range(0, len(chunk), 1280)]

    vad.reset_states()
    second_pass = [vad.predict(chunk[i : i + 1280]) for i in range(0, len(chunk), 1280)]

    assert first_pass == pytest.approx(second_pass, abs=1e-6)


def test_buffering_across_sub_window_chunks(vad: VAD) -> None:
    """Feeding 100-sample chunks (smaller than Silero's 512-sample window)
    must still produce meaningful probabilities — the internal buffer has
    to bridge calls so that completed windows fire in whichever call first
    fills the buffer past 512 samples. The contract we lock in here is:
    both a one-shot and a fragmented feed detect speech in the same clip.
    """
    audio = _load_speech()[: 1280 * 4]

    vad.reset_states()
    one_shot_mean = vad.predict(audio)

    vad.reset_states()
    fragmented = [vad.predict(audio[i : i + 100]) for i in range(0, len(audio), 100)]
    fragmented_positive = [p for p in fragmented if p > 0]

    assert one_shot_mean > 0.1, f"one-shot predict returned {one_shot_mean:.3f} on speech"
    assert fragmented_positive, "no window completed during fragmented feed"
    assert max(fragmented_positive) > 0.5, (
        f"fragmented feed peaked at {max(fragmented_positive):.3f} on speech audio; "
        "buffering may be dropping samples between calls."
    )


def test_call_appends_to_prediction_buffer(vad: VAD) -> None:
    """__call__ must also push the mean probability into ``prediction_buffer``
    so callers that inspect recent history (wake-word VAD gate) keep working."""
    vad.reset_states()
    vad.prediction_buffer.clear()
    audio = _load_speech()[:1280]
    vad(audio)
    assert len(vad.prediction_buffer) == 1


def test_rejects_unsupported_sample_rates() -> None:
    with pytest.raises(ValueError, match="16000 or 8000"):
        VAD(model_path=str(_VAD_MODEL), sample_rate=44100)
