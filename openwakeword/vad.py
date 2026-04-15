# Copyright 2022 David Scripka. All rights reserved.
# Copyright 2026 Dan Korotin. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#######################
# Silero VAD License
#######################

# MIT License
#
# Copyright (c) 2020-present Silero Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

########################################

"""Silero VAD v6 integration.

Upstream openWakeWord shipped a Silero v4 integration that took a separate
``(h, c)`` LSTM state pair and accepted variable-size audio chunks. Silero v5+
reworked the model:

* State is a single ``(2, batch, 128)`` tensor.
* Inference requires fixed 512-sample windows at 16 kHz (256 at 8 kHz).
* Each window must be prefixed with a 64-sample rolling context carried over
  from the previous window (32 samples at 8 kHz). Initial context is zeros.

If you call the v6 model with anything other than the fixed window plus
context, it silently returns near-zero probabilities even on clean speech —
this is the bug that cost us a cycle during the Marvin v4→v6 migration.

This module preserves the upstream ``VAD`` class signature (``predict``,
``__call__``, ``reset_states``, ``prediction_buffer``) so downstream code that
imports ``from openwakeword.vad import VAD`` keeps working, but internally the
class buffers incoming audio and runs one v6 inference per complete 512-sample
window.
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np
import onnxruntime as ort

# Silero v5+ requires fixed window sizes per sample rate.
_WINDOW_SIZES: dict[int, int] = {16000: 512, 8000: 256}

# v6 prepends a rolling context (last N samples of previous window).
_CONTEXT_SIZES: dict[int, int] = {16000: 64, 8000: 32}


def _make_session(model_path: str, n_threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = n_threads
    opts.intra_op_num_threads = n_threads
    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


class VAD:
    """Silero VAD v6 wake-word pre-filter.

    Stateful; feed audio of any size via :meth:`predict` or :meth:`__call__`.
    The underlying v6 model is invoked once per completed 512-sample window
    (at 16 kHz) or 256-sample window (at 8 kHz). Incomplete windows are
    buffered internally until enough samples arrive.
    """

    # Upstream v4 default was 480 samples at 16 kHz (30 ms). v6 requires
    # exactly 512 samples at 16 kHz. Keeping the ``frame_size`` keyword for
    # backward compatibility, but its value no longer controls model I/O —
    # only how the caller chunks audio before passing it in.
    def __init__(
        self,
        model_path: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "resources",
            "models",
            "silero_vad.onnx",
        ),
        n_threads: int = 1,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize the VAD model.

        Args:
            model_path: Path to a Silero VAD v5+ ONNX model.
            n_threads: Intra- and inter-op thread count for the ORT session.
            sample_rate: 16000 (default) or 8000. Silero v6 supports no others.
        """
        if sample_rate not in _WINDOW_SIZES:
            raise ValueError(
                f"Silero VAD v6 supports only 16000 or 8000 Hz, got {sample_rate}."
            )

        self._session = _make_session(model_path, n_threads)
        self._sample_rate = sample_rate
        self._window = _WINDOW_SIZES[sample_rate]
        self._context_size = _CONTEXT_SIZES[sample_rate]
        self._sr_tensor = np.array(sample_rate, dtype=np.int64)
        # Buffer length of ~10 s at 25 Hz update rate, matching upstream v4 class.
        self.prediction_buffer: deque = deque(maxlen=125)
        self.reset_states()

    # -- state management ---------------------------------------------------

    def reset_states(self, batch_size: int = 1) -> None:
        """Clear model state and internal audio buffer."""
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros((batch_size, self._context_size), dtype=np.float32)
        self._pending: list[float] = []

    # -- inference ----------------------------------------------------------

    def predict(self, x: np.ndarray, frame_size: int | None = None) -> float:
        """Run VAD on ``x`` and return the mean probability across windows.

        Args:
            x: Audio samples. May be ``int16`` (PCM) or ``float32`` in the
                range [-1, 1]. Length is not required to be a multiple of the
                Silero window size — leftover samples are buffered for the
                next call.
            frame_size: Ignored. Kept for API compatibility with the v4 VAD
                class. Silero v6 imposes a fixed window size internally.

        Returns:
            Mean of the per-window probabilities produced by this call. Returns
            ``0.0`` if no complete window was processed (audio buffered only).
        """
        del frame_size  # unused; kept for backward compat

        probs = self._run_windows(x)
        if not probs:
            return 0.0
        return float(np.mean(probs))

    def __call__(self, x: np.ndarray, frame_size: int | None = None) -> float:
        """Run VAD and append the mean probability to :attr:`prediction_buffer`."""
        mean_prob = self.predict(x, frame_size=frame_size)
        self.prediction_buffer.append(mean_prob)
        return mean_prob

    # -- internals ----------------------------------------------------------

    def _run_windows(self, x: np.ndarray) -> list[float]:
        if x.dtype == np.int16:
            audio_f = x.astype(np.float32) / 32767.0
        else:
            audio_f = np.asarray(x, dtype=np.float32)

        self._pending.extend(audio_f.tolist())

        probs: list[float] = []
        while len(self._pending) >= self._window:
            window = np.asarray(self._pending[: self._window], dtype=np.float32)
            del self._pending[: self._window]
            # v6 prepends context (last N samples of previous window).
            inp = np.concatenate([self._context, window.reshape(1, -1)], axis=1)
            out, self._state = self._session.run(
                None,
                {"input": inp, "state": self._state, "sr": self._sr_tensor},
            )
            self._context = inp[:, -self._context_size :]
            probs.append(float(out[0][0]))
        return probs
