# Change Log

## v0.7.0 - 2026-04-15 — `dankorotin/openWakeWord` fork

First release of the `dankorotin/openWakeWord` fork. Starts from upstream commit
`368c037` (post-v0.6.0 main) and diverges on packaging, VAD, and default backend.

### Added

* **Silero VAD v6.2.1 support** via a full port of the v5+ I/O contract
  (single combined `(2, 1, 128)` state tensor, fixed 512-sample windows at
  16 kHz, 64-sample rolling context). Upstream shipped v4 only. The public
  `VAD` class keeps its signature (`predict`, `__call__`, `reset_states`,
  `prediction_buffer`) so downstream code works unchanged.
* `tests/test_vad.py` — new unit tests that lock in the v6 protocol:
  non-zero scores on speech, near-zero on silence, state resets cleanly,
  arbitrary chunk sizes work via internal buffering.
* `[tflite]`, `[speex]` install extras for the optional backends.

### Changed

* **Requires Python ≥ 3.13.** Drops 3.10–3.12 and Windows. Simpler matrix.
  Users on older Python can continue to use upstream.
* **Default inference framework flipped from `"tflite"` to `"onnx"`.** ONNX
  is the only universally-supported backend on modern Python; `ai-edge-litert`
  has no 3.13 wheel as of 2026-04. TFLite remains available via the
  `[tflite]` extra for users with existing `.tflite` assets.
* **Packaging migrated to `pyproject.toml`-only** (PEP 621); `setup.py`
  removed. Ruff replaces flake8; MyPy kept.
* Custom verifier module uses `from __future__ import annotations` so the
  `openwakeword.Model` reference in its signatures no longer creates an
  import-order dependency inside the package.
* CI reduced to a single Linux × Python 3.13 job.

### Removed

* `setup.py` and the PyPI publish workflow — this fork is intended to be
  consumed directly from Git (`pip install git+https://github.com/dankorotin/openWakeWord`).
* `tensorflow`, `tensorflow_probability`, `onnx_tf` and related 2022-era pins
  from the `[full]`/`[train]` extra. Training is torch-only; ONNX is the
  export target.
* `speexdsp-ns` and `ai-edge-litert` from the default install. Both are
  still reachable via the `[speex]` and `[tflite]` extras respectively.

### Known follow-ups (not in this release)

* Gradient-accumulation bug in `train.py` (upstream issue #316) and the
  data-loader batch-repeat bug (upstream PR #202) — will be fixed in a
  dedicated commit with regression tests.
* Bugbear (`B*`) lint sweep on the legacy files `data.py`, `train.py`,
  `model.py`, `utils.py`. Currently suppressed per-file to keep CI green
  without behaviour changes.

---

## v0.6.0 - 2023/06/15

### Added

* Various bug fixes, and some new functionality in `model.py` to control repeated detections

### Changed

* Models are no longer included in the PyPi package, and must be downloaded separately

### Removed

## v0.5.0 - 2023/06/15

### Added

* A new wakeword model, "hey rhasspy"
* Added support for tflite versions of the melspectrogram model, embedding model, and pre-trained wakeword models
* Added an inference framework argument to allow users to select either ONNX or tflite as the inference framework
* The `detect_from_microphone.py` example now supports additional arguments and has improved console formatting

### Changed

* Made tflite the default inference framework for linux platforms due to improved efficiency, with windows still using ONNX as the default given the lack of pre-built Windows WHLs for the tflite runtime (https://pypi.org/project/tflite/)
* Adjusted the default provider arguments for onnx models to avoid warnings (https://github.com/dscripka/openWakeWord/issues/27)

### Removed