import os

from openwakeword.custom_verifier_model import train_custom_verifier
from openwakeword.model import Model
from openwakeword.vad import VAD

__all__ = ["Model", "VAD", "train_custom_verifier"]

# Resource directory where downloaded / bundled model files live.
_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "models")


def _resource(name: str) -> str:
    return os.path.join(_RESOURCES, name)


# Feature extractors (mel spectrogram + embedding). ONNX is the default and
# only universally-supported format in this fork; tflite variants remain
# available at the same upstream release but must be opted into explicitly.
FEATURE_MODELS = {
    "embedding": {
        "model_path": _resource("embedding_model.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
    },
    "melspectrogram": {
        "model_path": _resource("melspectrogram.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    },
}

# Silero VAD v6.2.1 (https://github.com/snakers4/silero-vad). v6 uses a single
# combined state tensor and fixed 512-sample windows at 16 kHz — see
# ``openwakeword.vad`` for the full protocol.
VAD_MODELS = {
    "silero_vad": {
        "model_path": _resource("silero_vad.onnx"),
        "download_url": "https://github.com/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx",
    },
}

# Pre-trained wake-word models. Kept at the upstream v0.5.1 URLs; downloading
# prefers the ``.onnx`` variant.
MODELS = {
    "alexa": {
        "model_path": _resource("alexa_v0.1.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.onnx",
    },
    "hey_mycroft": {
        "model_path": _resource("hey_mycroft_v0.1.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_mycroft_v0.1.onnx",
    },
    "hey_jarvis": {
        "model_path": _resource("hey_jarvis_v0.1.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",
    },
    "hey_rhasspy": {
        "model_path": _resource("hey_rhasspy_v0.1.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_rhasspy_v0.1.onnx",
    },
    "timer": {
        "model_path": _resource("timer_v0.1.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/timer_v0.1.onnx",
    },
    "weather": {
        "model_path": _resource("weather_v0.1.onnx"),
        "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/weather_v0.1.onnx",
    },
}

model_class_mappings = {
    "timer": {
        "1": "1_minute_timer",
        "2": "5_minute_timer",
        "3": "10_minute_timer",
        "4": "20_minute_timer",
        "5": "30_minute_timer",
        "6": "1_hour_timer",
    },
}


def get_pretrained_model_paths(inference_framework: str = "onnx") -> list[str]:
    """Return the list of bundled model paths for the given inference framework.

    The default was ``"tflite"`` in upstream; this fork defaults to ``"onnx"``
    — the only universally-supported backend on modern Python. Pass
    ``"tflite"`` explicitly if you have tflite models and the ``[tflite]``
    extra installed.
    """
    if inference_framework == "onnx":
        return [m["model_path"] for m in MODELS.values()]
    if inference_framework == "tflite":
        return [m["model_path"].replace(".onnx", ".tflite") for m in MODELS.values()]
    raise ValueError(
        f"inference_framework must be 'onnx' or 'tflite', got {inference_framework!r}"
    )
