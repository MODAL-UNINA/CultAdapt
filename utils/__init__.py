"""Shared utilities, loaded lazily so data preparation needs no GPU stack."""
from importlib import import_module

_IMAGE_EXPORTS = {"composite_to_rgb", "safe_open_image"}
_MODEL_EXPORTS = {
    "GatingNetwork", "activate_adapter", "build_bnb_config", "build_lora_config",
    "build_multi_adapter_model", "extract_clip_embedding", "load_base_model",
    "load_inference_model", "load_processor",
}
__all__ = ["config", *sorted(_IMAGE_EXPORTS), *sorted(_MODEL_EXPORTS)]


def __getattr__(name):
    if name == "config":
        value = import_module(".config", __name__)
    elif name in _IMAGE_EXPORTS:
        value = getattr(import_module(".image_utils", __name__), name)
    elif name in _MODEL_EXPORTS:
        value = getattr(import_module(".model_utils", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
