"""Explicit model registry for recurrent LM baselines."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODEL_MODULES = {
    "llama_orin_ssm": "modeling.llama_orin_ssm",
}


def build_model(name: str, **kwargs: Any) -> Any:
    """Build a model implementation by registry key."""

    try:
        module_name = _MODEL_MODULES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_MODULES))
        raise ValueError(f"Unknown model implementation {name!r}. Available: {available}") from exc

    module = import_module(module_name)
    return module.build_model(**kwargs)
