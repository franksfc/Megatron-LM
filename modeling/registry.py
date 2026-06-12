"""Explicit model registry for recurrent LM baselines."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODEL_MODULES = {
    "llama_loop": "modeling.llama_loop",
    "llama_new": "modeling.llama_new",
    "llama_orin": "modeling.llama_orin",
    "llama_orin_ssm": "modeling.llama_orin_ssm",
    "llama_ours": "modeling.llama_ours",
    "llama_pause": "modeling.llama_pause",
    "modeling_llama_loop": "modeling.llama_loop",
    "modeling_llama_new": "modeling.llama_new",
    "modeling_llama_orin": "modeling.llama_orin",
    "modeling_llama_orin_legacylegacy": "modeling.llama_orin",
    "modeling_llama_ours": "modeling.llama_ours",
    "modeling_llama_pause": "modeling.llama_pause",
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
