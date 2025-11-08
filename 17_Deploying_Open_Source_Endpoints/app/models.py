"""Model utilities for constructing chat LLM clients.

Centralizes configuration of the default chat model and temperature so graphs can
import a single helper without repeating provider-specific wiring.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_together import ChatTogether


def get_chat_model(model_name: str | None = None, *, temperature: float = 0.7) -> Any:
    """Return a configured LangChain ChatTogether client.

    - model_name: optional override. If not provided, uses TOGETHER_MODEL env var, falling back to "openai/gpt-oss-20b".
    - temperature: sampling temperature for the chat model.

    Returns: a LangChain-compatible chat model instance.
    """
    name = model_name or os.environ.get("TOGETHER_MODEL", "openai/gpt-oss-20b")
    return ChatTogether(model=name, temperature=temperature)

