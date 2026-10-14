"""Small adapter for passing a verified application identity to Memori.

IMPORTANT: principal must come from trusted authentication middleware (for
example, a verified JWT/session), never directly from request JSON or a header
that the client can freely choose.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .api import memori


def memori_for_principal(
    principal: Mapping[str, Any],
    user_input: str,
    *,
    source: str = "user",
) -> dict:
    """Call Memori using the subject of an already-authenticated principal.

    This adapter deliberately has no user_id argument. The calling application
    must authenticate the request and pass the verified identity object.
    """
    user_id = principal.get("sub") or principal.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Authenticated principal must contain a non-empty sub or user_id.")
    return memori(user_id=user_id.strip(), user_input=user_input, source=source)
