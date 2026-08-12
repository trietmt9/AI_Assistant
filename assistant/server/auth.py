"""Bearer token auth.

CLAUDE.md safety rules and PLAN.md §2: this machine has a globally routable
address with no NAT, so the endpoint is one misconfiguration away from being
public. Two deliberate choices follow:

* **There is no unauthenticated mode.** If no token is configured, one is
  generated and written to `data/server_token` on first start rather than auth
  being skipped. An "auth off for now" flag is the kind of thing that survives
  into production.
* **`/health` is the only exempt route**, and it returns nothing but liveness —
  no model name, no document counts, nothing worth scraping.
"""

from __future__ import annotations

import logging
import secrets
from pathlib import Path

from fastapi import HTTPException, Request, status

from assistant.config import DATA_DIR, settings

log = logging.getLogger(__name__)

TOKEN_FILE = DATA_DIR / "server_token"
PUBLIC_PATHS = frozenset({"/health"})


def load_or_create_token() -> str:
    """Configured token, else the persisted one, else a fresh one."""
    if settings.server_token:
        return settings.server_token

    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token

    token = secrets.token_urlsafe(32)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)
    log.warning("generated a new server token at %s", TOKEN_FILE)
    return token


def check_bearer(request: Request, token: str) -> None:
    """Raise 401 unless the request carries the right bearer token.

    Accepts the token from `Authorization: Bearer ...` or, for WebSocket
    clients that cannot set headers easily, a `?token=` query parameter.
    """
    if request.url.path in PUBLIC_PATHS:
        return

    supplied = ""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        supplied = header[7:].strip()
    if not supplied:
        supplied = request.query_params.get("token", "")

    # Constant-time: a timing oracle on a 32-byte token is not a realistic
    # attack here, but the correct call costs nothing.
    if not supplied or not secrets.compare_digest(supplied, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def token_path() -> Path:
    return TOKEN_FILE


__all__ = ["load_or_create_token", "check_bearer", "token_path", "PUBLIC_PATHS"]
