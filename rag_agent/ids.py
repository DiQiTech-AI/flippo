from __future__ import annotations

import secrets
import time


def new_id(prefix: str) -> str:
    millis = int(time.time() * 1000)
    return f"{prefix}_{millis:x}{secrets.token_hex(6)}"

