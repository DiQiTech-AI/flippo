from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings


@dataclass(frozen=True)
class AvailableModel:
    id: str
    label: str


class ModelSelectionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _is_chat_model(item: dict[str, Any]) -> bool:
    """Exclude catalog rows explicitly identified as non-chat models."""
    task = str(item.get("task") or "").strip().casefold()
    if task and task not in {"chat", "conversational", "text-generation", "text_generation"}:
        return False

    kind = str(item.get("type") or "").strip().casefold()
    if kind and any(
        marker in kind
        for marker in ("audio", "speech", "transcription", "image", "video", "embedding", "rerank")
    ):
        return False

    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        enabled = {str(key).casefold() for key, value in capabilities.items() if value}
    elif isinstance(capabilities, list):
        enabled = {str(value).casefold() for value in capabilities}
    else:
        enabled = set()
    if enabled:
        chat_markers = {"chat", "completions", "chat.completions", "text-generation", "text_generation"}
        media_markers = {"audio", "speech", "transcription", "image", "video", "embeddings", "rerank"}
        if enabled.isdisjoint(chat_markers) and not enabled.isdisjoint(media_markers):
            return False
    return True


async def discover_unsloth_models(settings: Settings, *, timeout_seconds: float = 8.0) -> list[AvailableModel]:
    if not settings.unsloth_api_key:
        raise ModelSelectionError(
            "UNSLOTH_CREDENTIALS_MISSING",
            "UNSLOTH_API_KEY is missing; model discovery is unavailable.",
        )

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(
                _models_url(settings.unsloth_base_url),
                headers={"Authorization": f"Bearer {settings.unsloth_api_key}"},
            )
    except httpx.TimeoutException as exc:
        raise ModelSelectionError(
            "UNSLOTH_DISCOVERY_TIMEOUT",
            "Timed out while checking the models currently loaded in Unsloth Studio.",
        ) from exc
    except httpx.HTTPError as exc:
        raise ModelSelectionError(
            "UNSLOTH_UNAVAILABLE",
            "Could not connect to Unsloth Studio to check the currently loaded model.",
        ) from exc

    if response.status_code in {401, 403}:
        raise ModelSelectionError(
            "UNSLOTH_AUTH_FAILED",
            "Unsloth Studio rejected the configured API key.",
        )
    if response.status_code >= 400:
        raise ModelSelectionError(
            "UNSLOTH_DISCOVERY_FAILED",
            f"Unsloth Studio model discovery failed with HTTP {response.status_code}.",
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise ModelSelectionError(
            "UNSLOTH_DISCOVERY_INVALID",
            "Unsloth Studio returned an invalid model list.",
        ) from exc
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ModelSelectionError(
            "UNSLOTH_DISCOVERY_INVALID",
            "Unsloth Studio returned an unsupported model-list format.",
        )

    by_id: dict[str, AvailableModel] = {}
    supports_loaded_state = False
    for item in rows:
        if not isinstance(item, dict):
            continue
        if "loaded" in item:
            supports_loaded_state = True
        if item.get("loaded") is not True or not _is_chat_model(item):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        label = item.get("display_name")
        by_id.setdefault(
            model_id,
            AvailableModel(model_id, label.strip() if isinstance(label, str) and label.strip() else model_id),
        )

    if rows and not supports_loaded_state:
        raise ModelSelectionError(
            "UNSLOTH_LOADED_STATE_UNSUPPORTED",
            "Unsloth Studio did not identify which listed models are loaded; refusing to guess.",
        )
    return list(by_id.values())


def select_unsloth_model(
    models: list[AvailableModel],
    requested_model_id: str | None,
    configured_model_id: str | None,
) -> tuple[AvailableModel, str]:
    by_id = {model.id: model for model in models}
    if requested_model_id:
        selected = by_id.get(requested_model_id)
        if selected is None:
            raise ModelSelectionError(
                "UNSLOTH_MODEL_NOT_LOADED",
                f"The selected model '{requested_model_id}' is no longer loaded in Unsloth Studio. Refresh the model list.",
            )
        return selected, "explicit"
    if not models:
        raise ModelSelectionError(
            "UNSLOTH_NO_LOADED_MODEL",
            "No chat model is currently loaded in Unsloth Studio.",
        )
    if len(models) == 1:
        return models[0], "only_loaded"
    if configured_model_id and configured_model_id in by_id:
        return by_id[configured_model_id], "configured_preference"
    raise ModelSelectionError(
        "UNSLOTH_MODEL_SELECTION_REQUIRED",
        "Multiple chat models are loaded in Unsloth Studio. Select a model for this run.",
    )


async def resolve_unsloth_model(
    settings: Settings, requested_model_id: str | None
) -> tuple[AvailableModel, str]:
    models = await discover_unsloth_models(settings)
    return select_unsloth_model(models, requested_model_id, settings.unsloth_model_id)
