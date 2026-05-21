from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from .config import AppConfig


@dataclass(frozen=True)
class FlagScaleStatus:
    available: bool
    message: str


def check_flagscale() -> FlagScaleStatus:
    spec = importlib.util.find_spec("flagscale")
    if spec is None:
        return FlagScaleStatus(
            available=False,
            message=(
                "FlagScale is not installed in this environment. The competition requires FlagScale "
                "for valid Track 3 results, so final inference must run in an environment with FlagScale."
            ),
        )
    return FlagScaleStatus(available=True, message="FlagScale package is available.")


def describe_runtime(config: AppConfig) -> dict[str, str]:
    status = check_flagscale()
    return {
        "framework": config.model.framework,
        "model_name": config.model.model_name,
        "flagscale_available": str(status.available),
        "message": status.message,
    }
