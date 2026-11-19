"""
src/core/deprecations.py
────────────────────────
Startup warnings for settings that still use an old name.
"""

from src.configs.config import Settings
from src.core.logger import get_logger

log = get_logger(__name__)


def warn_deprecated_settings(settings: Settings) -> list[str]:
    """Log one warning per deprecated setting in use and return their names."""
    used = settings.deprecated_storage_settings()
    for name in used:
        log.warning(
            "deprecated_setting",
            setting=name,
            note="The old name still works but will be removed in a later release.",
        )
    return used
