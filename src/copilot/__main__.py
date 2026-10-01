"""Entry point: python -m copilot"""

from __future__ import annotations

import logging

from copilot.config import get_settings
from copilot.logging import configure_logging


def main() -> None:
    """Start the application."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logging.getLogger(__name__).info("started in %s", settings.app_env)


if __name__ == "__main__":
    main()
