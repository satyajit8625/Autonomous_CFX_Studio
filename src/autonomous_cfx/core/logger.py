"""Autonomous CFX Studio - Structured Logger and Telemetry.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


class CFXLogger:
    """Centralized logger for Autonomous CFX Studio."""

    _logger: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: str = "AutonomousCFX") -> logging.Logger:
        if cls._logger is None:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            cls._logger = logger
        return cls._logger


logger = CFXLogger.get_logger()
