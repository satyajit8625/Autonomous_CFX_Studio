"""Autonomous CFX Studio - Assets Package."""

from autonomous_cfx.assets.procedural import (
    generate_procedural_jacket,
    generate_procedural_animated_body,
    generate_procedural_chair,
)
from autonomous_cfx.assets.asset_manager import AssetManager

__all__ = [
    "generate_procedural_jacket",
    "generate_procedural_animated_body",
    "generate_procedural_chair",
    "AssetManager",
]
