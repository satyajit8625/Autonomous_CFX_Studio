"""Core package for Autonomous CFX Studio."""

from autonomous_cfx.core.geometry import MeshGeometry, BoundingBox
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.project import ProjectManager, ProjectConfig
from autonomous_cfx.core.logger import logger

__all__ = [
    "MeshGeometry",
    "BoundingBox",
    "SimulationCache",
    "ProjectManager",
    "ProjectConfig",
    "logger",
]
