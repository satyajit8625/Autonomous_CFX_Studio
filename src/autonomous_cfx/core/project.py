"""Autonomous CFX Studio - Project and Asset Storage Management.
Maintains clean project hierarchy: shots, takes, versioned caches, and manifests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomous_cfx.core.logger import logger


@dataclass
class ProjectConfig:
    name: str = "CFX_Project"
    root_dir: str = ""
    fps: float = 24.0
    created_at: str = ""
    description: str = "Autonomous CFX Studio Project"


class ProjectManager:
    """Manages project directories, shots, asset manifests, and versioned caches."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.shots_dir = self.root_dir / "shots"
        self.assets_dir = self.root_dir / "assets"
        self.caches_dir = self.root_dir / "caches"
        self.config_file = self.root_dir / "project.json"
        self.config = ProjectConfig(root_dir=str(self.root_dir))
        self._initialize_structure()

    def _initialize_structure(self) -> None:
        """Create project directory tree if not present."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.caches_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_file.exists():
            self.save_config()
        else:
            self.load_config()

    def save_config(self) -> None:
        """Persist project configuration to JSON."""
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, indent=2)

    def load_config(self) -> None:
        """Load project configuration from JSON."""
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.config = ProjectConfig(**data)
        except Exception as e:
            logger.warning(f"Failed to load project config: {e}. Keeping defaults.")

    def get_shot_dir(self, shot_name: str) -> Path:
        """Get or create directory for a specific shot."""
        shot_path = self.shots_dir / shot_name
        shot_path.mkdir(parents=True, exist_ok=True)
        return shot_path

    def get_cache_version_dir(self, shot_name: str, cloth_name: str, version: int = 1) -> Path:
        """Get versioned directory for simulated caches: shots/<shot>/caches/<cloth>/v001/"""
        cache_path = self.get_shot_dir(shot_name) / "caches" / cloth_name / f"v{version:03d}"
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def get_next_cache_version(self, shot_name: str, cloth_name: str) -> int:
        """Determine next available cache version number."""
        parent_path = self.get_shot_dir(shot_name) / "caches" / cloth_name
        if not parent_path.exists():
            return 1
        existing = [
            int(p.name[1:])
            for p in parent_path.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
        ]
        return max(existing) + 1 if existing else 1
