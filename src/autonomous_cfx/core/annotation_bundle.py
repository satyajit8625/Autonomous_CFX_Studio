"""Autonomous CFX Studio - Versioned Viewport Annotation Bundle.

Captures multi-layer annotation data (clean render, annotation overlay, depth buffer,
face ID buffer, camera view/projection matrices, time/frame metadata) for image-guided
agent corrections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CameraParameters:
    """View and projection parameters corresponding to the viewport capture."""
    yaw: float
    pitch: float
    distance: float
    target: Tuple[float, float, float]
    fov: float
    width: int
    height: int
    view_matrix: List[List[float]]  # 4x4
    projection_matrix: List[List[float]]  # 4x4
    near_clip: float = 0.1
    far_clip: float = 100.0
    camera_name: str = "Free Camera"
    camera_type: str = "free"  # 'free' or 'shot_camera'
    focal_length_mm: float = 50.0
    horizontal_aperture_mm: float = 36.0
    vertical_aperture_mm: float = 24.0
    aspect_ratio: float = 1.5
    film_fit: str = "overscan"
    film_offset_x_mm: float = 0.0
    film_offset_y_mm: float = 0.0
    gate_rect: Optional[Tuple[int, int, int, int]] = None


@dataclass
class AnnotationRegion:
    """A distinct labeled annotation drawn by the artist on a specific frame."""
    region_id: str
    label: str  # e.g. "Left sleeve penetration", "Cuff shape"
    intent_description: str  # e.g. "Remove body penetration here during frames 1045–1060"
    drawing_tool: str  # 'brush', 'outline', 'arrow', 'text'
    stroke_points: List[Tuple[float, float]] = field(default_factory=list)
    color_rgba: Tuple[int, int, int, int] = (255, 60, 60, 200)
    is_protected: bool = False  # If True, region must NOT be altered


@dataclass
class AnnotationBundle:
    """Complete versioned bundle capturing visual and geometric context for correction."""
    bundle_id: str
    shot_name: str
    garment_name: str
    geometry_version: str
    frame: int
    time_seconds: float
    camera: CameraParameters
    regions: List[AnnotationRegion] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # Relative or absolute file paths for binary/image components
    clean_image_path: str = ""
    annotation_overlay_path: str = ""
    depth_buffer_path: str = ""
    face_id_buffer_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to JSON-serializable dictionary."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotationBundle:
        """Construct bundle from dictionary."""
        cam_data = data.pop("camera")
        camera = CameraParameters(**cam_data)
        regions_data = data.pop("regions", [])
        regions = [AnnotationRegion(**r) for r in regions_data]
        return cls(camera=camera, regions=regions, **data)

    def save(self, bundle_dir: str) -> str:
        """Persist metadata manifest and ensure target folder exists."""
        os.makedirs(bundle_dir, exist_ok=True)
        manifest_path = os.path.join(bundle_dir, f"{self.bundle_id}_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return manifest_path

    @classmethod
    def load(cls, manifest_path: str) -> AnnotationBundle:
        """Load annotation bundle from manifest file."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
