"""Autonomous CFX Studio - Animated Shot Camera System.

Implements native production camera optics, animated Alembic camera evaluation,
nested transform hierarchy resolution, film-fit letterboxing/pillarboxing,
navigation locking, character follow tracking, and exact view/projection matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from autonomous_cfx.core.annotation_bundle import CameraParameters
from autonomous_cfx.core.logger import logger


class FilmFitMode(str, Enum):
    """How the camera film aperture is fitted into the viewport window."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    FILL = "fill"
    OVERSCAN = "overscan"


class CameraMode(str, Enum):
    """Viewport viewing mode."""
    FREE_INSPECTION = "free_inspection"
    SHOT_CAMERA = "shot_camera"


@dataclass
class ShotCameraSample:
    """Evaluated camera state at a specific time sample."""
    frame: float
    time_seconds: float
    world_matrix: np.ndarray  # 4x4 row-major camera-to-world transform
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    near_clip: float = 0.1
    far_clip: float = 1000.0
    horizontal_offset_mm: float = 0.0
    vertical_offset_mm: float = 0.0
    lens_squeeze: float = 1.0

    @property
    def film_aspect_ratio(self) -> float:
        """Optical aspect ratio of the film aperture accounting for lens squeeze."""
        denom = max(1e-5, self.vertical_aperture_mm)
        return float((self.horizontal_aperture_mm * self.lens_squeeze) / denom)

    @property
    def eye(self) -> np.ndarray:
        """Camera optical center position in world space."""
        return self.world_matrix[3, :3].copy()

    @property
    def right(self) -> np.ndarray:
        """Camera local right (+X) axis in world space."""
        r = self.world_matrix[0, :3].copy()
        norm = np.linalg.norm(r)
        return r / max(1e-7, norm)

    @property
    def up(self) -> np.ndarray:
        """Camera local up (+Y) axis in world space."""
        u = self.world_matrix[1, :3].copy()
        norm = np.linalg.norm(u)
        return u / max(1e-7, norm)

    @property
    def forward(self) -> np.ndarray:
        """Camera optical look direction (-Z) in world space."""
        b = self.world_matrix[2, :3].copy()
        norm = np.linalg.norm(b)
        return -b / max(1e-7, norm)

    @property
    def target(self) -> np.ndarray:
        """Focal target point in front of camera along optical axis."""
        return self.eye + self.forward * 3.0

    @property
    def horizontal_fov_deg(self) -> float:
        """Horizontal field of view in degrees."""
        return float(2.0 * math.degrees(math.atan(self.horizontal_aperture_mm / (2.0 * max(1e-3, self.focal_length_mm)))))

    @property
    def vertical_fov_deg(self) -> float:
        """Vertical field of view in degrees."""
        return float(2.0 * math.degrees(math.atan(self.vertical_aperture_mm / (2.0 * max(1e-3, self.focal_length_mm)))))


@dataclass
class ShotCamera:
    """Animated shot camera imported from Alembic (.abc) archive."""
    name: str
    full_path: str
    fps: float = 24.0
    start_frame: int = 1
    samples: List[ShotCameraSample] = field(default_factory=list)
    film_fit: FilmFitMode = FilmFitMode.OVERSCAN
    unsupported_features: List[str] = field(default_factory=list)

    def evaluate(self, frame: float) -> ShotCameraSample:
        """Evaluate camera properties and world transform matrix at a specific shot frame."""
        if not self.samples:
            # Default fallback 50mm camera at (0, 1.5, 3)
            eye = np.array([0.0, 1.5, 3.0], dtype=np.float64)
            m = np.eye(4, dtype=np.float64)
            m[3, :3] = eye
            return ShotCameraSample(
                frame=frame,
                time_seconds=(frame - self.start_frame) / max(1.0, self.fps),
                world_matrix=m,
                focal_length_mm=50.0,
                horizontal_aperture_mm=36.0,
                vertical_aperture_mm=24.0,
            )

        if len(self.samples) == 1:
            s0 = self.samples[0]
            return ShotCameraSample(
                frame=frame,
                time_seconds=(frame - self.start_frame) / max(1.0, self.fps),
                world_matrix=s0.world_matrix.copy(),
                focal_length_mm=s0.focal_length_mm,
                horizontal_aperture_mm=s0.horizontal_aperture_mm,
                vertical_aperture_mm=s0.vertical_aperture_mm,
                near_clip=s0.near_clip,
                far_clip=s0.far_clip,
                horizontal_offset_mm=s0.horizontal_offset_mm,
                vertical_offset_mm=s0.vertical_offset_mm,
                lens_squeeze=s0.lens_squeeze,
            )

        # Locate surrounding samples by frame
        frames = [s.frame for s in self.samples]
        if frame <= frames[0]:
            return self.samples[0]
        if frame >= frames[-1]:
            return self.samples[-1]

        # Binary search / linear scan for interval
        idx = 0
        for i in range(len(frames) - 1):
            if frames[i] <= frame <= frames[i + 1]:
                idx = i
                break

        s_a = self.samples[idx]
        s_b = self.samples[idx + 1]
        span = s_b.frame - s_a.frame
        t = (frame - s_a.frame) / max(1e-6, span)

        # Linear matrix and scalar interpolation
        mat_interp = (1.0 - t) * s_a.world_matrix + t * s_b.world_matrix
        # Orthonormalize rotation basis
        r = mat_interp[0, :3]
        u = mat_interp[1, :3]
        f = mat_interp[2, :3]
        r = r / max(1e-7, np.linalg.norm(r))
        u = u - np.dot(u, r) * r
        u = u / max(1e-7, np.linalg.norm(u))
        f = np.cross(r, u)
        f = f / max(1e-7, np.linalg.norm(f))
        mat_interp[0, :3] = r
        mat_interp[1, :3] = u
        mat_interp[2, :3] = f

        return ShotCameraSample(
            frame=frame,
            time_seconds=(frame - self.start_frame) / max(1.0, self.fps),
            world_matrix=mat_interp,
            focal_length_mm=(1.0 - t) * s_a.focal_length_mm + t * s_b.focal_length_mm,
            horizontal_aperture_mm=(1.0 - t) * s_a.horizontal_aperture_mm + t * s_b.horizontal_aperture_mm,
            vertical_aperture_mm=(1.0 - t) * s_a.vertical_aperture_mm + t * s_b.vertical_aperture_mm,
            near_clip=s_a.near_clip,
            far_clip=s_a.far_clip,
            horizontal_offset_mm=(1.0 - t) * s_a.horizontal_offset_mm + t * s_b.horizontal_offset_mm,
            vertical_offset_mm=(1.0 - t) * s_a.vertical_offset_mm + t * s_b.vertical_offset_mm,
            lens_squeeze=s_a.lens_squeeze,
        )

    def compute_film_gate(
        self,
        viewport_width: int,
        viewport_height: int,
        sample: ShotCameraSample,
    ) -> Tuple[int, int, int, int]:
        """Compute pixel coordinates (x, y, w, h) of the film framing gate within the viewport."""
        w_view = max(1, viewport_width)
        h_view = max(1, viewport_height)
        r_view = float(w_view) / float(h_view)
        r_film = sample.film_aspect_ratio

        fit = self.film_fit
        if fit == FilmFitMode.OVERSCAN:
            # Entire film gate visible with letterbox/pillarbox
            if r_view >= r_film:
                # Viewport wider than film: pillarbox left/right
                gate_h = h_view
                gate_w = max(1, int(round(h_view * r_film)))
                gate_x = (w_view - gate_w) // 2
                gate_y = 0
            else:
                # Viewport taller than film: letterbox top/bottom
                gate_w = w_view
                gate_h = max(1, int(round(w_view / max(1e-5, r_film))))
                gate_x = 0
                gate_y = (h_view - gate_h) // 2
        elif fit == FilmFitMode.HORIZONTAL:
            gate_w = w_view
            gate_h = max(1, int(round(w_view / max(1e-5, r_film))))
            gate_x = 0
            gate_y = (h_view - gate_h) // 2
        elif fit == FilmFitMode.VERTICAL:
            gate_h = h_view
            gate_w = max(1, int(round(h_view * r_film)))
            gate_x = (w_view - gate_w) // 2
            gate_y = 0
        elif fit == FilmFitMode.FILL:
            # Fill entire viewport (crop if necessary)
            if r_view >= r_film:
                gate_w = w_view
                gate_h = max(1, int(round(w_view / max(1e-5, r_film))))
                gate_x = 0
                gate_y = (h_view - gate_h) // 2
            else:
                gate_h = h_view
                gate_w = max(1, int(round(h_view * r_film)))
                gate_x = (w_view - gate_w) // 2
                gate_y = 0
        else:
            gate_x, gate_y, gate_w, gate_h = 0, 0, w_view, h_view

        return gate_x, gate_y, gate_w, gate_h

    def compute_view_matrix(self, sample: ShotCameraSample) -> np.ndarray:
        """Compute 4x4 view matrix transforming world space to camera view space."""
        # V = inv(world_matrix)
        try:
            return np.linalg.inv(sample.world_matrix)
        except Exception:
            return np.eye(4, dtype=np.float64)

    def compute_projection_matrix(
        self,
        sample: ShotCameraSample,
    ) -> np.ndarray:
        """Compute standard OpenGL perspective projection matrix for the camera film gate."""
        f = max(1e-3, sample.focal_length_mm)
        w_film = sample.horizontal_aperture_mm
        h_film = sample.vertical_aperture_mm
        n = sample.near_clip
        far = sample.far_clip

        # Frustum bounds at near plane
        l = (-w_film * 0.5 + sample.horizontal_offset_mm) * (n / f)
        r = (w_film * 0.5 + sample.horizontal_offset_mm) * (n / f)
        b = (-h_film * 0.5 + sample.vertical_offset_mm) * (n / f)
        t = (h_film * 0.5 + sample.vertical_offset_mm) * (n / f)

        P = np.zeros((4, 4), dtype=np.float64)
        P[0, 0] = (2.0 * n) / max(1e-7, (r - l))
        P[1, 1] = (2.0 * n) / max(1e-7, (t - b))
        P[2, 0] = (r + l) / max(1e-7, (r - l))
        P[2, 1] = (t + b) / max(1e-7, (t - b))
        P[2, 2] = -(far + n) / max(1e-7, (far - n))
        P[2, 3] = -1.0
        P[3, 2] = -(2.0 * far * n) / max(1e-7, (far - n))

        return P

    def project_world_to_screen(
        self,
        points: np.ndarray,
        frame: float,
        viewport_width: int,
        viewport_height: int,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """Project 3D world points to 2D pixel coordinates and depth within camera framing gate."""
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), (0, 0, viewport_width, viewport_height)

        sample = self.evaluate(frame)
        gate_x, gate_y, gate_w, gate_h = self.compute_film_gate(viewport_width, viewport_height, sample)

        # View transform: points in world -> view space
        V = self.compute_view_matrix(sample)
        homo = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
        p_cam = homo @ V

        # View coordinates
        x_c = p_cam[:, 0]
        y_c = p_cam[:, 1]
        z_c = -p_cam[:, 2]  # positive in front of camera

        # Optical tangent projections
        f = max(1e-3, sample.focal_length_mm)
        tan_fov_x = (sample.horizontal_aperture_mm * 0.5) / f
        tan_fov_y = (sample.vertical_aperture_mm * 0.5) / f

        z_safe = np.maximum(z_c, sample.near_clip)
        x_ndc = (x_c - sample.horizontal_offset_mm * 0.001) / (z_safe * tan_fov_x)
        y_ndc = (y_c - sample.vertical_offset_mm * 0.001) / (z_safe * tan_fov_y)

        # Map NDC [-1, 1] into framing gate pixels
        x_screen = gate_x + (x_ndc + 1.0) * 0.5 * gate_w
        y_screen = gate_y + (1.0 - y_ndc) * 0.5 * gate_h

        screen_coords = np.column_stack([x_screen, y_screen]).astype(np.float32)
        return screen_coords, z_c.astype(np.float32), (gate_x, gate_y, gate_w, gate_h)

    def to_camera_parameters(
        self,
        frame: float,
        width: int,
        height: int,
    ) -> CameraParameters:
        """Construct a versioned CameraParameters dataclass corresponding to active shot frame."""
        sample = self.evaluate(frame)
        gate = self.compute_film_gate(width, height, sample)
        V = self.compute_view_matrix(sample)
        P = self.compute_projection_matrix(sample)

        # Compute nominal yaw, pitch, distance for inspection parity
        eye = sample.eye
        target = sample.target
        fwd = sample.forward
        pitch = float(np.degrees(np.arcsin(np.clip(fwd[1], -1.0, 1.0))))
        yaw = float(np.degrees(np.arctan2(fwd[0], fwd[2])))
        dist = float(np.linalg.norm(target - eye))

        return CameraParameters(
            yaw=yaw,
            pitch=pitch,
            distance=dist,
            target=(float(target[0]), float(target[1]), float(target[2])),
            fov=float(sample.horizontal_fov_deg),
            width=width,
            height=height,
            view_matrix=V.tolist(),
            projection_matrix=P.tolist(),
            near_clip=sample.near_clip,
            far_clip=sample.far_clip,
            camera_name=self.name,
            camera_type="shot_camera",
            focal_length_mm=sample.focal_length_mm,
            horizontal_aperture_mm=sample.horizontal_aperture_mm,
            vertical_aperture_mm=sample.vertical_aperture_mm,
            aspect_ratio=sample.film_aspect_ratio,
            film_fit=self.film_fit.value,
            film_offset_x_mm=sample.horizontal_offset_mm,
            film_offset_y_mm=sample.vertical_offset_mm,
            gate_rect=gate,
        )


@dataclass
class FreeCameraState:
    """Free inspection camera allowing orbit, pan, zoom, and character follow tracking."""
    yaw: float = 30.0
    pitch: float = 12.0
    distance: float = 1.8
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    follow_character: bool = False
    last_character_pos: Optional[np.ndarray] = None

    def update_character_tracking(self, current_char_center: np.ndarray) -> None:
        """Update target position preserving relative viewing vector without resetting zoom."""
        if not self.follow_character:
            self.last_character_pos = current_char_center.copy()
            return

        if self.last_character_pos is not None:
            delta = current_char_center - self.last_character_pos
            # Shift target by character displacement delta
            self.target = self.target + delta.astype(np.float32)

        self.last_character_pos = current_char_center.copy()

    def get_basis_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute eye position, right axis, and up axis."""
        rad_yaw = np.radians(self.yaw)
        rad_pitch = np.radians(self.pitch)

        eye_x = self.target[0] + self.distance * np.cos(rad_pitch) * np.sin(rad_yaw)
        eye_y = self.target[1] + self.distance * np.sin(rad_pitch)
        eye_z = self.target[2] + self.distance * np.cos(rad_pitch) * np.cos(rad_yaw)
        eye = np.array([eye_x, eye_y, eye_z], dtype=np.float32)

        forward = self.target - eye
        forward /= (np.linalg.norm(forward) + 1e-7)

        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_world)
        right /= (np.linalg.norm(right) + 1e-7)

        up = np.cross(right, forward)
        up /= (np.linalg.norm(up) + 1e-7)

        return eye, right, up

    def compute_view_matrix(self) -> np.ndarray:
        """Compute 4x4 row-major view matrix for free camera."""
        eye, right, up = self.get_basis_vectors()
        forward = np.cross(up, right)
        M_cam = np.eye(4, dtype=np.float64)
        M_cam[0, :3] = right
        M_cam[1, :3] = up
        M_cam[2, :3] = -forward
        M_cam[3, :3] = eye
        return np.linalg.inv(M_cam)

    def to_camera_parameters(self, width: int, height: int) -> CameraParameters:
        """Construct CameraParameters dataclass for free inspection camera."""
        V = self.compute_view_matrix()
        # Default perspective projection
        fov = 1.8
        aspect = width / float(max(1, height))
        P = np.zeros((4, 4), dtype=np.float64)
        P[0, 0] = 1.0 / (fov * aspect)
        P[1, 1] = 1.0 / fov
        P[2, 2] = -1.0
        P[2, 3] = -1.0
        P[3, 2] = -0.2

        return CameraParameters(
            yaw=self.yaw,
            pitch=self.pitch,
            distance=self.distance,
            target=(float(self.target[0]), float(self.target[1]), float(self.target[2])),
            fov=fov,
            width=width,
            height=height,
            view_matrix=V.tolist(),
            projection_matrix=P.tolist(),
            near_clip=0.1,
            far_clip=100.0,
            camera_name="Free Inspection Camera",
            camera_type="free",
            aspect_ratio=aspect,
            gate_rect=(0, 0, width, height),
        )
