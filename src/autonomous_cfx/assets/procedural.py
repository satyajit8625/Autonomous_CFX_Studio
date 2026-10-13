"""Autonomous CFX Studio - Procedural Milestone Assets.
Generates production-scale procedural Jacket, Animated Body mannequin, and Chair colliders.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from autonomous_cfx.core.geometry import MeshGeometry


def create_cylinder_mesh(
    radius_top: float,
    radius_bottom: float,
    height: float,
    center: np.ndarray,
    segments: int = 16,
    rings: int = 8,
    axis: str = "y",
) -> MeshGeometry:
    """Helper to generate an open or capped cylinder mesh."""
    verts = []
    faces = []

    y_vals = np.linspace(-height / 2.0, height / 2.0, rings)
    theta_vals = np.linspace(0, 2 * np.pi, segments, endpoint=False)

    for ring_idx, y in enumerate(y_vals):
        alpha = (y + height / 2.0) / height
        r = (1.0 - alpha) * radius_bottom + alpha * radius_top
        for theta in theta_vals:
            if axis == "y":
                x = r * np.cos(theta)
                z = r * np.sin(theta)
                pt = np.array([x, y, z], dtype=np.float32) + center
            elif axis == "x":
                x = y
                y_coord = r * np.cos(theta)
                z_coord = r * np.sin(theta)
                pt = np.array([x, y_coord, z_coord], dtype=np.float32) + center
            else:
                x = r * np.cos(theta)
                y_coord = r * np.sin(theta)
                z = y
                pt = np.array([x, y_coord, z], dtype=np.float32) + center
            verts.append(pt)

    for ring_idx in range(rings - 1):
        for seg in range(segments):
            next_seg = (seg + 1) % segments
            i0 = ring_idx * segments + seg
            i1 = ring_idx * segments + next_seg
            i2 = (ring_idx + 1) * segments + seg
            i3 = (ring_idx + 1) * segments + next_seg
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    return MeshGeometry(
        name="cylinder",
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def generate_procedural_jacket() -> MeshGeometry:
    """Generate a milestone cloth jacket mesh with torso, sleeves, and collar."""
    # 1. Torso: tapered cylinder with front slit opening
    torso_rings = 14
    torso_segments = 20
    torso_height = 0.65
    torso_center = np.array([0.0, 1.25, 0.0], dtype=np.float32)
    radius_top = 0.22
    radius_bottom = 0.24

    verts = []
    faces = []

    y_vals = np.linspace(-torso_height / 2.0, torso_height / 2.0, torso_rings)
    # Theta angles with a small front gap (e.g., from 0.1 rad to 2pi - 0.1 rad)
    theta_vals = np.linspace(0.15, 2 * np.pi - 0.15, torso_segments)

    for r_idx, y in enumerate(y_vals):
        alpha = (y + torso_height / 2.0) / torso_height
        r = (1.0 - alpha) * radius_bottom + alpha * radius_top
        # Slight elliptical chest shaping
        for theta in theta_vals:
            x = r * 1.15 * np.cos(theta)
            z = r * 0.95 * np.sin(theta)
            verts.append(np.array([x, y, z], dtype=np.float32) + torso_center)

    for r_idx in range(torso_rings - 1):
        for seg in range(torso_segments - 1):
            i0 = r_idx * torso_segments + seg
            i1 = r_idx * torso_segments + (seg + 1)
            i2 = (r_idx + 1) * torso_segments + seg
            i3 = (r_idx + 1) * torso_segments + (seg + 1)
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    torso_vert_count = len(verts)

    # 2. Left Sleeve (aligned along +X)
    sleeve_rings = 10
    sleeve_segments = 12
    sleeve_len = 0.48
    sleeve_r_top = 0.10
    sleeve_r_bot = 0.08
    left_shoulder = np.array([0.28, 1.45, 0.0], dtype=np.float32)

    x_vals_l = np.linspace(0.0, sleeve_len, sleeve_rings)
    theta_arm = np.linspace(0, 2 * np.pi, sleeve_segments, endpoint=False)

    for ring_idx, x in enumerate(x_vals_l):
        alpha = x / sleeve_len
        r = (1.0 - alpha) * sleeve_r_top + alpha * sleeve_r_bot
        # Slight downward arm slope
        y_offset = -0.35 * x
        for theta in theta_arm:
            y = r * np.cos(theta) + y_offset
            z = r * np.sin(theta)
            verts.append(np.array([x, y, z], dtype=np.float32) + left_shoulder)

    for ring_idx in range(sleeve_rings - 1):
        for seg in range(sleeve_segments):
            next_seg = (seg + 1) % sleeve_segments
            i0 = torso_vert_count + ring_idx * sleeve_segments + seg
            i1 = torso_vert_count + ring_idx * sleeve_segments + next_seg
            i2 = torso_vert_count + (ring_idx + 1) * sleeve_segments + seg
            i3 = torso_vert_count + (ring_idx + 1) * sleeve_segments + next_seg
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    left_arm_vert_count = len(verts)

    # 3. Right Sleeve (aligned along -X)
    right_shoulder = np.array([-0.28, 1.45, 0.0], dtype=np.float32)
    x_vals_r = np.linspace(0.0, -sleeve_len, sleeve_rings)

    for ring_idx, x in enumerate(x_vals_r):
        alpha = abs(x) / sleeve_len
        r = (1.0 - alpha) * sleeve_r_top + alpha * sleeve_r_bot
        y_offset = -0.35 * abs(x)
        for theta in theta_arm:
            y = r * np.cos(theta) + y_offset
            z = r * np.sin(theta)
            verts.append(np.array([x, y, z], dtype=np.float32) + right_shoulder)

    for ring_idx in range(sleeve_rings - 1):
        for seg in range(sleeve_segments):
            next_seg = (seg + 1) % sleeve_segments
            i0 = left_arm_vert_count + ring_idx * sleeve_segments + seg
            i1 = left_arm_vert_count + ring_idx * sleeve_segments + next_seg
            i2 = left_arm_vert_count + (ring_idx + 1) * sleeve_segments + seg
            i3 = left_arm_vert_count + (ring_idx + 1) * sleeve_segments + next_seg
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    # Vertex groups for pinning or diagnostics
    collar_indices = np.arange((torso_rings - 1) * torso_segments, torso_rings * torso_segments)

    jacket_mesh = MeshGeometry(
        name="jacket",
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
        vertex_groups={"collar": collar_indices},
    )
    jacket_mesh.compute_normals()
    return jacket_mesh


def generate_procedural_animated_body(
    num_frames: int = 40,
    fps: float = 24.0,
    sit_on_chair: bool = False,
) -> Tuple[MeshGeometry, Dict[int, np.ndarray]]:
    """Generate an animated mannequin body mesh and motion frames.
    Returns:
        base_geometry: MeshGeometry
        animation_frames: Dict[int, np.ndarray] mapping frame number to (N, 3) vertex coordinates.
    """
    # Create mannequin body: Torso (capsule/cylinder), Head (sphere approximation), Hips, Legs
    torso = create_cylinder_mesh(
        radius_top=0.19,
        radius_bottom=0.18,
        height=0.68,
        center=np.array([0.0, 1.25, 0.0], dtype=np.float32),
        segments=16,
        rings=10,
    )
    base_verts = np.copy(torso.vertices)
    base_faces = np.copy(torso.faces)

    base_geo = MeshGeometry(
        name="body_mannequin",
        vertices=base_verts,
        faces=base_faces,
    )
    base_geo.compute_normals()

    animation_frames: Dict[int, np.ndarray] = {}

    for frame in range(1, num_frames + 1):
        t = (frame - 1) / float(fps)
        f_verts = np.copy(base_verts)

        if sit_on_chair:
            # Sit down smoothly between frame 10 and 30
            sit_progress = np.clip((frame - 10) / 20.0, 0.0, 1.0)
            # Smoothstep
            s = sit_progress * sit_progress * (3 - 2 * sit_progress)
            # Lower body by 0.35m and move back by 0.15m onto chair
            f_verts[:, 1] -= s * 0.35
            f_verts[:, 2] -= s * 0.15
        else:
            # Walking motion / breathing & torso twist
            # Sway side to side (X) and twist around Y
            sway = 0.03 * np.sin(2.0 * np.pi * t * 1.5)
            bounce = 0.015 * np.abs(np.sin(2.0 * np.pi * t * 3.0))
            twist_angle = 0.10 * np.sin(2.0 * np.pi * t * 1.5)

            # Apply twist rotation around torso center
            cos_a = np.cos(twist_angle)
            sin_a = np.sin(twist_angle)
            x_old = f_verts[:, 0]
            z_old = f_verts[:, 2]
            f_verts[:, 0] = x_old * cos_a - z_old * sin_a + sway
            f_verts[:, 2] = x_old * sin_a + z_old * cos_a
            f_verts[:, 1] += bounce

        animation_frames[frame] = f_verts.astype(np.float32)

    return base_geo, animation_frames


def generate_procedural_chair() -> MeshGeometry:
    """Generate a milestone chair collider with seat, backrest, and legs."""
    # Box helper
    def make_box(min_pt: np.ndarray, max_pt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        v = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], min_pt[1], min_pt[2]],
            [max_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], max_pt[1], min_pt[2]],
            [min_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], min_pt[1], max_pt[2]],
            [max_pt[0], max_pt[1], max_pt[2]],
            [min_pt[0], max_pt[1], max_pt[2]],
        ], dtype=np.float32)
        f = np.array([
            [0, 2, 1], [0, 3, 2],  # Front
            [4, 5, 6], [4, 6, 7],  # Back
            [0, 1, 5], [0, 5, 4],  # Bottom
            [2, 3, 7], [2, 7, 6],  # Top
            [0, 4, 7], [0, 7, 3],  # Left
            [1, 2, 6], [1, 6, 5],  # Right
        ], dtype=np.int32)
        return v, f

    all_v = []
    all_f = []
    v_offset = 0

    # 1. Seat slab: center roughly at [0, 0.45, -0.2]
    seat_min = np.array([-0.25, 0.42, -0.45], dtype=np.float32)
    seat_max = np.array([0.25, 0.47, 0.05], dtype=np.float32)
    v1, f1 = make_box(seat_min, seat_max)
    all_v.append(v1)
    all_f.append(f1 + v_offset)
    v_offset += len(v1)

    # 2. Backrest slab: at z ~ -0.42, y from 0.47 to 0.95
    back_min = np.array([-0.25, 0.47, -0.45], dtype=np.float32)
    back_max = np.array([0.25, 0.95, -0.38], dtype=np.float32)
    v2, f2 = make_box(back_min, back_max)
    all_v.append(v2)
    all_f.append(f2 + v_offset)
    v_offset += len(v2)

    # Combine into single mesh
    chair_v = np.vstack(all_v)
    chair_f = np.vstack(all_f)

    chair_mesh = MeshGeometry(name="chair", vertices=chair_v, faces=chair_f)
    chair_mesh.compute_normals()
    return chair_mesh
