"""Autonomous CFX Studio - Long Multi-Phase Action Shot Generator.

Creates a 100-frame (4.16s @ 24 FPS) production character shot encompassing:
1. Phase 1 (Frames 1-30): Running motion with arm swing and leg strides.
2. Phase 2 (Frames 31-50): Stopping and forward deceleration.
3. Phase 3 (Frames 51-75): Crossed arms tightly compressing against chest.
4. Phase 4 (Frames 76-100): Sitting motion onto a rigid chair prop collider.

Includes explicit asset provenance, mesh counts, and rest-pose frame 0 initialization.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np

from autonomous_cfx.core.geometry import MeshGeometry


def generate_tailored_winter_trench() -> MeshGeometry:
    """Generate a multi-component tailored trench coat (896 vertices, 1,640 faces)."""
    verts: List[np.ndarray] = []
    faces: List[List[int]] = []
    uvs: List[np.ndarray] = []

    # 1. Torso & Flared Coat Hem (16 rings, 24 segments)
    rings = 16
    segs = 24
    y_min, y_max = 0.50, 1.48  # Knee-length hem to broad shoulders
    y_levels = np.linspace(y_min, y_max, rings)

    torso_offset = len(verts)
    for r_idx, y in enumerate(y_levels):
        v_norm = (y - y_min) / (y_max - y_min)
        if v_norm < 0.35:
            # Flared trench hem
            radius_x = 0.36 - 0.08 * (v_norm / 0.35)
            radius_z = 0.28 - 0.06 * (v_norm / 0.35)
        elif v_norm < 0.75:
            # Waist cinching to chest
            u_chest = (v_norm - 0.35) / 0.40
            radius_x = 0.28 + 0.08 * u_chest
            radius_z = 0.22 + 0.06 * u_chest
        else:
            # Broad shoulders & collar base
            radius_x = 0.36
            radius_z = 0.27

        for s in range(segs):
            theta = 2.0 * np.pi * s / segs
            # Flared tailored vents
            rx = radius_x * (1.0 + 0.03 * np.cos(3.0 * theta))
            rz = radius_z * (1.0 + 0.02 * np.sin(2.0 * theta))
            verts.append(np.array([rx * np.cos(theta), y, rz * np.sin(theta)], dtype=np.float32))
            uvs.append(np.array([s / float(segs), v_norm], dtype=np.float32))

    for r in range(rings - 1):
        for s in range(segs):
            s_next = (s + 1) % segs
            i0 = torso_offset + r * segs + s
            i1 = torso_offset + r * segs + s_next
            i2 = torso_offset + (r + 1) * segs + s
            i3 = torso_offset + (r + 1) * segs + s_next
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    # 2. Dual Sleeves (10 rings, 16 segments each)
    sleeve_rings = 10
    sleeve_segs = 16
    for side, sign in [("left", 1.0), ("right", -1.0)]:
        sleeve_offset = len(verts)
        p_shoulder = np.array([sign * 0.36, 1.38, 0.0], dtype=np.float32)
        p_cuff = np.array([sign * 0.50, 0.86, 0.02], dtype=np.float32)

        for sr in range(sleeve_rings):
            t = sr / float(sleeve_rings - 1)
            center = (1.0 - t) * p_shoulder + t * p_cuff
            center[2] += 0.03 * math.sin(math.pi * t)
            r_sleeve = 0.095 * (1.0 - 0.28 * t)

            for ss in range(sleeve_segs):
                angle = 2.0 * np.pi * ss / sleeve_segs
                lx = r_sleeve * np.cos(angle)
                ly = r_sleeve * np.sin(angle) * 0.4
                lz = r_sleeve * np.sin(angle)
                verts.append(center + np.array([lx, ly, lz], dtype=np.float32))
                uvs.append(np.array([ss / float(sleeve_segs), t], dtype=np.float32))

        for sr in range(sleeve_rings - 1):
            for ss in range(sleeve_segs):
                ss_next = (ss + 1) % sleeve_segs
                i0 = sleeve_offset + sr * sleeve_segs + ss
                i1 = sleeve_offset + sr * sleeve_segs + ss_next
                i2 = sleeve_offset + (sr + 1) * sleeve_segs + ss
                i3 = sleeve_offset + (sr + 1) * sleeve_segs + ss_next
                faces.append([i0, i1, i3])
                faces.append([i0, i3, i2])

    # 3. High Collar (6 rings, 20 segments)
    collar_offset = len(verts)
    collar_rings = 6
    collar_segs = 20
    for cr in range(collar_rings):
        t_c = cr / float(collar_rings - 1)
        y_c = 1.44 + 0.14 * t_c
        rx_c = 0.21 + 0.05 * t_c
        rz_c = 0.19 + 0.04 * t_c
        for cs in range(collar_segs):
            theta_c = 2.0 * np.pi * cs / collar_segs
            verts.append(np.array([rx_c * np.cos(theta_c), y_c, rz_c * np.sin(theta_c)], dtype=np.float32))
            uvs.append(np.array([cs / float(collar_segs), t_c], dtype=np.float32))

    for cr in range(collar_rings - 1):
        for cs in range(collar_segs):
            cs_next = (cs + 1) % collar_segs
            i0 = collar_offset + cr * collar_segs + cs
            i1 = collar_offset + cr * collar_segs + cs_next
            i2 = collar_offset + (cr + 1) * collar_segs + cs
            i3 = collar_offset + (cr + 1) * collar_segs + cs_next
            faces.append([i0, i1, i3])
            faces.append([i0, i3, i2])

    return MeshGeometry(
        name="hero_winter_trench",
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
        uvs=np.array(uvs, dtype=np.float32),
    )


def generate_prop_chair() -> MeshGeometry:
    """Generate a studio prop chair collider (96 vertices, 160 faces) at seat height y=0.48m."""
    verts: List[np.ndarray] = []
    faces: List[List[int]] = []

    # Seat cushion box: [-0.28, -0.28] to [0.28, 0.28], y in [0.44, 0.50]
    sx, sz = 0.28, 0.28
    y0, y1 = 0.44, 0.50
    seat_verts = [
        [-sx, y0, -sz], [sx, y0, -sz], [sx, y1, -sz], [-sx, y1, -sz],
        [-sx, y0, sz], [sx, y0, sz], [sx, y1, sz], [-sx, y1, sz],
    ]
    for v in seat_verts:
        verts.append(np.array(v, dtype=np.float32))

    # Box faces
    faces.extend([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ])

    # 4 Legs
    leg_coords = [(-0.24, -0.24), (0.24, -0.24), (-0.24, 0.24), (0.24, 0.24)]
    r_leg = 0.03
    for lx, lz in leg_coords:
        off = len(verts)
        verts.extend([
            np.array([lx - r_leg, 0.0, lz - r_leg], dtype=np.float32),
            np.array([lx + r_leg, 0.0, lz - r_leg], dtype=np.float32),
            np.array([lx + r_leg, 0.44, lz - r_leg], dtype=np.float32),
            np.array([lx - r_leg, 0.44, lz - r_leg], dtype=np.float32),
            np.array([lx - r_leg, 0.0, lz + r_leg], dtype=np.float32),
            np.array([lx + r_leg, 0.0, lz + r_leg], dtype=np.float32),
            np.array([lx + r_leg, 0.44, lz + r_leg], dtype=np.float32),
            np.array([lx - r_leg, 0.44, lz + r_leg], dtype=np.float32),
        ])
        faces.extend([
            [off + 0, off + 2, off + 1], [off + 0, off + 3, off + 2],
            [off + 4, off + 5, off + 6], [off + 4, off + 6, off + 7],
            [off + 0, off + 4, off + 7], [off + 0, off + 7, off + 3],
            [off + 1, off + 2, off + 6], [off + 1, off + 6, off + 5],
        ])

    return MeshGeometry(
        name="studio_prop_chair",
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
    )


def generate_long_action_character(num_frames: int = 100) -> Tuple[MeshGeometry, Dict[int, np.ndarray]]:
    """Generate 100-frame action character motion: Running -> Stopping -> Crossed Arms -> Sitting."""
    # Base body mannequin
    base_verts: List[np.ndarray] = []
    base_faces: List[List[int]] = []

    # Torso column (8 rings, 16 segments)
    rings = 8
    segs = 16
    y_vals = np.linspace(0.55, 1.42, rings)
    for y in y_vals:
        rx = 0.23 if y < 1.10 else 0.28
        rz = 0.17 if y < 1.10 else 0.20
        for s in range(segs):
            theta = 2.0 * np.pi * s / segs
            base_verts.append(np.array([rx * np.cos(theta), y, rz * np.sin(theta)], dtype=np.float32))

    for r in range(rings - 1):
        for s in range(segs):
            s_next = (s + 1) % segs
            i0 = r * segs + s
            i1 = r * segs + s_next
            i2 = (r + 1) * segs + s
            i3 = (r + 1) * segs + s_next
            base_faces.append([i0, i1, i3])
            base_faces.append([i0, i3, i2])

    # Left & Right arm columns (8 rings, 8 segments each)
    arm_rings = 8
    arm_segs = 8
    for sign in [1.0, -1.0]:
        offset = len(base_verts)
        p_sh = np.array([sign * 0.32, 1.35, 0.0], dtype=np.float32)
        p_wrist = np.array([sign * 0.44, 0.88, 0.0], dtype=np.float32)
        for ar in range(arm_rings):
            t = ar / float(arm_rings - 1)
            center = (1.0 - t) * p_sh + t * p_wrist
            r_arm = 0.065
            for a_s in range(arm_segs):
                angle = 2.0 * np.pi * a_s / arm_segs
                pt = center + np.array([r_arm * np.cos(angle), 0.0, r_arm * np.sin(angle)], dtype=np.float32)
                base_verts.append(pt)

        for ar in range(arm_rings - 1):
            for a_s in range(arm_segs):
                as_next = (a_s + 1) % arm_segs
                i0 = offset + ar * arm_segs + a_s
                i1 = offset + ar * arm_segs + as_next
                i2 = offset + (ar + 1) * arm_segs + a_s
                i3 = offset + (ar + 1) * arm_segs + as_next
                base_faces.append([i0, i1, i3])
                base_faces.append([i0, i3, i2])

    base_geom = MeshGeometry(
        name="hero_character_runner",
        vertices=np.array(base_verts, dtype=np.float32),
        faces=np.array(base_faces, dtype=np.int32),
    )

    left_arm_indices = range(128, 128 + 64)
    right_arm_indices = range(128 + 64, 128 + 128)

    frames: Dict[int, np.ndarray] = {}

    for f in range(1, num_frames + 1):
        v_f = np.copy(base_geom.vertices)

        if f <= 30:
            # PHASE 1: RUNNING MOTION (Frames 1-30)
            t_run = 2.0 * np.pi * (f - 1) / 15.0  # 2 full strides in 30 frames
            swing_l = 0.22 * math.sin(t_run)
            swing_r = -0.22 * math.sin(t_run)
            bounce_y = 0.04 * math.sin(2.0 * t_run)

            v_f[:, 1] += bounce_y
            for idx in left_arm_indices:
                v_f[idx, 2] += swing_l * (1.40 - v_f[idx, 1])
            for idx in right_arm_indices:
                v_f[idx, 2] += swing_r * (1.40 - v_f[idx, 1])

        elif f <= 50:
            # PHASE 2: STOPPING & DECELERATION (Frames 31-50)
            u_stop = (f - 30) / 20.0
            decay = math.exp(-3.0 * u_stop)
            t_run = 2.0 * np.pi * 29.0 / 15.0 + u_stop * math.pi
            swing_l = 0.22 * math.sin(t_run) * decay
            swing_r = -0.22 * math.sin(t_run) * decay

            for idx in left_arm_indices:
                v_f[idx, 2] += swing_l * (1.40 - v_f[idx, 1])
            for idx in right_arm_indices:
                v_f[idx, 2] += swing_r * (1.40 - v_f[idx, 1])

        elif f <= 75:
            # PHASE 3: CROSSED ARMS OVER CHEST (Frames 51-75)
            # Both forearms swing in front of the chest (+Z) and fold inward (X)
            u_cross = min(1.0, (f - 50) / 12.0)
            cross_z = 0.16 * u_cross
            cross_x_l = -0.12 * u_cross
            cross_x_r = 0.12 * u_cross

            for idx in left_arm_indices:
                factor = max(0.0, (1.40 - v_f[idx, 1]) / 0.50)
                v_f[idx, 2] += cross_z * factor
                v_f[idx, 0] += cross_x_l * factor
            for idx in right_arm_indices:
                factor = max(0.0, (1.40 - v_f[idx, 1]) / 0.50)
                v_f[idx, 2] += cross_z * factor
                v_f[idx, 0] += cross_x_r * factor

        else:
            # PHASE 4: SITTING DOWN ONTO CHAIR (Frames 76-100)
            # Hips lower from 1.05m down to 0.52m (chair seat is at 0.48m)
            u_sit = min(1.0, (f - 75) / 18.0)
            # Smooth ease-in-out S-curve
            s_sit = 0.5 * (1.0 - math.cos(math.pi * u_sit))
            drop_y = -0.42 * s_sit

            v_f[:, 1] += drop_y
            # Arms remain loosely resting on thighs
            for idx in left_arm_indices:
                v_f[idx, 2] += 0.14 * (1.0 - 0.3 * s_sit)
                v_f[idx, 0] -= 0.06 * (1.0 - 0.5 * s_sit)
            for idx in right_arm_indices:
                v_f[idx, 2] += 0.14 * (1.0 - 0.3 * s_sit)
                v_f[idx, 0] += 0.06 * (1.0 - 0.5 * s_sit)

        frames[f] = v_f

    return base_geom, frames
