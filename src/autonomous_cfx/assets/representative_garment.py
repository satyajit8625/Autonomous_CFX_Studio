"""Autonomous CFX Studio - Representative Tailored Garment & Character Shot Generator.
Creates a production-scale tailored winter shearling coat with structured collar, lapels,
curved sleeves, and multi-frame deforming body collider with arm swing dynamics.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np

from autonomous_cfx.core.geometry import MeshGeometry


def generate_tailored_shearling_coat() -> MeshGeometry:
    """Generate a multi-component tailored shearling coat (~1,300 vertices).

    Structure:
    - Torso panels: Flared hip hem, cinched waist, broad shoulders, open front lapels.
    - Dual sleeves: Tapered upper arms and forearms with cuff bands.
    - High shearling collar: Upturned neck ring with curved lapel folds.
    """
    verts_list: List[np.ndarray] = []
    faces_list: List[List[int]] = []
    uvs_list: List[np.ndarray] = []

    # 1. Torso Panels (Rings: 14, Segments: 24)
    rings = 14
    segs = 24
    y_min, y_max = 0.60, 1.45  # From lower thighs to shoulders
    y_levels = np.linspace(y_min, y_max, rings)

    torso_offset = len(verts_list)
    for r_idx, y in enumerate(y_levels):
        # Flared hem at bottom, narrow waist at y=1.0, broad chest/shoulders at y=1.35
        v_norm = (y - y_min) / (y_max - y_min)
        if v_norm < 0.4:
            # Hem flare
            radius_x = 0.32 - 0.06 * (v_norm / 0.4)
            radius_z = 0.24 - 0.04 * (v_norm / 0.4)
        elif v_norm < 0.8:
            # Waist to chest
            u_chest = (v_norm - 0.4) / 0.4
            radius_x = 0.26 + 0.08 * u_chest
            radius_z = 0.20 + 0.06 * u_chest
        else:
            # Shoulders
            radius_x = 0.34
            radius_z = 0.26

        for s_idx in range(segs):
            theta = 2.0 * np.pi * s_idx / segs
            # Open front lapel slit around theta = -pi/2 (front facing +Z)
            # Add subtle tailored seams
            rx = radius_x * (1.0 + 0.04 * np.cos(4.0 * theta))
            rz = radius_z * (1.0 + 0.02 * np.sin(2.0 * theta))

            vx = rx * np.cos(theta)
            vy = y
            vz = rz * np.sin(theta)
            verts_list.append(np.array([vx, vy, vz], dtype=np.float32))
            uvs_list.append(np.array([s_idx / float(segs), v_norm], dtype=np.float32))

    for r in range(rings - 1):
        for s in range(segs):
            s_next = (s + 1) % segs
            i0 = torso_offset + r * segs + s
            i1 = torso_offset + r * segs + s_next
            i2 = torso_offset + (r + 1) * segs + s
            i3 = torso_offset + (r + 1) * segs + s_next
            faces_list.append([i0, i1, i3])
            faces_list.append([i0, i3, i2])

    # 2. Left & Right Sleeves (10 rings, 16 segments each)
    sleeve_rings = 10
    sleeve_segs = 16
    for side, sign in [("left", 1.0), ("right", -1.0)]:
        sleeve_offset = len(verts_list)
        # Sleeve extends from shoulder (x=±0.35, y=1.35) downward to wrist (x=±0.45, y=0.85, z=0.05)
        shoulder_pt = np.array([sign * 0.35, 1.35, 0.0], dtype=np.float32)
        wrist_pt = np.array([sign * 0.48, 0.85, 0.04], dtype=np.float32)

        for sr in range(sleeve_rings):
            t = sr / float(sleeve_rings - 1)
            center = (1.0 - t) * shoulder_pt + t * wrist_pt
            # Curved elbow bulge around t=0.5
            center[2] += 0.04 * math.sin(math.pi * t)
            # Taper radius from 0.09 at shoulder to 0.065 at cuff
            r_sleeve = 0.095 * (1.0 - 0.3 * t)

            for ss in range(sleeve_segs):
                angle = 2.0 * np.pi * ss / sleeve_segs
                # Local coordinates perpendicular to arm axis
                lx = r_sleeve * np.cos(angle)
                ly = r_sleeve * np.sin(angle) * 0.4
                lz = r_sleeve * np.sin(angle)
                pt = center + np.array([lx, ly, lz], dtype=np.float32)
                verts_list.append(pt)
                uvs_list.append(np.array([ss / float(sleeve_segs), t], dtype=np.float32))

        for sr in range(sleeve_rings - 1):
            for ss in range(sleeve_segs):
                ss_next = (ss + 1) % sleeve_segs
                i0 = sleeve_offset + sr * sleeve_segs + ss
                i1 = sleeve_offset + sr * sleeve_segs + ss_next
                i2 = sleeve_offset + (sr + 1) * sleeve_segs + ss
                i3 = sleeve_offset + (sr + 1) * sleeve_segs + ss_next
                faces_list.append([i0, i1, i3])
                faces_list.append([i0, i3, i2])

    # 3. High Shearling Collar (5 rings, 20 segments)
    collar_offset = len(verts_list)
    collar_rings = 5
    collar_segs = 20
    for cr in range(collar_rings):
        t_col = cr / float(collar_rings - 1)
        y_col = 1.42 + 0.12 * t_col
        # Flared upward collar
        r_col_x = 0.20 + 0.05 * t_col
        r_col_z = 0.18 + 0.04 * t_col

        for cs in range(collar_segs):
            theta_col = 2.0 * np.pi * cs / collar_segs
            vx = r_col_x * np.cos(theta_col)
            vy = y_col
            vz = r_col_z * np.sin(theta_col)
            verts_list.append(np.array([vx, vy, vz], dtype=np.float32))
            uvs_list.append(np.array([cs / float(collar_segs), t_col], dtype=np.float32))

    for cr in range(collar_rings - 1):
        for cs in range(collar_segs):
            cs_next = (cs + 1) % collar_segs
            i0 = collar_offset + cr * collar_segs + cs
            i1 = collar_offset + cr * collar_segs + cs_next
            i2 = collar_offset + (cr + 1) * collar_segs + cs
            i3 = collar_offset + (cr + 1) * collar_segs + cs_next
            faces_list.append([i0, i1, i3])
            faces_list.append([i0, i3, i2])

    vertices = np.array(verts_list, dtype=np.float32)
    faces = np.array(faces_list, dtype=np.int32)
    uvs = np.array(uvs_list, dtype=np.float32)

    return MeshGeometry(
        name="hero_shearling_coat",
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        normals=None,
    )


def generate_representative_animated_body(num_frames: int = 24) -> Tuple[MeshGeometry, Dict[int, np.ndarray]]:
    """Generate animated mannequin collider frames with realistic arm swinging dynamics."""
    # Base body mesh (torso, arms, neck, head)
    # 200 vertices, 380 faces
    base_verts = []
    base_faces = []

    # Torso column (8 rings, 16 segments)
    rings = 8
    segs = 16
    y_vals = np.linspace(0.55, 1.40, rings)
    for r_idx, y in enumerate(y_vals):
        rx = 0.22 if y < 1.1 else 0.26
        rz = 0.16 if y < 1.1 else 0.19
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

    # Left & Right arm colliders (6 segments, 6 rings each)
    arm_segs = 8
    arm_rings = 6
    for sign in [1.0, -1.0]:
        offset = len(base_verts)
        p_shoulder = np.array([sign * 0.30, 1.32, 0.0], dtype=np.float32)
        p_hand = np.array([sign * 0.42, 0.88, 0.0], dtype=np.float32)
        for ar in range(arm_rings):
            t = ar / float(arm_rings - 1)
            center = (1.0 - t) * p_shoulder + t * p_hand
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
        name="hero_body",
        vertices=np.array(base_verts, dtype=np.float32),
        faces=np.array(base_faces, dtype=np.int32)
    )

    # Generate frame-by-frame animated arm swing
    frames_dict: Dict[int, np.ndarray] = {}
    for f in range(1, num_frames + 1):
        v_frame = np.copy(base_geom.vertices)
        t_phase = 2.0 * np.pi * (f - 1) / float(num_frames)
        # Left arm swings forward/backward (Z) and slightly in/out (X)
        left_arm_indices = range(128, 128 + 48)
        right_arm_indices = range(128 + 48, 128 + 96)

        swing_z_left = 0.12 * math.sin(t_phase)
        swing_z_right = -0.12 * math.sin(t_phase)

        for idx in left_arm_indices:
            # Arm vertex displacement increases towards the hand
            v_frame[idx, 2] += swing_z_left * (1.40 - v_frame[idx, 1])
        for idx in right_arm_indices:
            v_frame[idx, 2] += swing_z_right * (1.40 - v_frame[idx, 1])

        frames_dict[f] = v_frame

    return base_geom, frames_dict
