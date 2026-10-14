"""Autonomous CFX Studio - Performance Profiling Harness.

Profiles individual subsystems independently on the target workstation:
1. Application & Module Startup
2. Alembic Ingestion (Shot 020 24f, Shot 030 100f)
3. Viewport Frame Paint & Timeline Scrubbing
4. AI Intent Parsing & Parameter Configuration
5. XPBD Solver Progression & Substep Kernel Breakdown
6. Collision Detection Queries & Broadphase
7. Corrective Layer & Smoothing
8. Quality Evaluation
9. Alembic Geometry Export
10. Workstation Telemetry (CPU, Peak RAM, GPU VRAM)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import psutil

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from autonomous_cfx.agent.behavior_parser import BehaviorParser
from autonomous_cfx.agent.evaluator import SimulationEvaluator
from autonomous_cfx.agent.orchestrator import AutonomousCFXOrchestrator
from autonomous_cfx.assets.asset_manager import AssetManager
from autonomous_cfx.core.alembic_io import (
    ALEMBIC_AVAILABLE,
    AlembicObjectRole,
    AlembicShotExporter,
    AlembicShotImporter,
)
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.geometry import MeshGeometry
from autonomous_cfx.core.project import ProjectManager
from autonomous_cfx.simulation.colliders import AnimatedMeshCollider, GroundPlaneCollider
from autonomous_cfx.simulation.corrective_layer import CorrectiveLayerEngine
from autonomous_cfx.simulation.engine import SimulationEngine
from autonomous_cfx.simulation.properties import ClothMaterialProperties
from autonomous_cfx.simulation.xpbd_solver import XPBDSolver


def get_gpu_memory_info() -> Dict[str, float]:
    """Query NVIDIA GPU memory utilization using nvidia-smi."""
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = [float(x.strip()) for x in res.stdout.strip().split(",")]
        return {"total_mb": parts[0], "used_mb": parts[1], "free_mb": parts[2]}
    except Exception:
        return {"total_mb": 0.0, "used_mb": 0.0, "free_mb": 0.0}


def get_process_memory_mb() -> float:
    """Get current resident set size (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


class CFXProfiler:
    """Profiles and records execution metrics across all CFX subsystems."""

    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path
        self.results: Dict[str, Any] = {
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "os": sys.platform,
                "python_version": sys.version,
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "gpu": get_gpu_memory_info(),
            },
            "subsystems": {},
        }

    def profile_startup(self) -> Dict[str, Any]:
        """Measure startup and module import latency."""
        print("\n[1/9] Profiling Startup Latency...")
        t0 = time.perf_counter()
        import autonomous_cfx
        import autonomous_cfx.agent.orchestrator
        import autonomous_cfx.core.alembic_io
        import autonomous_cfx.simulation.xpbd_solver
        t_import = time.perf_counter() - t0

        t0 = time.perf_counter()
        from PySide6.QtWidgets import QApplication
        from autonomous_cfx.ui.app import AutonomousCFXMainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        window = AutonomousCFXMainWindow(project_dir="evidence/reports/profiling_project")
        t_ui_init = time.perf_counter() - t0

        metrics = {
            "module_import_seconds": round(t_import, 4),
            "ui_window_instantiation_seconds": round(t_ui_init, 4),
            "total_startup_seconds": round(t_import + t_ui_init, 4),
            "peak_rss_mb": round(get_process_memory_mb(), 2),
        }
        print(f"  Imports: {metrics['module_import_seconds']}s | UI Init: {metrics['ui_window_instantiation_seconds']}s | RAM: {metrics['peak_rss_mb']} MB")
        self.results["subsystems"]["startup"] = metrics
        return metrics

    def profile_alembic_io(self) -> Dict[str, Any]:
        """Measure Alembic inspection and geometry ingestion for representative assets."""
        print("\n[2/9] Profiling Alembic Ingestion...")
        metrics: Dict[str, Any] = {}
        assets = [
            ("shot020", "evidence/exports/shot020_production_input.abc"),
            ("shot030", "evidence/exports/shot030_action_long_input.abc"),
        ]
        importer = AlembicShotImporter()

        for shot_name, path in assets:
            if not os.path.exists(path):
                print(f"  Asset {path} not found, skipping.")
                continue

            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            ram_before = get_process_memory_mb()

            # 1. Inspection latency
            t0 = time.perf_counter()
            summary = importer.inspect_archive(path)
            t_inspect = time.perf_counter() - t0

            # 2. Ingestion latency
            t0 = time.perf_counter()
            shot_data = importer.load_shot(path)
            t_load = time.perf_counter() - t0
            ram_after = get_process_memory_mb()

            total_verts = sum(n.num_vertices for n in summary.nodes)
            total_faces = sum(n.num_faces for n in summary.nodes)

            metrics[shot_name] = {
                "file_path": path,
                "file_size_mb": round(file_size_mb, 3),
                "total_frames": summary.total_samples,
                "total_vertices": total_verts,
                "total_faces": total_faces,
                "inspect_seconds": round(t_inspect, 4),
                "load_all_samples_seconds": round(t_load, 4),
                "samples_per_second": round(summary.total_samples / max(1e-5, t_load), 1),
                "ram_delta_mb": round(ram_after - ram_before, 2),
            }
            print(
                f"  {shot_name} ({summary.total_samples}f, {total_verts}v): "
                f"Inspect={metrics[shot_name]['inspect_seconds']}s, "
                f"Load={metrics[shot_name]['load_all_samples_seconds']}s "
                f"({metrics[shot_name]['samples_per_second']} samples/s), "
                f"RAM Delta={metrics[shot_name]['ram_delta_mb']} MB"
            )

        self.results["subsystems"]["alembic_io"] = metrics
        return metrics

    def profile_viewport(self) -> Dict[str, Any]:
        """Measure viewport paint latency and scrubbing throughput."""
        print("\n[3/9] Profiling Viewport Navigation & Scrubbing...")
        from PySide6.QtWidgets import QApplication
        from autonomous_cfx.ui.viewport_widget import Viewport3DWidget

        app = QApplication.instance() or QApplication(sys.argv)
        vp = Viewport3DWidget()
        vp.resize(1280, 720)

        am = AssetManager(Path("evidence/reports/profiling_assets"))
        jacket, body_col, chair_col = am.load_milestone_package(num_frames=24, fps=24.0, include_chair=True)
        vp.set_cloth_geometry(jacket)
        colliders = [body_col.geometry]
        if chair_col:
            colliders.append(chair_col.geometry)
        vp.set_colliders(colliders)

        # Single frame paint latency (benchmark 20 paint iterations)
        from PySide6.QtGui import QImage, QPainter, QColor

        img = QImage(1280, 720, QImage.Format_ARGB32)
        painter = QPainter()

        face_color = QColor(245, 158, 11, 210)
        wire_color = QColor(251, 191, 36, 170)
        col_color = QColor(51, 65, 85, 120)

        paint_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            painter.begin(img)
            # Render internal paint pass directly
            vp._draw_mesh(painter, jacket, face_color, wire_color)
            for c in colliders:
                vp._draw_mesh(painter, c, col_color, None)
            painter.end()
            paint_times.append(time.perf_counter() - t0)

        mean_paint_ms = float(np.mean(paint_times)) * 1000.0
        fps_paint = 1000.0 / max(0.1, mean_paint_ms)

        # Orbit simulation (change angle and redraw 30 times)
        orbit_times = []
        for i in range(30):
            vp.yaw = (vp.yaw + 3.0) % 360.0
            vp.pitch = np.clip(vp.pitch + 0.5, -85.0, 85.0)
            t0 = time.perf_counter()
            painter.begin(img)
            vp._draw_mesh(painter, jacket, face_color, wire_color)
            painter.end()
            orbit_times.append(time.perf_counter() - t0)

        mean_orbit_ms = float(np.mean(orbit_times)) * 1000.0
        fps_orbit = 1000.0 / max(0.1, mean_orbit_ms)

        metrics = {
            "resolution": "1280x720",
            "cloth_verts": jacket.num_vertices,
            "cloth_faces": jacket.num_faces,
            "collider_count": len(colliders),
            "single_frame_paint_ms": round(mean_paint_ms, 2),
            "estimated_paint_fps": round(fps_paint, 1),
            "orbit_redraw_ms": round(mean_orbit_ms, 2),
            "estimated_orbit_fps": round(fps_orbit, 1),
        }
        print(f"  Frame Paint: {metrics['single_frame_paint_ms']} ms ({metrics['estimated_paint_fps']} FPS) | Orbit: {metrics['orbit_redraw_ms']} ms ({metrics['estimated_orbit_fps']} FPS)")
        self.results["subsystems"]["viewport"] = metrics
        return metrics

    def profile_ai_inference(self) -> Dict[str, Any]:
        """Measure prompt parsing and tool call generation."""
        print("\n[4/9] Profiling AI Intent Parsing...")
        parser = BehaviorParser()
        prompts = [
            "heavy leather jacket, stiff collar, low stretch",
            "light silk scarf fluttering in mountain wind at 8 m/s",
            "thick woolen cloak with heavy gravity drape and moderate friction",
        ]
        times = []
        for p in prompts:
            t0 = time.perf_counter()
            res = parser.parse(p)
            times.append(time.perf_counter() - t0)

        mean_parse_ms = float(np.mean(times)) * 1000.0
        metrics = {
            "prompts_tested": len(prompts),
            "mean_parse_time_ms": round(mean_parse_ms, 2),
            "parses_per_second": round(1000.0 / max(0.1, mean_parse_ms), 1),
        }
        print(f"  Behavior parsing: {metrics['mean_parse_time_ms']} ms ({metrics['parses_per_second']} parses/s)")
        self.results["subsystems"]["ai_inference"] = metrics
        return metrics

    def profile_simulation(self) -> Dict[str, Any]:
        """Measure full simulation runs and substep kernel breakdowns on Shot 020 and Shot 030."""
        print("\n[5/9] Profiling Simulation Engine & XPBD Kernels...")
        metrics: Dict[str, Any] = {}
        importer = AlembicShotImporter()

        cases = [
            ("shot020_24f", "evidence/exports/shot020_production_input.abc", 24),
            ("shot030_50f", "evidence/exports/shot030_action_long_input.abc", 50),
        ]

        for case_name, path, max_frames in cases:
            if not os.path.exists(path):
                continue

            summary = importer.inspect_archive(path)
            # Find garment and colliders
            garment_node = next((n for n in summary.nodes if n.node_type == "PolyMesh" and ("coat" in n.name or "trench" in n.name)), None)
            body_node = next((n for n in summary.nodes if n.node_type == "PolyMesh" and ("body" in n.name or "runner" in n.name)), None)

            if not garment_node or not body_node:
                continue

            # Load actual meshes
            roles = {
                garment_node.full_path: AlembicObjectRole.GARMENT,
                body_node.full_path: AlembicObjectRole.BODY,
            }
            shot_data = importer.load_shot(path, role_assignments=roles)
            garment_geo = shot_data.garment_meshes[garment_node.full_path][0]
            body_samples = {f: geo.vertices for f, geo in enumerate(shot_data.body_colliders[body_node.full_path], start=1)}
            body_col = AnimatedMeshCollider(body_node.name, shot_data.body_colliders[body_node.full_path][0], body_samples, fps=24.0)

            props = ClothMaterialProperties(
                density=0.45,
                stretch_compliance=1e-6,
                bend_compliance=1e-4,
                solver_substeps=16,
                solver_iterations=4,
            )

            engine = SimulationEngine(
                cloth_geometry=garment_geo,
                properties=props,
                colliders=[body_col],
                pinned_vertices=[0, 1, 2, 3],
                fps=24.0,
                start_frame=1,
                end_frame=max_frames,
            )

            ram_before = get_process_memory_mb()
            t0 = time.perf_counter()
            cache = engine.run()
            t_sim = time.perf_counter() - t0
            ram_after = get_process_memory_mb()

            fps_sim = max_frames / max(1e-5, t_sim)
            per_frame_ms = (t_sim / max_frames) * 1000.0

            metrics[case_name] = {
                "frames": max_frames,
                "substeps": props.solver_substeps,
                "iterations": props.solver_iterations,
                "cloth_vertices": garment_geo.num_vertices,
                "cloth_faces": garment_geo.num_faces,
                "total_simulation_seconds": round(t_sim, 4),
                "per_frame_ms": round(per_frame_ms, 2),
                "simulation_fps": round(fps_sim, 2),
                "ram_delta_mb": round(ram_after - ram_before, 2),
            }
            print(
                f"  {case_name}: {max_frames} frames in {metrics[case_name]['total_simulation_seconds']}s "
                f"({metrics[case_name]['simulation_fps']} sim FPS, {metrics[case_name]['per_frame_ms']} ms/frame)"
            )

        # Micro-benchmark solver kernel components
        print("  Running micro-benchmarks on XPBD kernels (100 substeps)...")
        geo = MeshGeometry(
            vertices=np.random.randn(800, 3).astype(np.float32),
            faces=np.random.randint(0, 800, (1400, 3), dtype=np.int32),
        )
        solver = XPBDSolver(geo, ClothMaterialProperties(solver_substeps=16, solver_iterations=4))

        # 1. Distance constraint projection
        p_test = np.copy(solver.positions)
        t0 = time.perf_counter()
        for _ in range(100):
            solver.distance_constraints.project(p_test, solver.inv_masses, dt=0.001)
        t_dist = time.perf_counter() - t0

        # 2. Bending constraint projection
        t0 = time.perf_counter()
        for _ in range(100):
            solver.bending_constraints.project(p_test, solver.inv_masses, dt=0.001)
        t_bend = time.perf_counter() - t0

        metrics["microbenchmarks"] = {
            "distance_100_evals_ms": round(t_dist * 1000.0, 2),
            "bending_100_evals_ms": round(t_bend * 1000.0, 2),
        }
        print(f"  Kernel: Distance x100={metrics['microbenchmarks']['distance_100_evals_ms']}ms | Bending x100={metrics['microbenchmarks']['bending_100_evals_ms']}ms")

        self.results["subsystems"]["simulation"] = metrics
        return metrics

    def profile_collision_detection(self) -> Dict[str, Any]:
        """Measure isolated collision detection query latency."""
        print("\n[6/9] Profiling Collision Detection Structures...")
        col_v = np.random.randn(250, 3).astype(np.float32)
        col_f = np.random.randint(0, 250, (400, 3), dtype=np.int32)
        col_geo = MeshGeometry(vertices=col_v, faces=col_f)
        anim_frames = {f: col_v + f * 0.01 for f in range(1, 30)}
        col = AnimatedMeshCollider("test_col", col_geo, anim_frames, fps=24.0)

        cloth_pts = np.random.randn(800, 3).astype(np.float32)

        # Measure 100 queries
        t0 = time.perf_counter()
        for f in range(1, 101):
            col.update_frame(float(f % 25 + 1))
            col.project_points(cloth_pts, margin=0.01)
        t_query = time.perf_counter() - t0

        per_query_ms = (t_query / 100.0) * 1000.0
        metrics = {
            "collider_vertices": 250,
            "cloth_query_points": 800,
            "queries_executed": 100,
            "total_query_seconds": round(t_query, 4),
            "per_query_ms": round(per_query_ms, 3),
            "queries_per_second": round(100.0 / max(1e-5, t_query), 1),
        }
        print(f"  Proximity query: {metrics['per_query_ms']} ms/query ({metrics['queries_per_second']} queries/s)")
        self.results["subsystems"]["collision_detection"] = metrics
        return metrics

    def profile_corrections_and_quality(self) -> Dict[str, Any]:
        """Measure corrective layer Laplacian smoothing and quality metric evaluation."""
        print("\n[7/9] Profiling Corrective Layer & Quality Evaluation...")
        am = AssetManager(Path("evidence/reports/profiling_assets"))
        jacket, body_col, _ = am.load_milestone_package(num_frames=20, fps=24.0, include_chair=False)

        # 1. Corrective layer
        from autonomous_cfx.agent.correction_agent import CorrectionIntent, StructuredCorrectionRequest
        from autonomous_cfx.core.surface_mapping import SurfaceSelection

        cloth_frames = [jacket.copy() for _ in range(20)]
        body_frames = [body_col.geometry.copy() for _ in range(20)]
        target_sel = SurfaceSelection(
            selected_faces=[5, 6],
            selected_vertices=[40, 41, 42],
            vertex_weights={40: 1.0, 41: 0.8, 42: 0.5},
            falloff_radius=0.05,
        )
        req = StructuredCorrectionRequest(
            request_id="prof_001",
            garment_name="jacket",
            geometry_version="v001",
            intent=CorrectionIntent.REPAIR_BODY_PENETRATION,
            target_region=target_sel,
            frame_range=(5, 15),
            fade_frames=(2, 2),
        )

        t0 = time.perf_counter()
        for _ in range(5):
            CorrectiveLayerEngine.apply_correction(cloth_frames, req, colliders_cache=body_frames)
        t_corr = time.perf_counter() - t0
        per_corr_ms = (t_corr / 5.0) * 1000.0

        # 2. Quality Evaluation across 100 frames
        verts = np.copy(jacket.vertices)
        cache = SimulationCache(base_faces=jacket.faces)
        for f in range(1, 101):
            cache.add_frame(f, verts + np.sin(f * 0.1) * 0.05, metrics={"contact_events": 10, "max_depth": 0.005})

        t0 = time.perf_counter()
        for _ in range(20):
            SimulationEvaluator.evaluate(cache, jacket)
        t_eval = time.perf_counter() - t0
        per_eval_ms = (t_eval / 20.0) * 1000.0

        metrics = {
            "corrective_layer_20f_ms": round(per_corr_ms, 2),
            "quality_evaluation_100f_ms": round(per_eval_ms, 2),
        }
        print(f"  Corrective Layer (20f): {metrics['corrective_layer_20f_ms']} ms | Evaluator (100f): {metrics['quality_evaluation_100f_ms']} ms")
        self.results["subsystems"]["corrections_and_quality"] = metrics
        return metrics

    def profile_export(self) -> Dict[str, Any]:
        """Measure Alembic geometry export latency and write throughput."""
        print("\n[8/9] Profiling Alembic Geometry Export...")
        am = AssetManager(Path("evidence/reports/profiling_assets"))
        jacket, body_col, _ = am.load_milestone_package(num_frames=24, fps=24.0, include_chair=False)

        garment_frames = [
            MeshGeometry(name="jacket", vertices=jacket.vertices + (f * 0.002), faces=jacket.faces)
            for f in range(1, 25)
        ]
        export_dir = Path("evidence/reports/profiling_exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        out_abc = str(export_dir / "profile_export_24f.abc")

        exporter = AlembicShotExporter()
        t0 = time.perf_counter()
        exporter.export_shot(
            output_path=out_abc,
            simulated_garments={"hero_jacket": garment_frames},
            fps=24.0,
            start_frame=1,
        )
        t_export = time.perf_counter() - t0

        size_kb = os.path.getsize(out_abc) / 1024.0 if os.path.exists(out_abc) else 0.0

        metrics = {
            "exported_frames": 24,
            "export_seconds": round(t_export, 4),
            "frames_per_second": round(24.0 / max(1e-5, t_export), 1),
            "file_size_kb": round(size_kb, 2),
        }
        print(f"  Alembic Export (24f): {metrics['export_seconds']}s ({metrics['frames_per_second']} frames/s, {metrics['file_size_kb']} KB)")
        self.results["subsystems"]["export"] = metrics
        return metrics

    def run_all(self) -> Dict[str, Any]:
        """Execute full profiling suite and persist report."""
        print("=================================================================")
        print("  AUTONOMOUS CFX STUDIO - PERFORMANCE PROFILING SUITE")
        print("=================================================================")
        print(f"Workstation: {self.results['environment']['cpu_count_physical']} cores ({self.results['environment']['cpu_count_logical']} threads), {self.results['environment']['total_ram_gb']} GB RAM")
        gpu = self.results['environment']['gpu']
        print(f"GPU: {gpu['total_mb']} MB total, {gpu['used_mb']} MB in use, {gpu['free_mb']} MB free")

        self.profile_startup()
        self.profile_alembic_io()
        self.profile_viewport()
        self.profile_ai_inference()
        self.profile_simulation()
        self.profile_collision_detection()
        self.profile_corrections_and_quality()
        self.profile_export()

        # Overall summary
        final_gpu = get_gpu_memory_info()
        peak_ram = get_process_memory_mb()
        self.results["workstation_summary"] = {
            "peak_process_rss_mb": round(peak_ram, 2),
            "initial_gpu_used_mb": gpu["used_mb"],
            "final_gpu_used_mb": final_gpu["used_mb"],
            "gpu_vram_delta_mb": round(final_gpu["used_mb"] - gpu["used_mb"], 2),
        }

        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
            print(f"\n[DONE] Baseline profiling report saved to: {self.output_path}")

        return self.results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Autonomous CFX Studio")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="evidence/reports/performance_profiling_baseline.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    profiler = CFXProfiler(output_path=args.output)
    profiler.run_all()
