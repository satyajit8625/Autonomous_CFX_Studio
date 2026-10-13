"""Autonomous CFX Studio - Main Desktop Application Window.
Provides a comprehensive UI workspace combining 3D viewport, agent orchestration, timeline, and review.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from autonomous_cfx.agent.orchestrator import AutonomousCFXOrchestrator, WorkflowExecutionResult
from autonomous_cfx.assets.asset_manager import AssetManager
from autonomous_cfx.core.cache_io import SimulationCache
from autonomous_cfx.core.project import ProjectManager
from autonomous_cfx.jobs.job_runner import JobRunner
from autonomous_cfx.ui.agent_panel import AgentPanel
from autonomous_cfx.ui.review_panel import ReviewPanel
from autonomous_cfx.ui.timeline_widget import TimelineWidget
from autonomous_cfx.ui.viewport_widget import Viewport3DWidget
from autonomous_cfx.core.logger import logger


class WorkerBridge(QObject):
    """Qt signal bridge for thread-safe UI updates from background workers."""
    progress_signal = Signal(str, int, int, dict)
    log_signal = Signal(str)
    finished_signal = Signal(object)


class AutonomousCFXMainWindow(QMainWindow):
    """Autonomous CFX Studio - Standalone Desktop GUI."""

    def __init__(self, project_dir: Optional[str | Path] = None):
        super().__init__()
        self.setWindowTitle("Autonomous CFX Studio v1.0.0")
        self.resize(1380, 880)

        # Initialize Subsystems
        self.project_manager = ProjectManager(project_dir or Path("."))
        self.asset_manager = AssetManager(self.project_manager.assets_dir)
        self.job_runner = JobRunner()
        self.orchestrator = AutonomousCFXOrchestrator(self.project_manager)

        # Runtime State
        self.active_cache: Optional[SimulationCache] = None
        self.last_workflow_result: Optional[WorkflowExecutionResult] = None
        self.current_job_id: Optional[str] = None

        # Thread communication bridge
        self.bridge = WorkerBridge()
        self.bridge.progress_signal.connect(self._on_worker_progress)
        self.bridge.log_signal.connect(self._on_worker_log)
        self.bridge.finished_signal.connect(self._on_worker_finished)

        self._build_ui()
        self._load_default_milestone_scene()

    def _build_ui(self):
        # 1. Menu Bar
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("&File")
        act_new = QAction("New Project", self)
        act_new.triggered.connect(self._on_new_project)
        menu_file.addAction(act_new)

        act_load_obj = QAction("Import Cloth OBJ...", self)
        act_load_obj.triggered.connect(self._on_import_obj)
        menu_file.addAction(act_load_obj)

        menu_file.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        # Simulation Menu
        menu_sim = menubar.addMenu("&Simulation")
        act_run = QAction("Run Autonomous Workflow", self)
        act_run.triggered.connect(self._on_run_simulation)
        menu_sim.addAction(act_run)

        # 2. Main Central Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Horizontal Splitter: [Agent Panel] | [3D Viewport] | [Review Panel]
        splitter_h = QSplitter(Qt.Horizontal)

        # Left: Agent Panel
        self.agent_panel = AgentPanel()
        self.agent_panel.run_requested.connect(self._on_run_with_prompt)
        self.agent_panel.cancel_requested.connect(self._on_cancel_simulation)
        splitter_h.addWidget(self.agent_panel)

        # Middle: Viewport and Timeline
        middle_container = QWidget()
        vbox_middle = QVBoxLayout(middle_container)
        vbox_middle.setContentsMargins(0, 0, 0, 0)
        vbox_middle.setSpacing(4)

        self.viewport = Viewport3DWidget()
        vbox_middle.addWidget(self.viewport, stretch=1)

        self.timeline = TimelineWidget()
        self.timeline.frame_changed.connect(self._on_timeline_frame_changed)
        vbox_middle.addWidget(self.timeline)

        splitter_h.addWidget(middle_container)

        # Right: Review & Quality Panel
        self.review_panel = ReviewPanel()
        self.review_panel.approve_clicked.connect(self._on_artist_approve)
        self.review_panel.reject_clicked.connect(self._on_artist_reject)
        self.review_panel.export_clicked.connect(self._on_export_caches)
        splitter_h.addWidget(self.review_panel)

        splitter_h.setStretchFactor(0, 1)
        splitter_h.setStretchFactor(1, 3)
        splitter_h.setStretchFactor(2, 1)

        main_layout.addWidget(splitter_h, stretch=1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Autonomous CFX Studio ready.")

    def _load_default_milestone_scene(self):
        """Loads canonical milestone setup: Jacket cloth, Animated Body collider, Chair collider."""
        jacket, body_col, chair_col = self.asset_manager.load_milestone_package(
            num_frames=40,
            fps=24.0,
            include_chair=True,
        )
        self.jacket_mesh = jacket
        self.body_collider = body_col
        self.chair_collider = chair_col

        self.viewport.set_cloth_geometry(self.jacket_mesh)
        cols_to_view = [self.body_collider.geometry]
        if self.chair_collider:
            cols_to_view.append(self.chair_collider.geometry)
        self.viewport.set_colliders(cols_to_view)

        self.agent_panel.append_log(
            f"Loaded Milestone Assets: Jacket ({jacket.num_vertices} verts), "
            f"Animated Body ({body_col.geometry.num_vertices} verts), Chair collider."
        )

    def _on_timeline_frame_changed(self, frame_num: int):
        self.viewport.current_frame = frame_num

        # Update animated colliders to frame
        if hasattr(self, "body_collider") and self.body_collider:
            self.body_collider.update_frame(float(frame_num))

        cols = [self.body_collider.geometry]
        if hasattr(self, "chair_collider") and self.chair_collider:
            cols.append(self.chair_collider.geometry)
        self.viewport.set_colliders(cols)

        # Update simulated cloth geometry
        if self.active_cache and frame_num in self.active_cache.frames:
            geo = self.active_cache.get_frame_geometry(frame_num)
            self.viewport.set_cloth_geometry(geo)

        self.viewport.update()

    def _on_run_simulation(self):
        prompt = self.agent_panel.txt_prompt.toPlainText().strip()
        self._on_run_with_prompt(prompt)

    def _on_run_with_prompt(self, prompt: str):
        self.agent_panel.set_running_state(True)
        self.status_bar.showMessage("Running Autonomous CFX Workflow in background...")

        colliders = [self.body_collider]
        if self.chair_collider:
            colliders.append(self.chair_collider)

        pinned = list(self.jacket_mesh.vertex_groups.get("collar", []))

        def background_task():
            def progress_cb(stage, cur, total, extra):
                self.bridge.progress_signal.emit(stage, cur, total, extra)
                if extra and "prompt" in extra:
                    self.bridge.log_signal.emit(f"Prompt: {extra['prompt']}")
                elif stage.startswith("Repair"):
                    self.bridge.log_signal.emit(f"[Agent Repair] {stage} ({cur}%)")

            res = self.orchestrator.execute_workflow(
                cloth_geometry=self.jacket_mesh,
                colliders=colliders,
                behavior_description=prompt,
                pinned_vertices=pinned,
                fps=24.0,
                start_frame=1,
                end_frame=40,
                auto_approve_if_passed=True,
                export_on_approval=False,  # Wait for artist review/approval in UI
                progress_callback=progress_cb,
            )
            return res

        job = self.job_runner.submit_job(
            name="Autonomous_CFX_Run",
            target_fn=background_task,
            on_complete=lambda j: self.bridge.finished_signal.emit(j.result),
        )
        self.current_job_id = job.job_id

    @Slot(str, int, int, dict)
    def _on_worker_progress(self, stage: str, cur: int, total: int, extra: dict):
        self.agent_panel.update_progress(cur, f"{stage} ({cur}%)")

    @Slot(str)
    def _on_worker_log(self, text: str):
        self.agent_panel.append_log(text)

    @Slot(object)
    def _on_worker_finished(self, result: Optional[WorkflowExecutionResult]):
        self.agent_panel.set_running_state(False)
        self.status_bar.showMessage("Simulation completed. Review ready.")

        if result is None:
            self.agent_panel.append_log("Workflow failed or was cancelled.")
            return

        self.last_workflow_result = result
        self.active_cache = result.simulation_cache

        # Update Review Panel
        self.review_panel.update_report(result.final_report)

        # Log summary in agent console
        rep = result.final_report
        self.agent_panel.append_log("-----------------------------------------")
        self.agent_panel.append_log(f"Final Assessment: {rep.status.value} (Score: {rep.score}/100)")
        self.agent_panel.append_log(f"Max Penetration: {rep.max_penetration_depth*1000.0:.1f} mm | Max Strain: {rep.max_stretch_ratio*100.0:.1f}%")
        if result.repair_history:
            self.agent_panel.append_log(f"Self-Healing: {len(result.repair_history)} repair iteration(s) applied.")
            for r in result.repair_history:
                self.agent_panel.append_log(f"  Attempt {r.attempt_index}: Score {r.pre_score} -> {r.post_score} (Resolved: {r.resolved})")
        self.agent_panel.append_log("-----------------------------------------")

        # Set timeline to frame 1 and play
        self.timeline.set_frame(1)
        self._on_timeline_frame_changed(1)

    def _on_cancel_simulation(self):
        if self.current_job_id:
            self.job_runner.cancel_job(self.current_job_id)
            self.agent_panel.set_running_state(False)
            self.status_bar.showMessage("Simulation cancelled.")

    def _on_artist_approve(self):
        self.review_panel.mark_approved(True)
        self.agent_panel.append_log("[Artist Review] Cache approved by artist.")
        self.status_bar.showMessage("Cache approved. Export unlocked.")

    def _on_artist_reject(self):
        self.review_panel.mark_approved(False)
        self.agent_panel.append_log("[Artist Review] Cache rejected by artist.")
        self.status_bar.showMessage("Cache rejected.")

    def _on_export_caches(self):
        if not self.active_cache:
            QMessageBox.warning(self, "Export", "No active simulation cache to export.")
            return

        shot_name = "shot010"
        cloth_name = self.active_cache.name
        next_ver = self.project_manager.get_next_cache_version(shot_name, cloth_name)
        cache_dir = self.project_manager.get_cache_version_dir(shot_name, cloth_name, next_ver)

        # Export point cache
        vpcache_file = cache_dir / f"{cloth_name}_v{next_ver:03d}.vpcache"
        self.active_cache.save_point_cache(vpcache_file)

        # Export OBJ sequence
        obj_dir = cache_dir / "obj_sequence"
        obj_files = self.active_cache.export_obj_sequence(obj_dir, prefix=cloth_name)

        # Export Manifest
        manifest_file = self.active_cache.export_interchange_manifest(
            output_dir=cache_dir,
            alembic_equivalent_name=f"{cloth_name}_v{next_ver:03d}.abc",
        )

        self.review_panel.set_export_success(str(cache_dir), len(obj_files))
        self.agent_panel.append_log(f"[Export] Saved {len(obj_files)} OBJ frames & .vpcache to: {cache_dir}")
        QMessageBox.information(
            self,
            "Export Complete",
            f"Successfully exported approved caches:\n\n"
            f"• Directory: {cache_dir}\n"
            f"• Point Cache: {vpcache_file.name}\n"
            f"• Wavefront OBJ Sequence: {len(obj_files)} frames\n"
            f"• Interchange Manifest: {Path(manifest_file).name}",
        )

    def _on_new_project(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Root")
        if dir_path:
            self.project_manager = ProjectManager(dir_path)
            self.status_bar.showMessage(f"Loaded project: {dir_path}")

    def _on_import_obj(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Cloth Mesh", "", "3D Geometry (*.obj *.stl *.ply)")
        if file_path:
            try:
                mesh = self.asset_manager.load_mesh_file(file_path)
                self.jacket_mesh = mesh
                self.viewport.set_cloth_geometry(mesh)
                self.agent_panel.append_log(f"Imported custom cloth mesh: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to load mesh: {e}")


def launch_studio():
    """Main application launcher."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = AutonomousCFXMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(launch_studio())
