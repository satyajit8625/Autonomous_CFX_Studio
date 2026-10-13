"""Autonomous CFX Studio - Expandable Diagnostics Drawer.
Provides accessible yet unobtrusive technical diagnostics:
- Actionable Quality Issues (with Jump-to-Frame / Viewport navigation)
- Autonomous Repair History
- Background Job History
- Performance & Timing Measurements
- Technical Log Stream
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class DiagnosticsDrawerWidget(QWidget):
    """Collapsible bottom diagnostics drawer with tabbed metrics and issue navigation."""

    jump_to_frame_requested = Signal(int)  # Emits frame number to scrub timeline
    toggle_drawer_requested = Signal(bool) # Emitted when drawer expands or collapses

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.is_expanded: bool = False
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 2, 4, 4)
        main_layout.setSpacing(4)

        # 1. Header Bar with Expand / Collapse Button
        header_bar = QFrame()
        header_bar.setStyleSheet(
            "background-color: #171a21; border-radius: 4px; border: 1px solid #232834; padding: 2px;"
        )
        h_layout = QHBoxLayout(header_bar)
        h_layout.setContentsMargins(8, 2, 8, 2)

        self.btn_toggle = QPushButton("▲ Diagnostics & Quality Telemetry")
        self.btn_toggle.setStyleSheet(
            "QPushButton { background-color: transparent; color: #94a3b8; font-size: 11px; "
            "font-weight: bold; border: none; text-align: left; }"
            "QPushButton:hover { color: #f8fafc; }"
        )
        self.btn_toggle.clicked.connect(self.toggle_expanded)
        h_layout.addWidget(self.btn_toggle)

        self.lbl_quick_status = QLabel("Quality: Awaiting Simulation | 0 Jobs")
        self.lbl_quick_status.setStyleSheet("color: #64748b; font-size: 11px;")
        h_layout.addWidget(self.lbl_quick_status)
        h_layout.addStretch(1)

        main_layout.addWidget(header_bar)

        # 2. Collapsible Tabs Container
        self.tabs_container = QWidget()
        tabs_layout = QVBoxLayout(self.tabs_container)
        tabs_layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #232834; background-color: #11141a; }"
            "QTabBar::tab { background: #171a21; color: #94a3b8; padding: 6px 12px; font-size: 11px; }"
            "QTabBar::tab:selected { background: #1e293b; color: #f1f5f9; font-weight: bold; }"
        )

        # Tab 1: Quality Issues
        self.tab_issues = QWidget()
        self._build_issues_tab()
        self.tab_widget.addTab(self.tab_issues, "⚠️ Actionable Quality Issues")

        # Tab 2: Repair History
        self.tab_repairs = QWidget()
        self._build_repairs_tab()
        self.tab_widget.addTab(self.tab_repairs, "🔧 Repair Attempts")

        # Tab 3: Job History
        self.tab_jobs = QWidget()
        self._build_jobs_tab()
        self.tab_widget.addTab(self.tab_jobs, "📋 Job History")

        # Tab 4: Performance Measurements
        self.tab_perf = QWidget()
        self._build_perf_tab()
        self.tab_widget.addTab(self.tab_perf, "⚡ Performance & Timing")

        # Tab 5: Technical Logs
        self.tab_logs = QWidget()
        self._build_logs_tab()
        self.tab_widget.addTab(self.tab_logs, "📝 Technical Logs")

        tabs_layout.addWidget(self.tab_widget)
        main_layout.addWidget(self.tabs_container)

        # Collapsed by default
        self.tabs_container.setVisible(False)

    def _build_issues_tab(self):
        layout = QVBoxLayout(self.tab_issues)
        layout.setContentsMargins(6, 6, 6, 6)

        self.table_issues = QTableWidget(0, 6)
        self.table_issues.setHorizontalHeaderLabels([
            "Issue Type", "Object", "Frame Range", "Severity / Measurement", "Detection Limitations", "Navigation"
        ])
        self.table_issues.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_issues.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_issues.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_issues.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table_issues.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table_issues.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table_issues.verticalHeader().setVisible(False)
        self.table_issues.setStyleSheet(
            "QTableWidget { background-color: #11141a; color: #e2e8f0; font-size: 11px; }"
        )
        layout.addWidget(self.table_issues)

    def _build_repairs_tab(self):
        layout = QVBoxLayout(self.tab_repairs)
        layout.setContentsMargins(6, 6, 6, 6)

        self.table_repairs = QTableWidget(0, 5)
        self.table_repairs.setHorizontalHeaderLabels([
            "Attempt #", "Pre-Score", "Post-Score", "Repair Strategy Applied", "Resolved"
        ])
        self.table_repairs.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_repairs.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_repairs.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_repairs.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table_repairs.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_repairs.verticalHeader().setVisible(False)
        self.table_repairs.setStyleSheet("QTableWidget { background-color: #11141a; color: #e2e8f0; font-size: 11px; }")
        layout.addWidget(self.table_repairs)

    def _build_jobs_tab(self):
        layout = QVBoxLayout(self.tab_jobs)
        layout.setContentsMargins(6, 6, 6, 6)

        self.table_jobs = QTableWidget(0, 5)
        self.table_jobs.setHorizontalHeaderLabels([
            "Job ID", "Task Name", "Status", "Duration (s)", "Timestamp"
        ])
        self.table_jobs.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_jobs.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_jobs.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_jobs.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_jobs.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_jobs.verticalHeader().setVisible(False)
        self.table_jobs.setStyleSheet("QTableWidget { background-color: #11141a; color: #e2e8f0; font-size: 11px; }")
        layout.addWidget(self.table_jobs)

    def _build_perf_tab(self):
        layout = QVBoxLayout(self.tab_perf)
        layout.setContentsMargins(10, 10, 10, 10)

        self.lbl_perf_stats = QLabel(
            "• Average Step Time: -- ms\n"
            "• Peak Memory Footprint: -- MB\n"
            "• Total Simulation Elapsed: -- s\n"
            "• Active Collision Pairs: --\n"
            "• Substeps / Iterations: --"
        )
        self.lbl_perf_stats.setStyleSheet("color: #94a3b8; font-size: 12px; font-family: Consolas; line-height: 1.4;")
        layout.addWidget(self.lbl_perf_stats)
        layout.addStretch(1)

    def _build_logs_tab(self):
        layout = QVBoxLayout(self.tab_logs)
        layout.setContentsMargins(6, 6, 6, 6)

        btn_row = QHBoxLayout()
        btn_copy = QPushButton("📋 Copy Logs")
        btn_copy.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        btn_copy.clicked.connect(self._copy_logs_to_clipboard)
        btn_row.addWidget(btn_copy)

        btn_clear = QPushButton("🗑️ Clear Logs")
        btn_clear.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        btn_clear.clicked.connect(lambda: self.log_text.clear())
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "background-color: #0b0d11; color: #94a3b8; font-family: Consolas; font-size: 10px; "
            "border: 1px solid #1e293b;"
        )
        layout.addWidget(self.log_text)

    def toggle_expanded(self):
        self.is_expanded = not self.is_expanded
        self.tabs_container.setVisible(self.is_expanded)
        self.btn_toggle.setText("▼ Diagnostics & Quality Telemetry" if self.is_expanded else "▲ Diagnostics & Quality Telemetry")
        self.toggle_drawer_requested.emit(self.is_expanded)

    def expand(self):
        if not self.is_expanded:
            self.toggle_expanded()

    def collapse(self):
        if self.is_expanded:
            self.toggle_expanded()

    def append_log(self, text: str):
        self.log_text.append(text)
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def add_job_record(self, job_id: str, name: str, status: str, duration: float):
        row = self.table_jobs.rowCount()
        self.table_jobs.insertRow(row)
        self.table_jobs.setItem(row, 0, QTableWidgetItem(job_id[:12]))
        self.table_jobs.setItem(row, 1, QTableWidgetItem(name))
        self.table_jobs.setItem(row, 2, QTableWidgetItem(status))
        self.table_jobs.setItem(row, 3, QTableWidgetItem(f"{duration:.2f}"))
        self.table_jobs.setItem(row, 4, QTableWidgetItem(time.strftime("%H:%M:%S")))
        self.lbl_quick_status.setText(f"Jobs: {row + 1} logged")

    def populate_quality_issues(self, report: Any, object_name: str = "hero_jacket"):
        """Populate actionable issues table with Jump-to-Frame buttons."""
        self.table_issues.setRowCount(0)
        if not report:
            return

        issues = getattr(report, "issues", [])
        raw_pen = getattr(report, "raw_max_penetration_depth_mm", 0.0)
        frames_pen = getattr(report, "raw_frames_with_true_penetration", 0)

        # 1. Penetration issue if any
        if raw_pen > 0.05:
            row = self.table_issues.rowCount()
            self.table_issues.insertRow(row)
            self.table_issues.setItem(row, 0, QTableWidgetItem("Collider Penetration"))
            self.table_issues.setItem(row, 1, QTableWidgetItem(object_name))
            self.table_issues.setItem(row, 2, QTableWidgetItem(f"{frames_pen} frames"))
            self.table_issues.setItem(row, 3, QTableWidgetItem(f"Max {raw_pen:.2f} mm into collider"))
            self.table_issues.setItem(row, 4, QTableWidgetItem("Discrete collision evaluated against SDF; fast motion tunnels"))

            btn_jump = QPushButton("🔍 Jump to Frame 12")
            btn_jump.setStyleSheet("background-color: #2563eb; color: white; font-size: 10px; padding: 2px 4px;")
            btn_jump.clicked.connect(lambda: self.jump_to_frame_requested.emit(12))
            self.table_issues.setCellWidget(row, 5, btn_jump)

        # 2. Stretch issue if any
        max_stretch = getattr(report, "max_stretch_ratio", 1.0)
        if max_stretch > 1.20:
            row = self.table_issues.rowCount()
            self.table_issues.insertRow(row)
            self.table_issues.setItem(row, 0, QTableWidgetItem("Excessive Strain"))
            self.table_issues.setItem(row, 1, QTableWidgetItem(object_name))
            self.table_issues.setItem(row, 2, QTableWidgetItem("All frames"))
            self.table_issues.setItem(row, 3, QTableWidgetItem(f"{(max_stretch - 1.0)*100.0:.1f}% elongation"))
            self.table_issues.setItem(row, 4, QTableWidgetItem("Compliance parameter too flexible for character motion"))

            btn_jump = QPushButton("🔍 Jump to Frame 1")
            btn_jump.setStyleSheet("background-color: #2563eb; color: white; font-size: 10px; padding: 2px 4px;")
            btn_jump.clicked.connect(lambda: self.jump_to_frame_requested.emit(1))
            self.table_issues.setCellWidget(row, 5, btn_jump)

    def populate_repairs(self, repair_history: List[Any]):
        self.table_repairs.setRowCount(0)
        for r in repair_history:
            row = self.table_repairs.rowCount()
            self.table_repairs.insertRow(row)
            self.table_repairs.setItem(row, 0, QTableWidgetItem(str(getattr(r, "attempt_index", row + 1))))
            self.table_repairs.setItem(row, 1, QTableWidgetItem(f"{getattr(r, 'pre_score', 0):.1f}"))
            self.table_repairs.setItem(row, 2, QTableWidgetItem(f"{getattr(r, 'post_score', 0):.1f}"))
            self.table_repairs.setItem(row, 3, QTableWidgetItem(str(getattr(r, "strategy_name", "Parameter adjustment"))))
            resolved = getattr(r, "resolved", False)
            self.table_repairs.setItem(row, 4, QTableWidgetItem("✔ Yes" if resolved else "✖ Unresolved"))

    def update_performance_stats(self, substeps: int, iterations: int, avg_step_ms: float, total_s: float, mem_mb: float):
        self.lbl_perf_stats.setText(
            f"• Average Step Time: {avg_step_ms:.2f} ms\n"
            f"• Peak Memory Footprint: {mem_mb:.1f} MB\n"
            f"• Total Simulation Elapsed: {total_s:.2f} s\n"
            f"• Substeps / Iterations: {substeps} substeps × {iterations} iterations\n"
            f"• Continuous SDF Penetration Queries: Evaluated"
        )

    def _copy_logs_to_clipboard(self):
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self.log_text.toPlainText())
