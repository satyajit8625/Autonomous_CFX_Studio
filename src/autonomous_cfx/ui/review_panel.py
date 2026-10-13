"""Autonomous CFX Studio - Artist Review and Export Panel.
Enables CFX artists to inspect quality metrics, approve or reject simulation results, and trigger cache export.
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from autonomous_cfx.agent.evaluator import QualityMetricsReport, QualityStatus


class ReviewPanel(QWidget):
    """Panel for reviewing simulation quality and exporting approved caches."""

    approve_clicked = Signal()
    reject_clicked = Signal()
    export_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # 1. Quality Score Header
        grp_score = QGroupBox("Quality Diagnostic Assessment")
        vbox_s = QVBoxLayout(grp_score)

        self.lbl_score_badge = QLabel("Score: -- / 100")
        self.lbl_score_badge.setAlignment(Qt.AlignCenter)
        self.lbl_score_badge.setStyleSheet(
            "background-color: #2b303c; color: #a0a8b4; font-size: 18px; font-weight: bold; "
            "padding: 10px; border-radius: 6px;"
        )
        vbox_s.addWidget(self.lbl_score_badge)

        self.lbl_status = QLabel("Awaiting Simulation")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #728299; font-size: 12px; font-weight: 500;")
        vbox_s.addWidget(self.lbl_status)

        main_layout.addWidget(grp_score)

        # 2. Detailed Metrics Table
        grp_metrics = QGroupBox("Production CFX Tolerances")
        vbox_m = QVBoxLayout(grp_metrics)

        self.table_metrics = QTableWidget(5, 2)
        self.table_metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table_metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_metrics.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_metrics.verticalHeader().setVisible(False)
        self.table_metrics.setEditTriggers(QTableWidget.NoEditTriggers)

        metric_names = [
            "Max Penetration Depth",
            "Frames w/ Penetration",
            "Max Stretch Ratio",
            "Peak Velocity",
            "Numerical Stability",
        ]
        for row, name in enumerate(metric_names):
            self.table_metrics.setItem(row, 0, QTableWidgetItem(name))
            self.table_metrics.setItem(row, 1, QTableWidgetItem("--"))

        vbox_m.addWidget(self.table_metrics)
        main_layout.addWidget(grp_metrics)

        # 3. Artist Review & Approval
        grp_review = QGroupBox("Artist Review & Approval")
        vbox_r = QVBoxLayout(grp_review)

        btn_row = QHBoxLayout()
        self.btn_approve = QPushButton("✔ Approve Cache")
        self.btn_approve.setStyleSheet(
            "background-color: #208b47; color: white; font-weight: bold; padding: 8px; border-radius: 4px;"
        )
        self.btn_approve.clicked.connect(self.approve_clicked.emit)
        btn_row.addWidget(self.btn_approve)

        self.btn_reject = QPushButton("✖ Reject")
        self.btn_reject.setStyleSheet(
            "background-color: #a83232; color: white; padding: 8px; border-radius: 4px;"
        )
        self.btn_reject.clicked.connect(self.reject_clicked.emit)
        btn_row.addWidget(self.btn_reject)
        vbox_r.addLayout(btn_row)

        self.btn_export = QPushButton("💾 Export Approved Caches (OBJ / .vpcache / Manifest)")
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("padding: 8px; font-weight: 500;")
        self.btn_export.clicked.connect(self.export_clicked.emit)
        vbox_r.addWidget(self.btn_export)

        self.lbl_export_status = QLabel("No cache approved for export")
        self.lbl_export_status.setStyleSheet("color: #728299; font-size: 10px; font-style: italic;")
        self.lbl_export_status.setWordWrap(True)
        vbox_r.addWidget(self.lbl_export_status)

        main_layout.addWidget(grp_review, stretch=1)

    def update_report(self, report: QualityMetricsReport):
        """Update metrics display from QualityMetricsReport."""
        score_val = report.score
        if report.status == QualityStatus.APPROVED:
            self.lbl_score_badge.setText(f"Score: {score_val:.1f} / 100")
            self.lbl_score_badge.setStyleSheet(
                "background-color: #1a4d2e; color: #4ade80; font-size: 18px; font-weight: bold; "
                "padding: 10px; border-radius: 6px;"
            )
            self.lbl_status.setText("STATUS: APPROVED (Within Tolerances)")
            self.lbl_status.setStyleSheet("color: #4ade80; font-weight: bold;")
            self.btn_export.setEnabled(True)
            self.lbl_export_status.setText("Cache passed quality checks. Ready for export.")
        elif report.status == QualityStatus.NEEDS_REPAIR:
            self.lbl_score_badge.setText(f"Score: {score_val:.1f} / 100")
            self.lbl_score_badge.setStyleSheet(
                "background-color: #5c3b0d; color: #fbbf24; font-size: 18px; font-weight: bold; "
                "padding: 10px; border-radius: 6px;"
            )
            self.lbl_status.setText("STATUS: NEEDS REPAIR")
            self.lbl_status.setStyleSheet("color: #fbbf24; font-weight: bold;")
        else:
            self.lbl_score_badge.setText(f"Score: {score_val:.1f} / 100")
            self.lbl_score_badge.setStyleSheet(
                "background-color: #5c1818; color: #f87171; font-size: 18px; font-weight: bold; "
                "padding: 10px; border-radius: 6px;"
            )
            self.lbl_status.setText("STATUS: CRITICAL FAILURE")
            self.lbl_status.setStyleSheet("color: #f87171; font-weight: bold;")

        # Populate Table
        self.table_metrics.setItem(0, 1, QTableWidgetItem(f"{report.max_penetration_depth*1000.0:.2f} mm"))
        self.table_metrics.setItem(1, 1, QTableWidgetItem(f"{report.frames_with_penetration} / {report.total_frames}"))
        self.table_metrics.setItem(2, 1, QTableWidgetItem(f"{report.max_stretch_ratio*100.0:.1f} %"))
        self.table_metrics.setItem(3, 1, QTableWidgetItem(f"{report.peak_velocity:.1f} m/s"))
        self.table_metrics.setItem(4, 1, QTableWidgetItem("Stable" if report.is_numerically_stable else "Unstable"))

    def mark_approved(self, is_approved: bool = True):
        self.btn_export.setEnabled(is_approved)
        if is_approved:
            self.lbl_export_status.setText("Artist manually approved cache. Export unlocked.")
        else:
            self.lbl_export_status.setText("Cache rejected by artist.")

    def set_export_success(self, path: str, count: int):
        self.lbl_export_status.setText(f"Exported {count} OBJ frames and point cache to: {path}")
        self.lbl_export_status.setStyleSheet("color: #4ade80; font-size: 10px;")
