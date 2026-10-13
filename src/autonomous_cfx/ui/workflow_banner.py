"""Autonomous CFX Studio - Workflow Banner and Stage Stepper Widget.
Displays the 7-step artist workflow:
Import Alembic -> Assign Roles -> Describe Behavior -> Simulate -> Mark Corrections -> Review -> Export Alembic.
Highlights current active stage and explains any blocked actions.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class WorkflowStage(str, Enum):
    IMPORT_ALEMBIC = "1. Import Alembic"
    ASSIGN_ROLES = "2. Assign Roles"
    DESCRIBE_BEHAVIOR = "3. Describe Behavior"
    SIMULATE = "4. Simulate"
    MARK_CORRECTIONS = "5. Mark Corrections"
    REVIEW = "6. Review"
    EXPORT_ALEMBIC = "7. Export Alembic"


STAGE_ORDER = [
    WorkflowStage.IMPORT_ALEMBIC,
    WorkflowStage.ASSIGN_ROLES,
    WorkflowStage.DESCRIBE_BEHAVIOR,
    WorkflowStage.SIMULATE,
    WorkflowStage.MARK_CORRECTIONS,
    WorkflowStage.REVIEW,
    WorkflowStage.EXPORT_ALEMBIC,
]


class WorkflowBannerWidget(QWidget):
    """Horizontal workflow stepper showing the 7 stages and active status."""

    stage_clicked = Signal(WorkflowStage)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_stage: WorkflowStage = WorkflowStage.IMPORT_ALEMBIC
        self.stage_labels: dict[WorkflowStage, QLabel] = {}
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 4, 6, 4)
        main_layout.setSpacing(4)

        # 1. Stepper row
        stepper_frame = QFrame()
        stepper_frame.setStyleSheet(
            "background-color: #171a21; border-radius: 6px; border: 1px solid #2a313d;"
        )
        layout_stepper = QHBoxLayout(stepper_frame)
        layout_stepper.setContentsMargins(8, 4, 8, 4)
        layout_stepper.setSpacing(4)

        for i, stage in enumerate(STAGE_ORDER):
            lbl = QLabel(stage.value)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(self._get_stage_style(stage, is_active=(i == 0)))
            layout_stepper.addWidget(lbl, stretch=1)
            self.stage_labels[stage] = lbl

            if i < len(STAGE_ORDER) - 1:
                arr = QLabel("→")
                arr.setStyleSheet("color: #4b5563; font-weight: bold; font-size: 11px;")
                layout_stepper.addWidget(arr)

        main_layout.addWidget(stepper_frame)

        # 2. Blocked / Status explanation banner
        self.lbl_status = QLabel("Ready to start: Import an animated Alembic shot or load demo scene.")
        self.lbl_status.setStyleSheet(
            "color: #94a3b8; font-size: 11px; padding: 2px 6px; font-weight: 500;"
        )
        main_layout.addWidget(self.lbl_status)

    def set_stage(self, stage: WorkflowStage, status_text: Optional[str] = None, is_blocked: bool = False):
        """Update active workflow stage and explanation."""
        self.current_stage = stage
        curr_idx = STAGE_ORDER.index(stage)

        for i, s in enumerate(STAGE_ORDER):
            lbl = self.stage_labels[s]
            is_active = (s == stage)
            is_completed = (i < curr_idx)
            lbl.setStyleSheet(self._get_stage_style(s, is_active=is_active, is_completed=is_completed))

        if status_text:
            if is_blocked:
                self.lbl_status.setText(f"⚠️ {status_text}")
                self.lbl_status.setStyleSheet("color: #f59e0b; font-size: 11px; padding: 2px 6px; font-weight: 600;")
            else:
                self.lbl_status.setText(f"✓ {status_text}")
                self.lbl_status.setStyleSheet("color: #38bdf8; font-size: 11px; padding: 2px 6px; font-weight: 500;")

    def _get_stage_style(self, stage: WorkflowStage, is_active: bool = False, is_completed: bool = False) -> str:
        if is_active:
            return (
                "background-color: #2563eb; color: #ffffff; font-weight: bold; "
                "font-size: 11px; padding: 4px 8px; border-radius: 4px;"
            )
        elif is_completed:
            return (
                "background-color: #1e293b; color: #10b981; font-weight: bold; "
                "font-size: 11px; padding: 4px 8px; border-radius: 4px;"
            )
        else:
            return (
                "background-color: transparent; color: #64748b; font-size: 11px; "
                "padding: 4px 8px; border-radius: 4px;"
            )
