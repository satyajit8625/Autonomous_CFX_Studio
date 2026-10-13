"""Autonomous CFX Studio - Persistent AI Agent Conversation Panel.
Replaces technical logs with an interactive conversational feed showing:
- Artist requests
- Interpreted intent (material, stretch, bend, friction, damping, margins)
- Selected objects and frame range
- Proposed actions
- Clarification questions (with interactive answer buttons)
- Current activity and progress
- Remaining problems & reviewable quality result
Displays actual runtime transparently: Local model, Disconnected, or Rule-based mode.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from autonomous_cfx.agent.llm_client import LocalLLMClient


class AgentConversationPanel(QWidget):
    """Interactive conversational agent panel with persistent structured cards."""

    prompt_submitted = Signal(str)      # Emitted when artist sends a prompt
    run_sim_requested = Signal(str)     # Emitted when artist clicks Run Simulation
    cancel_requested = Signal()         # Emitted when artist cancels active job
    clarification_answered = Signal(str) # Emitted when artist selects an option for clarification

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumWidth(380)

        # Local LLM client for runtime status
        self.llm_client = LocalLLMClient()
        self.runtime_mode = "Rule-based mode (Deterministic grammar)"
        self.runtime_color = "#38bdf8"
        self._check_runtime_status()

        self.is_simulating: bool = False
        self.has_valid_setup: bool = False
        self.last_parsed_props: Optional[Dict[str, Any]] = None

        self._build_ui()

    def _check_runtime_status(self):
        """Detect whether a live local LLM endpoint is connected or if we run in deterministic rule mode."""
        if self.llm_client.backend in ("ollama", "openai_compatible"):
            try:
                import urllib.request
                req = urllib.request.Request(self.llm_client.endpoint_url, timeout=1.0)
                with urllib.request.urlopen(req) as resp:
                    if resp.status == 200:
                        self.runtime_mode = f"Local model: {self.llm_client.model_name}"
                        self.runtime_color = "#34d399"
                        return
            except Exception:
                pass
            self.runtime_mode = "Disconnected (Endpoint unreachable)"
            self.runtime_color = "#f87171"
        else:
            self.runtime_mode = "Rule-based mode (Deterministic grammar)"
            self.runtime_color = "#38bdf8"

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # 1. Top Runtime Badge Header
        header_frame = QFrame()
        header_frame.setStyleSheet(
            "background-color: #171a21; border-radius: 6px; border: 1px solid #232834; padding: 4px;"
        )
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(8, 4, 8, 4)

        lbl_badge_title = QLabel("Agent Runtime:")
        lbl_badge_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
        h_layout.addWidget(lbl_badge_title)

        self.lbl_runtime_status = QLabel(f"● {self.runtime_mode}")
        self.lbl_runtime_status.setStyleSheet(
            f"color: {self.runtime_color}; font-weight: bold; font-size: 11px;"
        )
        self.lbl_runtime_status.setToolTip(
            "Offline deterministic rule-based grammar engine.\n"
            "Never presents deterministic fallback as neural-model execution."
        )
        h_layout.addWidget(self.lbl_runtime_status, stretch=1)

        main_layout.addWidget(header_frame)

        # 2. Persistent Conversation Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet(
            "QScrollArea { background-color: #11141a; border: 1px solid #232834; border-radius: 6px; }"
        )

        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background-color: transparent;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch(1)

        self.scroll_area.setWidget(self.chat_container)
        main_layout.addWidget(self.scroll_area, stretch=1)

        # 3. Input & Presets Section
        grp_input = QGroupBox("Describe Cloth Behavior")
        grp_input.setStyleSheet("QGroupBox { font-size: 12px; font-weight: bold; color: #cbd5e1; }")
        vbox_in = QVBoxLayout(grp_input)
        vbox_in.setContentsMargins(8, 6, 8, 6)
        vbox_in.setSpacing(6)

        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText(
            "Describe physical behavior (e.g. heavy leather jacket, stiff collar, low stretch, high friction with body)..."
        )
        self.txt_prompt.setMaximumHeight(65)
        self.txt_prompt.setStyleSheet(
            "background-color: #171a21; color: #f1f5f9; border: 1px solid #334155; "
            "border-radius: 4px; padding: 4px; font-size: 12px;"
        )
        self.txt_prompt.setText("heavy leather jacket, stiff collar, low stretch, high friction with body, high damping")
        vbox_in.addWidget(self.txt_prompt)

        # Preset chips
        chip_layout = QHBoxLayout()
        chip_layout.setSpacing(4)
        presets = [
            ("Leather", "heavy leather jacket, stiff collar, low stretch, high friction with body, high damping"),
            ("Silk", "lightweight silk blouse, loose drape, soft fluid folds, low friction"),
            ("Denim", "rugged denim jacket, medium weight, tight stretch resistance"),
            ("Wool", "heavy wool overcoat, thick folds, high damping, structured drape"),
        ]
        for name, p_text in presets:
            btn = QPushButton(name)
            btn.setStyleSheet(
                "QPushButton { background-color: #232834; color: #94a3b8; font-size: 10px; "
                "padding: 3px 6px; border-radius: 3px; border: 1px solid #334155; }"
                "QPushButton:hover { background-color: #334155; color: #f8fafc; }"
            )
            btn.clicked.connect(lambda checked=False, t=p_text: self.txt_prompt.setText(t))
            chip_layout.addWidget(btn)
        chip_layout.addStretch(1)
        vbox_in.addLayout(chip_layout)

        main_layout.addWidget(grp_input)

        # 4. Simulation Execution Controls
        btn_row = QHBoxLayout()
        self.btn_run_sim = QPushButton("⚡ Run Autonomous Simulation")
        self.btn_run_sim.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; font-weight: bold; font-size: 12px; "
            "padding: 9px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
            "QPushButton:disabled { background-color: #1e293b; color: #64748b; }"
        )
        self.btn_run_sim.setEnabled(False)
        self.btn_run_sim.setToolTip("Please import geometry and confirm roles in the Scene Browser first.")
        self.btn_run_sim.clicked.connect(self._on_run_sim_clicked)
        btn_row.addWidget(self.btn_run_sim, stretch=2)

        self.btn_cancel = QPushButton("⏹ Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "QPushButton { background-color: #475569; color: white; font-weight: 500; padding: 9px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #dc2626; }"
            "QPushButton:disabled { background-color: #1e293b; color: #475569; }"
        )
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        btn_row.addWidget(self.btn_cancel, stretch=1)

        main_layout.addLayout(btn_row)

        # Initial Welcome Message
        self.add_system_message(
            "Autonomous CFX Physical Agent ready. Import an animated Alembic shot or load demo assets to begin."
        )

    def set_setup_validity(self, has_garment: bool, target_garment_name: str = "jacket", frame_range: str = "1-40"):
        """Enable or disable simulation button based on scene validation."""
        self.has_valid_setup = has_garment
        if has_garment:
            self.btn_run_sim.setEnabled(not self.is_simulating)
            self.btn_run_sim.setToolTip(f"Ready to simulate {target_garment_name} across frames {frame_range}.")
        else:
            self.btn_run_sim.setEnabled(False)
            self.btn_run_sim.setToolTip("No garment confirmed. Confirm roles in the Scene Browser.")

    def set_running_state(self, is_running: bool):
        """Update button states during background simulation."""
        self.is_simulating = is_running
        self.btn_run_sim.setEnabled(not is_running and self.has_valid_setup)
        self.btn_cancel.setEnabled(is_running)

    def _on_run_sim_clicked(self):
        prompt = self.txt_prompt.toPlainText().strip()
        if not prompt:
            return

        # Add artist user message card
        self.add_artist_message(prompt)

        # Trigger intent interpretation
        assistant_text, tool_calls = self.llm_client.generate_response(prompt)

        for tc in tool_calls:
            if tc.tool_name == "request_clarification":
                self.add_clarification_card(tc.arguments)
                return
            elif tc.tool_name == "configure_simulation_parameters":
                self.last_parsed_props = tc.arguments
                self.add_intent_card(tc.arguments)

        self.run_sim_requested.emit(prompt)

    # Message & Card Builders
    def add_artist_message(self, text: str):
        card = QFrame()
        card.setStyleSheet(
            "background-color: #1e3a5f; border-radius: 8px; border: 1px solid #2563eb; padding: 6px;"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        lbl_header = QLabel("👤 Artist Request")
        lbl_header.setStyleSheet("color: #93c5fd; font-size: 11px; font-weight: bold;")
        layout.addWidget(lbl_header)

        lbl_body = QLabel(text)
        lbl_body.setWordWrap(True)
        lbl_body.setStyleSheet("color: #ffffff; font-size: 12px;")
        layout.addWidget(lbl_body)

        self._insert_chat_card(card)

    def add_intent_card(self, args: Dict[str, Any]):
        card = QFrame()
        card.setStyleSheet(
            "background-color: #1a2230; border-radius: 8px; border: 1px solid #334155; padding: 6px;"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        lbl_header = QLabel(f"🤖 Interpreted Intent: {args.get('material_name', 'Cloth').upper()}")
        lbl_header.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
        layout.addWidget(lbl_header)

        lbl_reason = QLabel(args.get("reasoning", ""))
        lbl_reason.setWordWrap(True)
        lbl_reason.setStyleSheet("color: #cbd5e1; font-size: 11px; font-style: italic;")
        layout.addWidget(lbl_reason)

        # Details grid
        details = (
            f"• Density: {args.get('density')} kg/m²  |  "
            f"Stretch Compliance: {args.get('stretch_compliance'):.1e}\n"
            f"• Bend Compliance: {args.get('bend_compliance'):.1e}  |  "
            f"Damping: {args.get('damping')}\n"
            f"• Friction: {args.get('friction')}  |  "
            f"Margin: {args.get('collision_margin')*1000.0:.1f} mm  |  "
            f"Substeps: {args.get('solver_substeps')}"
        )
        lbl_details = QLabel(details)
        lbl_details.setStyleSheet("color: #94a3b8; font-size: 11px; font-family: Consolas;")
        layout.addWidget(lbl_details)

        self._insert_chat_card(card)

    def add_clarification_card(self, clarif: Dict[str, Any]):
        card = QFrame()
        card.setStyleSheet(
            "background-color: #3b2808; border-radius: 8px; border: 1px solid #d97706; padding: 6px;"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        lbl_header = QLabel("⚠️ Contradictory Intent - Clarification Required")
        lbl_header.setStyleSheet("color: #fbbf24; font-size: 11px; font-weight: bold;")
        layout.addWidget(lbl_header)

        lbl_q = QLabel(clarif.get("clarification_question", ""))
        lbl_q.setWordWrap(True)
        lbl_q.setStyleSheet("color: #ffffff; font-size: 12px;")
        layout.addWidget(lbl_q)

        # Option buttons
        for opt in clarif.get("suggested_options", []):
            btn_opt = QPushButton(opt)
            btn_opt.setStyleSheet(
                "QPushButton { background-color: #78350f; color: #fde68a; font-size: 11px; "
                "text-align: left; padding: 6px; border-radius: 4px; border: 1px solid #b45309; }"
                "QPushButton:hover { background-color: #92400e; color: #ffffff; }"
            )
            opt_text = opt
            btn_opt.clicked.connect(lambda checked=False, o=opt_text: self._on_clarification_choice(o))
            layout.addWidget(btn_opt)

        self._insert_chat_card(card)

    def _on_clarification_choice(self, chosen_option: str):
        self.add_artist_message(f"Selected resolution: {chosen_option}")
        self.clarification_answered.emit(chosen_option)
        # Re-interpret with selected preference
        if "structured" in chosen_option.lower() or "leather" in chosen_option.lower():
            self.txt_prompt.setText("heavy leather jacket, stiff collar, low stretch, high friction with body, high damping")
        else:
            self.txt_prompt.setText("lightweight silk, fluid drape, loose folds, low friction")
        self._on_run_sim_clicked()

    def add_activity_card(self, stage: str, percent: int) -> QProgressBar:
        card = QFrame()
        card.setStyleSheet(
            "background-color: #171d28; border-radius: 8px; border: 1px solid #2d3748; padding: 6px;"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        lbl_header = QLabel(f"⚙️ Simulation Activity: {stage}")
        lbl_header.setStyleSheet("color: #60a5fa; font-size: 11px; font-weight: bold;")
        layout.addWidget(lbl_header)

        pbar = QProgressBar()
        pbar.setRange(0, 100)
        pbar.setValue(percent)
        pbar.setStyleSheet("QProgressBar { height: 14px; text-align: center; font-size: 10px; }")
        layout.addWidget(pbar)

        self._insert_chat_card(card)
        return pbar

    def add_result_card(self, status: str, score: float, max_pen_mm: float, max_strain_pct: float, repairs_count: int = 0):
        card = QFrame()
        is_pass = "pass" in status.lower() or "approved" in status.lower()
        bg_col = "#064e3b" if is_pass else "#451a03"
        border_col = "#10b981" if is_pass else "#f59e0b"
        title_col = "#34d399" if is_pass else "#fbbf24"

        card.setStyleSheet(
            f"background-color: {bg_col}; border-radius: 8px; border: 1px solid {border_col}; padding: 6px;"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        lbl_header = QLabel(f"📊 Quality Result: {status.upper()} (Score: {score:.1f}/100)")
        lbl_header.setStyleSheet(f"color: {title_col}; font-size: 12px; font-weight: bold;")
        layout.addWidget(lbl_header)

        info = (
            f"• Max Raw Penetration: {max_pen_mm:.2f} mm\n"
            f"• Peak Stretch Strain: {max_strain_pct:.1f}%\n"
            f"• Bounded Repairs Applied: {repairs_count}"
        )
        lbl_info = QLabel(info)
        lbl_info.setStyleSheet("color: #f1f5f9; font-size: 11px; font-family: Consolas;")
        layout.addWidget(lbl_info)

        lbl_action = QLabel("Inspect timeline & viewport. Mark corrections if local defects exist, or Approve cache.")
        lbl_action.setWordWrap(True)
        lbl_action.setStyleSheet("color: #cbd5e1; font-size: 11px; margin-top: 4px;")
        layout.addWidget(lbl_action)

        self._insert_chat_card(card)

    def add_system_message(self, text: str):
        card = QFrame()
        card.setStyleSheet("background-color: #171a21; border-radius: 6px; padding: 4px;")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(6, 4, 6, 4)
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(lbl)
        self._insert_chat_card(card)

    def _insert_chat_card(self, widget: QWidget):
        # Insert before bottom stretch
        count = self.chat_layout.count()
        self.chat_layout.insertWidget(count - 1, widget)
        # Scroll to bottom
        QApplication = widget.parent()
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
