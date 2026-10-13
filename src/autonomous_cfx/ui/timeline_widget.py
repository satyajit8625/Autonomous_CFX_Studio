"""Autonomous CFX Studio - Timeline and Playback Scrubber Widget.
Supports dynamic frame ranges, variable FPS playback, frame stepping, and visual issue markers.
"""

from __future__ import annotations

from typing import List, Optional
from PySide6.QtCore import QPointF, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)


class MarkedSlider(QSlider):
    """QSlider subclass that draws visual issue markers on violating frames."""

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.issue_frames: List[int] = []

    def set_issue_frames(self, frames: List[int]):
        self.issue_frames = frames
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.issue_frames:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(239, 68, 68), 2)
        painter.setPen(pen)

        min_val = self.minimum()
        max_val = self.maximum()
        span = max_val - min_val if max_val > min_val else 1

        w = self.width() - 20
        h = self.height()

        for f in self.issue_frames:
            if min_val <= f <= max_val:
                t = (f - min_val) / float(span)
                x = 10 + t * w
                # Draw small red indicator line
                painter.drawLine(int(x), h - 6, int(x), h - 1)


class TimelineWidget(QWidget):
    """Timeline controller for scrubbing, playing, and marking issue frames."""

    frame_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.fps: float = 24.0
        self.is_playing: bool = False
        self.start_frame: int = 1
        self.end_frame: int = 40
        self.issue_frames: List[int] = []

        # Playback timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Step Back
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(32)
        self.btn_prev.setToolTip("Step backward 1 frame")
        self.btn_prev.clicked.connect(self.step_backward)
        layout.addWidget(self.btn_prev)

        # Play/Pause
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setFixedWidth(68)
        self.btn_play.clicked.connect(self.toggle_play)
        layout.addWidget(self.btn_play)

        # Step Forward
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(32)
        self.btn_next.setToolTip("Step forward 1 frame")
        self.btn_next.clicked.connect(self.step_forward)
        layout.addWidget(self.btn_next)

        # Current Frame SpinBox
        self.spinbox = QSpinBox()
        self.spinbox.setRange(1, 40)
        self.spinbox.setValue(1)
        self.spinbox.setFixedWidth(60)
        self.spinbox.valueChanged.connect(self.set_frame)
        layout.addWidget(self.spinbox)

        # Frame Slider with issue markers
        self.slider = MarkedSlider(Qt.Horizontal)
        self.slider.setRange(1, 40)
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, stretch=1)

        # Range & FPS Label
        self.lbl_info = QLabel("Frame: 1 / 40  |  24.0 FPS")
        self.lbl_info.setStyleSheet("color: #94a3b8; font-size: 11px; font-family: Consolas;")
        layout.addWidget(self.lbl_info)

    def set_frame_range(self, start_frame: int, end_frame: int, fps: Optional[float] = None):
        """Configure frame limits and optionally frame rate."""
        self.start_frame = start_frame
        self.end_frame = max(start_frame, end_frame)
        if fps:
            self.fps = fps

        self.slider.setRange(self.start_frame, self.end_frame)
        self.spinbox.setRange(self.start_frame, self.end_frame)
        self._update_label(self.slider.value())

    def set_fps(self, fps: float):
        self.fps = fps
        self._update_label(self.slider.value())
        if self.is_playing:
            self.timer.setInterval(int(1000.0 / self.fps))

    def set_issue_markers(self, frames: List[int]):
        """Highlight frames with detected defects."""
        self.issue_frames = frames
        self.slider.set_issue_frames(frames)

    def get_current_frame(self) -> int:
        return self.slider.value()

    def set_frame(self, frame_num: int):
        clamped = max(self.start_frame, min(self.end_frame, frame_num))
        if self.slider.value() != clamped:
            self.slider.setValue(clamped)
        if self.spinbox.value() != clamped:
            self.spinbox.setValue(clamped)
        self._update_label(clamped)
        self.frame_changed.emit(clamped)

    def _on_slider_changed(self, val: int):
        if self.spinbox.value() != val:
            self.spinbox.setValue(val)
        self._update_label(val)
        self.frame_changed.emit(val)

    def _update_label(self, cur_frame: int):
        marker_txt = f"  [⚠️ {len(self.issue_frames)} Issue Frame(s)]" if self.issue_frames else ""
        self.lbl_info.setText(f"Frame: {cur_frame} / {self.end_frame}  |  {self.fps:.1f} FPS{marker_txt}")

    def toggle_play(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        self.is_playing = True
        self.btn_play.setText("❚❚ Pause")
        interval_ms = max(5, int(1000.0 / self.fps))
        self.timer.start(interval_ms)

    def pause(self):
        self.is_playing = False
        self.btn_play.setText("▶ Play")
        self.timer.stop()

    def step_forward(self):
        cur = self.slider.value()
        if cur < self.end_frame:
            self.set_frame(cur + 1)
        else:
            self.set_frame(self.start_frame)

    def step_backward(self):
        cur = self.slider.value()
        if cur > self.start_frame:
            self.set_frame(cur - 1)
        else:
            self.set_frame(self.end_frame)

    def _on_tick(self):
        self.step_forward()
