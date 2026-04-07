"""
app.py — QApplication setup, dark stylesheet, entry point.
"""
from __future__ import annotations
import sys
from qtpy.QtWidgets import QApplication
from .main_window import MainWindow

DARK_STYLESHEET = """
/* ── Base ───────────────────────────── */
QWidget {
    background-color: #111214;
    color: #F0F0F2;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 12px;
}
QMainWindow, QDialog {
    background-color: #111214;
}

/* ── Splitter handles ────────────────── */
QSplitter::handle {
    background: #2E3038;
}
QSplitter::handle:horizontal { width: 4px; }
QSplitter::handle:vertical { height: 4px; }
QSplitter::handle:hover { background: #4A8BEF; }

/* ── Buttons ─────────────────────────── */
QPushButton {
    background-color: transparent;
    border: 0.5px solid #3D3F48;
    color: #9B9DA6;
    padding: 5px 12px;
    border-radius: 5px;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #1E2025;
    border-color: #5A5C65;
    color: #F0F0F2;
}
QPushButton:pressed {
    background-color: #282A30;
}
QPushButton:disabled {
    color: #3A3C44;
    border-color: #2E3038;
}

/* ── Labels ──────────────────────────── */
QLabel {
    color: #9B9DA6;
    font-size: 12px;
}

/* ── Inputs ──────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #18191C;
    border: 0.5px solid #3D3F48;
    border-radius: 4px;
    padding: 3px 6px;
    color: #F0F0F2;
    font-size: 12px;
}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #5A5C65;
}

/* ── ComboBox ────────────────────────── */
QComboBox {
    background-color: #18191C;
    border: 0.5px solid #3D3F48;
    border-radius: 4px;
    padding: 3px 8px;
    color: #F0F0F2;
    min-height: 22px;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background-color: #282A30;
    color: #F0F0F2;
    selection-background-color: rgba(46, 109, 212, 0.25);
}

/* ── GroupBox ────────────────────────── */
QGroupBox {
    border: 0.5px solid #2E3038;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: bold;
    color: #9B9DA6;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

/* ── Scrollbars ──────────────────────── */
QScrollBar:vertical {
    background: #18191C;
    width: 6px;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #3D3F48;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #5A5C65; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* ── Progress bar ────────────────────── */
QProgressBar {
    background-color: #18191C;
    border: 0.5px solid #3D3F48;
    border-radius: 4px;
    text-align: center;
    color: #9B9DA6;
    font-size: 11px;
    height: 8px;
}
QProgressBar::chunk {
    background-color: #2E6DD4;
    border-radius: 3px;
}

/* ── Status bar ──────────────────────── */
QStatusBar {
    color: #5A5C65;
    font-size: 11px;
    border-top: 0.5px solid #2E3038;
    background: #111214;
    padding: 2px 8px;
}
"""


def create_app(argv) -> tuple[QApplication, MainWindow]:
    """Build QApplication, apply stylesheet, return (app, window)."""
    app = QApplication(argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    return app, window


def run(argv=None, default_dat=None, default_n_channels=None):
    """Full entry: create_app → show → exec."""
    if argv is None:
        argv = sys.argv
    app, window = create_app(argv)

    # Apply CLI defaults
    if default_dat:
        window.default_dat = default_dat
    if default_n_channels:
        window.default_n_channels = default_n_channels

    window.show()
    sys.exit(app.exec_())
