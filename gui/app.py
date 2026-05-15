"""
app.py — QApplication setup and entry point.

The dark stylesheet lives in gui/theme.py.
"""
from __future__ import annotations
import sys
from qtpy.QtWidgets import QApplication
from .main_window import MainWindow
from .theme import DARK_STYLESHEET


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
