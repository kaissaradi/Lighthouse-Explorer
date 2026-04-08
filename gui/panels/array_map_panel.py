"""
array_map_panel.py — Channel selector with grouped views and progress bar.
"""
from __future__ import annotations
from typing import Optional
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QColor, QBrush, QFont
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QComboBox,
)


class ArrayMapPanel(QWidget):
    """Channel selector with groups: All / LH Found / Uncertain / No LH."""

    channel_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._n_channels: int = 0
        self._selected_ch: Optional[int] = None
        self._qc_results: dict = {}  # {ch: QCResult}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Top toolbar: group filter + search + Go
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["All", "LH Found", "Uncertain", "No LH"])
        self._view_combo.currentTextChanged.connect(self._rebuild_list)
        toolbar.addWidget(self._view_combo)
        toolbar.addWidget(QLabel("CH:"))
        self._ch_input = QLineEdit()
        self._ch_input.setPlaceholderText("#")
        self._ch_input.returnPressed.connect(self._on_go)
        self._ch_input.setMaximumWidth(50)
        toolbar.addWidget(self._ch_input)
        self._go_btn = QPushButton("Go")
        self._go_btn.clicked.connect(self._on_go)
        toolbar.addWidget(self._go_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Progress bar (shown during batch QC)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumHeight(14)
        layout.addWidget(self._progress_bar)

        # Progress label
        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("color: #888; font-size: 10px;")
        self._progress_lbl.setVisible(False)
        layout.addWidget(self._progress_lbl)

        # Channel list
        self._channel_list = QListWidget()
        self._channel_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._channel_list)

        # Status label
        self._status_lbl = QLabel("No recording loaded.")
        self._status_lbl.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status_lbl)

    def set_array(self, ei_positions):
        """
        Populate channels after data load.
        ei_positions: [C, 2] or [C, 1] — we use C (num channels).
        """
        import numpy as np

        positions = np.asarray(ei_positions)
        self._n_channels = positions.shape[0]
        self._selected_ch = None
        self._qc_results.clear()

        self._status_lbl.setText(f"{self._n_channels} channels loaded.")
        self._rebuild_list()

    def set_progress(self, current: int, total: int, message: str = ""):
        """Show progress during batch QC."""
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_lbl.setVisible(True)
        self._progress_lbl.setText(message or f"CH {current}/{total}")

    def hide_progress(self):
        """Hide progress bar after batch QC completes."""
        self._progress_bar.setVisible(False)
        self._progress_lbl.setVisible(False)

    def update_channel_result(self, ch: int, result):
        """Add a completed QC result and refresh the list."""
        self._qc_results[ch] = result
        self._rebuild_list()

    def set_qc_result_color(self, ch: int, miss_rate: Optional[float]):
        """
        Update a channel's display after QC. Called from main window.
        We rebuild the whole list to keep grouping consistent.
        """
        # Just trigger a rebuild — colors are applied in _rebuild_list
        self._rebuild_list()

    def set_selected_channel(self, ch: int):
        """Highlight the selected channel in the list."""
        self._selected_ch = ch
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.data(Qt.UserRole) == ch:
                item.setSelected(True)
                self._channel_list.scrollToItem(item)
                break

    def set_firing_rates(self, fr_values: dict[int, float]):
        """Not used — no-op."""
        pass

    def clear(self):
        """Reset the channel list."""
        self._n_channels = 0
        self._selected_ch = None
        self._qc_results.clear()
        self._channel_list.clear()
        self._status_lbl.setText("No recording loaded.")
        self.hide_progress()

    # ── internal ───────────────────────────────────────────────────

    def _get_group_channels(self) -> list[int]:
        """Return channels matching current view filter."""
        view = self._view_combo.currentText()
        if view == "All":
            return list(range(self._n_channels))

        channels = []
        for ch in range(self._n_channels):
            result = self._qc_results.get(ch)
            if result is None:
                continue

            n_lh = result.n_lh
            n_total = result.n_total

            if view == "LH Found":
                if n_lh > 0:
                    channels.append(ch)
            elif view == "Uncertain":
                if result.n_uncertain > 0:
                    channels.append(ch)
            elif view == "No LH":
                if n_lh == 0 and n_total > 0:
                    channels.append(ch)

        return channels

    def _rebuild_list(self):
        """Rebuild the channel list based on current view filter."""
        self._channel_list.clear()
        channels = self._get_group_channels()

        for ch in channels:
            result = self._qc_results.get(ch)
            if result:
                label = f"CH {ch} — {result.n_lh} LH, {result.n_soup} soup, {result.n_uncertain} unc"
                # Color based on LH ratio
                if result.n_total > 0:
                    lh_ratio = result.n_lh / result.n_total
                else:
                    lh_ratio = 0
                r = int(255 * (1.0 - lh_ratio))
                g = int(255 * lh_ratio)
                color = QColor(r, g, 40)
            else:
                label = f"CH {ch} — pending"
                color = QColor(136, 136, 136)

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, ch)
            item.setForeground(QBrush(color))
            self._channel_list.addItem(item)

        # Update status
        shown = len(channels)
        total = self._n_channels
        self._status_lbl.setText(f"Showing {shown}/{total} channels.")

        # Re-select if current channel is visible
        if self._selected_ch is not None:
            self.set_selected_channel(self._selected_ch)

    def _on_item_clicked(self, item: QListWidgetItem):
        ch = item.data(Qt.UserRole)
        if ch is not None:
            self.channel_selected.emit(int(ch))

    def _on_go(self):
        """Jump to a specific channel by number."""
        text = self._ch_input.text().strip()
        if not text:
            return
        try:
            ch = int(text)
        except ValueError:
            return

        if 0 <= ch < self._n_channels:
            # Switch to "All" view so the channel is visible
            self._view_combo.setCurrentText("All")
            self._select_channel_by_index(ch)
            self._ch_input.clear()

    def _select_channel_by_index(self, ch: int):
        """Programmatically select a channel and emit signal."""
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.data(Qt.UserRole) == ch:
                self._channel_list.setCurrentItem(item)
                self._channel_list.scrollToItem(item)
                self.channel_selected.emit(ch)
                break
