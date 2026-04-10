"""
array_map_panel.py — Channel selector with grouped views and progress bar.
"""
from __future__ import annotations
from typing import Optional
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QColor, QBrush
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


class _ChannelListWidget(QListWidget):
    """QListWidget that emits on arrow-key navigation in addition to clicks."""

    channel_activated = Signal(int)  # emitted on arrow key move or Enter

    def keyPressEvent(self, event):
        super().keyPressEvent(event)  # let Qt move the selection first
        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Return, Qt.Key_Enter):
            item = self.currentItem()
            if item is not None:
                ch = item.data(Qt.UserRole)
                if ch is not None:
                    self.channel_activated.emit(int(ch))


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

        # Top toolbar
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

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumHeight(14)
        layout.addWidget(self._progress_bar)

        # Progress label
        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("color: #888; font-size: 10px;")
        self._progress_lbl.setVisible(False)
        layout.addWidget(self._progress_lbl)

        # Channel list — custom subclass handles keyboard
        self._channel_list = _ChannelListWidget()
        self._channel_list.setFocusPolicy(Qt.StrongFocus)
        self._channel_list.enterEvent = lambda e: self._channel_list.setFocus()
        self._channel_list.itemClicked.connect(self._on_item_clicked)
        self._channel_list.channel_activated.connect(self.channel_selected)
        layout.addWidget(self._channel_list)

        # Status label
        self._status_lbl = QLabel("No recording loaded.")
        self._status_lbl.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status_lbl)

    # ── public API ─────────────────────────────────────────────────

    def set_array(self, ei_positions):
        import numpy as np
        positions = np.asarray(ei_positions)
        self._n_channels = positions.shape[0]
        self._selected_ch = None
        self._qc_results.clear()
        self._status_lbl.setText(f"{self._n_channels} channels loaded.")
        self._rebuild_list()

    def set_progress(self, current: int, total: int, message: str = ""):
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_lbl.setVisible(True)
        self._progress_lbl.setText(message or f"CH {current}/{total}")

    def hide_progress(self):
        self._progress_bar.setVisible(False)
        self._progress_lbl.setVisible(False)

    def update_channel_result(self, ch: int, result):
        """Update a single channel's label+color without rebuilding the whole list."""
        self._qc_results[ch] = result
        view = self._view_combo.currentText()

        # Check if this channel should be visible in the current view
        visible = self._ch_passes_filter(ch, result, view)

        # Find existing item for this channel
        existing_row = None
        for i in range(self._channel_list.count()):
            if self._channel_list.item(i).data(Qt.UserRole) == ch:
                existing_row = i
                break

        if visible:
            label, color = self._make_label_color(ch, result)
            if existing_row is not None:
                # Update in place — no clear(), selection preserved
                item = self._channel_list.item(existing_row)
                item.setText(label)
                item.setForeground(QBrush(color))
            else:
                # New item — insert in channel-number order
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, ch)
                item.setForeground(QBrush(color))
                insert_pos = self._find_insert_pos(ch)
                self._channel_list.insertItem(insert_pos, item)
        else:
            # Channel no longer passes filter — remove it
            if existing_row is not None:
                self._channel_list.takeItem(existing_row)

        self._update_status()

    def set_qc_result_color(self, ch: int, miss_rate: Optional[float]):
        result = self._qc_results.get(ch)
        if result:
            self.update_channel_result(ch, result)

    def set_selected_channel(self, ch: int):
        self._selected_ch = ch
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.data(Qt.UserRole) == ch:
                self._channel_list.setCurrentItem(item)
                self._channel_list.scrollToItem(item)
                break

    def clear(self):
        self._n_channels = 0
        self._selected_ch = None
        self._qc_results.clear()
        self._channel_list.clear()
        self._status_lbl.setText("No recording loaded.")
        self.hide_progress()

    # ── internals ──────────────────────────────────────────────────

    def _ch_passes_filter(self, ch: int, result, view: str) -> bool:
        if view == "All":
            return True
        if result is None:
            return False
        if view == "LH Found":
            return result.n_lh > 0
        if view == "Uncertain":
            return result.n_uncertain > 0
        if view == "No LH":
            return result.n_lh == 0 and result.n_total > 0
        return True

    def _make_label_color(self, ch: int, result):
        if result:
            label = f"CH {ch} — {result.n_lh} LH, {result.n_soup} soup, {result.n_uncertain} unc"
            lh_ratio = result.n_lh / result.n_total if result.n_total > 0 else 0
            color = QColor(int(255 * (1.0 - lh_ratio)), int(255 * lh_ratio), 40)
        else:
            label = f"CH {ch} — pending"
            color = QColor(136, 136, 136)
        return label, color

    def _find_insert_pos(self, ch: int) -> int:
        """Binary-search for insertion index maintaining channel order."""
        lo, hi = 0, self._channel_list.count()
        while lo < hi:
            mid = (lo + hi) // 2
            if self._channel_list.item(mid).data(Qt.UserRole) < ch:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _get_group_channels(self) -> list[int]:
        view = self._view_combo.currentText()
        if view == "All":
            return list(range(self._n_channels))
        channels = []
        for ch in range(self._n_channels):
            result = self._qc_results.get(ch)
            if self._ch_passes_filter(ch, result, view):
                channels.append(ch)
        return channels

    def _rebuild_list(self):
        """Full rebuild — only called on view-filter change or initial load."""
        # Remember current item so we can restore it
        cur_item = self._channel_list.currentItem()
        cur_ch = cur_item.data(Qt.UserRole) if cur_item else self._selected_ch

        self._channel_list.blockSignals(True)
        self._channel_list.clear()
        channels = self._get_group_channels()
        for ch in channels:
            result = self._qc_results.get(ch)
            label, color = self._make_label_color(ch, result)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, ch)
            item.setForeground(QBrush(color))
            self._channel_list.addItem(item)
        self._channel_list.blockSignals(False)

        self._update_status()

        # Restore selection
        if cur_ch is not None:
            self.set_selected_channel(cur_ch)

    def _update_status(self):
        shown = self._channel_list.count()
        self._status_lbl.setText(f"Showing {shown}/{self._n_channels} channels.")

    def _on_item_clicked(self, item: QListWidgetItem):
        ch = item.data(Qt.UserRole)
        if ch is not None:
            self.channel_selected.emit(int(ch))

    def _on_go(self):
        text = self._ch_input.text().strip()
        if not text:
            return
        try:
            ch = int(text)
        except ValueError:
            return
        if 0 <= ch < self._n_channels:
            self._view_combo.setCurrentText("All")
            self._select_channel_by_index(ch)
            self._ch_input.clear()

    def _select_channel_by_index(self, ch: int):
        for i in range(self._channel_list.count()):
            item = self._channel_list.item(i)
            if item.data(Qt.UserRole) == ch:
                self._channel_list.setCurrentItem(item)
                self._channel_list.scrollToItem(item)
                self.channel_selected.emit(ch)
                break