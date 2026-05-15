
import pytest
import numpy as np
from gui.main_window import MainWindow

@pytest.fixture
def main_window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_state_reset_on_new_load(main_window, qtbot, mocker):
    # 1. Manually populate state to simulate a previous run
    main_window.raw_data = np.zeros((100, 10))
    main_window.qc_results = {0: "dummy_result"}
    main_window.sorter_spike_times = {0: np.array([1, 2, 3])}
    main_window.sorter_unit_map = {1: np.array([1, 2, 3])}
    main_window.sorter_dom_channel = {1: 0}
    
    # Mock LoaderWorker and its signals
    mock_loader_cls = mocker.patch("gui.main_window.LoaderWorker")
    mock_loader_inst = mock_loader_cls.return_value
    mock_loader_inst.progress = mocker.Mock()
    mock_loader_inst.finished = mocker.Mock()
    mock_loader_inst.error = mocker.Mock()
    mock_loader_inst.aborted = mocker.Mock()
    
    # 2. Trigger a new load request
    params = {"dat_path": "dummy.dat", "n_channels": 10}
    main_window.on_load_requested(params)
    
    # Simulate loader finishing
    new_raw_data = np.zeros((200, 5))
    main_window._on_loader_finished(new_raw_data)
    
    # 3. Verify state is reset
    assert main_window.raw_data.shape == (200, 5)
    assert len(main_window.qc_results) == 0, "qc_results should be cleared on new load"
    assert len(main_window.sorter_spike_times) == 0, "sorter_spike_times should be cleared on new load"
    assert len(main_window.sorter_unit_map) == 0, "sorter_unit_map should be cleared on new load"
    assert len(main_window.sorter_dom_channel) == 0, "sorter_dom_channel should be cleared on new load"
