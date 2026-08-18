
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

    # 2. Trigger a new load request (no auto-batch — avoids stray pool tasks)
    params = {
        "dat_path": "dummy.dat",
        "n_channels": 10,
        "fs": 20000,
        "start_min": 0.0,
        "auto_batch_qc": False,
    }
    main_window.on_load_requested(params)

    # Simulate loader finishing
    new_raw_data = np.zeros((200, 5))
    main_window._on_loader_finished(new_raw_data)

    # 3. Verify recording/QC state is reset, but KS is preserved
    assert main_window.raw_data.shape == (200, 5)
    assert len(main_window.qc_results) == 0, "qc_results should be cleared on new load"
    # KS may be loaded before or after recording — must survive recording reload
    assert len(main_window.sorter_spike_times) == 1, "sorter maps must persist across recording load"
    assert len(main_window.sorter_unit_map) == 1
    assert len(main_window.sorter_dom_channel) == 1


def test_reattach_sorter_updates_existing_results(main_window):
    from tests.factories import make_qc_result

    r = make_qc_result(channel=0, n_lh=3, n_soup=0)
    r.n_sorter_spikes = -1
    main_window.qc_results[0] = r
    main_window.current_channel = 0
    main_window.raw_data = np.zeros((10_000, 2), dtype=np.int16)
    main_window.lh_params["start_sample"] = 0
    main_window.lh_params["fs"] = 20_000

    # Absolute KS times inside window
    main_window.sorter_unit_map = {7: np.array([100, 200, 300], dtype=np.int64)}
    main_window.sorter_dom_channel = {7: 0}
    main_window.sorter_spike_times = {0: np.array([100, 200, 300], dtype=np.int64)}
    main_window.sorter_units_by_channel = {0: [7]}
    main_window.sorter_unit_labels = {7: "good"}

    main_window._reattach_sorter_to_all_results()
    assert r.n_sorter_spikes == 3
    assert r.sorter_times is not None and r.sorter_times.size == 3
    assert 7 in r.sorter_unit_map
    assert getattr(r, "sorter_unit_labels", {}).get(7) == "good"


def test_ks_channel_alignment_drops_oor(main_window):
    """KS channels outside recording width must be dropped with a warning note."""
    main_window.raw_data = np.zeros((1000, 4), dtype=np.int16)
    main_window.sorter_spike_times = {
        0: np.array([1, 2], dtype=np.int64),
        10: np.array([3, 4], dtype=np.int64),  # OOR for 4-ch recording
    }
    main_window.sorter_units_by_channel = {0: [1], 10: [2]}
    main_window.sorter_unit_map = {
        1: np.array([1, 2], dtype=np.int64),
        2: np.array([3, 4], dtype=np.int64),
    }
    main_window.sorter_dom_channel = {1: 0, 2: 10}
    main_window.sorter_channel_meta = {
        "channel_map_min": 0,
        "channel_map_max": 10,
        "channel_map_n": 11,
        "channel_map_is_identity": False,
    }
    note = main_window._validate_sorter_channel_alignment()
    assert "dropped" in note or "outside" in note
    assert 10 not in main_window.sorter_spike_times
    assert 0 in main_window.sorter_spike_times
    assert 2 not in main_window.sorter_unit_map


def test_ks_channel_alignment_ok_identity(main_window):
    main_window.raw_data = np.zeros((1000, 8), dtype=np.int16)
    main_window.sorter_spike_times = {0: np.array([1]), 7: np.array([2])}
    main_window.sorter_channel_meta = {
        "channel_map_min": 0,
        "channel_map_max": 7,
        "channel_map_n": 8,
        "channel_map_is_identity": True,
    }
    note = main_window._validate_sorter_channel_alignment()
    assert "aligned" in note
