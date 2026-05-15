import pytest
import numpy as np
import psutil
import os
import gc
from gui.workers.batch_qc_worker import BatchQCWorker
from qtpy.QtCore import QCoreApplication

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # In MB

@pytest.mark.parametrize("iterations", [5])
def test_batch_qc_worker_memory_leak(qtbot, iterations):
    # Mock raw data: 1000 samples, 10 channels
    raw_data = np.random.randint(-1000, 1000, (1000, 10), dtype=np.int16)
    params = {}
    
    initial_mem = get_memory_usage()
    
    for i in range(iterations):
        worker = BatchQCWorker(raw_data=raw_data, params=params)
        
        with qtbot.waitSignal(worker.finished, timeout=10000):
            worker.run()
        
        # Explicitly delete and collect
        del worker
        gc.collect()
        QCoreApplication.processEvents()
    
    final_mem = get_memory_usage()
    
    # Allow some growth, but if it's substantial, it's a leak.
    assert final_mem - initial_mem < 5.0, f"Memory leak detected: {final_mem - initial_mem:.2f} MB increase"
