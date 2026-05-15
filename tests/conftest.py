from __future__ import annotations

import os

from core import configure_native_thread_environment

configure_native_thread_environment()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_data: batch QC against an on-disk lab recording (skips if missing)",
    )


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
