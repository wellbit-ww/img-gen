from __future__ import annotations

import threading

_engine_execution_lock = threading.Lock()


def gpu_job_lock() -> threading.Lock:
    return _engine_execution_lock
