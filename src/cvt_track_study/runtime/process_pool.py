"""Process-pool helpers with prompt interruption semantics.

CPU-bound drivetrain simulations should use separate processes rather than Python
threads.  This wrapper also makes Ctrl+C/error handling prompt on Windows instead
of waiting for long-running worker calls to finish.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any


def terminate_process_pool(executor: ProcessPoolExecutor) -> None:
    """Cancel queued work and terminate running workers without a long wait."""

    processes_obj = getattr(executor, "_processes", None)
    processes = list(processes_obj.values()) if isinstance(processes_obj, dict) else []

    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:  # pragma: no cover - compatibility with older Python
        executor.shutdown(wait=False)

    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (AttributeError, OSError):
            pass

    for process in processes:
        try:
            process.join(timeout=0.25)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=0.25)
        except (AttributeError, OSError):
            pass


@contextmanager
def interruptible_process_pool(*args: Any, **kwargs: Any) -> Iterator[ProcessPoolExecutor]:
    """Yield a process pool that is force-stopped if the parent is interrupted."""

    executor = ProcessPoolExecutor(*args, **kwargs)
    try:
        yield executor
    except BaseException:
        terminate_process_pool(executor)
        raise
    else:
        executor.shutdown(wait=True)
