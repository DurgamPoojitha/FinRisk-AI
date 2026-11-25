"""
Asynchronous Background Job Execution Manager:
- Offloads heavy computational tasks (Monte Carlo, Rolling Backtests, Stress Tests)
- Tracks execution status: PENDING -> RUNNING -> SUCCESS / FAILED
- Supports progress tracking and result retrieval
"""
from __future__ import annotations
import uuid
import time
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.workers")

_executor = ThreadPoolExecutor(max_workers=4)

# In-memory job store (backed by Redis in production cluster)
_JOBS: Dict[str, Dict[str, Any]] = {}


def submit_background_job(
    task_name: str,
    target_func: Callable,
    *args,
    **kwargs,
) -> str:
    """Submits a long-running numerical task to the background executor."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    _JOBS[job_id] = {
        "job_id": job_id,
        "task_name": task_name,
        "status": "PENDING",
        "progress_pct": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }

    def _wrapper():
        _JOBS[job_id]["status"] = "RUNNING"
        _JOBS[job_id]["progress_pct"] = 20
        logger.info(f"Started background job {job_id} ({task_name}).")
        try:
            res = target_func(*args, **kwargs)
            _JOBS[job_id]["status"] = "SUCCESS"
            _JOBS[job_id]["progress_pct"] = 100
            _JOBS[job_id]["result"] = res
            _JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"Completed background job {job_id} successfully.")
        except Exception as e:
            _JOBS[job_id]["status"] = "FAILED"
            _JOBS[job_id]["error"] = str(e)
            _JOBS[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.error(f"Background job {job_id} failed: {e}")

    _executor.submit(_wrapper)
    return job_id


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Returns the current execution status and result of a background job."""
    return _JOBS.get(job_id)
