"""
FastAPI Routes: Asynchronous Background Jobs Status & Control.
"""
from fastapi import APIRouter, HTTPException, status
from backend.app.workers.job_manager import get_job_status

router = APIRouter(prefix="/jobs", tags=["Background Compute Jobs"])


@router.get("/{job_id}", summary="Get background job status and result")
def get_job(job_id: str):
    job = get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    return job
