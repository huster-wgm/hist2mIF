"""Inference jobs API."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from hist2mif.core.config import settings
from hist2mif.services.jobs import (
    create_job_with_file,
    delete_job,
    format_time_ago,
    list_job_ids,
    read_meta,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _job_summary(job_id: str) -> dict:
    m = read_meta(job_id) or {}
    done = int(m.get("progress_done") or 0)
    total = int(m.get("progress_total") or 0)
    status = str(m.get("status") or "-")
    if total > 0 and status in {"running", "complete"}:
        progress_label = f"{done}/{total}"
    else:
        progress_label = "-"

    rel = m.get("composite_png_rel") or m.get("preview_png")
    thumb_url = f"/api/jobs/{job_id}/composite" if status == "complete" and rel else None

    return {
        "id": job_id,
        "filename": m.get("filename") or "-",
        "mag": m.get("mag") or "-",
        "status": status,
        "progress_done": done,
        "progress_total": total,
        "progress_label": progress_label,
        "created": m.get("created") or "",
        "updated": m.get("updated") or "",
        "time_ago": format_time_ago(m),
        "error": m.get("error"),
        "thumbnail_url": thumb_url,
        "composite_url": thumb_url,
    }


@router.get("/")
def api_list_jobs() -> list[dict]:
    return [_job_summary(jid) for jid in list_job_ids()]


@router.get("/{job_id}")
def api_get_job(job_id: str) -> dict:
    if read_meta(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_summary(job_id)


@router.post("/")
async def api_create_job(
    file: UploadFile = File(...),
    mag: str = Form("20x"),
) -> JSONResponse:
    if mag not in {"10x", "20x"}:
        raise HTTPException(status_code=400, detail='mag must be "10x" or "20x"')

    name = file.filename or "upload.tif"
    suf = Path(name).suffix.lower()
    if suf not in {".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Only .tif / .tiff uploads are supported")

    fd, str_path = tempfile.mkstemp(suffix=suf)
    tmp_path = Path(str_path)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(await file.read())
        job_id = create_job_with_file(tmp_path, mag)
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse({"job_id": job_id}, status_code=201)


@router.get("/{job_id}/composite")
def api_job_composite(job_id: str) -> FileResponse:
    m = read_meta(job_id)
    if m is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if m.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Composite not ready")
    rel = m.get("composite_png_rel") or m.get("preview_png")
    if not rel:
        raise HTTPException(status_code=404, detail="No composite path")
    path = settings.data_root / str(rel)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Composite file missing")
    return FileResponse(path, media_type="image/png", filename=path.name)


@router.delete("/{job_id}")
def api_delete_job(job_id: str) -> dict[str, bool]:
    if read_meta(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    ok = delete_job(job_id)
    return {"ok": ok}
