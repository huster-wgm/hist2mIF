"""Job persistence and background inference worker."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import tifffile
from PIL import Image

from hist2mif.core.config import settings
from hist2mif.services import inference

logger = logging.getLogger(__name__)

_model: object | None = None
_model_device: object | None = None
_model_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relative_to_data(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(settings.data_root.resolve()))
    except ValueError:
        return str(p)


def job_dir(job_id: str) -> Path:
    return settings.jobs_dir / job_id


def read_meta(job_id: str) -> dict | None:
    p = job_dir(job_id) / "meta.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_meta(job_id: str, meta: dict) -> None:
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    meta.setdefault("updated", _utc_now())
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def update_meta(job_id: str, **patch: object) -> dict:
    meta = read_meta(job_id) or {}
    meta.update(patch)
    meta["updated"] = _utc_now()
    write_meta(job_id, meta)
    return meta


def list_job_ids() -> list[str]:
    root = settings.jobs_dir
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in dirs]


def format_time_ago(meta: dict) -> str:
    ts = meta.get("created") or meta.get("updated") or ""
    if not ts:
        return "-"
    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return f"{seconds}s ago"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 48:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except ValueError:
        return ts


def _get_or_load_model():
    global _model, _model_device
    with _model_lock:
        if _model is None:
            device = inference.get_device()
            logger.info("Loading GigaTIME on %s", device)
            _model_device = device
            _model = inference.load_model(device)
        return _model, _model_device


def run_inference_to_png(
    src_path: Path,
    output_path: Path,
    mag: str,
    *,
    max_side: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Run the shared TIFF -> virtual mIF composite PNG workflow synchronously."""
    if mag not in inference.MAG_INPUT_SIZE:
        raise ValueError(f"Unknown magnification: {mag}")

    target_max_side = max_side or settings.max_side
    rgb = inference.load_he_image(src_path, max_side=target_max_side)
    rgb, scale = inference.downsample_if_needed(rgb, target_max_side)

    model, device = _get_or_load_model()
    probs, qmeta = inference.predict_image_quilt(rgb, model, device, mag, progress_cb=progress_cb)

    comp_u8 = inference.composite_virtual_mif_rgb_u8(probs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(comp_u8).save(output_path)

    qmeta["downsample_scale"] = scale
    return {
        "output_path": str(output_path),
        "quilt_meta": qmeta,
    }


def run_inference_to_tif_and_snapshot(
    src_path: Path,
    output_tif_path: Path,
    snapshot_png_path: Path,
    mag: str,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Stream full-resolution TIFF inference and write a 1/10 PNG snapshot."""
    if mag not in inference.MAG_INPUT_SIZE:
        raise ValueError(f"Unknown magnification: {mag}")

    h, w = inference.get_he_image_shape(src_path)
    tile_size = inference.MAG_INPUT_SIZE[mag]
    nh = (h + tile_size - 1) // tile_size
    nw = (w + tile_size - 1) // tile_size
    total = nh * nw

    output_tif_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_png_path.parent.mkdir(parents=True, exist_ok=True)
    output_tif_path.unlink(missing_ok=True)
    snapshot_png_path.unlink(missing_ok=True)

    output = tifffile.memmap(
        output_tif_path,
        shape=(h, w, 3),
        dtype=np.uint8,
        photometric="rgb",
        bigtiff=True,
    )
    snapshot = np.zeros(((h + 9) // 10, (w + 9) // 10, 3), dtype=np.uint8)

    model, device = _get_or_load_model()
    transform = inference.build_val_transform(tile_size)

    done = 0
    for yi in range(nh):
        for xi in range(nw):
            y0 = yi * tile_size
            x0 = xi * tile_size
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            ph, pw = y1 - y0, x1 - x0

            patch = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            patch[:ph, :pw, :] = inference.read_he_region(src_path, y0, y1, x0, x1)

            x = inference.preprocess_patch(patch, transform).to(dtype=torch.float32)
            logits = inference.do_inference_windows(x, model, device, window_size=inference.WINDOW)
            probs = torch.sigmoid(logits)[0].float().cpu().numpy()
            comp = inference.composite_virtual_mif_rgb_u8(probs[:, :ph, :pw], rescale=False)

            output[y0:y1, x0:x1, :] = comp
            _write_snapshot_region(snapshot, comp, y0, x0)

            done += 1
            if done % max(nw, 1) == 0:
                output.flush()
            if progress_cb is not None:
                progress_cb(done, total)

    output.flush()
    Image.fromarray(snapshot).save(snapshot_png_path)

    meta = {
        "input_hw": tile_size,
        "tile_size": tile_size,
        "grid": [nh, nw],
        "image_hw": [h, w],
        "snapshot_scale": 0.1,
    }
    return {
        "output_tif_path": str(output_tif_path),
        "snapshot_png_path": str(snapshot_png_path),
        "quilt_meta": meta,
    }


def _write_snapshot_region(
    snapshot: np.ndarray,
    comp: np.ndarray,
    y0: int,
    x0: int,
) -> None:
    ph, pw = comp.shape[:2]
    ys = np.arange(y0, y0 + ph)
    xs = np.arange(x0, x0 + pw)
    keep_y = ys[ys % 10 == 0]
    keep_x = xs[xs % 10 == 0]
    if keep_y.size == 0 or keep_x.size == 0:
        return
    tile_y = keep_y - y0
    tile_x = keep_x - x0
    snapshot[np.ix_(keep_y // 10, keep_x // 10)] = comp[np.ix_(tile_y, tile_x)]


def _run_job_worker(job_id: str, saved_input: Path, mag: str) -> None:
    stem = saved_input.stem
    try:
        def cb(done: int, total: int) -> None:
            update_meta(job_id, progress_done=done, progress_total=total, status="running")

        composite_path = job_dir(job_id) / f"{stem}_virtual_mIF_composite.png"
        result = run_inference_to_png(saved_input, composite_path, mag, progress_cb=cb)
        qmeta = result["quilt_meta"]

        total = int(qmeta["grid"][0] * qmeta["grid"][1])
        update_meta(
            job_id,
            status="complete",
            progress_done=total,
            progress_total=total,
            composite_png_rel=relative_to_data(composite_path),
            quilt_meta=qmeta,
        )
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        update_meta(job_id, status="failed", error=str(e))


def create_job_with_file(src_path: Path, mag: str) -> str:
    """Copy validated upload into a new job directory and start inference in a background thread."""
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)

    job_id = str(uuid.uuid4())
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)

    suffix = src_path.suffix.lower()
    saved_input = d / f"input{suffix}"
    shutil.copyfile(src_path, saved_input)

    filename = src_path.stem + suffix

    update_meta(
        job_id,
        id=job_id,
        filename=filename,
        mag=mag,
        status="running",
        progress_done=0,
        progress_total=1,
        created=_utc_now(),
        error=None,
        composite_png_rel=None,
        input_rel=relative_to_data(saved_input),
    )

    t = threading.Thread(target=_run_job_worker, args=(job_id, saved_input, mag), daemon=True)
    t.start()
    return job_id


def delete_job(job_id: str) -> bool:
    path = job_dir(job_id)
    if not path.is_dir():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True
