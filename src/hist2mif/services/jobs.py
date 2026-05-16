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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from hist2mif.core.config import settings
from hist2mif.domain.channels import legend_entries
from hist2mif.services import inference

logger = logging.getLogger(__name__)

DEFAULT_CLI_BATCH_SIZE = 128
DEFAULT_CLI_NUM_WORKERS = 4
DEFAULT_CLI_PIN_MEMORY = True

# CLI snapshot is written at 1/SNAPSHOT_STEP of the full-resolution composite.
SNAPSHOT_STEP = 50

# Candidate paths probed by _load_font for legend text. Order matters: the
# first readable TTF wins so the snapshot still renders if a host lacks
# DejaVu (Ubuntu container, minimal Debian image, etc.).
_LEGEND_FONT_PATHS: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

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
    threshold: float | None = inference.DEFAULT_ACTIVATION_THRESHOLD,
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

    comp_u8 = inference.composite_virtual_mif_rgb_u8(probs, threshold=threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(comp_u8).save(output_path)

    qmeta["downsample_scale"] = scale
    qmeta["activation_threshold"] = threshold
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
    batch_size: int = DEFAULT_CLI_BATCH_SIZE,
    num_workers: int = DEFAULT_CLI_NUM_WORKERS,
    pin_memory: bool = DEFAULT_CLI_PIN_MEMORY,
    threshold: float | None = inference.DEFAULT_ACTIVATION_THRESHOLD,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Stream full-resolution TIFF inference and write a 1/SNAPSHOT_STEP PNG snapshot."""
    if mag not in inference.MAG_INPUT_SIZE:
        raise ValueError(f"Unknown magnification: {mag}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if threshold is not None and not (0.0 < float(threshold) < 1.0):
        raise ValueError("threshold must be in (0, 1) or None for continuous blending")

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
    snapshot = np.zeros(
        ((h + SNAPSHOT_STEP - 1) // SNAPSHOT_STEP, (w + SNAPSHOT_STEP - 1) // SNAPSHOT_STEP, 3),
        dtype=np.uint8,
    )

    dataset = _TiffTileDataset(src_path, h, w, tile_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    loader_iter = iter(loader)

    model, device = _get_or_load_model()

    done = 0
    with torch.inference_mode():
        for batch, y0s, x0s, phs, pws in loader_iter:
            batch = batch.to(device=device, dtype=torch.float32, non_blocking=pin_memory)
            logits = model(batch)
            probs_batch = torch.sigmoid(logits).float().detach().cpu().numpy()

            for i in range(probs_batch.shape[0]):
                y0 = int(y0s[i])
                x0 = int(x0s[i])
                ph = int(phs[i])
                pw = int(pws[i])
                y1 = y0 + ph
                x1 = x0 + pw
                comp = inference.composite_virtual_mif_rgb_u8(
                    probs_batch[i, :, :ph, :pw],
                    rescale=False,
                    threshold=threshold,
                )
                output[y0:y1, x0:x1, :] = comp
                _write_snapshot_region(snapshot, comp, y0, x0)

                done += 1
                if progress_cb is not None:
                    progress_cb(done, total)

            if done % max(nw, 1) == 0:
                output.flush()

    output.flush()
    snapshot_with_legend = _attach_legend(snapshot)
    Image.fromarray(snapshot_with_legend).save(snapshot_png_path)

    meta = {
        "input_h": tile_size,
        "input_w": tile_size,
        "input_hw": tile_size,
        "tile_size": tile_size,
        "window_size": inference.WINDOW,
        "grid": [nh, nw],
        "image_hw": [h, w],
        "snapshot_scale": 1.0 / SNAPSHOT_STEP,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "activation_threshold": threshold,
    }
    return {
        "output_tif_path": str(output_tif_path),
        "snapshot_png_path": str(snapshot_png_path),
        "quilt_meta": meta,
    }


class _TiffTileDataset(Dataset):
    def __init__(self, src_path: Path, height: int, width: int, tile_size: int) -> None:
        self.src_path = src_path
        self.height = height
        self.width = width
        self.tile_size = tile_size
        self.nw = (width + tile_size - 1) // tile_size
        self.total = ((height + tile_size - 1) // tile_size) * self.nw
        self.transform = inference.build_val_transform(tile_size)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int, int]:
        yi = idx // self.nw
        xi = idx % self.nw
        y0 = yi * self.tile_size
        x0 = xi * self.tile_size
        y1 = min(y0 + self.tile_size, self.height)
        x1 = min(x0 + self.tile_size, self.width)
        ph, pw = y1 - y0, x1 - x0

        patch = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
        patch[:ph, :pw, :] = inference.read_he_region(self.src_path, y0, y1, x0, x1)
        tensor = inference.preprocess_patch(patch, self.transform).squeeze(0)
        return tensor, y0, x0, ph, pw


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _LEGEND_FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_legend(target_height: int) -> np.ndarray:
    """Render the paper color legend at a size proportional to ``target_height``."""
    entries = legend_entries()
    n = len(entries)

    row_h = max(12, min(48, target_height // (n + 3)))
    swatch = max(8, int(row_h * 0.7))
    font_size = max(10, int(row_h * 0.6))
    padding = max(6, row_h // 3)

    font = _load_font(font_size)
    text_w = 0
    for name, _ in entries:
        bbox = font.getbbox(name)
        text_w = max(text_w, bbox[2] - bbox[0])

    legend_w = padding + swatch + padding + text_w + padding
    legend_h = max(target_height, n * row_h + 2 * padding)

    img = Image.new("RGB", (legend_w, legend_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    total_rows_h = n * row_h
    start_y = (legend_h - total_rows_h) // 2

    for i, (name, color) in enumerate(entries):
        row_y = start_y + i * row_h
        sy = row_y + (row_h - swatch) // 2
        sx = padding
        draw.rectangle([sx, sy, sx + swatch, sy + swatch], fill=color)

        bbox = font.getbbox(name)
        text_h = bbox[3] - bbox[1]
        tx = sx + swatch + padding
        ty = row_y + (row_h - text_h) // 2 - bbox[1]
        draw.text((tx, ty), name, fill=(255, 255, 255), font=font)

    return np.array(img, dtype=np.uint8)


def _attach_legend(snapshot: np.ndarray) -> np.ndarray:
    """Place the marker legend to the right of ``snapshot`` on a black background."""
    legend = _render_legend(snapshot.shape[0])

    h_snap, w_snap = snapshot.shape[:2]
    h_leg, w_leg = legend.shape[:2]
    out_h = max(h_snap, h_leg)

    canvas = np.zeros((out_h, w_snap + w_leg, 3), dtype=np.uint8)
    canvas[:h_snap, :w_snap, :] = snapshot
    canvas[:h_leg, w_snap : w_snap + w_leg, :] = legend
    return canvas


def _write_snapshot_region(
    snapshot: np.ndarray,
    comp: np.ndarray,
    y0: int,
    x0: int,
) -> None:
    ph, pw = comp.shape[:2]
    ys = np.arange(y0, y0 + ph)
    xs = np.arange(x0, x0 + pw)
    keep_y = ys[ys % SNAPSHOT_STEP == 0]
    keep_x = xs[xs % SNAPSHOT_STEP == 0]
    if keep_y.size == 0 or keep_x.size == 0:
        return
    tile_y = keep_y - y0
    tile_x = keep_x - x0
    snapshot[np.ix_(keep_y // SNAPSHOT_STEP, keep_x // SNAPSHOT_STEP)] = comp[np.ix_(tile_y, tile_x)]


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
