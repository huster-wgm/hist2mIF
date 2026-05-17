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

import cv2
import numpy as np
import torch
import tifffile
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from hist2mif.core.config import settings
from hist2mif.domain.channels import (
    CHANNEL_DISPLAY_NAMES,
    CHANNEL_NAMES_23,
    EXPORT_CHANNELS,
    LEGEND_ORDER,
    legend_entries,
)
from hist2mif.services import inference

logger = logging.getLogger(__name__)

DEFAULT_CLI_BATCH_SIZE = 128
DEFAULT_CLI_NUM_WORKERS = 4
DEFAULT_CLI_PIN_MEMORY = True

# Pipeline scales (all cv2.INTER_NEAREST):
#   probs (256x256)  --binarize-->  mask (21, 256, 256)
#       --cv2.resize 1/MASK_TIF_DOWNSAMPLE NEAREST-->  mask_small (21, 16, 16)
#   mask_buffer (21, H/16, W/16) is then used to produce:
#     mIF TIFF      = 21 mask pages, JPEG q=90, same 1/16 resolution
#     snapshot PNG  = composite(mask_buffer) * brightness gain + paper legend
#                     (also at 1/16; no extra resize)
#     thumbnail PNG = 4x6 grid of per-channel colored masks (paper palette)
#                     with white padding between cells; empty cells are black
MASK_TIF_DOWNSAMPLE = 16

# JPEG quality used for the per-channel mask TIFF pages (libjpeg level).
OUTPUT_JPEG_QUALITY = 90

# Thumbnail grid layout (6 cols x 4 rows = 24 cells; 21 markers + 3 black).
THUMBNAIL_GRID_COLS = 6
THUMBNAIL_GRID_ROWS = 4
THUMBNAIL_CELL_WIDTH = 400
THUMBNAIL_PADDING = 12
THUMBNAIL_PAD_COLOR: tuple[int, int, int] = (255, 255, 255)

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
    thumbnail_png_path: Path | None = None,
    batch_size: int = DEFAULT_CLI_BATCH_SIZE,
    num_workers: int = DEFAULT_CLI_NUM_WORKERS,
    pin_memory: bool = DEFAULT_CLI_PIN_MEMORY,
    threshold: float | None = inference.DEFAULT_ACTIVATION_THRESHOLD,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict:
    """Stream tile inference and write a 21-channel binary-mask TIFF + composite PNG + thumbnail grid."""
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
    if tile_size % MASK_TIF_DOWNSAMPLE != 0:
        raise ValueError(
            f"tile_size={tile_size} must be divisible by MASK_TIF_DOWNSAMPLE "
            f"({MASK_TIF_DOWNSAMPLE})"
        )
    mask_block = tile_size // MASK_TIF_DOWNSAMPLE

    nh = (h + tile_size - 1) // tile_size
    nw = (w + tile_size - 1) // tile_size
    total = nh * nw

    output_tif_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_png_path.parent.mkdir(parents=True, exist_ok=True)
    output_tif_path.unlink(missing_ok=True)
    snapshot_png_path.unlink(missing_ok=True)
    if thumbnail_png_path is not None:
        thumbnail_png_path.parent.mkdir(parents=True, exist_ok=True)
        thumbnail_png_path.unlink(missing_ok=True)

    export_idxs = np.asarray([i for i, _ in EXPORT_CHANNELS], dtype=np.int64)
    channel_names = [CHANNEL_NAMES_23[i] for i in export_idxs]
    n_channels = export_idxs.size

    mask_buffer = np.zeros((n_channels, nh * mask_block, nw * mask_block), dtype=np.uint8)

    mask_threshold = float(
        threshold if threshold is not None else inference.DEFAULT_ACTIVATION_THRESHOLD
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
                yi = y0 // tile_size
                xi = x0 // tile_size

                probs_tile = probs_batch[i]  # (23, tile_size, tile_size)
                if ph < tile_size:
                    probs_tile[:, ph:, :] = 0.0
                if pw < tile_size:
                    probs_tile[:, :, pw:] = 0.0

                # Binarize 21 export channels then resize 1/MASK_TIF_DOWNSAMPLE.
                mask_full_chw = (probs_tile[export_idxs] > mask_threshold).astype(np.uint8) * 255
                mask_full_hwc = np.ascontiguousarray(mask_full_chw.transpose(1, 2, 0))
                mask_small_hwc = cv2.resize(
                    mask_full_hwc,
                    (mask_block, mask_block),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_buffer[
                    :,
                    yi * mask_block : (yi + 1) * mask_block,
                    xi * mask_block : (xi + 1) * mask_block,
                ] = mask_small_hwc.transpose(2, 0, 1)

                done += 1
                if progress_cb is not None:
                    progress_cb(done, total)

    with tifffile.TiffWriter(output_tif_path) as tw:
        for ch_idx in range(n_channels):
            tw.write(
                mask_buffer[ch_idx],
                photometric="minisblack",
                compression="jpeg",
                compressionargs={"level": OUTPUT_JPEG_QUALITY},
                description=channel_names[ch_idx],
                metadata={"channel": channel_names[ch_idx], "index": int(ch_idx)},
            )

    # Snapshot: full-resolution (same as mask TIFF) RGB composite + paper legend.
    snapshot = _composite_full_mask(mask_buffer)
    snapshot_with_legend = _attach_legend(snapshot)
    Image.fromarray(snapshot_with_legend).save(snapshot_png_path)

    # Thumbnail: 4x6 grid of per-channel colored masks with white padding.
    thumb_hw: list[int] | None = None
    if thumbnail_png_path is not None:
        thumbnail = _build_channel_thumbnail(mask_buffer)
        Image.fromarray(thumbnail).save(thumbnail_png_path)
        thumb_hw = list(thumbnail.shape[:2])

    meta = {
        "input_h": tile_size,
        "input_w": tile_size,
        "input_hw": tile_size,
        "tile_size": tile_size,
        "window_size": inference.WINDOW,
        "grid": [nh, nw],
        "image_hw": [h, w],
        "snapshot_scale": 1.0 / MASK_TIF_DOWNSAMPLE,
        "snapshot_hw": list(snapshot.shape[:2]),
        "output_tif_scale": 1.0 / MASK_TIF_DOWNSAMPLE,
        "output_tif_hw": [mask_buffer.shape[1], mask_buffer.shape[2]],
        "output_tif_compression": "jpeg",
        "output_tif_jpeg_quality": OUTPUT_JPEG_QUALITY,
        "output_tif_channels": channel_names,
        "output_tif_threshold": mask_threshold,
        "thumbnail_hw": thumb_hw,
        "thumbnail_grid": [THUMBNAIL_GRID_ROWS, THUMBNAIL_GRID_COLS],
        "downsample": "cv2.INTER_NEAREST",
        "composite_divisor": int(n_channels),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "activation_threshold": threshold,
    }
    return {
        "output_tif_path": str(output_tif_path),
        "snapshot_png_path": str(snapshot_png_path),
        "thumbnail_png_path": str(thumbnail_png_path) if thumbnail_png_path is not None else None,
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


def _composite_full_mask(mask_buffer: np.ndarray, *, row_chunk: int = 256) -> np.ndarray:
    """Composite (C, H, W) mask buffer into (H, W, 3) RGB via composite_mask_rgb_u8.

    Row-chunked to bound peak memory: a single ``composite_mask_rgb_u8`` call on
    a full mask buffer would materialize a ``(C, H, W)`` float32 copy and an
    ``einsum`` intermediate (both ~2 GB for typical slides), so we slice the
    height into ``row_chunk`` strips and composite them independently.
    """
    _, height, width = mask_buffer.shape
    snapshot = np.zeros((height, width, 3), dtype=np.uint8)
    for y0 in range(0, height, row_chunk):
        y1 = min(y0 + row_chunk, height)
        snapshot[y0:y1] = inference.composite_mask_rgb_u8(mask_buffer[:, y0:y1, :])
    return snapshot


def _build_channel_thumbnail(
    mask_buffer: np.ndarray,
    *,
    cols: int = THUMBNAIL_GRID_COLS,
    rows: int = THUMBNAIL_GRID_ROWS,
    cell_width: int = THUMBNAIL_CELL_WIDTH,
    padding: int = THUMBNAIL_PADDING,
    pad_color: tuple[int, int, int] = THUMBNAIL_PAD_COLOR,
) -> np.ndarray:
    """Render a cols x rows grid of per-channel masks painted with paper colors.

    Channels are drawn in `LEGEND_ORDER` so the grid reads in the same
    top-to-bottom order as the snapshot's legend column. Empty cells (when the
    grid is larger than the channel count) are filled black so the surrounding
    white padding still reads as a "between cells" separator.
    """
    n_channels, full_h, full_w = mask_buffer.shape
    if full_h == 0 or full_w == 0:
        raise ValueError("mask buffer must have non-zero spatial extent")

    cell_h = max(1, int(round(cell_width * full_h / full_w)))
    grid_w = cols * cell_width + (cols + 1) * padding
    grid_h = rows * cell_h + (rows + 1) * padding
    grid = np.full((grid_h, grid_w, 3), pad_color, dtype=np.uint8)

    pal_u8 = (inference.get_export_palette() * 255.0).astype(np.uint8)  # (C, 3)
    name_to_pos = {CHANNEL_NAMES_23[i]: pos for pos, (i, _) in enumerate(EXPORT_CHANNELS)}
    ordered = [(name, name_to_pos[name]) for name in LEGEND_ORDER if name in name_to_pos]

    font_size = max(12, cell_h // 18)
    font = _load_font(font_size)
    text_pad = max(4, padding // 2)

    for cell_idx in range(rows * cols):
        r, c = divmod(cell_idx, cols)
        y0 = padding + r * (cell_h + padding)
        x0 = padding + c * (cell_width + padding)
        y1 = y0 + cell_h
        x1 = x0 + cell_width

        if cell_idx >= len(ordered):
            grid[y0:y1, x0:x1] = 0
            continue

        display_name, ch_pos = ordered[cell_idx]
        label = CHANNEL_DISPLAY_NAMES.get(display_name, display_name)

        mask = mask_buffer[ch_pos]
        cell_mask = cv2.resize(mask, (cell_width, cell_h), interpolation=cv2.INTER_NEAREST)
        cell = np.zeros((cell_h, cell_width, 3), dtype=np.uint8)
        cell[cell_mask > 127] = pal_u8[ch_pos]

        # Draw a small swatch + label in the top-left so each cell self-identifies.
        cell_img = Image.fromarray(cell)
        draw = ImageDraw.Draw(cell_img)
        swatch = max(font_size, 16)
        draw.rectangle(
            [text_pad, text_pad, text_pad + swatch, text_pad + swatch],
            fill=tuple(int(v) for v in pal_u8[ch_pos]),
            outline=(255, 255, 255),
            width=1,
        )
        draw.text(
            (text_pad + swatch + text_pad, text_pad - 2),
            label,
            fill=(255, 255, 255),
            font=font,
        )
        grid[y0:y1, x0:x1] = np.asarray(cell_img, dtype=np.uint8)

    return grid


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
