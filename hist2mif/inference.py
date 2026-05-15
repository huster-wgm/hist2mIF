"""GigaTIME inference helpers (patch quilt + tiled forward)."""

from __future__ import annotations

import math
import os
import zipfile
from pathlib import Path
from typing import Callable

import albumentations as geometric
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import tifffile
from albumentations.augmentations import transforms as alb_transforms
from albumentations.core.composition import Compose
from huggingface_hub import snapshot_download
from PIL import Image

from hist2mif.archs import gigatime
from hist2mif.channels import CHANNEL_NAMES_23, EXPORT_CHANNELS

NUM_CLASSES = 23
WINDOW = 256

# Match scripts/gigatime_testing.ipynb defaults unless overridden by magnification.
MAG_INPUT_SIZE: dict[str, int] = {
    "20x": 512,
    "10x": 384,
}

_REPO_ID = "prov-gigatime/GigaTIME"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    device: torch.device,
    *,
    hf_token: str | None = None,
    local_files_only: bool = False,
) -> torch.nn.Module:
    token = hf_token or os.environ.get("HF_TOKEN")
    local_dir = snapshot_download(
        repo_id=_REPO_ID,
        token=token if token else None,
        local_files_only=local_files_only,
    )
    weights_path = os.path.join(local_dir, "model.pth")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing model weights at {weights_path}")

    model = gigatime(NUM_CLASSES, 3)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_he_image(path: Path) -> npt.NDArray[np.uint8]:
    """Load a TIFF/SVS-style raster as RGB uint8 (H, W, 3)."""
    arr = tifffile.imread(path)
    rgb = _to_rgb_uint8(arr)
    return rgb


def _to_rgb_uint8(arr: npt.NDArray[np.floating | np.integer]) -> npt.NDArray[np.uint8]:
    if arr.ndim == 2:
        x = arr.astype(np.float32)
        x = _scale_to_01(x)
        u8 = (x * 255.0).clip(0, 255).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=-1)

    if arr.ndim == 3:
        # Prefer obvious HWC RGB/RGBA; otherwise treat small leading dim as CHW.
        if arr.shape[-1] in (3, 4):
            hwc = np.asarray(arr)
        elif arr.shape[0] in (1, 3, 4):
            chw = np.asarray(arr)
            if chw.shape[0] == 1:
                x = chw[0].astype(np.float32)
                x = _scale_to_01(x)
                u8 = (x * 255.0).clip(0, 255).astype(np.uint8)
                return np.stack([u8, u8, u8], axis=-1)
            if chw.shape[0] >= 3:
                r = chw[0].astype(np.float32)
                g = chw[1].astype(np.float32)
                b = chw[2].astype(np.float32)
                r, g, b = (_scale_to_01(x) for x in (r, g, b))
                rgb = np.stack(
                    [
                        (r * 255.0).clip(0, 255),
                        (g * 255.0).clip(0, 255),
                        (b * 255.0).clip(0, 255),
                    ],
                    axis=-1,
                ).astype(np.uint8)
                return rgb
            raise ValueError(f"Unsupported CHW array shape: {arr.shape}")

        hwc = np.asarray(arr)
        if hwc.shape[-1] == 1:
            x = hwc[..., 0].astype(np.float32)
            x = _scale_to_01(x)
            u8 = (x * 255.0).clip(0, 255).astype(np.uint8)
            return np.stack([u8, u8, u8], axis=-1)
        if hwc.shape[-1] >= 3:
            x = hwc[..., :3].astype(np.float32)
            x = _scale_to_01(x)
            return (x * 255.0).clip(0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported array shape for H&E: {arr.shape}")


def _scale_to_01(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normalize arbitrary dtype/range to roughly [0, 1]."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    mn = float(finite.min())
    mx = float(finite.max())
    if math.isclose(mx, mn):
        return np.zeros_like(x, dtype=np.float32)
    if mx <= 1.5 and mn >= -0.5:
        # Likely already 0..1 floats
        return np.clip(x, 0.0, 1.0)
    return np.clip((x - mn) / (mx - mn), 0.0, 1.0)


def downsample_if_needed(
    rgb: npt.NDArray[np.uint8],
    max_side: int,
) -> tuple[npt.NDArray[np.uint8], float]:
    h, w = rgb.shape[0], rgb.shape[1]
    m = max(h, w)
    if m <= max_side:
        return rgb, 1.0
    scale = max_side / m
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(rgb)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8), scale


def build_val_transform(input_hw: int) -> Compose:
    return Compose(
        [
            geometric.Resize(input_hw, input_hw),
            alb_transforms.Normalize(),
        ]
    )


def preprocess_patch(patch_hwc: npt.NDArray[np.uint8], transform: Compose) -> torch.Tensor:
    aug = transform(image=patch_hwc)
    chw = aug["image"].astype(np.float32).transpose(2, 0, 1)
    return torch.from_numpy(chw).unsqueeze(0)


def pad_to_multiple(x: torch.Tensor, mult: int) -> torch.Tensor:
    _, _, h, w = x.shape
    nh = ((h + mult - 1) // mult) * mult
    nw = ((w + mult - 1) // mult) * mult
    if nh == h and nw == w:
        return x
    pad_h = nh - h
    pad_w = nw - w
    return F.pad(x, (0, pad_w, 0, pad_h))


@torch.inference_mode()
def do_inference_windows(
    x_bchw: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    *,
    window_size: int = WINDOW,
) -> torch.Tensor:
    """Slide non-overlapping windows of size window_size across a square input."""
    b, _c, h, w = x_bchw.shape
    if b != 1:
        raise ValueError("Batch size 1 only")
    if h != w:
        raise ValueError(f"Expected square input, got {h}x{w}")

    x = x_bchw.to(device)
    padded = pad_to_multiple(x, window_size)
    _, _, hp, wp = padded.shape

    out = torch.empty((b, NUM_CLASSES, hp, wp), device=device, dtype=padded.dtype)
    for i in range(0, hp, window_size):
        for j in range(0, wp, window_size):
            win = padded[:, :, i : i + window_size, j : j + window_size]
            logits = model(win)
            out[:, :, i : i + window_size, j : j + window_size] = logits
    return out[:, :, :h, :w]


def predict_image_quilt(
    rgb: npt.NDArray[np.uint8],
    model: torch.nn.Module,
    device: torch.device,
    magnification: str,
    *,
    tile_size: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[npt.NDArray[np.float32], dict]:
    """
    Tile the input image into patches of `tile_size` (default = model input size),
    run GigaTIME per patch, stitch outputs to full-resolution probabilities.
    Returns probs as float32 (23, H, W) in [0, 1].
    """
    if magnification not in MAG_INPUT_SIZE:
        raise ValueError(f"Unknown magnification: {magnification}")

    input_hw = MAG_INPUT_SIZE[magnification]
    ts = tile_size or input_hw
    if ts <= 0:
        raise ValueError("tile_size must be positive")

    transform = build_val_transform(input_hw)
    h, w = rgb.shape[0], rgb.shape[1]

    nh = math.ceil(h / ts)
    nw = math.ceil(w / ts)
    total = nh * nw

    out = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    done = 0
    for yi in range(nh):
        for xi in range(nw):
            y0 = yi * ts
            x0 = xi * ts
            y1 = min(y0 + ts, h)
            x1 = min(x0 + ts, w)

            patch = np.zeros((ts, ts, 3), dtype=np.uint8)
            ph, pw = y1 - y0, x1 - x0
            patch[:ph, :pw, :] = rgb[y0:y1, x0:x1, :]

            x = preprocess_patch(patch, transform).to(dtype=torch.float32)
            logits = do_inference_windows(x, model, device, window_size=WINDOW)
            probs = torch.sigmoid(logits)[0].float().cpu().numpy()

            pr = probs[:, :ph, :pw]
            out[:, y0:y1, x0:x1] += pr
            counts[y0:y1, x0:x1] += 1.0

            done += 1
            if progress_cb is not None:
                progress_cb(done, total)

    counts = np.maximum(counts, 1.0)
    out = out / counts[np.newaxis, :, :]

    meta = {
        "input_hw": input_hw,
        "tile_size": ts,
        "grid": [nh, nw],
        "image_hw": [h, w],
    }
    return out, meta


def write_export_tifs(
    probs23_hw: npt.NDArray[np.float32],
    out_dir: Path,
    *,
    stem: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx, safe in EXPORT_CHANNELS:
        name = CHANNEL_NAMES_23[idx]
        fp = out_dir / f"{stem}__{safe}__{name}.tif"
        tifffile.imwrite(fp, probs23_hw[idx].astype(np.float32), photometric="minisblack", compression=None)
        written.append(fp)
    return written


def zip_exports(tif_paths: list[Path], zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in tif_paths:
            zf.write(p, arcname=p.name)
    return zip_path


def dapi_preview_u8(probs: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Channel 0 == DAPI probability map -> uint8 for UI."""
    dapi = probs[0]
    x = np.clip(dapi, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)
