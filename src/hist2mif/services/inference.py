"""GigaTIME inference helpers (patch quilt + tiled forward)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

# Avoid noisy update-check warnings in CLI/API logs.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

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

from hist2mif.domain.channels import EXPORT_CHANNELS, export_channel_colors_u8
from hist2mif.models.gigatime import gigatime

NUM_CLASSES = 23
WINDOW = 256
DEFAULT_INPUT_HW = WINDOW

# Per-pixel binary-activation threshold used by the official GigaTIME
# `scripts/gigatime_testing.ipynb` (``pred = (probs > 0.5).float()``).
# Continuous sigmoid outputs from H&E are over-confident and saturate any
# blend to white; thresholding first yields the paper's dark-background +
# per-marker color look.
DEFAULT_ACTIVATION_THRESHOLD = 0.5

# GigaTIME model forward operates on 256x256 windows. For whole-slide CLI
# inference, read those 256x256 regions directly from TIFF instead of resizing
# larger tiles and splitting them again.
MAG_INPUT_SIZE: dict[str, int] = {
    "20x": DEFAULT_INPUT_HW,
    "10x": DEFAULT_INPUT_HW,
}

_REPO_ID = "prov-gigatime/GigaTIME"

# Deployment policy: GPU server ships PyTorch wheels linked against CUDA >= 12.8 (cu128+).
MIN_TORCH_CUDA_MAJOR = 12
MIN_TORCH_CUDA_MINOR = 8

_PALETTE_21: npt.NDArray[np.float32] | None = None


def _parse_torch_cuda_version(version: str | None) -> tuple[int, int] | None:
    if not version:
        return None
    parts = version.strip().split(".")
    if not parts:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return major, minor
    except ValueError:
        return None


def assert_cuda_runtime_ok() -> torch.device:
    """Require NVIDIA CUDA per server policy (torch.version.cuda >= 12.8)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Hist2mIF requires an NVIDIA GPU with CUDA enabled (torch.cuda.is_available() is False)."
        )
    ver = _parse_torch_cuda_version(torch.version.cuda)
    if ver is None:
        raise RuntimeError(
            "Hist2mIF requires a CUDA-enabled PyTorch wheel (torch.version.cuda is empty). "
            "Install from https://download.pytorch.org/whl/cu128"
        )
    if ver < (MIN_TORCH_CUDA_MAJOR, MIN_TORCH_CUDA_MINOR):
        raise RuntimeError(
            f"Hist2mIF requires PyTorch built with CUDA >= {MIN_TORCH_CUDA_MAJOR}.{MIN_TORCH_CUDA_MINOR} "
            f"(torch.version.cuda={torch.version.cuda!r}). "
            "Reinstall torch/torchvision from the cu128 wheel index."
        )
    return torch.device("cuda")


def get_device() -> torch.device:
    return assert_cuda_runtime_ok()


def get_export_palette() -> npt.NDArray[np.float32]:
    """One RGB color per exported channel (same order as EXPORT_CHANNELS).

    Colors are sourced from the GigaTIME paper legend (see
    `hist2mif.domain.channels.CHANNEL_COLORS_RGB`) so the composite snapshot
    matches the reference figure rather than a synthetic HSV spread.
    """
    global _PALETTE_21
    if _PALETTE_21 is None:
        colors_u8 = np.asarray(export_channel_colors_u8(), dtype=np.float32)
        if colors_u8.shape != (len(EXPORT_CHANNELS), 3):
            raise RuntimeError(
                f"Export palette shape mismatch: got {colors_u8.shape}, "
                f"expected ({len(EXPORT_CHANNELS)}, 3)"
            )
        _PALETTE_21 = colors_u8 / 255.0
    return _PALETTE_21


def composite_virtual_mif_rgb_u8(
    probs23_hw: npt.NDArray[np.float32],
    *,
    rescale: bool = False,
    threshold: float | None = DEFAULT_ACTIVATION_THRESHOLD,
) -> npt.NDArray[np.uint8]:
    """Blend exported marker probabilities into one RGB image (H, W, 3) uint8.

    Follows the paper's binary-activation interpretation
    (`scripts/gigatime_testing.ipynb`): a pixel is "active" for channel c iff
    ``sigmoid(logits[c]) > threshold``. Each active pixel is painted with that
    channel's color from `CHANNEL_COLORS_RGB`, and channels are combined via a
    per-pixel max across (mask * color). Pixels with no active marker stay
    black, which gives the dark background + colored markers look of the paper
    Figure 1A/2D / 3H virtual mIF panels.

    Pass ``threshold=None`` to skip binarization and blend continuous
    probabilities directly (use only for sanity checks; the result will
    typically wash out because the model is over-confident on H&E).

    Background channels TRITC/Cy5 are excluded via EXPORT_CHANNELS.
    """
    if probs23_hw.shape[0] != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} channels, got {probs23_hw.shape[0]}")
    idxs = [i for i, _ in EXPORT_CHANNELS]
    sel = probs23_hw[idxs, :, :].astype(np.float32, copy=False)  # (C, H, W)
    if threshold is not None:
        sel = (sel > float(threshold)).astype(np.float32)
    pal = get_export_palette()  # (C, 3) in [0, 1]

    h, w = sel.shape[1], sel.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(sel.shape[0]):
        contrib = sel[c, :, :, None] * pal[c, None, None, :]
        np.maximum(rgb, contrib, out=rgb)

    mx = float(rgb.max())
    if rescale and mx > 1e-6:
        rgb = rgb / mx
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


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


def load_he_image(path: Path, *, max_side: int | None = None) -> npt.NDArray[np.uint8]:
    """Load a TIFF/SVS-style raster as RGB uint8 (H, W, 3).

    If a pyramidal TIFF is provided, decode the smallest level that still covers
    max_side so very large whole-slide images are not read at full resolution.
    """
    arr = _read_tiff_level(path, max_side=max_side)
    rgb = _to_rgb_uint8(arr)
    return rgb


def get_he_image_shape(path: Path) -> tuple[int, int]:
    """Return the full-resolution H, W shape without decoding image pixels."""
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"No image series found in {path}")
        hw = _shape_hw(tif.series[0].levels[0].shape)
        if hw is None:
            raise ValueError(f"Unsupported TIFF shape: {tif.series[0].levels[0].shape}")
        return hw


def read_he_region(path: Path, y0: int, y1: int, x0: int, x1: int) -> npt.NDArray[np.uint8]:
    """Decode only the requested full-resolution region as RGB uint8."""
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"No image series found in {path}")
        shape = tif.series[0].levels[0].shape

    if len(shape) == 2:
        selection = (slice(y0, y1), slice(x0, x1))
    elif len(shape) == 3 and shape[-1] in (1, 3, 4):
        selection = (slice(y0, y1), slice(x0, x1), slice(None))
    elif len(shape) == 3 and shape[0] in (1, 3, 4):
        selection = (slice(None), slice(y0, y1), slice(x0, x1))
    else:
        raise ValueError(f"Unsupported TIFF shape: {shape}")

    arr = tifffile.imread(path, selection=selection)
    return _to_rgb_uint8(arr)


def _read_tiff_level(path: Path, *, max_side: int | None) -> npt.NDArray[np.floating | np.integer]:
    if max_side is None or max_side <= 0:
        return tifffile.imread(path)

    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"No image series found in {path}")
        series = tif.series[0]
        levels = list(getattr(series, "levels", None) or [series])
        level = _select_pyramid_level(levels, max_side)
        return level.asarray()


def _select_pyramid_level(levels: list, max_side: int):
    shaped_levels = [(level, _shape_hw(level.shape)) for level in levels]
    covering = [
        (level, hw)
        for level, hw in shaped_levels
        if hw is not None and max(hw) >= max_side
    ]
    if covering:
        return min(covering, key=lambda item: max(item[1]))[0]

    valid = [(level, hw) for level, hw in shaped_levels if hw is not None]
    if valid:
        return max(valid, key=lambda item: max(item[1]))[0]
    return levels[0]


def _shape_hw(shape: tuple[int, ...]) -> tuple[int, int] | None:
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            return shape[0], shape[1]
        if shape[0] in (1, 3, 4):
            return shape[1], shape[2]
    return None


def _to_rgb_uint8(arr: npt.NDArray[np.floating | np.integer]) -> npt.NDArray[np.uint8]:
    if arr.ndim == 2:
        x = arr.astype(np.float32)
        x = _scale_to_01(x)
        u8 = (x * 255.0).clip(0, 255).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=-1)

    if arr.ndim == 3:
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
    """Run non-overlapping windows as one batched model forward."""
    b, _c, h, w = x_bchw.shape
    if h != w:
        raise ValueError(f"Expected square input, got {h}x{w}")

    x = x_bchw.to(device)
    padded = pad_to_multiple(x, window_size)
    _, c, hp, wp = padded.shape
    nh = hp // window_size
    nw = wp // window_size

    windows = (
        padded.unfold(2, window_size, window_size)
        .unfold(3, window_size, window_size)
        .permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .view(b * nh * nw, c, window_size, window_size)
    )
    logits = model(windows)
    out = (
        logits.view(b, nh, nw, NUM_CLASSES, window_size, window_size)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(b, NUM_CLASSES, hp, wp)
    )
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
        "input_h": input_hw,
        "input_w": input_hw,
        "input_hw": input_hw,
        "tile_size": ts,
        "window_size": WINDOW,
        "grid": [nh, nw],
        "image_hw": [h, w],
    }
    return out, meta
