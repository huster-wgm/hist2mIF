"""CLI for one-shot TIFF to PNG inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _default_output_tif_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_virtual_mIF.tif")


def _default_snapshot_png_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_virtual_mIF_snapshot.png")


def _build_parser() -> argparse.ArgumentParser:
    from hist2mif.services.inference import DEFAULT_ACTIVATION_THRESHOLD
    from hist2mif.services.jobs import (
        DEFAULT_CLI_BATCH_SIZE,
        DEFAULT_CLI_NUM_WORKERS,
        MASK_TIF_DOWNSAMPLE,
        SNAPSHOT_DOWNSAMPLE,
    )

    parser = argparse.ArgumentParser(
        prog="hist2mif-cli",
        description="Run Hist2mIF locally on a .tif/.tiff file and write TIFF + PNG outputs.",
    )
    parser.add_argument("--file", required=True, type=Path, help="Input .tif/.tiff H&E image")
    parser.add_argument(
        "--mag",
        choices=("10x", "20x"),
        required=True,
        help="Input magnification profile to use for tiling/preprocessing",
    )
    parser.add_argument(
        "--output-tif",
        type=Path,
        help=(
            f"21-channel binary-mask TIFF at 1/{MASK_TIF_DOWNSAMPLE} scale "
            f"(JPEG-compressed, one grayscale uint8 page per marker, cv2.INTER_NEAREST "
            f"per-tile resize; default: *_virtual_mIF.tif)"
        ),
    )
    parser.add_argument(
        "--snapshot-png",
        type=Path,
        help=(
            f"1/{SNAPSHOT_DOWNSAMPLE} scale PNG snapshot with marker legend "
            f"(cv2.INTER_NEAREST per-tile resize; "
            f"default: *_virtual_mIF_snapshot.png)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CLI_BATCH_SIZE,
        help=f"Number of tiles to infer per GPU batch (default: {DEFAULT_CLI_BATCH_SIZE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CLI_NUM_WORKERS,
        help=f"DataLoader worker processes for TIFF region loading (default: {DEFAULT_CLI_NUM_WORKERS})",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable DataLoader pin_memory (enabled by default)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ACTIVATION_THRESHOLD,
        help=(
            f"Per-pixel sigmoid activation threshold used to binarize each marker "
            f"before max-blending into the composite (paper default: {DEFAULT_ACTIVATION_THRESHOLD}). "
            f"Raise toward 1.0 for sparser, paper-figure-style activations; "
            f"pass --threshold 0 to blend continuous probabilities (debug only)."
        ),
    )
    return parser


def _validate_args(
    input_path: Path,
    output_tif_path: Path,
    snapshot_png_path: Path,
    batch_size: int,
    workers: int,
    threshold: float | None,
) -> None:
    if not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Only .tif / .tiff inputs are supported")
    if output_tif_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Output TIFF path must end with .tif or .tiff")
    if snapshot_png_path.suffix.lower() != ".png":
        raise ValueError("Snapshot PNG path must end with .png")
    if output_tif_path == input_path:
        raise ValueError("Output TIFF path must not overwrite the input file")
    if snapshot_png_path == input_path:
        raise ValueError("Snapshot PNG path must not overwrite the input file")
    if output_tif_path == snapshot_png_path:
        raise ValueError("Output TIFF and snapshot PNG paths must be different")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if workers < 0:
        raise ValueError("Workers must be non-negative")
    if threshold is not None and not (0.0 <= threshold < 1.0):
        raise ValueError("Threshold must be in [0, 1)")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = args.file.expanduser().resolve()
    output_tif_path = (
        args.output_tif.expanduser().resolve() if args.output_tif else _default_output_tif_path(input_path)
    )
    snapshot_png_path = (
        args.snapshot_png.expanduser().resolve()
        if args.snapshot_png
        else _default_snapshot_png_path(input_path)
    )

    threshold: float | None = None if args.threshold == 0 else float(args.threshold)

    try:
        _validate_args(
            input_path,
            output_tif_path,
            snapshot_png_path,
            args.batch_size,
            args.workers,
            args.threshold,
        )
    except ValueError as exc:
        parser.error(str(exc))

    from hist2mif.services.jobs import run_inference_to_tif_and_snapshot

    def progress(done: int, total: int) -> None:
        print(f"\rProcessing tiles: {done}/{total}", end="", flush=True)

    try:
        result = run_inference_to_tif_and_snapshot(
            input_path,
            output_tif_path,
            snapshot_png_path,
            args.mag,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=not args.no_pin_memory,
            threshold=threshold,
            progress_cb=progress,
        )
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"Wrote {result['output_tif_path']}")
    print(f"Wrote {result['snapshot_png_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
