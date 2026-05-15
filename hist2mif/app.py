"""Gradio web UI for Hist2mIF (GigaTIME)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import tifffile
from PIL import Image

from hist2mif.channels import CHANNEL_NAMES_23, EXPORT_CHANNELS
from hist2mif.inference import (
    dapi_preview_u8,
    downsample_if_needed,
    get_device,
    load_he_image,
    load_model,
    predict_image_quilt,
    write_export_tifs,
    zip_exports,
)

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("HIST2MIF_DATA", "data")).resolve()
JOBS_DIR = DATA_ROOT / "jobs"
MAX_SIDE = int(os.environ.get("HIST2MIF_MAX_SIDE", "8192"))

_model: object | None = None
_model_device: object | None = None
_model_lock = threading.Lock()

EXPORT_LABELS: list[str] = [CHANNEL_NAMES_23[i] for i, _ in EXPORT_CHANNELS]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relative(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(DATA_ROOT.resolve()))
    except ValueError:
        return str(p)


def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


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
    if not JOBS_DIR.is_dir():
        return []
    dirs = [p for p in JOBS_DIR.iterdir() if p.is_dir() and (p / "meta.json").is_file()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in dirs]


def _format_progress(done: int, total: int) -> str:
    if total <= 0:
        return "0/0"
    return f"{done}/{total}"


def _format_time_ago(meta: dict) -> str:
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


def build_queue_rows_and_choices() -> tuple[list[list[str | None]], list[str]]:
    rows: list[list[str | None]] = []
    choices: list[str] = []
    for jid in list_job_ids():
        m = read_meta(jid) or {}
        fname = str(m.get("filename") or "-")
        mag = str(m.get("mag") or "-")
        status = str(m.get("status") or "-")
        done = int(m.get("progress_done") or 0)
        total = int(m.get("progress_total") or 0)

        if status == "running" and total > 0:
            prog = _format_progress(done, total)
        elif status == "complete" and total > 0:
            prog = _format_progress(done, total)
        else:
            prog = "-"

        preview = m.get("preview_png")
        if preview:
            preview_path = str((DATA_ROOT / str(preview)).resolve())
        else:
            preview_path = None

        rows.append([preview_path, fname, mag, status, prog, _format_time_ago(m)])
        choices.append(f"{jid[:8]}… | {fname}")

    return rows, choices


def refresh_queue(selected_label: str | None = None):
    rows, choices = build_queue_rows_and_choices()
    if not choices:
        return [], gr.update(choices=["(no tasks yet)"], value=None)

    if selected_label and selected_label in choices:
        value = selected_label
    else:
        value = choices[0]
    return rows, gr.update(choices=choices, value=value)


def _parse_selected_job(label: str | None) -> str | None:
    if not label or label == "(no tasks yet)":
        return None
    prefix8 = label.split("…", 1)[0].strip()
    for jid in list_job_ids():
        if jid.startswith(prefix8):
            return jid
    return None


def _get_or_load_model():
    global _model, _model_device
    with _model_lock:
        if _model is None:
            device = get_device()
            logger.info("Loading GigaTIME on %s", device)
            _model_device = device
            _model = load_model(device)
        return _model, _model_device


def run_inference(upload: str | None, mag: str, progress: gr.Progress = gr.Progress()):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    if not upload:
        raise gr.Error("Please upload a .tif file")

    src_path = Path(upload)
    if src_path.suffix.lower() not in {".tif", ".tiff"}:
        raise gr.Error("Please upload a .tif / .tiff file")

    job_id = str(uuid.uuid4())
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)

    stem = src_path.stem
    suffix = src_path.suffix.lower()
    saved_input = d / f"input{suffix}"
    shutil.copyfile(src_path, saved_input)

    filename = stem + suffix

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
        preview_png=None,
        zip_rel=None,
        input_rel=_relative(saved_input),
    )

    err_msg = ""
    try:
        rgb = load_he_image(saved_input)
        rgb, _scale = downsample_if_needed(rgb, MAX_SIDE)

        def cb(done: int, total: int) -> None:
            progress(done / max(total, 1), desc="Running inference (tiles)")
            update_meta(job_id, progress_done=done, progress_total=total, status="running")

        model, device = _get_or_load_model()
        probs, qmeta = predict_image_quilt(rgb, model, device, mag, progress_cb=cb)

        export_dir = d / "exports"
        tifs = write_export_tifs(probs, export_dir, stem=stem)
        zip_path = zip_exports(tifs, d / f"{stem}_virtual_mIF.zip")

        prev_u8 = dapi_preview_u8(probs)
        preview_path = d / "preview_dapi.png"
        Image.fromarray(prev_u8).save(preview_path)

        total = int(qmeta["grid"][0] * qmeta["grid"][1])
        update_meta(
            job_id,
            status="complete",
            progress_done=total,
            progress_total=total,
            preview_png=_relative(preview_path),
            zip_rel=_relative(zip_path),
            quilt_meta=qmeta,
        )

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        update_meta(job_id, status="failed", error=str(e))
        err_msg = f"**Error:** `{e}`"

    selected_label = f"{job_id[:8]}… | {filename}"
    rows, dd_update = refresh_queue(selected_label)
    return rows, dd_update, gr.update(value=err_msg)


def on_select_job(label: str | None, channel: str):
    jid = _parse_selected_job(label)
    if not jid:
        return None, gr.update(), gr.update(value="Select a task.")

    m = read_meta(jid) or {}
    exports = job_dir(jid) / "exports"
    stem = Path(str(m.get("filename", "input.tif"))).stem

    if m.get("status") == "failed":
        return None, gr.update(), gr.update(value=f"**Failed:** {m.get('error','unknown error')}")

    # DAPI preview PNG is fast to load
    if channel == "DAPI":
        p = job_dir(jid) / "preview_dapi.png"
        if p.is_file():
            img = np.array(Image.open(p))
            return img, gr.update(), gr.update(value="")

    for idx, safe in EXPORT_CHANNELS:
        name = CHANNEL_NAMES_23[idx]
        if name != channel:
            continue
        fp = exports / f"{stem}__{safe}__{name}.tif"
        if fp.is_file():
            arr = tifffile.imread(fp)
            a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)
            # If exports look like 0..255 uints, normalize defensively
            if float(a.max()) > 1.5:
                a = a / 255.0
            u8 = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
            return u8, gr.update(), gr.update(value="")

    return None, gr.update(), gr.update(value="Preview not available yet.")


def download_zip(label: str | None):
    jid = _parse_selected_job(label)
    if not jid:
        return None
    m = read_meta(jid) or {}
    if m.get("status") != "complete":
        return None
    zr = m.get("zip_rel")
    if not zr:
        return None
    return str(DATA_ROOT / zr)


def delete_job(label: str | None):
    jid = _parse_selected_job(label)
    if jid:
        shutil.rmtree(job_dir(jid), ignore_errors=True)
    rows, dd_update = refresh_queue(None)
    return rows, dd_update, None


def build_app() -> gr.Blocks:
    purple = gr.themes.Soft(primary_hue="purple", secondary_hue="purple")
    css = """
    .hist-title { letter-spacing: 0.08em; font-weight: 700; }
    .hist-sub { opacity: 0.75; }
    """

    with gr.Blocks(title="Hist2mIF", theme=purple, css=css) as demo:
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    <div class="hist-title">Hist2mIF</div>
                    <div class="hist-sub">H&E to multiplexed immunofluorescence (virtual mIF via GigaTIME)</div>
                    """
                )
            with gr.Column(scale=1, min_width=120):
                gr.Markdown("<div style='text-align:right;opacity:0.7'>Gradio theme toggle is on the top-right (☾ / ☀)</div>")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("##### UPLOAD SLIDE")
                upload = gr.File(label="Upload", file_types=[".tif", ".tiff"], type="filepath")
                mag = gr.Radio(choices=["10x", "20x"], value="20x", label="Magnification")
                run_btn = gr.Button("Run Inference", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("##### PREVIEW")
                channel = gr.Dropdown(
                    choices=EXPORT_LABELS,
                    value="DAPI",
                    label="Channel (virtual mIF)",
                )
                preview = gr.Image(label="Prediction preview", type="numpy")

        gr.Markdown("##### TASK QUEUE")
        queue = gr.Dataframe(
            headers=["PREVIEW", "FILE", "MAG", "STATUS", "PROGRESS", "TIME"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )
        selected = gr.Dropdown(label="Selected task", interactive=True)
        with gr.Row():
            dl = gr.File(label="Download all channels (.zip of 21× .tif)", interactive=False)
            download_btn = gr.Button("Download")
            refresh_btn = gr.Button("Refresh queue")
            delete_btn = gr.Button("Delete task", variant="stop")

        err = gr.Markdown("")

        run_btn.click(
            fn=run_inference,
            inputs=[upload, mag],
            outputs=[queue, selected, err],
        )
        refresh_btn.click(fn=lambda: refresh_queue(None), outputs=[queue, selected])
        selected.change(fn=on_select_job, inputs=[selected, channel], outputs=[preview, dl, err])
        channel.change(fn=on_select_job, inputs=[selected, channel], outputs=[preview, dl, err])
        download_btn.click(fn=download_zip, inputs=[selected], outputs=[dl])

        delete_btn.click(
            fn=delete_job,
            inputs=[selected],
            outputs=[queue, selected, preview],
        )

        demo.load(fn=lambda: refresh_queue(None), outputs=[queue, selected])

    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("HIST2MIF_HOST", "0.0.0.0")
    port = int(os.environ.get("HIST2MIF_PORT", "7860"))
    share = os.environ.get("HIST2MIF_SHARE", "").lower() in {"1", "true", "yes"}

    app = build_app()
    app.queue()
    app.launch(server_name=host, server_port=port, share=share, allowed_paths=[str(DATA_ROOT)])


if __name__ == "__main__":
    main()
