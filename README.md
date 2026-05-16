# Hist2mIF

Web UI for **H&E → virtual multiplex immunofluorescence (mIF)** using Microsoft’s **[GigaTIME](https://github.com/prov-gigatime/GigaTIME)** model (research use per upstream license).

The interface is a **React + Vite** SPA (light, tech-forward styling) backed by a **FastAPI** service. A separate **Vue** bundle is not included—maintaining two frameworks would duplicate work; if you need Vue, we can port the same API contract.

## Features

- Upload `.tif` / `.tiff` H&E slides
- Choose **10× / 20×** (affects internal resize / tiling settings)
- Live task queue with thumbnails and tile-level progress
- **Single RGB composite** (21 markers, pseudo‑color blend) — download as PNG
- **CUDA ≥ 12.8** enforced at runtime via `torch.version.cuda`

## Repository layout

Standard **src layout** + separate frontend:

```
hist2mIF/
├── pyproject.toml
├── README.md
├── src/hist2mif/           # Python package (installable)
│   ├── main.py             # CLI → Uvicorn
│   ├── api/
│   │   ├── app.py          # FastAPI factory + SPA mount
│   │   └── routers/        # route handlers only
│   ├── core/config.py      # env-backed settings
│   ├── domain/channels.py  # marker metadata
│   ├── models/gigatime.py  # vendored GigaTIME U-Net
│   └── services/           # inference + job worker
├── frontend/               # React (Vite + Tailwind)
└── data/                   # runtime jobs (gitignored by default)
```

## Requirements

- **Linux + NVIDIA GPU**, PyTorch CUDA build **≥ 12.8**
- Python **3.11+**, **`uv`**, **`npm`** (for building the UI)
- Hugging Face **read token** for `prov-gigatime/GigaTIME` (**HF_TOKEN**)

## Quick start (development)

Terminal A — API (loads model on first job):

```bash
cd hist2mIF
uv sync
export HF_TOKEN="***"
export HIST2MIF_DATA="/path/to/data"
uv run hist2mif
# listens on 0.0.0.0:7860 by default
```

Local one-shot CLI:

```bash
cd hist2mIF
uv run hist2mif-cli --file /path/to/input.tif --mag 20x
# writes /path/to/input_virtual_mIF.tif at original resolution
# and /path/to/input_virtual_mIF_snapshot.png at 1/50 scale + paper legend
# reads 256x256 TIFF regions directly through a DataLoader
# and runs model inference with batch_size=128, workers=4, pin_memory=True

uv run hist2mif-cli --file /path/to/input.tiff --mag 10x \
  --output-tif /path/to/output.tif \
  --snapshot-png /path/to/output_snapshot.png \
  --batch-size 128 \
  --workers 4 \
  --threshold 0.5
```

The composite follows the paper's binary-activation rule
(`scripts/gigatime_testing.ipynb`, `pred = (probs > 0.5)`): each pixel is
"active" for a marker iff its sigmoid output exceeds `--threshold`, and
channels are combined via per-pixel max of (mask × paper color). Raise
`--threshold` toward 1.0 for the sparser dark-background look of Figure 1A /
2D; lower it for denser tissue overlays.

Terminal B — React UI (proxies `/api` → backend):

```bash
cd hist2mIF/frontend
npm install
npm run dev
# open http://127.0.0.1:5173
```

## Production (single port)

Build the SPA, then serve API + static files from FastAPI:

```bash
cd hist2mIF/frontend && npm run build
cd ..
uv run hist2mif
# visit http://<host>:7860/ — API remains under /api/*
```

Optional: `HIST2MIF_CORS` comma-separated origins if the UI is hosted on another origin.

## PyTorch / CUDA

`pyproject.toml` already pins **`torch` / `torchvision`** to PyTorch's official **cu128** wheel index ([uv ↔ PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)):

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch       = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
```

`uv sync` then pulls the proper CUDA 12.8 build; the wheel bundles cuDNN/NCCL/CUDA runtime and PyTorch loads them itself — no `LD_LIBRARY_PATH` tweaks, no manual `nvidia-*` reinstalls.

To switch CUDA majors (cu126 / cu130 / cpu / rocm): change the index URL + name in those two blocks and re-run `uv sync`.

## Environment variables

Copy `.env.example` to **`.env`** in the repository root. Variables are read automatically on startup (existing shell environment takes precedence over `.env`).

See [.env.example](.env.example).

## License / use

Upstream GigaTIME is **research-only** and **not for clinical use**. Review the [GigaTIME README](https://github.com/prov-gigatime/GigaTIME/blob/main/README.md) and [LICENSE](https://github.com/prov-gigatime/GigaTIME/blob/main/LICENSE).
