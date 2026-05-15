# Hist2mIF

Web UI for **H&E → virtual multiplex immunofluorescence (mIF)** using Microsoft’s **[GigaTIME](https://github.com/prov-gigatime/GigaTIME)** model (research use per upstream license).

## Features

- Upload `.tif` / `.tiff` H&E slides
- Choose **10× / 20×** (affects internal resize / tiling settings)
- Run tiled inference and show a **DAPI** preview by default (switch channel to browse all exported markers)
- Export **21** usable marker channels as `.tif` (Background channels **TRITC** + **Cy5** are omitted) plus a **`.zip`** bundle

## Requirements

- **Linux + NVIDIA GPU** is strongly recommended (CUDA). macOS Intel is not supported by recent PyTorch wheels; do dev/deploy on the GPU server.
- Python **3.11+**
- `uv` ([Astral](https://docs.astral.sh/uv/))
- Hugging Face **read token** with access to `prov-gigatime/GigaTIME` (**HF_TOKEN**)

## Quick start (GPU server)

```bash
mkdir -p /home/ubuntu/gohiroaki
cd /home/ubuntu/gohiroaki
git clone https://github.com/huster-wgm/hist2mIF.git
cd hist2mIF

uv sync

# NVIDIA GPU (strongly recommended): reinstall CUDA wheels matching your driver (example cu124)
uv pip uninstall -y torch torchvision
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

export HF_TOKEN="***"  # read token with access to prov-gigatime/GigaTIME
export HIST2MIF_DATA="/home/ubuntu/gohiroaki/hist2mIF/data"

uv run hist2mif
```

Then open `http://<server-ip>:7860`.

## Environment variables

See [.env.example](.env.example).

## Implementation notes

- Model I/O follows `scripts/gigatime_testing.ipynb` in GigaTIME: **ImageNet normalization** (`albumentations.augmentations.transforms.Normalize`), **256×256** sliding windows inside each **square** patch.
- Outputs are **sigmoid probabilities** in \[0, 1\] saved as **float32** single-channel TIFFs (plus a zip).

## License / use

The upstream GigaTIME release is **research-only** and **not for clinical use**. Review:

- [GigaTIME README](https://github.com/prov-gigatime/GigaTIME/blob/main/README.md)
- [GigaTIME LICENSE](https://github.com/prov-gigatime/GigaTIME/blob/main/LICENSE)

This repository’s UI code is provided as-is for integrating that upstream model.
