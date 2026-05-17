#! /bin/bash
set -euo pipefail

# Skip files this CLI itself produced (BMCL_2_virtual_mIF.tif etc.) so we don't
# recursively re-infer on our own mask TIFF outputs.
for file in "$(pwd)"/../tiff/*.tif; do
    case "$file" in
        *_virtual_mIF.tif|*_virtual_mIF_*.tif) continue ;;
    esac
    uv run hist2mif-cli --file "$file" --mag 20x
done