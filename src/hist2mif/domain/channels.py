"""GigaTIME channel order (model output indices 0..22).

See prov-gigatime/GigaTIME issue #8 / scripts/db_test.py common_channel_list.
TRITC and Cy5 are background channels; the UI composites the remaining 21 markers
into one RGB overlay (see `hist2mif.services.inference.composite_virtual_mif_rgb_u8`).
"""

from __future__ import annotations

# Full model output order (23 channels)
CHANNEL_NAMES_23: list[str] = [
    "DAPI",
    "TRITC",  # background
    "Cy5",  # background
    "PD-1",
    "CD14",
    "CD4",
    "T-bet",
    "CD34",
    "CD68",
    "CD16",
    "CD11c",
    "CD138",
    "CD20",
    "CD3",
    "CD8",
    "PD-L1",
    "CK",
    "Ki67",
    "Tryptase",
    "Actin-D",
    "Caspase3-D",
    "PHH3-B",
    "Transgelin",
]

BACKGROUND_INDICES = {1, 2}


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


# 21 exported channels: (index, safe_filename)
EXPORT_CHANNELS: list[tuple[int, str]] = [
    (i, _safe_name(CHANNEL_NAMES_23[i]))
    for i in range(len(CHANNEL_NAMES_23))
    if i not in BACKGROUND_INDICES
]
