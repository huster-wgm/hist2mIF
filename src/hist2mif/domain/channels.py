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


# Cleaner labels for the composite legend (strips the `-D` / `-B` model suffixes).
CHANNEL_DISPLAY_NAMES: dict[str, str] = {
    "Actin-D": "Actin",
    "Caspase3-D": "Caspase 3",
    "PHH3-B": "PHH3",
}


# Top-to-bottom order used when drawing the legend next to the composite snapshot.
# Mirrors the GigaTIME paper figure so users can cross-reference the published palette.
LEGEND_ORDER: list[str] = [
    "CD8",
    "PD-1",
    "Tryptase",
    "PHH3-B",
    "CD16",
    "CD14",
    "CD138",
    "Transgelin",
    "CD11c",
    "Actin-D",
    "CD20",
    "CD34",
    "Caspase3-D",
    "T-bet",
    "CK",
    "DAPI",
    "CD68",
    "CD3",
    "PD-L1",
    "Ki67",
    "CD4",
]


# Per-marker composite colors aligned to the GigaTIME paper legend (uint8 RGB).
# Keys match CHANNEL_NAMES_23 entries; only the 21 non-background markers are used.
CHANNEL_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "DAPI":       (240, 230, 130),
    "PD-1":       (140,  35,  35),
    "CD14":       (235, 225, 170),
    "CD4":        ( 40,  85, 220),
    "T-bet":      (125,  60, 185),
    "CD34":       (110, 165, 220),
    "CD68":       (255, 230,  30),
    "CD16":       (235, 160, 165),
    "CD11c":      (165, 205,  40),
    "CD138":      ( 20,  90,  90),
    "CD20":       (225,  35,  35),
    "CD3":        (225,  60, 135),
    "CD8":        (230,  70,  40),
    "PD-L1":      ( 20, 100,  50),
    "CK":         (215, 195, 155),
    "Ki67":       ( 60, 180,  60),
    "Tryptase":   (150, 140,  35),
    "Actin-D":    (175, 220, 230),
    "Caspase3-D": ( 30,  30,  85),
    "PHH3-B":     (210,  90, 165),
    "Transgelin": (230, 120,  30),
}


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


# 21 exported channels: (index, safe_filename)
EXPORT_CHANNELS: list[tuple[int, str]] = [
    (i, _safe_name(CHANNEL_NAMES_23[i]))
    for i in range(len(CHANNEL_NAMES_23))
    if i not in BACKGROUND_INDICES
]


def export_channel_colors_u8() -> list[tuple[int, int, int]]:
    """Return per-export-channel RGB colors (uint8), in EXPORT_CHANNELS order."""
    out: list[tuple[int, int, int]] = []
    for i, _safe in EXPORT_CHANNELS:
        name = CHANNEL_NAMES_23[i]
        color = CHANNEL_COLORS_RGB.get(name)
        if color is None:
            raise KeyError(f"Missing composite color for channel {name!r}")
        out.append(color)
    return out


def legend_entries() -> list[tuple[str, tuple[int, int, int]]]:
    """Return (display_name, RGB uint8) pairs in paper legend order."""
    entries: list[tuple[str, tuple[int, int, int]]] = []
    for name in LEGEND_ORDER:
        color = CHANNEL_COLORS_RGB.get(name)
        if color is None:
            raise KeyError(f"Missing composite color for channel {name!r}")
        display = CHANNEL_DISPLAY_NAMES.get(name, name)
        entries.append((display, color))
    return entries
