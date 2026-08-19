"""Assemble six Figure 1 panels into a fixed composite PNG."""

import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DRAFTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "figure1"
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else DRAFTS / "Figure_1_concept.png"
GUTTER_X = 48
GUTTER_Y = 56

ROWS = (
    ("Figure_1A_passive.png", "Figure_1B_perturbation.png",
     "Figure_1C_shared.png"),
    ("cell4D_time_series.png", "cell4E_structure.png",
     "cell4F_response.png"),
)


def main():
    panels = [[Image.open(DRAFTS / name).convert("RGB") for name in row]
              for row in ROWS]
    widths = {panel.width for row in panels for panel in row}
    if len(widths) != 1:
        raise ValueError(f"Panel widths must match, got {sorted(widths)}")

    panel_width = widths.pop()
    row_heights = [max(panel.height for panel in row) for row in panels]
    canvas = Image.new(
        "RGB",
        (3 * panel_width + 2 * GUTTER_X, sum(row_heights) + GUTTER_Y),
        "white",
    )

    y = 0
    for row, row_height in zip(panels, row_heights):
        for column, panel in enumerate(row):
            # Top-align panels within each row.
            canvas.paste(panel, (column * (panel_width + GUTTER_X), y))
        y += row_height + GUTTER_Y

    canvas.save(OUTPUT, dpi=(400, 400), optimize=True)
    print(f"wrote {OUTPUT} ({canvas.width} x {canvas.height} px)")


if __name__ == "__main__":
    main()
