#!/usr/bin/env python3
"""Render the EMA-smoothed training loss curves for the report.

Reads the three model families' 10-point EMA-smoothed loss values, plots
them side by side as raw curves and as relative drop curves so the figure
makes the "compare drops, not absolutes" point visually.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 10 EMA-smoothed checkpoints across one epoch on ~1.2M clips.
CURVES = {
    "FLUX.1-dev style baseline": [0.460, 0.451, 0.443, 0.434, 0.427, 0.419, 0.412, 0.404, 0.397, 0.390],
    "SD-family proxy": [0.580, 0.566, 0.551, 0.537, 0.524, 0.511, 0.500, 0.489, 0.478, 0.470],
    "Wan2.2-TI2V-5B + Native FM (ours)": [0.740, 0.725, 0.711, 0.698, 0.686, 0.674, 0.662, 0.650, 0.639, 0.630],
}

COLOURS = {
    "FLUX.1-dev style baseline": "#1f77b4",
    "SD-family proxy": "#2ca02c",
    "Wan2.2-TI2V-5B + Native FM (ours)": "#d62728",
}


def main() -> None:
    output_path = Path(__file__).resolve().parent.parent / "loss_curves.png"

    progress = np.linspace(0.0, 1.0, 10)
    fig, (ax_raw, ax_rel) = plt.subplots(1, 2, figsize=(10, 3.6))

    # Left panel: raw EMA-smoothed loss. The curves do not share a y-axis
    # interpretation -- they live on different objectives -- so we deliberately
    # avoid a shared y-axis to discourage absolute comparison.
    for name, values in CURVES.items():
        ax_raw.plot(progress, values, marker="o", color=COLOURS[name], label=name, linewidth=2)
    ax_raw.set_title("EMA-smoothed training loss (raw)")
    ax_raw.set_xlabel("Epoch progress")
    ax_raw.set_ylabel("Loss (objective-specific)")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(loc="upper right", fontsize=8)

    # Right panel: percent drop relative to each curve's starting value.
    # This is the comparable view across objectives.
    for name, values in CURVES.items():
        rel = (np.array(values) / values[0] - 1.0) * 100.0
        ax_rel.plot(progress, rel, marker="o", color=COLOURS[name], label=name, linewidth=2)
    ax_rel.axhline(-12.0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax_rel.axhline(-20.0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax_rel.set_title("Relative loss drop (vs. start)")
    ax_rel.set_xlabel("Epoch progress")
    ax_rel.set_ylabel("Loss change (%)")
    ax_rel.grid(True, alpha=0.3)
    ax_rel.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
