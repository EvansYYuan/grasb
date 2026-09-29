#!/usr/bin/env python
"""Render transparent encoder-side GRN pushforward equation assets."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[3]
OUTDIR = PROJECT / "manuscript/figures/components"
CORAL = "#F06455"


def render(tex: str, filename: str, fontsize: float) -> None:
    fig = plt.figure(figsize=(3.0, 0.75))
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, tex, ha="center", va="center",
             fontsize=fontsize, color=CORAL)
    common = dict(bbox_inches="tight", pad_inches=0.035, transparent=True)
    base = OUTDIR / filename
    fig.savefig(base.with_suffix(".png"), dpi=600, **common)
    fig.savefig(base.with_suffix(".svg"), **common)
    fig.savefig(base.with_suffix(".pdf"), **common)
    plt.close(fig)


def main() -> None:
    mpl.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
    })
    OUTDIR.mkdir(parents=True, exist_ok=True)
    render(r"$J_E\,v_{\mathrm{GRN}}$", "encoder_grn_pushforward_tex", 30)
    render(
        r"$J_E(x)\,v_{\mathrm{GRN}}^{x}(x,t)$",
        "encoder_grn_pushforward_explicit_tex",
        27,
    )


if __name__ == "__main__":
    main()
