"""Per-query figures for the report: where the final system wins and loses (Phase 9).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.plot_per_query --config configs/plot_per_query_hi.yaml

Writes three PNG figures to ``output_dir`` from the saved per-query scores:

1. ``scatter``: per-query ``metric`` of the final system (y) against a
   baseline (x). Points above the diagonal are queries the final system improves.
2. ``components``: per-query ``metric`` of the hybrid's lexical run (x)
   against its dense run (y), each point coloured by whether the hybrid beats
   the dense run on that query (win / tie / loss). Shows that the hybrid gains
   where the lexical run is strong and dense is weak.
3. ``wtl``: one horizontal bar per comparison, split into wins, ties and losses
   of the final system, with the counts printed in the bars.

Colours follow the diverging convention (blue = win, red = loss, neutral grey =
tie), so the bar and scatter figures read the same way. Static images for the
report: no hover layer; every count shown is also printed on the figure.
"""

import argparse

import matplotlib
import matplotlib.lines
import matplotlib.patches
import matplotlib.ticker

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import pandas as pd

from src.eval.evaluate import read_per_query
from src.utils.config import load_config, resolve_path

# Reference palette (light mode), see the dataviz guidance: diverging blue <-> red, grey midpoint
SURFACE = "#fcfcfb"
TEXT, TEXT_2, MUTED = "#0b0b0b", "#52514e", "#a3a29d"
WIN, LOSS, TIE = "#2a78d6", "#e34948", "#d6d5d0"


def style_axes(ax) -> None:
    """Recessive axes: no top/right spines, light grid behind the data, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)


def legend_below(ax, labels: dict[str, str], title: str, y: float = -0.12) -> None:
    """Legend in win / tie / loss order, in one row under the axes (it never covers data)."""
    handles = [matplotlib.lines.Line2D([], [], marker="o", linestyle="", markersize=6,
                                       color=colour, label=labels[kind])
               for kind, colour in (("win", WIN), ("tie", TIE), ("loss", LOSS))]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, y), fontsize=9, handletextpad=0.3, columnspacing=1.2,
              title=title, title_fontsize=9)


PLURAL = {"win": "wins", "tie": "ties", "loss": "losses"}


def outcome(a: pd.Series, b: pd.Series) -> pd.Series:
    """'win' / 'tie' / 'loss' of per-query scores a against b."""
    diff = a - b.loc[a.index]
    return diff.map(lambda d: "win" if d > 0 else ("loss" if d < 0 else "tie"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    split, metric = cfg["split"], cfg["metric"]
    out_dir = resolve_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def per_query(name: str) -> pd.Series:
        return read_per_query(resolve_path(cfg["per_query_dir"]) / f"{name}_{split}.csv")[metric]

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

    # 1. Final system vs a baseline, one point per query
    sc = cfg["scatter"]
    y, x = per_query(sc["system"]), per_query(sc["baseline"])
    x = x.loc[y.index]
    res = outcome(y, x)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    style_axes(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", zorder=1)
    for kind, colour in (("tie", TIE), ("loss", LOSS), ("win", WIN)):
        sel = res == kind
        ax.scatter(x[sel], y[sel], s=30, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=2)
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect="equal",
           xlabel=f"{sc['baseline_label']}  ({metric})", ylabel=f"{sc['system_label']}  ({metric})")
    ax.set_title(sc["title"], fontsize=11, loc="left", color=TEXT)
    legend_below(ax, {k: f"{PLURAL[k]} ({int((res == k).sum())})" for k in PLURAL},
                 title="hybrid A against the baseline, per query")
    fig.tight_layout()
    fig.savefig(out_dir / sc["file"], dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. The hybrid's components against each other, coloured by the hybrid's outcome vs dense
    co = cfg["components"]
    hybrid = per_query(co["hybrid"])
    lex, dense = per_query(co["lexical"]).loc[hybrid.index], per_query(co["dense"]).loc[hybrid.index]
    res = outcome(hybrid, dense)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    style_axes(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", zorder=1)
    for kind, colour in (("tie", TIE), ("loss", LOSS), ("win", WIN)):
        sel = res == kind
        ax.scatter(lex[sel], dense[sel], s=30, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=2)
    ax.text(0.97, 0.06, "lexical better", ha="right", color=TEXT_2, fontsize=9)
    ax.text(0.03, 0.96, "dense better", ha="left", va="top", color=TEXT_2, fontsize=9)
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect="equal",
           xlabel=f"{co['lexical_label']}  ({metric})", ylabel=f"{co['dense_label']}  ({metric})")
    ax.set_title(co["title"], fontsize=11, loc="left", color=TEXT)
    legend_below(ax, {k: f"{PLURAL[k]} ({int((res == k).sum())})" for k in PLURAL},
                 title="hybrid A against the dense route, per query")
    fig.tight_layout()
    fig.savefig(out_dir / co["file"], dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Wins / ties / losses of the final system against several runs
    wt = cfg["wtl"]
    rows = []
    for comp in wt["comparisons"]:
        res = outcome(per_query(comp["system"]), per_query(comp["baseline"]))
        rows.append((comp["label"], *(int((res == k).sum()) for k in ("win", "tie", "loss"))))
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.2))
    style_axes(ax)
    ax.grid(False)
    for i, (label, *counts) in enumerate(reversed(rows)):
        total, left = sum(counts), 0.0
        for count, colour, text_colour in zip(counts, (WIN, TIE, LOSS), ("white", TEXT, "white"), strict=True):
            share = count / total
            ax.barh(i, share, left=left, height=0.62, color=colour, edgecolor=SURFACE, linewidth=2)
            if share >= 0.06:
                ax.text(left + share / 2, i, str(count), ha="center", va="center", fontsize=9, color=text_colour)
            left += share
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=9, color=TEXT)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(f"share of queries ({metric}, per query)")
    ax.set_title(wt["title"], fontsize=11, loc="left", color=TEXT)
    handles = [matplotlib.patches.Patch(color=c, label=k) for k, c in (("win", WIN), ("tie", TIE), ("loss", LOSS))]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / wt["file"], dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figures written to {cfg['output_dir']}: {sc['file']}, {co['file']}, {wt['file']}")


if __name__ == "__main__":
    main()
