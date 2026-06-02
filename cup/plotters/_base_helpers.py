"""
cup.plotters._base_helpers
==========================
Low-level drawing helpers shared across plotter classes.
These are *not* plotters themselves — they are imported by the concrete
plotter modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mplhep as hep
import numpy as np
import pandas as pd
from hist import Hist
import hist


# ---------------------------------------------------------------------------
# Style resolution
# ---------------------------------------------------------------------------

_DEFAULT_STYLE = {"histtype": "step", "color": "k", "yerr": False, "linewidth": 2}


def resolve_style(style_name: Optional[str], styles: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the style dict for *style_name*, or sensible defaults."""
    if style_name and style_name in styles:
        return styles[style_name].style_kw.copy()
    if style_name == "default" or style_name is None:
        return _DEFAULT_STYLE.copy()
    return {}


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    if not filters:
        return df
    for f in filters:
        df = f.apply(df)
    return df


def describe_filters(filters) -> List[str]:
    if not filters:
        return []
    return [f.describe() for f in filters if f.describe() is not None]


# ---------------------------------------------------------------------------
# hep label / annotation helpers
# ---------------------------------------------------------------------------

def add_exp_label(ax, global_cfg, analysis_cfg):
    hep.label.exp_text(
        exp=global_cfg.project_name,
        text=global_cfg.project_label,
        supp=analysis_cfg.analysis_supplementaltext,
        fontsize=global_cfg.fontsize,
        ax=ax,
    )


def add_filter_text(ax, filter_text: List[str], fontsize: float):
    if filter_text:
        ax.text(
            1, 1.05,
            "\n".join(filter_text),
            transform=ax.transAxes,
            va="bottom", ha="right",
            fontsize=fontsize * 0.85,
        )


def add_unmerged_warning(ax, fontsize: float):
    ax.text(
        0, 1.055,
        "All events (not only commons)",
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=fontsize * 0.85,
        color="red",
    )


# ---------------------------------------------------------------------------
# 1D histogram drawing
# ---------------------------------------------------------------------------

def draw_hist1d(
    ax,
    df: pd.DataFrame,
    product: str,
    binning_cfg,
    label: str,
    showmedian: Optional[str],
    style_name: Optional[str],
    styles: Dict[str, Any],
    density: bool = False,
):
    x = df[product].dropna()
    axis = binning_cfg.create(product)
    H = Hist(axis, storage=hist.storage.Weight())
    H.fill(x.values)

    if density:
        total = H.values().sum()
        if total > 0:
            H = H / total

    if showmedian:
        unit = f" {binning_cfg.unit}" if binning_cfg.unit else ""
        legend_label = f"{label} ({format(np.median(x.values), showmedian)}{unit})"
    else:
        legend_label = label

    style = resolve_style(style_name, styles)
    hep.histplot(H, ax=ax, label=legend_label, flow=binning_cfg.flow, **style)


# ---------------------------------------------------------------------------
# Ratio panel drawing
# ---------------------------------------------------------------------------

def draw_ratio(ax, a: pd.DataFrame, b: pd.DataFrame, product: str,
               binning_cfg, ylabel, comparison, style, color, alpha, styles):
    Da = a[product].dropna()
    Db = b[product].dropna()
    axis = binning_cfg.create(product)

    Ha = Hist(axis, storage=hist.storage.Weight())
    Ha.fill(Da.values)
    Hb = Hist(axis, storage=hist.storage.Weight())
    Hb.fill(Db.values)

    cvalues, cerrorslo, cerrorshi = hep.comp.get_comparison(Ha, Hb, comparison=comparison)
    cerrors = np.array([cerrorslo, cerrorshi])

    if comparison in ("ratio", "split_ratio", "efficiency"):
        style = style or "errorbar"
        ax.axhline(1, ls="--", color="k")
    else:
        style = style or "bar"
        ax.axhline(0, ls="--", color="k")

    if style == "bar":
        ax.bar(axis.centers, cvalues, width=axis.widths,
               color=color, alpha=alpha if alpha is not None else 0.5)
    elif style == "errorbar":
        ax.errorbar(axis.centers, cvalues, yerr=cerrors,
                    color=color, markersize=5, marker="o",
                    xerr=False, ls="",
                    alpha=alpha if alpha is not None else 1)
    else:
        raise NotImplementedError(f'Ratio style must be "bar" or "errorbar", got "{style}"')

    ax.set_ylabel(ylabel if ylabel is not None else comparison.capitalize().replace("_", " "))


# ---------------------------------------------------------------------------
# Profile drawing
# ---------------------------------------------------------------------------

def draw_profile(
    ax,
    df: pd.DataFrame,
    x_product: str,
    y_product: str,
    x_binning_cfg,
    label: str,
    style_name: Optional[str],
    styles: Dict[str, Any],
    stat: str = "median",
    show_band: bool = True,
):
    x = df[x_product].dropna()
    y = df[y_product].dropna()
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]

    axis = x_binning_cfg.create(x_product)
    edges = axis.edges
    centers = axis.centers
    widths = np.diff(edges) / 2

    bin_idx = np.digitize(x.values, edges)
    n_bins = len(centers)
    central = np.full(n_bins, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)

    for i in range(1, n_bins + 1):
        vals = y.values[bin_idx == i]
        if len(vals) == 0:
            continue
        if stat == "median":
            central[i - 1] = np.median(vals)
            lo[i - 1] = np.percentile(vals, 25)
            hi[i - 1] = np.percentile(vals, 75)
        elif stat == "mean":
            central[i - 1] = np.mean(vals)
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            lo[i - 1] = central[i - 1] - sem
            hi[i - 1] = central[i - 1] + sem
        else:
            raise ValueError(f"stat must be 'median' or 'mean', got '{stat}'")

    valid = ~np.isnan(central)
    color = resolve_style(style_name, styles).get("color", None)

    if show_band:
        ax.fill_between(centers[valid], lo[valid], hi[valid],
                        step="mid", alpha=0.20, color=color, linewidth=0)
        ax.step(centers[valid], lo[valid], where="mid", lw=0.8, ls="--", color=color, alpha=0.4)
        ax.step(centers[valid], hi[valid], where="mid", lw=0.8, ls="--", color=color, alpha=0.4)

    ax.errorbar(centers[valid], central[valid], xerr=widths[valid],
                ls="", marker="o", markersize=5, linewidth=1.8,
                color=color, label=label)
