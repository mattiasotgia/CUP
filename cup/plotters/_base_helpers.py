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
import matplotlib.pyplot as plt
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

def add_exp_label(ax: plt.Axes, global_cfg, analysis_cfg):
    ax.set_title(f'{global_cfg.project_name}\n{global_cfg.project_label}', loc="left", fontsize=global_cfg.fontsize, color='gray')

    ax.text(
        1.05, 1, analysis_cfg.analysis_supplementaltext,
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=global_cfg.fontsize * 0.5,
        rotation=90,
        color='gray',
    )


def add_filter_text(ax, filter_text: List[str], fontsize: float):
    if filter_text:
        ax.set_title(
            "\n".join(filter_text),
            loc='right',
            fontsize=fontsize * 0.85,
        )


def add_unmerged_warning(ax, fontsize: float, global_cfg):
    ax.set_title(
        f'{global_cfg.project_name}\n{global_cfg.project_label} (not only common events!)',
        loc='left',
        fontsize=global_cfg.fontsize,
        color='red',
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
    show_band: bool = None,
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
    # color = resolve_style(style_name, styles).get("color", None)

    # Extract style configurations comprehensively
    style = resolve_style(style_name, styles)

    draw_style = style.get("histtype", "errorbar")

    color = style.get("color", None)
    marker = style.get("marker", "o")
    linestyle = style.get("linestyle", "-")
    linewidth = style.get("linewidth", 1.5)
    alpha = style.get("alpha", 1.0)
    fill_alpha = style.get("fill_alpha", 0.20)

    # Backward compatibility handler for the legacy `show_band` boolean
    if show_band is True and draw_style == "errorbar":
        draw_style = "band"

    # --- Style 1: Shaded Band ---
    if draw_style == "band":
        # Draw the central line
        line, = ax.plot(centers[valid], central[valid], ls=linestyle, lw=linewidth, color=color, label=label)
        actual_color = line.get_color()  # Capture auto-cycle color if color is None
        
        # Fill the uncertainty area (using "mid" step matching your legacy look)
        ax.fill_between(
            centers[valid], 
            lo[valid], 
            hi[valid],
            step="mid", 
            alpha=fill_alpha, 
            color=actual_color, 
            linewidth=0
        )
        # Optional: Add faint boundary lines to the band
        ax.step(centers[valid], lo[valid], where="mid", lw=0.8, ls="--", color=actual_color, alpha=0.4)
        ax.step(centers[valid], hi[valid], where="mid", lw=0.8, ls="--", color=actual_color, alpha=0.4)

    # --- Style 2 & 3: Line / Step Profile with Vertical Error Lines ---
    elif draw_style in ["line", "step"]:
        if draw_style == "step":
            line, = ax.step(centers[valid], central[valid], where="mid", ls=linestyle, lw=linewidth, color=color, label=label)
        else:
            line, = ax.plot(centers[valid], central[valid], ls=linestyle, lw=linewidth, color=color, label=label)
            
        actual_color = line.get_color()
        # Draw vertical error bars mapping the uncertainty bounds
        ax.vlines(
            centers[valid],
            lo[valid],
            hi[valid],
            colors=actual_color,
            lw=linewidth,
            alpha=alpha
        )

    # --- Style 4: Classic Errorbar (Default) ---
    else:  # "errorbar"
        ax.errorbar(
            centers[valid], 
            central[valid], 
            yerr=[(central - lo)[valid], (hi - central)[valid]],
            xerr=widths[valid],
            ls="", 
            marker=marker, 
            markersize=style.get("markersize", 5), 
            linewidth=linewidth,
            color=color, 
            alpha=alpha,
            label=label
        )
