"""
cup.plotters.plotter_efficiency
================================
Efficiency plot: ratio of a numerator dataset (passing) to a denominator
dataset (total), drawn as Clopper–Pearson intervals.

Triggered when ``plot_cfg.efficiency == True``.

YAML configuration example
--------------------------
.. code-block:: yaml

    plots:
      - product: reco_energy
        label: "Reco energy (GeV)"
        binning:
          bins: 20
          limits: [0, 5]
          unit: "GeV"
        efficiency: true
        efficiency_numerator: "signal_passing"
        efficiency_denominator: "signal_all"

The two named datasets must both be present in the analysis ``datasets`` list.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from hist import Hist
import hist

from cup.plotters import register_plotter, BasePlotter, FigureContext
from cup.plotters._base_helpers import (
    apply_filters, describe_filters,
    resolve_style,
    add_exp_label, add_filter_text, add_unmerged_warning,
)


def _clopper_pearson(k: np.ndarray, n: np.ndarray, alpha: float = 0.317):
    """
    Return (efficiency, lo_err, hi_err) using the Clopper–Pearson interval.
    alpha=0.317 gives ~68% confidence (1-sigma equivalent).
    """
    from scipy.stats import beta as beta_dist  # soft dependency

    eff = np.where(n > 0, k / n, np.nan)
    lo = np.where(n > 0, eff - beta_dist.ppf(alpha / 2, k, n - k + 1), np.nan)
    hi = np.where(n > 0, beta_dist.ppf(1 - alpha / 2, k + 1, n - k) - eff, np.nan)
    lo = np.clip(lo, 0, None)
    hi = np.clip(hi, 0, None)
    return eff, lo, hi


@register_plotter("efficiency", priority=2)
class PlotterEfficiency(BasePlotter):
    """
    TEfficiency-style plot comparing passing / total across datasets.

    Each dataset is drawn independently.  Efficiency is the ratio of
    ``efficiency_numerator`` to ``efficiency_denominator`` datasets (both
    must be keys in ``dfs``), **or** a single dataset can carry a boolean
    ``passed`` column (future extension).
    """

    def can_handle(self, plot_cfg, analysis_cfg) -> bool:
        products = plot_cfg.product
        if not isinstance(products, list):
            products = [products]
        return len(products) == 1 and getattr(plot_cfg, "efficiency", False)

    # ------------------------------------------------------------------
    def make(self, ctx: FigureContext) -> None:
        ax = ctx.ax[0]
        p = ctx.products[0]
        binning = ctx.binnings[0]
        label_axis = ctx.labels[0]
        plot_cfg = ctx.plot_cfg
        analysis_cfg = ctx.analysis_cfg
        global_cfg = ctx.global_cfg

        num_names = getattr(plot_cfg, "efficiency_numerator", None)
        den_names = getattr(plot_cfg, "efficiency_denominator", None)

        if num_names is None or den_names is None:
            raise ValueError(
                "Efficiency plot requires 'efficiency_numerator' and "
                "'efficiency_denominator' to be set in the plot config."
            )

        # Normalise to lists so single-pair and multi-pair share one code path
        if isinstance(num_names, str):
            num_names = [num_names]
        if isinstance(den_names, str):
            den_names = [den_names]

        if len(num_names) != len(den_names):
            raise ValueError(
                f"efficiency_numerator and efficiency_denominator must have the "
                f"same length, got {len(num_names)} and {len(den_names)}."
            )

        axis = binning.create(p)

        def _fill(dataset_name) -> Hist:
            if dataset_name not in ctx.dfs:
                raise KeyError(
                    f"Dataset '{dataset_name}' not found. Available: {list(ctx.dfs)}"
                )
            data = apply_filters(ctx.dfs[dataset_name]["data"], plot_cfg.filter)
            data = apply_filters(data, analysis_cfg.filter)
            x = data[p].dropna()
            H = Hist(axis, storage=hist.storage.Weight())
            H.fill(x.values)
            return H

        for num_name, den_name in zip(num_names, den_names):
            H_num = _fill(num_name)
            H_den = _fill(den_name)

            k = H_num.values()
            n = H_den.values()
            eff, lo_err, hi_err = _clopper_pearson(k, n)

            num_label = ctx.dfs[num_name]["label"]
            den_label = ctx.dfs[den_name]["label"]

            # Use numerator dataset's style; fall back gracefully for extra pairs
            style = resolve_style(ctx.dfs[num_name]["style"], ctx.styles)
            color = style.get("color", None)   # None → matplotlib auto-cycles

            ax.errorbar(
                axis.centers,
                eff,
                yerr=np.array([lo_err, hi_err]),
                xerr=axis.widths / 2,
                fmt="o",
                markersize=5,
                linewidth=1.5,
                color=color,
                label=f"{num_label} / {den_label}",
            )

        ax.axhline(1.0, ls="--", color="grey", lw=0.8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Efficiency")
        ax.set_xlabel(label_axis)
        ax.legend(title=analysis_cfg.name)

        if binning.scale and binning.scale_ax:
            ax.set_xscale(binning.scale)
        if getattr(plot_cfg, "grid", False):
            ax.grid(True)

        add_exp_label(ax, global_cfg, analysis_cfg)

        filter_text = describe_filters(plot_cfg.filter) + describe_filters(analysis_cfg.filter)
        add_filter_text(ax, filter_text, global_cfg.fontsize)

        if not analysis_cfg.merge_on:
            add_unmerged_warning(ax, global_cfg.fontsize)

        ctx.save()