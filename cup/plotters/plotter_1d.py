"""
cup.plotters.plotter_1d
=======================
1D overlaid histogram plot, with optional ratio sub-panels.

Handles any ``PlotConfig`` with a single product (``len(products) == 1``).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from cup.plotters import register_plotter, BasePlotter, FigureContext
from cup.plotters._base_helpers import (
    apply_filters, describe_filters,
    draw_hist1d, draw_ratio,
    add_exp_label, add_filter_text, add_unmerged_warning,
)


@register_plotter("1d", priority=0)
class Plotter1D(BasePlotter):
    """1D overlaid histograms with optional ratio panels."""

    def can_handle(self, plot_cfg, analysis_cfg) -> bool:
        products = plot_cfg.product
        if not isinstance(products, list):
            products = [products]
        return (
            len(products) == 1
            and not getattr(plot_cfg, "profile", False)
            and not getattr(plot_cfg, "efficiency", False)
        )

    # ------------------------------------------------------------------
    def build_figure(self, plot_cfg, analysis_cfg, global_cfg):
        n_ratios = len(plot_cfg.ratio) if plot_cfg.ratio else 0
        w, h = analysis_cfg.figsize
        total_h = h + n_ratios * global_cfg.ratio_height
        height_ratios = [h] + [global_cfg.ratio_height] * n_ratios

        fig, axes = plt.subplots(
            nrows=1 + n_ratios,
            sharex=True,
            figsize=(w, total_h),
            gridspec_kw={"height_ratios": height_ratios},
        )
        return fig, np.atleast_1d(axes)

    # ------------------------------------------------------------------
    def make(self, ctx: FigureContext) -> None:
        ax = ctx.ax
        p = ctx.products[0]
        binning = ctx.binnings[0]
        label_axis = ctx.labels[0]
        plot_cfg = ctx.plot_cfg
        analysis_cfg = ctx.analysis_cfg
        global_cfg = ctx.global_cfg

        # Y-axis label
        if binning.unit:
            ylabel = (
                f"{plot_cfg.ylabel if not analysis_cfg.density else f'Normalised {plot_cfg.ylabel.casefold()}'}"
                f" / {binning.create(binning.scale).widths[0]:.2f} {binning.unit}"
            )
        else:
            ylabel = (
                f"{plot_cfg.ylabel if not analysis_cfg.density else f'Normalised {plot_cfg.ylabel.casefold()}'}"
                f" / {binning.create(binning.scale).widths[0]:.2f}"
            )

        # Draw each dataset
        for dname, dinfo in ctx.dfs.items():
            data = apply_filters(dinfo["data"], plot_cfg.filter)
            data = apply_filters(data, analysis_cfg.filter)

            draw_hist1d(
                ax=ax[0],
                df=data,
                product=p,
                binning_cfg=binning,
                label=dinfo["label"],
                showmedian=plot_cfg.showmedian,
                style_name=dinfo["style"],
                styles=ctx.styles,
                density=analysis_cfg.density,
            )

        # Axes formatting
        if plot_cfg.yscale:
            ax[0].set_yscale(plot_cfg.yscale)
        if binning.scale and binning.scale_ax:
            ax[-1].set_xscale(binning.scale)
        if binning.integer:
            ax[-1].tick_params(axis="x", which="minor", bottom=False, top=False)
        if plot_cfg.grid:
            ax[0].grid(True)

        ax[0].set_xlabel("")
        ax[-1].set_xlabel(label_axis)
        ax[0].set_ylabel(ylabel)
        ax[0].legend(title=analysis_cfg.label if analysis_cfg.label else analysis_cfg.name)

        add_exp_label(ax[0], global_cfg, analysis_cfg)

        filter_text = describe_filters(plot_cfg.filter) + describe_filters(analysis_cfg.filter)
        add_filter_text(ax[0], filter_text, global_cfg.fontsize)

        if not analysis_cfg.merge_on:
            add_unmerged_warning(ax[0], global_cfg.fontsize, global_cfg)

        # Ratio panels
        if plot_cfg.ratio:
            for i, r in enumerate(plot_cfg.ratio, start=1):
                name_a, name_b = r.compare
                data_a = apply_filters(ctx.dfs[name_a]["data"], plot_cfg.filter)
                data_a = apply_filters(data_a, analysis_cfg.filter)
                data_b = apply_filters(ctx.dfs[name_b]["data"], plot_cfg.filter)
                data_b = apply_filters(data_b, analysis_cfg.filter)

                draw_ratio(
                    ax=ax[i],
                    a=data_a, b=data_b,
                    product=p,
                    binning_cfg=binning,
                    ylabel=r.ylabel,
                    comparison=r.comparison,
                    style=r.style,
                    color=r.color,
                    alpha=r.alpha,
                    styles=ctx.styles,
                )

        ctx.save()