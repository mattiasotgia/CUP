"""
cup.plotters.plotter_profile
=============================
Profile plot: bin x-variable, compute median or mean of y-variable per bin.

Triggered when ``plot_cfg.profile == True`` and ``len(products) == 2``.
"""

from __future__ import annotations

from cup.plotters import register_plotter, BasePlotter, FigureContext
from cup.plotters._base_helpers import (
    apply_filters, describe_filters,
    draw_profile,
    add_exp_label, add_filter_text, add_unmerged_warning,
)


@register_plotter("profile", priority=1)
class PlotterProfile(BasePlotter):
    """Profile plot (median/mean ± band) of y vs. binned x."""

    def can_handle(self, plot_cfg, analysis_cfg) -> bool:
        products = plot_cfg.product
        if not isinstance(products, list):
            products = [products]
        return len(products) == 2 and getattr(plot_cfg, "profile", False)

    # ------------------------------------------------------------------
    def make(self, ctx: FigureContext) -> None:
        ax = ctx.ax[0]
        x_product, y_product = ctx.products
        x_binning = ctx.binnings[0]
        x_label, y_label = ctx.labels
        plot_cfg = ctx.plot_cfg
        analysis_cfg = ctx.analysis_cfg
        global_cfg = ctx.global_cfg

        stat = getattr(plot_cfg, "profile_stat", "median")
        show_band = getattr(plot_cfg, "profile_band", True)

        x_label_axis = f"{x_label} ({x_binning.unit})" if x_binning.unit else x_label

        for dname, dinfo in ctx.dfs.items():
            data = apply_filters(dinfo["data"], plot_cfg.filter)
            data = apply_filters(data, analysis_cfg.filter)

            draw_profile(
                ax=ax,
                df=data,
                x_product=x_product,
                y_product=y_product,
                x_binning_cfg=x_binning,
                label=dinfo["label"],
                style_name=dinfo["style"],
                styles=ctx.styles,
                stat=stat,
                # show_band=show_band,
            )

        if getattr(plot_cfg, "yscale", None):
            ax.set_yscale(plot_cfg.yscale)
        if x_binning.scale and x_binning.scale_ax:
            ax.set_xscale(x_binning.scale)
        if getattr(plot_cfg, "grid", False):
            ax.grid(True)

        band_legend = "IQR (25–75%)" if stat == "median" else "SEM"
        ax.set_xlabel(x_label_axis)
        # ax.set_ylabel(f"{stat.capitalize()} $\\pm$ {band_legend}, {y_label.lower()}")
        ax.set_ylabel(f"{stat.capitalize()}, {y_label.lower()}")
        ax.legend(title=analysis_cfg.label if analysis_cfg.label else analysis_cfg.name)

        add_exp_label(ax, global_cfg, analysis_cfg)

        filter_text = describe_filters(plot_cfg.filter) + describe_filters(analysis_cfg.filter)
        add_filter_text(ax, filter_text, global_cfg.fontsize)

        if not analysis_cfg.merge_on:
            add_unmerged_warning(ax, global_cfg.fontsize, global_cfg)

        ctx.save()