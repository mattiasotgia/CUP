"""
cup — Configurable Unified Plotter
===================================
A modular histogram / profile / efficiency plotting framework for HEP analyses.

Typical programmatic usage::

    from cup import PlotManager
    from cup.core.parser import Config

    cfg = Config.load("my_analysis.yaml")
    pm  = PlotManager(cfg)
    pm.run()

Extending with a new plot type::

    from cup.plotters import register_plotter, BasePlotter

    @register_plotter("my_plot_type")
    class MyPlotter(BasePlotter):
        def can_handle(self, plot_cfg, analysis_cfg):
            return getattr(plot_cfg, "my_plot_type", False)

        def make(self, fig_ctx):
            ...
"""

from cup.core.manager import PlotManager  # noqa: F401 — re-export for convenience

__all__ = ["PlotManager"]