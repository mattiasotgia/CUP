"""
cup.plotters
============
Plugin registry for plot types.

Each plot type is a class that inherits from ``BasePlotter`` and registers
itself with ``@register_plotter("name")``.

The ``PlotManager`` iterates over registered plotters in priority order and
delegates each ``PlotConfig`` to the first plotter whose ``can_handle``
returns ``True``.

Built-in plotters (imported at the bottom of this file so they self-register):

- ``Plotter1D``     — 1D overlaid histograms with optional ratio panels
- ``PlotterProfile``— profile (median/mean per bin) vs. a second variable
- ``PlotterEfficiency`` — TEfficiency-style plots comparing two datasets

Adding a third-party plotter
-----------------------------
In your own package::

    from cup.plotters import register_plotter, BasePlotter

    @register_plotter("my_special_plot", priority=5)
    class MyPlotter(BasePlotter):
        def can_handle(self, plot_cfg, analysis_cfg):
            return getattr(plot_cfg, "my_special_plot", False)

        def make(self, fig_ctx):
            ...  # draw into fig_ctx.ax[0], save via fig_ctx.save()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------

_PLOTTER_REGISTRY: List[Dict[str, Any]] = []   # [{"name", "cls", "priority"}, ...]


def register_plotter(name: str, priority: int = 0):
    """
    Class decorator that registers a ``BasePlotter`` subclass.

    Parameters
    ----------
    name:
        Unique identifier (used for logging / error messages).
    priority:
        Higher priority plotters are tried first.  Default built-ins use 0.
        Use a positive integer to override a built-in for a specific flag.
    """
    def decorator(cls: Type[BasePlotter]) -> Type[BasePlotter]:
        _PLOTTER_REGISTRY.append({"name": name, "cls": cls, "priority": priority})
        _PLOTTER_REGISTRY.sort(key=lambda e: e["priority"], reverse=True)
        return cls
    return decorator


def get_plotter(plot_cfg, analysis_cfg) -> Optional["BasePlotter"]:
    """Return the first registered plotter that can handle this config."""
    for entry in _PLOTTER_REGISTRY:
        instance = entry["cls"].__new__(entry["cls"])
        if instance.can_handle(plot_cfg, analysis_cfg):
            return instance
    return None


def list_plotters() -> List[str]:
    """Return names of all registered plotters in priority order."""
    return [e["name"] for e in _PLOTTER_REGISTRY]


# ---------------------------------------------------------------------------
# Figure context — passed into every plotter's make() method
# ---------------------------------------------------------------------------

@dataclass
class FigureContext:
    """
    Everything a plotter needs to produce and save a figure.

    Attributes
    ----------
    fig, ax:
        Pre-created Matplotlib figure and axes array (``np.atleast_1d``).
    dfs:
        ``{dataset_name: {"data": df, "data_raw": df, "label": str, "style": str}}``
    analysis_cfg, plot_cfg:
        Config objects for this analysis and plot.
    global_cfg:
        The ``GlobalConfig`` (for output path, labels, etc.)
    styles:
        Full styles dict from ``Config.styles``.
    products, binnings, labels:
        Normalised lists derived from ``plot_cfg``.
    outpath:
        ``Path`` where the figure should be saved.
    """
    fig: plt.Figure
    ax: np.ndarray          # shape (nrows,)
    dfs: Dict[str, Any]
    analysis_cfg: Any
    plot_cfg: Any
    global_cfg: Any
    styles: Dict[str, Any]
    products: List[str]
    binnings: List[Any]
    labels: List[str]
    outpath: Any            # pathlib.Path

    # Helpers ---------------------------------------------------------------

    def save(self):
        kw = {}
        if self.global_cfg.file_dpi:
            kw["dpi"] = self.global_cfg.file_dpi
        self.fig.tight_layout()
        self.fig.savefig(self.outpath, dpi=150, bbox_inches="tight", **kw)
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Base plotter
# ---------------------------------------------------------------------------

class BasePlotter(ABC):
    """
    Abstract base for all plot types.

    Subclasses must implement:

    - ``can_handle(plot_cfg, analysis_cfg) -> bool``
    - ``make(fig_ctx: FigureContext)``

    Subclasses may override ``build_figure(plot_cfg, analysis_cfg, global_cfg)``
    to customise the figure / axes layout (default: single axes).
    """

    @abstractmethod
    def can_handle(self, plot_cfg, analysis_cfg) -> bool:
        """Return True if this plotter should handle the given config."""

    @abstractmethod
    def make(self, ctx: FigureContext) -> None:
        """Draw into ctx.fig / ctx.ax and call ctx.save()."""

    def build_figure(self, plot_cfg, analysis_cfg, global_cfg):
        """
        Create and return (fig, ax_array).
        Override for multi-panel layouts (e.g. ratio panels).
        Default: single axes with analysis figsize.
        """
        fig, ax = plt.subplots(figsize=analysis_cfg.figsize)
        return fig, np.atleast_1d(ax)


# ---------------------------------------------------------------------------
# Lazy import of built-in plotters (triggers self-registration)
# ---------------------------------------------------------------------------

from cup.plotters.plotter_1d import Plotter1D              # noqa: E402, F401
from cup.plotters.plotter_profile import PlotterProfile    # noqa: E402, F401
from cup.plotters.plotter_efficiency import PlotterEfficiency  # noqa: E402, F401