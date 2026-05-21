"""
cup.core.manager
================
``PlotManager`` — the main entry point for running an analysis.

Responsibilities:

1. Load datasets (with optional merge-on-common-keys logic).
2. For each ``PlotConfig``, find the registered plotter that can handle it.
3. Build the figure context and delegate to the plotter's ``make()`` method.

The manager itself contains **no drawing code** — all rendering lives in
the ``cup.plotters`` modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import uproot

from cup.core.parser import Config, AnalysisConfig, DatasetConfig, PlotConfig
from cup.plotters import FigureContext, get_plotter, list_plotters

hep.style.use("DUNE")


class PlotManager:
    """
    Orchestrates dataset loading, figure creation, and plot dispatch.

    Parameters
    ----------
    config:
        Parsed ``Config`` object (from ``Config.load("analysis.yaml")``).
    """

    def __init__(self, config: Config):
        self.config = config
        self.outdir = config.config.outdir
        self.outdir.mkdir(parents=True, exist_ok=True)

        plt.rcParams["font.size"] = config.config.labelfontsize
        plt.rcParams["axes.titlesize"] = config.config.labelfontsize
        plt.rcParams["legend.fontsize"] = config.config.fontsize
        plt.rcParams["axes.ymargin"] = 0.1

    # =========================================================================
    # Dataset loading
    # =========================================================================

    def load_dataset(self, dataset_cfg: DatasetConfig, analysis_cfg: AnalysisConfig) -> pd.DataFrame:
        """
        Load one dataset.  Override this method to support custom data sources
        (e.g. parquet, HDF5, numpy) without changing any other code.
        """
        where = self.config.config.dataset_path.format(
            analysis=analysis_cfg.name,
            dataset=dataset_cfg.name,
        )
        print(f"  Loading: {where}")
        return uproot.open(self.config.config.file)[where].arrays(library="pd")

    def load_all_datasets(
        self,
        analysis_cfg: AnalysisConfig,
        merge_on=None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load all datasets for an analysis, optionally intersecting on common keys.

        Returns
        -------
        dict of ``{dataset_name: {"data": df, "data_raw": df, "label": str, "style": str}}``
        """
        merge_on = merge_on if merge_on is not None else analysis_cfg.merge_on
        merge_cols = ([merge_on] if isinstance(merge_on, str) else list(merge_on)) if merge_on else None

        common_keys = None
        if merge_cols:
            common_keys = self.load_dataset(analysis_cfg.dataset[0], analysis_cfg)[merge_cols].drop_duplicates()
            for ds in analysis_cfg.dataset[1:]:
                common_keys = pd.merge(
                    common_keys,
                    self.load_dataset(ds, analysis_cfg)[merge_cols].drop_duplicates(),
                    on=merge_cols,
                    how="inner",
                )

        out: Dict[str, Dict[str, Any]] = {}
        for ds in analysis_cfg.dataset:
            df_raw = self.load_dataset(ds, analysis_cfg)
            df = df_raw.merge(common_keys, on=merge_cols, how="inner") if merge_cols else df_raw
            out[ds.name] = {
                "data_raw": df_raw,
                "data": df,
                "label": ds.label,
                "style": ds.style,
            }
        return out

    # =========================================================================
    # Main run loop
    # =========================================================================

    def run(self):
        """Run all analyses defined in the config and produce output plots."""
        for name, analysis_cfg in self.config.analysis.items():
            print(f"\n→ Analysis: {name}")
            dfs = self.load_all_datasets(analysis_cfg)
            for plot_cfg in analysis_cfg.plot:
                self._dispatch(name, dfs, analysis_cfg, plot_cfg)

    # =========================================================================
    # Dispatcher
    # =========================================================================

    def _dispatch(
        self,
        analysis_name: str,
        dfs: Dict[str, Any],
        analysis_cfg: AnalysisConfig,
        plot_cfg: PlotConfig,
    ):
        """Find the appropriate plotter and invoke it."""

        # Normalise products / binnings / labels to lists
        products = plot_cfg.product if isinstance(plot_cfg.product, list) else [plot_cfg.product]
        binnings = plot_cfg.binning if isinstance(plot_cfg.binning, list) else [plot_cfg.binning]
        labels   = plot_cfg.label   if isinstance(plot_cfg.label,   list) else [plot_cfg.label]

        if binnings[0].unit:
            labels[0] = f"{labels[0]} ({binnings[0].unit})"

        plotter = get_plotter(plot_cfg, analysis_cfg)
        if plotter is None:
            raise NotImplementedError(
                f"No registered plotter can handle this plot config.\n"
                f"  products={products}, profile={getattr(plot_cfg, 'profile', False)}, "
                f"efficiency={getattr(plot_cfg, 'efficiency', False)}\n"
                f"  Registered plotters: {list_plotters()}"
            )

        fig, ax = plotter.build_figure(plot_cfg, analysis_cfg, self.config.config)

        outpath = self._build_outpath(analysis_name, products, binnings, plot_cfg)

        ctx = FigureContext(
            fig=fig,
            ax=ax,
            dfs=dfs,
            analysis_cfg=analysis_cfg,
            plot_cfg=plot_cfg,
            global_cfg=self.config.config,
            styles=self.config.styles,
            products=products,
            binnings=binnings,
            labels=labels,
            outpath=outpath,
        )

        plotter.make(ctx)
        print(f"  Saved: {outpath}")

    # =========================================================================
    # Output path builder
    # =========================================================================

    def _build_outpath(self, analysis_name: str, products: List[str],
                       binnings, plot_cfg: PlotConfig) -> Path:
        analysis_cfg = self.config.analysis[analysis_name]
        merged = (
            f"mergedOn_{'_'.join(analysis_cfg.merge_on)}"
            if analysis_cfg.merge_on
            else "unmerged"
        )
        if getattr(plot_cfg, "profile", False):
            tag = f"PROFILE_{'_vs_'.join(products)}"
        elif getattr(plot_cfg, "efficiency", False):
            tag = f"EFF_{products[0]}"
        else:
            tag = products[0]

        fname = (
            f"{merged}_{self.config.config.project}"
            f"_{analysis_name}_{tag}"
            f".{self.config.config.file_extension}"
        )
        return self.outdir / fname
    

# cup/core/manager.py

def merge_datasets(
    self,
    datasets: dict[str, pd.DataFrame],
    on: str | list[str],
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge any dict of DataFrames on common keys, adding a 'dataset' column
    to track origin.

    Parameters
    ----------
    datasets:
        {name: df} — can come from load_dataset() calls or anywhere else.
    on:
        Column(s) to intersect on.
    how:
        Passed to pd.merge — 'inner' keeps only common keys (default),
        'outer' keeps all.

    Returns
    -------
    A single merged DataFrame with a 'dataset' column.
    """
    merge_cols = [on] if isinstance(on, str) else list(on)

    # find common keys across all frames
    common = None
    for df in datasets.values():
        keys = df[merge_cols].drop_duplicates()
        common = keys if common is None else pd.merge(common, keys, on=merge_cols, how="inner")

    merged_frames = []
    for name, df in datasets.items():
        filtered = df.merge(common, on=merge_cols, how="inner")
        filtered = filtered.copy()
        filtered["dataset"] = name
        merged_frames.append(filtered)

    return pd.concat(merged_frames, ignore_index=True)


def get_dataset(
    self,
    analysis: str,
    dataset: str,
    filters=None,
) -> pd.DataFrame:
    """
    Load a single dataset by name, with optional filters applied.

    Parameters
    ----------
    analysis:
        Key in config.analysis (used for path formatting).
    dataset:
        Key in that analysis's dataset list.
    filters:
        List of FilterConfig, or None.

    Returns
    -------
    Filtered DataFrame.
    """
    from cup.core.filters import apply_filters  # already in _base_helpers

    analysis_cfg = self.config.analysis[analysis]
    ds_cfg = next((d for d in analysis_cfg.dataset if d.name == dataset), None)
    if ds_cfg is None:
        raise KeyError(f"Dataset '{dataset}' not found in analysis '{analysis}'. "
                       f"Available: {[d.name for d in analysis_cfg.dataset]}")

    df = self.load_dataset(ds_cfg, analysis_cfg)
    if filters:
        for f in filters:
            df = f.apply(df)
    return df

