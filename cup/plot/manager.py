import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import uproot

import matplotlib.pyplot as plt
import mplhep as hep

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from cup.core.parser import Config, DatasetConfig


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

from hist import Hist
import hist


class PlotManager:
    """
    Plot manager for 1D overlayed and 2D multi-panel histograms using hist.Hist.

    Features:
    ---------
    - Dataset loading hook (ROOT, parquet, etc.)
    - Style handling from TOML
    - 1D overlay plots using mplhep.histplot
    - 2D multi-panel comparisons using hist.plot2d_full
    - Density normalization
    - Filter application
    - merge_on hook if needed
    """

    def __init__(self, config):
        self.config: Config = config
        self.outdir = config.config.outdir
        self.outdir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # ---------------------- DATA LOADING -------------------------
    # ============================================================

    def load_dataset(self, dataset_config: DatasetConfig):
        """
        Hook for loading dataset (ROOT)
        Example:
            return uproot.open(self.config.config.file)[dataset_config.name].arrays(library='pd')
        """
        # raise NotImplementedError("Implement dataset loading logic here.")
        return uproot.open(self.config.config.file)[dataset_config.name].arrays(library='pd')

    def load_all_datasets(self, analysis_cfg):
        """
        Load all datasets for an analysis section.
        Returns:
            { dataset_name: { 'df': df, 'label': ..., 'style': ... } }
        """
        out = {}
        for dcfg in analysis_cfg.dataset:
            df = self.load_dataset(dcfg)
            out[dcfg.name] = {
                "df": df,
                "label": dcfg.label,
                "style": dcfg.style
            }
        return out

    # ============================================================
    # ------------------------ FILTERS ----------------------------
    # ============================================================

    def apply_filters(self, df, plot_cfg):
        """Apply zero, one, or multiple filters to dataframe."""
        flt = plot_cfg.filter
        if flt is None:
            return df

        filters = flt if isinstance(flt, list) else [flt]
        for f in filters:
            df = f.apply(df)
        return df

    # ============================================================
    # ------------------------ STYLES -----------------------------
    # ============================================================

    def resolve_style(self, style_name):
        """Return style dict (copied to avoid mutation)."""
        if style_name and style_name in self.config.styles:
            return self.config.styles[style_name].style_kw.copy()
        return {}

    # ============================================================
    # ---------------------- PLOT: 1D -----------------------------
    # ============================================================

    def plot_1d(self, ax, df, product, binning_cfg, label, style_name, density=False):
        """
        1D histogram using hist.Hist + mplhep.histplot.
        """
        x = df[product].dropna()

        # Use the axis directly.
        axis = binning_cfg.create(product)

        H = Hist(axis, storage=hist.storage.Weight())
        H.fill(**{product: x.values})

        # Density normalization
        if density:
            total = H.values().sum()
            if total > 0:
                H = H / total

        style = self.resolve_style(style_name)
        histtype = style.get("histtype", "step")  # do NOT pop

        hep.histplot(
            H,
            ax=ax,
            label=label,
            histtype=histtype,
            **style
        )

    # ============================================================
    # ---------------------- PLOT: 2D -----------------------------
    # ============================================================

    def plot_2d_single_panel(self, ax, df, products, binning_cfgs, label, density=False):
        """
        2D histogram using hist.Hist + hist.plot2d_full.
        """
        xname, yname = products

        x = df[xname].dropna()
        y = df[yname].dropna()

        ax_x = binning_cfgs[0].create(xname)
        ax_y = binning_cfgs[1].create(yname)

        H = Hist(ax_x, ax_y, storage=hist.storage.Weight())
        H.fill(**{xname: x.values, yname: y.values})

        if density:
            total = H.values().sum()
            if total > 0:
                H = H / total

        pcm = H.plot2d_full(ax=ax)
        ax.set_title(label)
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        return pcm

    # ============================================================
    # ---------------------- MAIN ENTRY --------------------------
    # ============================================================

    def run(self):
        """Run all analyses and produce plots."""
        for name, analysis_cfg in self.config.analysis.items():
            print(f"→ Running analysis: {name}")
            dfs = self.load_all_datasets(analysis_cfg)
            for plot_cfg in analysis_cfg.plot:
                self._make_plot(name, dfs, analysis_cfg, plot_cfg)

    # ============================================================
    # ------------------- PLOT DISPATCHER -------------------------
    # ============================================================

    def _make_plot(self, analysis_name, dfs, analysis_cfg, plot_cfg):

        products = plot_cfg.product
        binning = plot_cfg.binning
        labels = plot_cfg.label

        # Normalize to list
        if not isinstance(products, list):
            products = [products]
        if not isinstance(binning, list):
            binning = [binning]
        if not isinstance(labels, list):
            labels = [labels]

        # ------------------------
        # 1D Overlay
        # ------------------------
        if len(products) == 1:

            fig, ax = plt.subplots(figsize=(8, 6))
            for dname, dinfo in dfs.items():
                df = self.apply_filters(dinfo["df"], plot_cfg)
                self.plot_1d(
                    ax=ax,
                    df=df,
                    product=products[0],
                    binning_cfg=binning[0],
                    label=dinfo["label"],
                    style_name=dinfo["style"],
                    density=analysis_cfg.density,
                )

            ax.set_xlabel(products[0])
            ax.set_ylabel("Density" if analysis_cfg.density else "Counts")
            ax.set_title(labels[0])
            ax.legend()

            out = self.outdir / f"{analysis_name}_{products[0]}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            return

        # ------------------------
        # 2D Multi-Panel
        # ------------------------
        if len(products) == 2:

            dataset_names = list(dfs.keys())
            n_datasets = len(dataset_names)

            ncols = 2
            nrows = int(np.ceil(n_datasets / ncols))

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(7 * ncols, 6 * nrows),
                squeeze=False
            )

            axes = axes.flatten()
            pcm_last = None

            for i, dname in enumerate(dataset_names):
                dinfo = dfs[dname]
                df = self.apply_filters(dinfo["df"], plot_cfg)
                ax = axes[i]

                pcm_last = self.plot_2d_single_panel(
                    ax=ax,
                    df=df,
                    products=products,
                    binning_cfgs=binning,
                    label=dinfo["label"],
                    density=analysis_cfg.density
                )

            # Remove unused axes
            for k in range(n_datasets, len(axes)):
                fig.delaxes(axes[k])

            # Shared colorbar
            fig.colorbar(pcm_last, ax=axes.tolist(), shrink=0.85)

            out = self.outdir / f"{analysis_name}_MULTIPANEL_{products[0]}_{products[1]}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            return

        raise NotImplementedError("Plotting more than 2 products is not implemented.")
