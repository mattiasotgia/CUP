'''
'''

from hist import Hist
import hist
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
import numpy as np
import pandas as pd
import uproot
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

from cup.core.parser import Config, DatasetConfig, AnalysisConfig, BinningConfig, PlotConfig, FilterConfig
import cup.core


hep.style.use('DUNE')

class PlotManager:
    '''
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
    '''

    def __init__(self, config: Config):
        self.config: Config = config
        self.outdir = config.config.outdir
        self.outdir.mkdir(parents=True, exist_ok=True)

        plt.rcParams['font.size'] = self.config.config.labelfontsize
        plt.rcParams['axes.titlesize'] = self.config.config.labelfontsize
        plt.rcParams['legend.fontsize'] = self.config.config.fontsize
        plt.rcParams['axes.ymargin'] = 0.1

    # ============================================================
    # ---------------------- DATA LOADING -------------------------
    # ============================================================

    def load_dataset(self, dataset_config: DatasetConfig) -> pd.DataFrame:
        '''
        Load a single dataset using the dataset configuration
        '''
        # raise NotImplementedError('Implement dataset loading logic here.')
        return uproot.open(self.config.config.file)[dataset_config.name].arrays(library='pd')

    def load_all_datasets(self, analysis_cfg: AnalysisConfig, merge_on=None):
        '''
        Load all datasets for an analysis section.
        If merge_on is provided, keep only rows whose merge_on value(s)
        are shared across ALL datasets.

        Returns:
            { dataset_name: { 'df': df, 'label': ..., 'style': ... } }
        '''
        out = {}

        merge_on = analysis_cfg.merge_on

        common_keys = self.load_dataset(analysis_cfg.dataset[0])[merge_on].drop_duplicates()
        for adf in analysis_cfg.dataset[1:]:
            common_keys = pd.merge(
                common_keys,
                self.load_dataset(adf)[merge_on].drop_duplicates(),
                on=merge_on,
                how='inner'
            )


        for adf in analysis_cfg.dataset:
            out[adf.name] = {
                'data_raw': self.load_dataset(adf),
                'data': self.load_dataset(adf).merge(common_keys, on=merge_on, how='inner'),
                'label': adf.label,
                'style': adf.style
            }

        return out

    # ============================================================
    # ------------------------ FILTERS ----------------------------
    # ============================================================

    def apply_filters(self, df: pd.DataFrame, filters: Optional[List[FilterConfig]]):
        '''Apply zero or more filters to dataframe.'''
        if not filters:
            return df

        for f in filters:
            df = f.apply(df)
        return df

    def describe_filters(self, filters: Optional[List[FilterConfig]]) -> List[str]:
        if not filters:
            return []
        return [f.describe() for f in filters if f.describe() is not None]

    # ============================================================
    # ------------------------ STYLES -----------------------------
    # ============================================================

    def resolve_style(self, style_name):
        '''Return style dict (copied to avoid mutation).
        Also provide 'default' style if not declared '''
        if style_name and style_name in self.config.styles:
            return self.config.styles[style_name].style_kw.copy()
        if style_name and style_name == 'default':
            return {'histtype': 'step', 'color': 'k', 'yerr': False, 'linewidth': 2}
        return {}
    
    # ============================================================
    # ------------------------ RATIOS -----------------------------
    # ============================================================

    def plot_ratio(self, ax, a: pd.DataFrame, b: pd.DataFrame, product: str, binning_cfg: BinningConfig, 
                   ylabel: str, comparison: str, style: str, color: str, alpha: float | None):
        
        Da = a[product].dropna()
        Db = b[product].dropna()
        
        axis = binning_cfg.create(product)

        Ha = Hist(axis, storage=hist.storage.Weight())
        Ha.fill(Da.values)
        Hb = Hist(axis, storage=hist.storage.Weight())
        Hb.fill(Db.values)

        cvalues, cerrorslo, cerrorshi = hep.get_comparison(Ha, Hb, comparison=comparison)
        cerrors = np.array([cerrorslo, cerrorshi])

        if comparison in ['ratio', 'split_ratio', 'efficiency']:
            if style is None:
                style = 'errorbar'
            ax.axhline(1, ls='--', color='k')
        else:
            # in ['pull', 'difference', 'relative_difference', 'asymmetry']
            if style is None:
                style = 'bar'
            ax.axhline(0, ls='--', color='k')

        if style == 'bar':
            alpha=alpha if alpha is not None else 0.5
            ax.bar(axis.centers, cvalues, width=axis.widths, color=color, alpha=alpha)
        elif style == 'errorbar':
            alpha=alpha if alpha is not None else 1
            ax.errorbar(axis.centers, cvalues, yerr=cerrors, color=color, markersize=5, marker='o', xerr=False, ls='')
        else:
            raise NotImplementedError(f'Allowed styles are "bar" and "errorbar", you entered {style}')
        
        ax.set_ylabel(ylabel if ylabel is not None else comparison.capitalize().replace('_', ' '))


    # ============================================================
    # ---------------------- PLOT: 1D -----------------------------
    # ============================================================

    def plot_1d(self, ax, df: pd.DataFrame, label: str, showmedian: str | None, style_name: str,
                product: str, product_name: str, binning_cfg: BinningConfig, density=False):
        '''
        1D histogram using hist.Hist + mplhep.histplot.
        '''
        x = df[product].dropna()

        # Use the axis directly.
        axis = binning_cfg.create(product)
        # if binning_cfg.unit:
        #     product_name = f'{product_name} ({binning_cfg.unit})'

        H = Hist(axis, storage=hist.storage.Weight())
        H.fill(x.values)

        # Density normalization
        if density:
            total = H.values().sum()
            if total > 0:
                H = H / total

        style = self.resolve_style(style_name)

        hep.histplot(
            H,
            ax=ax,
            label=label if not showmedian else f'{label} ({format(np.median(x.values), showmedian)}{'' if not binning_cfg.unit else f" {binning_cfg.unit}"})',
            flow=binning_cfg.flow,
            **style
        )

    # ============================================================
    # ---------------------- PLOT: 2D -----------------------------
    # ============================================================

    def plot_2d_single_panel(self, ax, df: pd.DataFrame, style_name: str,
                             products: Tuple[str, str], products_names: Tuple[str, str], 
                             binning_cfgs: Tuple[BinningConfig, BinningConfig], density=False):
        '''
        2D histogram using hist.Hist + hist.plot2d_full.
        '''

        raise NotImplementedError('This is a WIP')
        xname, yname = products
        xlabel, ylabel = products_names

        x = df[xname].dropna()
        y = df[yname].dropna()

        binx, biny = binning_cfgs

        
        ax_x = binx.create(binx.scale if binx.scale else 'linear')
        ax_y = biny.create(biny.scale if biny.scale else 'linear')

        H = Hist(ax_x, ax_y, storage=hist.storage.Weight())
        H.fill(**{xname: x.values, yname: y.values})

        if density:
            total = H.values().sum()
            if total > 0:
                H = H / total

        pcm = H.plot2d_full(ax=ax)
        # ax.set_title(label)  
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        return pcm

    # ============================================================
    # ---------------------- MAIN ENTRY --------------------------
    # ============================================================

    def run(self):
        '''Run all analyses and produce plots.'''
        for name, analysis_cfg in self.config.analysis.items():
            print(f'-> Running analysis: {name}')
            dfs = self.load_all_datasets(analysis_cfg)
            for plot_cfg in analysis_cfg.plot:
                self._make_plot(name, dfs, analysis_cfg, plot_cfg)

    # ============================================================
    # ------------------- PLOT DISPATCHER -------------------------
    # ============================================================

    def _make_plot(self, analysis_name: str, dfs: Dict[str, Any], 
                   analysis_cfg: AnalysisConfig, plot_cfg: PlotConfig):

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

            if binning[0].unit:
                labels[0] = f'{labels[0]} ({binning[0].unit})'

            if binning[0].unit:
                ylabel = f'{plot_cfg.ylabel if not analysis_cfg.density else f"Normalised {plot_cfg.ylabel.casefold()}"} / {binning[0].create(binning[0].scale).widths[0]:.2f} {binning[0].unit}'
            else:
                ylabel = f'{plot_cfg.ylabel if not analysis_cfg.density else f"Normalised {plot_cfg.ylabel.casefold()}"} / {binning[0].create(binning[0].scale).widths[0]:.2f}'

            nRatios = len(plot_cfg.ratio) if plot_cfg.ratio is not None else 0
            ratio_height = self.config.config.ratio_height   # inches
            width, height = analysis_cfg.figsize
            main_height = height
            total_height = main_height + nRatios * ratio_height
            height_ratios = [main_height] + [ratio_height] * nRatios

            fig, ax = plt.subplots(
                nrows=1 + nRatios,
                sharex=True,
                figsize=(width, total_height),
                gridspec_kw={"height_ratios": height_ratios},
            )
            ax = np.atleast_1d(ax)
            
            for dname, dinfo in dfs.items():
                data = self.apply_filters(dinfo['data'], plot_cfg.filter)
                data = self.apply_filters(data, analysis_cfg.filter)
                
                self.plot_1d(
                    ax=ax[0],
                    df=data,
                    product=products[0],
                    product_name=labels[0],
                    binning_cfg=binning[0],
                    label=dinfo['label'],
                    showmedian=plot_cfg.showmedian,
                    style_name=dinfo['style'],
                    density=analysis_cfg.density,
                )

            if plot_cfg.yscale:
                ax[0].set_yscale(plot_cfg.yscale)

            if binning[0].scale and binning[0].scale_ax:
                ax[0].set_xscale(binning[0].scale)

            ax[0].set_xlabel('')
            ax[-1].set_xlabel(labels[0])
            ax[0].set_ylabel(ylabel)
            ax[0].legend(title=analysis_cfg.name)

            if binning[0].integer:
                ax[-1].tick_params(axis='x', which='minor', bottom=False, top=False)

            if plot_cfg.grid:
                ax[0].grid(True)

            hep.label.exp_text(
                exp=self.config.config.project_name, 
                text=self.config.config.project_label,
                supp=analysis_cfg.analysis_supplementaltext,
                fontsize = self.config.config.fontsize,
                ax=ax[0]
            )

            filter_text = []

            filter_text += self.describe_filters(plot_cfg.filter)
            filter_text += self.describe_filters(analysis_cfg.filter)

            if filter_text:
                ax[0].text(
                    1,
                    1.05,
                    '\n'.join(filter_text),
                    transform=ax[0].transAxes,
                    va='bottom',
                    ha='right',
                    fontsize=self.config.config.fontsize * 0.85,
                    # bbox=dict(
                    #     boxstyle='',
                    #     # facecolor='white',
                    #     edgecolor='none',
                    #     alpha=0.85
                    # )
                )

            if plot_cfg.ratio is not None:
                for i, r in enumerate(plot_cfg.ratio, start=1):

                    name_a, name_b = r.compare
                    data_a = dfs[name_a]['data']
                    data_a = self.apply_filters(data_a, plot_cfg.filter)
                    data_a = self.apply_filters(data_a, analysis_cfg.filter)
                    data_b = dfs[name_b]['data']
                    data_b = self.apply_filters(data_b, plot_cfg.filter)
                    data_b = self.apply_filters(data_b, analysis_cfg.filter)
                    

                    self.plot_ratio(
                        ax=ax[i], 
                        a=data_a,
                        b=data_b, 
                        product=products[0], 
                        binning_cfg=binning[0],
                        ylabel=r.ylabel,
                        comparison=r.comparison,
                        color=r.color,
                        alpha=r.alpha,
                        style=r.style
                    )

            out = self.outdir / f'{self.config.config.project}_{analysis_name}_{products[0]}.{self.config.config.file_extension}'
            fig.tight_layout()
            additional_kw = {}
            if self.config.config.file_dpi:
                additional_kw['dpi'] = self.config.config.file_dpi

            fig.tight_layout()
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return

        # ------------------------
        # 2D Multi-Panel
        # ------------------------
        if len(products) == 2:

            raise NotImplementedError('This is a WIP')
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
                df = self.apply_filters(dinfo['df'], plot_cfg)
                ax = axes[i]

                pcm_last = self.plot_2d_single_panel(
                    ax=ax,
                    df=df,
                    products=products,
                    binning_cfgs=binning,
                    label=dinfo['label'],
                    density=analysis_cfg.density
                )

            # Remove unused axes
            for k in range(n_datasets, len(axes)):
                fig.delaxes(axes[k])

            # Shared colorbar
            fig.colorbar(pcm_last, ax=axes.tolist(), shrink=0.85)

            out = self.outdir / f'{analysis_name}_MULTIPANEL_{products[0]}_{products[1]}.png'
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            return

        raise NotImplementedError('Plotting more than 2 products is not implemented.')
