# cup — Configurable Unified Plotter

A modular histogram / profile / efficiency plotting framework for HEP analyses.

## Installation

```bash
pip install -e .
```

This installs the `cup` package and the `cup-plot` CLI entry point.

## Quick start

```bash
# Run all analyses
cup-plot run analysis.yaml

# Run only specific analyses
cup-plot run analysis.yaml -a Muons -a NuMuCC

# Override output directory
cup-plot run analysis.yaml --outdir /tmp/plots

# Dry-run: see what would be produced
cup-plot run analysis.yaml --dry-run

# Validate config without running
cup-plot validate analysis.yaml

# List registered plotters
cup-plot list-plotters
```

Or programmatically:

```python
from cup import PlotManager
from cup.core.parser import Config

cfg = Config.load("analysis.yaml")
pm  = PlotManager(cfg)
pm.run()
```

## Architecture

```
cup/
├── __init__.py            re-exports PlotManager
├── core/
│   ├── parser.py          YAML → Config dataclasses
│   ├── registry.py        filter + binning registries (decorator API)
│   ├── filters.py         built-in filter implementations
│   ├── binnings.py        built-in axis factories (linear, log)
│   └── manager.py         PlotManager — loads data, dispatches to plotters
├── plotters/
│   ├── __init__.py        BasePlotter, FigureContext, register_plotter()
│   ├── _base_helpers.py   shared drawing utilities (no drawing state)
│   ├── plotter_1d.py      1D overlaid histograms + ratio panels
│   ├── plotter_profile.py profile (median/mean per bin)
│   └── plotter_efficiency.py  efficiency (Clopper–Pearson)
└── cli/
    └── main.py            cup-plot CLI (Click)
```

### Adding a new plot type

1. Create `cup/plotters/plotter_mytype.py`:

```python
from cup.plotters import register_plotter, BasePlotter, FigureContext

@register_plotter("my_type", priority=3)   # higher priority = tried first
class MyTypePlotter(BasePlotter):

    def can_handle(self, plot_cfg, analysis_cfg) -> bool:
        products = plot_cfg.product if isinstance(plot_cfg.product, list) else [plot_cfg.product]
        return len(products) == 1 and getattr(plot_cfg, "my_type", False)

    def make(self, ctx: FigureContext) -> None:
        ax = ctx.ax[0]
        # ... draw into ax ...
        ctx.save()
```

2. Import it in `cup/plotters/__init__.py` (or in your own package's `__init__.py`).

3. Add `my_type: true` to the plot YAML config.

### Adding a new filter

```python
from cup.core.registry import register_filter

@register_filter("my_filter", describe=lambda col: f"{col} cleaned")
def my_filter(df, col: str) -> pd.DataFrame:
    return df[df[col].notna()]
```

### Adding a new binning scale

```python
from cup.core.registry import register_binning
import numpy as np, hist

@register_binning("sqrt")
def binning_sqrt(bins, limits, flow, name):
    lo, hi = sorted(limits)
    edges = np.linspace(np.sqrt(lo), np.sqrt(hi), bins) ** 2
    return hist.axis.Variable(edges, name=name, flow=flow)
```

## YAML configuration reference

See `analysis.yaml` for a fully annotated example.  Key sections:

| Section | Purpose |
|---------|---------|
| `global` | file paths, project labels, output settings |
| `styles.<name>` | `mplhep.histplot` keyword arguments |
| `analyses.<name>` | datasets, per-analysis filters, figsize, density |
| `analyses.<name>.plots[*]` | individual plot configs |

### Plot-type flags

| Flag | Effect |
|------|--------|
| *(none)* | 1D overlay histogram |
| `profile: true` | Profile plot (requires 2 products) |
| `efficiency: true` | Efficiency plot (requires `efficiency_numerator` + `efficiency_denominator`) |