"""
cup.cli.main
============
``cup-plot`` command-line interface.

Usage
-----
.. code-block:: bash

    # Run all analyses in a config file
    cup-plot run analysis.yaml

    # Run only specific analyses
    cup-plot run analysis.yaml --analysis Muons --analysis Protons

    # List registered plotters
    cup-plot list-plotters

    # Validate config without running
    cup-plot validate analysis.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Ensure built-in filters / binnings are registered before anything else
import cup.core.filters   # noqa: F401
import cup.core.binnings  # noqa: F401


@click.group()
def cli():
    """cup — Configurable Unified Plotter for HEP analyses."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--analysis", "-a",
    multiple=True,
    help="Run only the named analysis/analyses (repeatable). "
         "Defaults to all analyses in the config.",
)
@click.option(
    "--outdir", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the output directory from the config.",
)
@click.option(
    "--dry-run", is_flag=True, default=False,
    help="Parse config and print what would be produced, without writing files.",
)
def run(config: Path, analysis: tuple, outdir: Path | None, dry_run: bool):
    """
    Run analyses defined in CONFIG (a YAML file) and produce plot files.

    \b
    Examples:
        cup-plot run analysis.yaml
        cup-plot run analysis.yaml -a Muons -a Protons
        cup-plot run analysis.yaml --outdir /tmp/plots --dry-run
    """
    from cup.core.parser import Config
    from cup.core.manager import PlotManager

    click.echo(f"Loading config: {config}")
    cfg = Config.load(config)

    if outdir is not None:
        cfg.config.outdir = outdir
        click.echo(f"Output directory overridden → {outdir}")

    if analysis:
        missing = set(analysis) - set(cfg.analysis)
        if missing:
            click.echo(f"ERROR: Unknown analysis/analyses: {sorted(missing)}", err=True)
            click.echo(f"Available: {sorted(cfg.analysis)}", err=True)
            sys.exit(1)
        # Filter to requested analyses only
        cfg.analysis = {k: v for k, v in cfg.analysis.items() if k in analysis}

    click.echo(f"Analyses to run: {list(cfg.analysis)}")

    if dry_run:
        _dry_run_summary(cfg)
        return

    pm = PlotManager(cfg)
    pm.run()
    click.echo(f"\nDone.  Output in: {cfg.config.outdir}")


# ---------------------------------------------------------------------------
# list-plotters
# ---------------------------------------------------------------------------

@cli.command("list-plotters")
def list_plotters_cmd():
    """List all registered plotter types in priority order."""
    import cup.plotters as _p  # noqa: F401 — triggers registration

    from cup.plotters import list_plotters, _PLOTTER_REGISTRY
    click.echo("Registered plotters (highest priority first):")
    for entry in _PLOTTER_REGISTRY:
        click.echo(f"  [{entry['priority']:+d}]  {entry['name']:20s}  ({entry['cls'].__module__}.{entry['cls'].__name__})")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
def validate(config: Path):
    """
    Parse CONFIG and report any structural errors without running.
    """
    from cup.core.parser import Config

    try:
        cfg = Config.load(config)
        click.echo(f"✓  Config is valid.")
        click.echo(f"   Project : {cfg.config.project}")
        click.echo(f"   Analyses: {list(cfg.analysis)}")
        click.echo(f"   Styles  : {list(cfg.styles)}")
        for name, a in cfg.analysis.items():
            click.echo(f"   [{name}] {len(a.dataset)} dataset(s), {len(a.plot)} plot(s)")
    except Exception as exc:
        click.echo(f"✗  Config error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dry-run helper
# ---------------------------------------------------------------------------

def _dry_run_summary(cfg):
    from cup.plotters import get_plotter

    click.echo("\nDry-run: would produce the following files:")
    for aname, acfg in cfg.analysis.items():
        click.echo(f"\n  [{aname}]")
        for plot_cfg in acfg.plot:
            products = plot_cfg.product if isinstance(plot_cfg.product, list) else [plot_cfg.product]
            binning_info = f"binning(s) = {plot_cfg.binning}" if plot_cfg.binning else ""
            plotter = get_plotter(plot_cfg, acfg)
            pname = type(plotter).__name__ if plotter else "NO PLOTTER FOUND"
            click.echo(f"    {pname:25s}  product(s) = {str(products):40s}{binning_info}")
