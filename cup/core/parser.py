"""
cup.core.parser
===============
Dataclasses and loader for YAML analysis configurations.

YAML → Config is the single entry-point::

    cfg = Config.load("analysis.yaml")
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from cup.core import registry


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GlobalConfig:
    project: str
    file: Path
    outdir: Path = field(default_factory=lambda: Path.cwd() / "plots")
    project_name: str = "ICARUS"
    project_label: str = "Work in progress"
    fontsize: int = 18
    labelfontsize: int = 15
    file_extension: str = "pdf"
    file_dpi: Optional[float] = None
    ratio_height: int = 2
    dataset_path: str = "{dataset}"


@dataclass
class StyleConfig:
    name: str
    style_kw: Dict[str, Any]


@dataclass
class DatasetConfig:
    name: str
    label: str
    style: Optional[str] = "default"


@dataclass
class FilterConfig:
    name: str
    params: Dict[str, Any]

    # ------------------------------------------------------------------
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the filter using the registry."""
        if self.name not in registry.FILTER_REGISTRY:
            raise ValueError(f"Unknown filter: '{self.name}'. "
                             f"Registered: {list(registry.FILTER_REGISTRY)}")
        return registry.FILTER_REGISTRY[self.name](df, **self.params)

    def describe(self) -> Optional[str]:
        """Human-readable description, delegated to the registry entry."""
        if self.name in registry.FILTER_DESCRIBE_REGISTRY:
            return registry.FILTER_DESCRIBE_REGISTRY[self.name](**self.params)
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params})" if params else self.name

    # ------------------------------------------------------------------
    @staticmethod
    def parse(raw) -> Optional[List["FilterConfig"]]:
        """Accept a single dict or a list of dicts."""
        if raw is None:
            return None
        if isinstance(raw, dict):
            raw = [raw]
        out = []
        for f in raw:
            f = f.copy()
            name = f.pop("name")
            out.append(FilterConfig(name=name, params=f))
        return out


@dataclass
class BinningConfig:
    bins: int
    limits: Tuple[float, float]
    unit: Optional[str] = None
    scale: str = "linear"
    flow: Optional[str] = None
    integer: bool = False
    scale_ax: bool = True

    def create(self, name: str):
        """Build a hist axis for this binning."""
        if self.scale not in registry.BINSCALE_REGISTRY:
            raise ValueError(f"Unknown binning scale: '{self.scale}'. "
                             f"Registered: {list(registry.BINSCALE_REGISTRY)}")
        return registry.BINSCALE_REGISTRY[self.scale](
            bins=self.bins,
            limits=self.limits,
            flow=bool(self.flow),
            name=name,
        )
    
    def __str__(self) -> str:
        unit_str = f" {self.unit}" if self.unit else ""
        flow_str = f", flow='{self.flow}'" if self.flow else ""
        return f"{self.scale}({self.bins} bins, limits={self.limits}{unit_str}{flow_str})"


@dataclass
class RatioPlotConfig:
    compare: Tuple[str, str]
    comparison: Literal[
        "ratio", "split_ratio", "pull",
        "difference", "relative_difference", "efficiency", "asymmetry",
    ] = "ratio"
    style: Optional[Literal["errorbar", "bar"]] = None
    color: str = "k"
    alpha: Optional[float] = None
    ylabel: Optional[str] = None

    @staticmethod
    def parse(raw) -> Optional[List["RatioPlotConfig"]]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            raw = [raw]
        return [RatioPlotConfig(**r) for r in raw]


@dataclass
class PlotConfig:
    label: str | List[str]
    product: str | List[str]
    binning: BinningConfig | List[BinningConfig]
    layout: Optional[Tuple[int, int]] = None
    yscale: Optional[str] = None
    ylabel: Optional[str] = "Entries"
    grid: bool = False
    showmedian: Optional[str] = None
    filter: Optional[List[FilterConfig]] = None
    ratio: Optional[List[RatioPlotConfig]] = None
    # --- plot-type flags ---
    profile: bool = False
    profile_stat: Literal["mean", "median"] = "median"
    profile_band: bool = True
    efficiency: bool = False
    efficiency_denominator: Optional[str] = None   # dataset name used as denominator
    efficiency_numerator: Optional[str] = None     # dataset name used as numerator


@dataclass
class AnalysisConfig:
    name: str
    dataset: List[DatasetConfig]
    plot: List[PlotConfig]
    merge_on: Optional[str | List[str]] = None
    density: bool = False
    figsize: Tuple[float, float] = (9, 7)
    filter: Optional[List[FilterConfig]] = None
    label: Optional[str] = None
    analysis_supplementaltext: str = ""


@dataclass
class Config:
    config: GlobalConfig
    analysis: Dict[str, AnalysisConfig]
    styles: Dict[str, StyleConfig]

    # ------------------------------------------------------------------
    @staticmethod
    def load(path: str | Path) -> "Config":
        """
        Load a YAML configuration file.

        Parameters
        ----------
        path:
            Path to the ``.yaml``, ``.yml``, or ``.toml`` configuration file.
        """

        path = Path(path)
        suffix = path.suffix.lower()

        if suffix not in [".yaml", ".yml", ".toml"]:
            raise ValueError(f"Unsupported file format: {suffix}. Must be .yaml, .yml, or .toml")

        with open(path, "r", encoding="utf-8") as fh:
            if suffix in [".yaml", ".yml"]:
                raw = yaml.safe_load(fh)
            elif suffix == ".toml":
                import toml
                raw = toml.load(fh)

        # --- global section ---
        setup_raw = raw["global"].copy()
        setup_raw["file"] = Path(setup_raw["file"])
        setup_raw.setdefault("outdir", Path.cwd() / "plots")
        setup_raw["outdir"] = Path(setup_raw["outdir"])
        global_cfg = GlobalConfig(**setup_raw)

        # --- styles ---
        styles: Dict[str, StyleConfig] = {}
        for k, v in raw.get("styles", {}).items():
            styles[k] = StyleConfig(name=k, style_kw=v)

        # --- analyses ---
        analyses: Dict[str, AnalysisConfig] = {}
        for k, v in raw.get("analyses", {}).items():
            v = v.copy()

            datasets = [DatasetConfig(**d) for d in v.pop("datasets", [])]
            raw_plots = v.pop("plots", [])

            plots: List[PlotConfig] = []
            for p in raw_plots:
                p = p.copy()
                p["filter"] = FilterConfig.parse(p.get("filter"))
                p["ratio"] = RatioPlotConfig.parse(p.get("ratio"))

                if "binning" in p:
                    if isinstance(p["binning"], list):
                        p["binning"] = [BinningConfig(**b) for b in p["binning"]]
                    else:
                        p["binning"] = BinningConfig(**p["binning"])

                plots.append(PlotConfig(**p))

            analysis_filter = FilterConfig.parse(v.pop("filter", None))

            analyses[k] = AnalysisConfig(
                name=k,
                dataset=datasets,
                plot=plots,
                filter=analysis_filter,
                **v,
            )

        return Config(config=global_cfg, analysis=analyses, styles=styles)