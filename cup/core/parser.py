'''
Docstring for cup.core.parser
'''

from dataclasses import dataclass
import pandas as pd
import toml
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from cup.core import registry

@dataclass
class GlobalConfig:
    project: str
    file: str
    outdir: Optional[Path] = Path.cwd() / 'plots'
    project_name: Optional[str] = 'ICARUS'
    project_label: Optional[str] = 'Work in progress'
    fontsize: Optional[int] = 18
    labelfontsize: Optional[int] = 15
    file_extension: Optional[str] = 'pdf'
    file_dpi: Optional[float] = None

@dataclass
class StyleConfig:
    name: str
    style_kw: dict

@dataclass
class DatasetConfig:
    name: str
    label: str
    style: Optional[str] = 'default'

@dataclass
class FilterConfig:
    name: str
    params: Dict[str, Any]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Apply the filter using the registry.'''
        if self.name not in registry.FILTER_REGISTRY:
            raise ValueError(f'Unknown filter: {self.name}')
        # print(f'Applied filter {self.name}')
        return registry.FILTER_REGISTRY[self.name](df, **self.params)

    def describe(self) -> str:
        '''
        Return a human-readable description of the filter.
        Delegated to the registry filter if available.
        '''
        if self.name in registry.FILTER_DESCRIBE_REGISTRY:
            return registry.FILTER_DESCRIBE_REGISTRY[self.name](**self.params)

        # Fallback
        params = ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f'{self.name}({params})' if params else self.name
    
def parse_filters(raw_filter) -> Optional[List[FilterConfig]]:
    if raw_filter is None:
        return None

    if isinstance(raw_filter, dict):
        raw_filter = [raw_filter]

    filters = []
    for f in raw_filter:
        f = f.copy()
        name = f.pop('name')
        filters.append(FilterConfig(name=name, params=f))

    return filters

@dataclass
class BinningConfig:
    bins: int
    limits: Tuple[int, int]
    unit: Optional[str] = None
    scale: Optional[str] = 'linear'
    flow: Optional[str] = None
    integer: Optional[bool] = False

    def create(self, name: str):
        return registry.BINSCALE_REGISTRY[self.scale](
            bins=self.bins,
            limits=self.limits,
            flow=bool(self.flow),
            name=name
        )
    
@dataclass
class PlotConfig:
    label: str | List[str]
    product: str | List[str]
    binning: BinningConfig | List[BinningConfig]
    layout: Optional[Tuple[int, int]] = None
    yscale: Optional[str] = None
    ylabel: Optional[str] = 'Entries'
    grid: Optional[bool] = False
    showmedian: Optional[str] = None
    filter: Optional[List[FilterConfig]] = None

@dataclass
class AnalysisConfig:
    name: str
    dataset: List[DatasetConfig]
    plot: List[PlotConfig]
    merge_on: Optional[str | List[str] | None] = None
    density: Optional[bool] = False
    figsize: Optional[Tuple[float, float]] = (9, 7)
    filter: Optional[List[FilterConfig]] = None
    analysis_supplementaltext: Optional[str] = ''


@dataclass
class Config:
    
    config: GlobalConfig
    analysis: Dict[str, AnalysisConfig]
    styles: Dict[str, StyleConfig]

    @staticmethod
    def load(table):
        '''
        Load a configuration from a TOML file

        Parameter
        ---
         - table: `str` path to the configuration table
        
        Return
        ---
        `cup.core.parser.Config` configuration
        '''
        with open(table, 'r', encoding='utf-8') as reader:
            raw = toml.load(reader)

        setup_raw = raw['global']

        # Force outdir into a Path
        if 'outdir' in setup_raw:
            setup_raw['outdir'] = Path(setup_raw['outdir'])
        else:
            setup_raw['outdir'] = Path.cwd() / 'plots'

        if 'file' in setup_raw:
            setup_raw['file'] = Path(setup_raw['file'])
        else:
            raise FileExistsError('Missing file for analysis')

        config = GlobalConfig(**setup_raw)
        styles: Dict[str, StyleConfig] = {}
        analysis: Dict[str, AnalysisConfig] = {}

        for k in raw.keys():
            if 'style' in k:
                styles[k] = StyleConfig(name=k, style_kw=raw[k])

            if 'analysis' in k:
                
                datasets = [
                    DatasetConfig(**d)
                    for d in raw[k].pop('dataset', {})
                ]

                # Parse plots
                plots = []
                for p in raw[k].pop('plot', {}):
                    
                    # p --> dictionary of the analysis_Muon.plot list
                    # keys: label, product, (filter --> FilterConfig)

                    p['filter'] = parse_filters(p.get('filter'))

                    if 'binning' in p:
                        if isinstance(p['binning'], list):
                            listOfBinning = []
                            for b in p['binning']:
                                listOfBinning.append(BinningConfig(**b))
                            p['binning'] = listOfBinning
                        else:
                            p['binning'] = BinningConfig(**p['binning'])

                    plots.append(PlotConfig(**p))

                analysis_filters = parse_filters(raw[k].pop('filter', None))

                analysis[k.replace('analysis_', '')] = AnalysisConfig(
                    name=k.replace('analysis_', ''),
                    dataset=datasets,
                    plot=plots,
                    filter=analysis_filters,
                    **raw[k]
                )
        
        return Config(
            config=config,
            analysis=analysis,
            styles=styles
        )