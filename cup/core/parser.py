import toml
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from cup.core.registry import FILTER_REGISTRY, BINSCALE_REGISTRY

import pandas as pd


@dataclass
class GlobalConfig:
    project: str
    file: str
    outdir: Optional[Path] = Path.cwd() / 'plots'
    project_name: Optional[str] = 'ICARUS'

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
        if self.name not in FILTER_REGISTRY:
            raise ValueError(f'Unknown filter: {self.name}')
        return FILTER_REGISTRY[self.name](df, **self.params)

@dataclass
class BinningConfig:
    bins: int
    limits: Tuple[int, int]
    scale: Optional[str] = 'linear'
    flow: Optional[bool] = True

    def create(self, name: str):
        return BINSCALE_REGISTRY[self.scale](
            nbins=self.bins,
            limits=self.limits,
            flow=self.flow,
            name=name
        )
    
@dataclass
class PlotConfig:
    label: str | List[str]
    product: str | List[str]
    binning: BinningConfig | List[BinningConfig]
    filter: Optional[FilterConfig |List[FilterConfig] | None] = None

@dataclass
class AnalysisConfig:
    name: str
    dataset: List[DatasetConfig]
    plot: List[PlotConfig]
    merge_on: Optional[str | List[str] | None] = None
    density: Optional[bool] = False


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
                    for d in raw[k].get('dataset', {})
                ]

                # Parse plots
                plots = []
                for p in raw[k].get('plot', {}):
                    
                    # p --> dictionary of the analysis_Muon.plot list
                    # keys: label, product, (filter --> FilterConfig)

                    if 'filter' in p:
                        filter_name = p['filter'].pop('name')
                        p['filter'] = FilterConfig(name=filter_name, params=p['filter'])

                    if 'binning' in p:
                        if isinstance(p['binning'], list):
                            listOfBinning = []
                            for b in p['binning']:
                                listOfBinning.append(BinningConfig(**b))
                            p['binning'] = listOfBinning
                        else:
                            p['binning'] = BinningConfig(**p['binning'])

                    plots.append(PlotConfig(**p))

                analysis[k.replace('analysis_', '')] = AnalysisConfig(
                    name=k.replace('analysis_', ''),
                    dataset=datasets,
                    plot=plots,
                    merge_on=raw[k].get('merge_on', False)
                )
        
        return Config(
            config=config,
            analysis=analysis,
            styles=styles
        )