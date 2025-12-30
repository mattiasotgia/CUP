import argparse

from cup.core.parser import Config
from cup.plot.manager import PlotManager

import matplotlib as mpl

def main():
    
    # Declare cliapp
    cliapp = argparse.ArgumentParser('cup', description='CAFAna Unified Plotter')
    cliapp.add_argument(
        '-c', '--configuration', help='TOML configuration for current analysis', 
        nargs=1, type=str, required=True)
    cliapp.add_argument('-s', '--show', help='If interactive, open analysis.plot with name <show>')
    cliapp.add_argument('-b', '--batch', help='If running over ssh, force backend to be Agg', action='store_true')
    
    args = cliapp.parse_args()

    if args.batch:
        mpl.use('Agg')

    # start cli routing here
    # print(args.configuration)
    try:
        configuration = Config.load(args.configuration[0])
    except TypeError as te:
        print(f'Key error: {te}')
        return
    except Exception as e:
        print(e)
        return
    
    # print(configuration)
    plot_manager = PlotManager(configuration) # TODO
    plot_manager.run()
