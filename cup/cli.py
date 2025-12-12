import argparse
from cup.core.parser import Config
from cup.plot.manager import PlotManager

def main():
    
    # Declare cliapp
    cliapp = argparse.ArgumentParser('cup', description='CAFAna Unified Plotter')
    cliapp.add_argument(
        '-c', '--configuration', help='TOML configuration for current analysis', 
        nargs=1, type=str, required=True)
    cliapp.add_argument('-s', '--show', help='If interactive, open analysis.plot with name <show>')

    
    args = cliapp.parse_args()

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