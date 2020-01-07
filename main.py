from plotters import CountsPlotter, CountsPlotDisplay, AssociationMetricPlotter
from miner import AssociationMiner
from cleaner import DataCleaner
from constants import *


def main():
    plotter = CountsPlotter("responses.tsv")
    # plotter.plot_music_band_by_age()
    # plotter.plot_chara_band_by_age()
    # plotter.plot_music_band_by_region(show_all=False)
    # plotter.plot_chara_band_by_region(show_all=False)
    # plotter.plot_music_band_by_gender()
    # plotter.plot_chara_band_by_gender()
    # plotter.plot_play_style_by_age()
    # plotter.plot_play_style_by_region(show_all=False)
    # plotter.plot_play_style_by_gender()

    miner = AssociationMiner("responses.tsv")
    rules = miner.mine_favorite_character_reasons(antecedent="reason")
    AssociationMetricPlotter.plot(rules, x_axis="support", y_axis="lift")


main()
