from plotters import CountsPlotter, CountsPlotDisplay, AssociationMetricPlotter
from miner import AssociationMiner
from snsplotters import HeatMapPlotter
from cleaner import DataCleaner
from constants import *


def main():
    plotter = CountsPlotter("responses.tsv", export_to_excel=True)
    # plotter.plot_music_band_by_age()
    # plotter.plot_chara_band_by_age()
    # plotter.plot_music_band_by_region(show_all=False)
    # plotter.plot_chara_band_by_region(show_all=False)
    # plotter.plot_music_band_by_gender()
    # plotter.plot_chara_band_by_gender()
    # plotter.plot_play_style_by_age()
    # plotter.plot_play_style_by_region(show_all=False)
    # plotter.plot_play_style_by_gender()
    # plotter.plot_participation_by_age()
    # plotter.plot_participation_by_region(show_all=False)
    # plotter.plot_participation_by_gender()

    miner = AssociationMiner("responses.tsv", export_to_excel=True)
    # miner.mine_favorite_characters()
    # miner.mine_favorite_band_members()
    # miner.mine_favorite_character_reasons(antecedent="character")
    # miner.mine_favorite_character_reasons(antecedent="reason")
    # miner.mine_age_favorite_characters()
    # miner.mine_gender_favorite_characters()
    # AssociationMetricPlotter.plot(rules, x_axis="support", y_axis="lift")


main()
