from plotters import CountsPlotter, CountsPlotDisplay, AssociationMetricPlotter
from miner import AssociationMiner
from snsplotters import HeatMapPlotter
from helpers import DataCleaner, ResponseParser
from constants import *


def main():
    plotter = CountsPlotter("responses.tsv", export_to_csv=True)
    sns_plotter = HeatMapPlotter("responses.tsv")
    # plotter.plot_music_band_by_age()
    # plotter.plot_chara_band_by_age()
    # plotter.plot_music_band_by_region()
    # plotter.plot_chara_band_by_region()
    # plotter.plot_music_band_by_gender()
    # plotter.plot_chara_band_by_gender()
    # plotter.plot_play_style_by_age()
    # plotter.plot_play_style_by_region()
    # plotter.plot_play_style_by_gender()
    # plotter.plot_participation_by_age()
    # plotter.plot_participation_by_region()
    # plotter.plot_participation_by_gender()
    # sns_plotter.draw_gender_vs_region()
    # sns_plotter.draw_age_vs_gender()
    # sns_plotter.draw_age_vs_region()

    miner = AssociationMiner("responses.tsv", export_to_csv=True)
    # miner.mine_favorite_characters()
    # miner.mine_favorite_band_members()
    # miner.mine_favorite_character_reasons(antecedent="character")
    # miner.mine_favorite_character_reasons(antecedent="reason")
    # miner.mine_age_favorite_characters()
    # miner.mine_gender_favorite_characters()
    # miner.mine_region_favorite_characters()
    # miner.mine_age_favorite_band_chara()
    # miner.mine_gender_favorite_band_chara()
    # miner.mine_region_favorite_band_chara()
    # AssociationMetricPlotter.plot(rules, x_axis="support", y_axis="lift")


main()
