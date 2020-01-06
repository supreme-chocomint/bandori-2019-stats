from plotters import CountsPlotter, CountsPlotDisplay
from miner import AssociationMiner


def main():
    plotter = CountsPlotter("responses.tsv")
    # plotter.plot_music_band_by_age()
    # plotter.plot_chara_band_by_age()
    # plotter.plot_music_band_by_region(show_all=False)
    # plotter.plot_chara_band_by_region(show_all=False)
    # plotter.plot_music_band_by_gender()
    # plotter.plot_chara_band_by_gender()

    miner = AssociationMiner("responses.tsv")
    rules = miner.mine_favorite_characters()
    results = miner.search_rules(rules, one_of=["Hikawa"], location="antecedents")


main()
