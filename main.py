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
    filtered_itemsets = miner.find_sets()
    rules = miner.find_rules(
        metric="confidence",
        metric_threshold=0.1,
        max_consequents=2,
        sort_by=["antecedent_len", "lift"],
        sort_ascending=[True, False]
    )


main()
