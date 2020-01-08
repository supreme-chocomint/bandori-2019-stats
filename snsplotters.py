import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from constants import *
from cleaner import DataCleaner


class HeatMapPlotter:
    """
    Makes heat map of a demographic.
    Right now, only draws a map of those who have Poppin'Party as a favorite, visualizing their gender and region.
    Each cell is proportion of fans in the region (listed in x-axis) that belong to the gender (listed in y-axis).
    """

    def __init__(self, tsv_path):
        self.df = DataCleaner.prepare_data_frame(tsv_path)

    def draw_heat_map(self):
        df = DataCleaner.filter_gender(self.df)
        likes_poppin_party = df[df[BANDS_MUSIC].str.contains("Poppin'Party")]
        pp_counts = pd.crosstab(likes_poppin_party[GENDER], likes_poppin_party[REGION], normalize="columns")
        print(pp_counts)

        plt.title("Poppin'Party as Favourite Band")
        ax = sns.heatmap(pp_counts, annot=True, cmap="coolwarm")
        self._fix_heat_map()
        ax.vlines(list(range(len(pp_counts.columns))), *ax.get_ylim())  # add lines to separate columns
        plt.show()

    @staticmethod
    def _fix_heat_map():
        """
        Fixes Seaborn bug that crops top/bottom of heatmap on show
        https://github.com/mwaskom/seaborn/issues/1773#issuecomment-546466986
        """
        # fix for mpl bug that cuts off top/bottom of seaborn viz
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
