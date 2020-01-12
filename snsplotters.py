import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from constants import *
from helpers import DataCleaner


class HeatMapPlotter:
    """
    Makes heat map for demographic frequencies.
    """

    def __init__(self, tsv_path):
        self.df = DataCleaner.prepare_data_frame(tsv_path)

    def draw_gender_vs_region(self):
        df = DataCleaner.filter_gender(self.df)
        df = DataCleaner.filter_region(df)
        counts = pd.crosstab(df[REGION], df[GENDER], normalize="index")
        print(pd.crosstab(df[REGION], df[GENDER]))
        self._draw(counts, border="horizontal")

    def draw_age_vs_region(self):
        df = DataCleaner.filter_age(self.df)
        df = DataCleaner.filter_region(df)
        counts = pd.crosstab(df[REGION], df[AGE], normalize="index")
        print(pd.crosstab(df[REGION], df[AGE]))
        self._draw(counts, border="horizontal")

    def draw_age_vs_gender(self):
        df = DataCleaner.filter_age(self.df)
        df = DataCleaner.filter_gender(df)
        counts = pd.crosstab(df[AGE], df[GENDER], normalize="index")
        print(pd.crosstab(df[AGE], df[GENDER]))
        self._draw(counts, border="horizontal")

    def _draw(self, counts, cmap="BuGn", border="none"):
        """
        Draws heat map for frequency table.
        :param counts: DataFrame with counts in each cell
        :param cmap: color map to use. See https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/
        :param border: "horizontal", "vertical", or "none": how to draw borders between rows or columns
        """
        ax = sns.heatmap(counts, annot=True, cmap=cmap)
        self._fix_heat_map()

        if border == "none":
            pass
        elif border == "horizontal":
            ax.hlines(list(range(len(counts.index))), *ax.get_xlim())
        elif border == "vertical":
            ax.vlines(list(range(len(counts.columns))), *ax.get_ylim())
        else:
            raise ValueError("invalid 'border' argument")

        plt.show()

    @staticmethod
    def _fix_heat_map():
        """
        Fixes Seaborn bug that crops top/bottom of heat map on show
        https://github.com/mwaskom/seaborn/issues/1773#issuecomment-546466986
        """
        # fix for mpl bug that cuts off top/bottom of seaborn viz
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
