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

    def draw(self, x, y, df=None, normalize=None, fmt="g"):
        """
        General function to plot heat maps.
        :param x: x-axis column name
        :param y: y-axis column name
        :param df: the DataFrame to make a map on
        :param normalize: "x", "y", "both", or None; which axis to show normalization of counts along
        :param fmt: String; formatting to use, general 6-digit precision being default; see _plot_heat_map()
        """
        if df is None:
            df = self.df

        # translate normalize argument
        norm_arg_to_pd = {
            "x": "index",
            "y": "columns",
            "both": True,
            None: False
        }
        norm_arg_to_border = {
            "x": "horizontal",
            "y": "vertical",
            "both": None,
            None: None,
        }

        counts = pd.crosstab(df[x], df[y], normalize=norm_arg_to_pd[normalize])
        print(pd.crosstab(df[x], df[y]))
        self._plot_heat_map(counts, border=norm_arg_to_border[normalize], fmt=fmt)

    def draw_gender_vs_region(self):
        """
        Cell annotations are percentage in region.
        """
        df = DataCleaner.filter_gender(self.df)
        df = DataCleaner.filter_region(df)
        counts = pd.crosstab(df[REGION], df[GENDER], normalize="index")
        print(pd.crosstab(df[REGION], df[GENDER]))
        self._plot_heat_map(counts, border="horizontal")

    def draw_age_vs_region(self):
        """
        Cell annotations are percentage in region.
        """
        df = DataCleaner.filter_age(self.df)
        df = DataCleaner.filter_region(df)
        counts = pd.crosstab(df[REGION], df[AGE], normalize="index")
        print(pd.crosstab(df[REGION], df[AGE]))
        self._plot_heat_map(counts, border="horizontal")

    def draw_age_vs_gender(self):
        """
        Cell annotations are percentage in age.
        """
        df = DataCleaner.filter_age(self.df)
        df = DataCleaner.filter_gender(df)
        counts = pd.crosstab(df[AGE], df[GENDER], normalize="index")
        print(pd.crosstab(df[AGE], df[GENDER]))
        self._plot_heat_map(counts, border="horizontal")

    def _plot_heat_map(self, counts, cmap="BuGn", border=None, fmt=".2g"):
        """
        Draws heat map for frequency table.
        :param counts: DataFrame with counts in each cell
        :param cmap: color map to use. See https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/
        :param border: "horizontal", "vertical", or None: which borders between rows and/or columns to draw
        :param fmt: https://docs.python.org/3/library/string.html#format-specification-mini-language
        """
        ax = sns.heatmap(counts, annot=True, fmt=fmt, cmap=cmap)
        self._fix_heat_map()

        if border is None:
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
