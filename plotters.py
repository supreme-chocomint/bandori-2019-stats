import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from constants import *
from cleaner import DataCleaner


class CountsPlotDisplay:

    def __init__(
            self,
            kind,
            title,
            x_label,
            y_label,
            transpose=False,
            colormap="viridis",
            annotation_size="xx-small"
    ):
        """
        https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html for plot kinds.
        https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html for possible color-maps.
        https://matplotlib.org/3.1.1/api/text_api.html#matplotlib.text.Text.set_fontsize for possible annotation sizes.

        Note that not all plot kinds are tested (as well, some are useless/meaningless representations).
        Bar definitely works; line and box look like they work.
        Inline annotations of raw values only confirmed to show up on bar graphs.

        :param kind: String; plot type (e.g. bar, line, box)
        :param title: String; title of plot
        :param x_label: String; text describing x-axis
        :param y_label: String; text describing y-axis
        :param transpose: Bool; switches the dimension represented on x-axis and hue/plots
        :param colormap: String; which colors to use for the plots
        :param annotation_size: String; font size for labelling raw values inside plots
        """
        self.kind = kind
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.transpose = transpose
        self.colormap = colormap
        self.annotation_size = annotation_size


class CountsPlotter:
    """
    Plots two dimensions against counts/proportions of people in the intersection of those dimensions.
    E.g. if the two dimensions are region and favorite characters, the y-axis will be the number of people
    with a specific character as a favorite and from a specific region, with the x-axis being either the
    characters or the regions (depending on the CountsPlotDisplay's attributes).
    """

    def __init__(self, tsv_path):
        self.display = None
        self.df = DataCleaner.prepare_data_frame(tsv_path)

    def plot_music_band_by_age(self, display=None):
        self.display = CountsPlotDisplay(
            kind="bar",
            title="Favorite Bands (Music) By Age Group",
            x_label="Band",
            y_label="Percent in Age Group With Band Listed",
            transpose=True
        ) if display is None else display

        self._plot_band_by_age(BANDS_MUSIC)
        self.display = None

    def plot_chara_band_by_age(self, display=None):
        self.display = CountsPlotDisplay(
            kind="bar",
            title="Favorite Bands (Characters) By Age Group",
            x_label="Band",
            y_label="Percent in Age Group With Band Listed",
            transpose=True
        ) if display is None else display

        self._plot_band_by_age(BANDS_CHARA)
        self.display = None

    def plot_music_band_by_region(self, display=None, show_all=True):
        self.display = CountsPlotDisplay(
            kind="bar",
            title="Favorite Bands (Music) By Region",
            x_label="Band",
            y_label="Percent in Region With Band Listed",
            transpose=True
        ) if display is None else display

        self._plot_band_by_region(BANDS_MUSIC, show_all=show_all)
        self.display = None

    def plot_chara_band_by_region(self, display=None, show_all=True):
        self.display = CountsPlotDisplay(
            kind="bar",
            title="Favorite Bands (Characters) By Region",
            x_label="Band",
            y_label="Percent in Region With Band Listed",
            transpose=True
        ) if display is None else display

        self._plot_band_by_region(BANDS_CHARA, show_all=show_all)
        self.display = None

    def plot_music_band_by_gender(self, display=None):
        self.display = CountsPlotDisplay(
            kind="bar",
            title="Favorite Bands (Music) By Gender",
            x_label="Band",
            y_label="Percent in Gender With Band Listed",
            transpose=True,
            annotation_size="medium"
        ) if display is None else display

        self._plot_band_by_gender(BANDS_MUSIC)
        self.display = None

    def plot_chara_band_by_gender(self, display=None):
        self.display = CountsPlotDisplay(
            kind="bar",
            title="Favorite Bands (Characters) By Gender",
            x_label="Band",
            y_label="Percent in Gender With Band Listed",
            transpose=True,
            annotation_size="medium"
        ) if display is None else display

        self._plot_band_by_gender(BANDS_CHARA)
        self.display = None

    def plot_play_style_by_age(self, display=None):
        self.display = CountsPlotDisplay(
            kind="bar",
            title="Play Style By Age Group",
            x_label="Play Style",
            y_label="Percent in Age Group with Play Style",
            transpose=True,
            annotation_size="x-small"
        ) if display is None else display

        df = DataCleaner.filter_age(self.df)
        raw, normalized = self._group_counts_for_answer(df, AGE, PLAY_STYLE)
        self._plot_group_counts_for_answer(raw, normalized, sort=self.sort_ages)

        self.display = None

    def plot_play_style_by_region(self, display=None, show_all=True):
        def sort(raw_counts, normalized_counts):
            return self.sort_regions(raw_counts, normalized_counts, data_has_all=show_all)

        self.display = CountsPlotDisplay(
            kind="bar",
            title="Play Style By Region",
            x_label="Play Style",
            y_label="Percent in Region with Play Style",
            transpose=True,
            annotation_size="x-small"
        ) if display is None else display

        df = DataCleaner.filter_region(self.df, keep_all_legal=False)
        raw, normalized = self._group_counts_for_answer(df, REGION, PLAY_STYLE)
        self._plot_group_counts_for_answer(raw, normalized, sort=sort)

        self.display = None

    def plot_play_style_by_gender(self, display=None):
        self.display = CountsPlotDisplay(
            kind="bar",
            title="Play Style By Gender",
            x_label="Play Style",
            y_label="Percent in Gender with Play Style",
            transpose=True,
            annotation_size="medium"
        ) if display is None else display

        df = DataCleaner.filter_gender(self.df)
        raw, normalized = self._group_counts_for_answer(df, GENDER, PLAY_STYLE)
        self._plot_group_counts_for_answer(raw, normalized)

        self.display = None

    def _plot_band_by_age(
            self,
            band_col
    ):
        df = DataCleaner.filter_age(self.df)
        counts, counts_norm = self._group_counts_for_answer(
            df, stat_col=AGE, answer_col=band_col, answer_values=ALL_BANDS
        )
        self._plot_group_counts_for_answer(
            counts, counts_norm, sort=self.sort_ages
        )

    def _plot_band_by_region(
            self,
            band_col,
            show_all
    ):
        def data_sort(c, c_norm):
            return self.sort_regions(c, c_norm, data_has_all=show_all)

        df = self.df.replace(to_replace="North Asia and Central Asia", value="North/Central Asia")
        df = DataCleaner.filter_region(df, show_all)
        counts, counts_norm = self._group_counts_for_answer(
            df, stat_col=REGION, answer_col=band_col, answer_values=ALL_BANDS
        )
        self._plot_group_counts_for_answer(
            counts, counts_norm, sort=data_sort
        )

    def _plot_band_by_gender(
            self,
            band_col
    ):
        df = DataCleaner.filter_gender(self.df)
        counts, counts_norm = self._group_counts_for_answer(
            df, stat_col=GENDER, answer_col=band_col, answer_values=ALL_BANDS
        )
        self._plot_group_counts_for_answer(
            counts, counts_norm
        )

    @staticmethod
    def _group_counts_for_answer(
            df,
            stat_col,
            answer_col,
            answer_values=None
    ):
        """
        For each statistical group (e.g. people from Oceania),
        make new DataFrames consisting of number of respondents that gave an
        answer to the question represented by answer_col (e.g. favorite characters).
        Each column is an answer value, and each row is a group in the statistic.

        The elements of answer_values should not be substrings of each other,
        or that will match false-positives.

        If answer_values is None, the responses are split on commas, and the resulting
        elements are taken as the possible answer values.

        E.g. stat_col=AGE, answer_col=BAND_MUSIC will give you tables with
        different ages in the rows and different bands in the columns, where
        each cell tells you how many people in that age listed that band as a
        favorite music-wise.

        :param df: DataFrame
        :param stat_col: String; column name
        :param answer_col: String; column name
        :param answer_values: List of Strings or None; all legal values for answer column
        :return: two DataFrames, one with raw counts and one with percentages in group
        """

        # If no answer values provided, get them by splitting responses on comma
        if answer_values is None:
            answers = df[answer_col].str.split(",", expand=True)  # split up answers
            answer_values = answers.stack().str.strip().unique()  # make into Series, clean, and get all unique

        rows = []
        rows_normalized = []
        groups = df[stat_col].unique()

        for group in groups:
            rows_in_stat_group = df.loc[df[stat_col] == group]
            num_answered = dict()
            num_answered_normalized = dict()

            for answer in answer_values:
                # * is reserved character, so escape it if required
                try:
                    answer_regex = list(answer)
                    answer_regex.insert(answer.index("*"), "\\")
                    answer_regex = "".join(answer_regex)
                except ValueError:
                    # no need to escape
                    answer_regex = answer

                try:
                    rows_with_answer = rows_in_stat_group[answer_col].str.contains(answer_regex)
                    num_answered[answer] = rows_with_answer.value_counts()[True]
                    num_answered_normalized[answer] = rows_with_answer.value_counts(normalize=True)[True]
                except KeyError:
                    # nobody with in this group listed this answer
                    num_answered[answer] = 0
                    num_answered_normalized[answer] = 0

            rows.append(num_answered)
            rows_normalized.append(num_answered_normalized)

        non_normalized = pd.DataFrame(rows, index=groups, columns=answer_values)
        normalized = pd.DataFrame(rows_normalized, index=groups, columns=answer_values)
        return non_normalized, normalized

    @staticmethod
    def sort_ages(
            counts,
            counts_normalized
    ):
        """
        https://stackoverflow.com/a/11067072 and https://stackoverflow.com/a/25122293
        """
        counts, counts_normalized = counts.transpose(), counts_normalized.transpose()
        counts.sort_index(axis=1, inplace=True)
        counts_normalized.sort_index(axis=1, inplace=True)
        for br_counts in [counts, counts_normalized]:
            under_13 = br_counts["Under 13"]
            br_counts.drop(labels=["Under 13"], axis=1, inplace=True)
            br_counts.insert(0, "Under 13", under_13)
        return counts.transpose(), counts_normalized.transpose()

    @staticmethod
    def sort_regions(
            counts,
            counts_normalized,
            data_has_all=True
    ):
        """
        :param counts: DataFrame
        :param counts_normalized: DataFrame
        :param data_has_all: Whether the DataFrames have all regions or just the core (i.e. large sample count) one
        """
        region_core_sort_order = [
            "North America",
            "Southeast Asia",
            "Europe",
            "South America",
            "Oceania"
        ]
        region_small_sample_sort_order = [
            "East Asia",
            "South Asia",
            "Middle East",
            "Central America",
            "Other",
            "North/Central Asia",  # not the original string in the survey/TSV
            "Africa"
        ]
        if data_has_all:
            region_sort_order = region_core_sort_order + region_small_sample_sort_order
        else:
            region_sort_order = region_core_sort_order

        return counts.reindex(region_sort_order), counts_normalized.reindex(region_sort_order)

    def _plot_group_counts_for_answer(
            self,
            counts,
            counts_normalized,
            sort=None
    ):
        if sort:
            counts, counts_normalized = sort(counts, counts_normalized)
        if self.display.transpose:
            counts, counts_normalized = counts.transpose(), counts_normalized.transpose()

        ax = counts_normalized.plot(kind=self.display.kind, colormap=self.display.colormap, rot=0)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))  # add tick every 5%
        plt.subplots_adjust(wspace=0.2)
        ax.set_title(self.display.title)
        ax.set_xlabel(self.display.x_label)
        ax.set_ylabel(self.display.y_label)

        self._label_bars_with_raw_values(ax, counts)
        plt.show()

    def _label_bars_with_raw_values(
            self,
            ax,
            band_region_counts
    ):
        for i, p in enumerate(ax.patches):
            age_index = i // len(band_region_counts.index)
            band_index = i % len(band_region_counts.index)
            _age = band_region_counts.columns[age_index]
            _band = band_region_counts.index[band_index]
            raw_value = band_region_counts.loc[_band][_age]
            ax.annotate(str(raw_value), (p.get_x(), p.get_height() * 1.005), size=self.display.annotation_size)


class AssociationMetricPlotter:

    @staticmethod
    def plot(
            rules,
            x_axis,
            y_axis
    ):
        """
        Plots scatter graph of association rule metrics.
        :param rules: Rules
        :param x_axis: String of column name
        :param y_axis: String of column name
        """
        x = rules.table[[x_axis]].to_numpy()
        y = rules.table[[y_axis]].to_numpy()
        plt.scatter(x, y)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()


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
