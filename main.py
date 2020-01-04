import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from constants import *


def main():
    df = pd.read_table("responses.tsv")
    df = df[[REGION, GENDER, AGE, BANDS_MUSIC, BANDS_CHARA, CHARACTERS, CHARACTER_REASONS,
             SONGS_ORIGINAL, SONGS_COVER, FRANCHISE_PARTICIPATION, PLAY_STYLE,
             OTHER_GAMES_IDOL, OTHER_GAMES_RHYTHM]]  # Filter out unneeded data

    # plot_music_band_by_age(df)
    # plot_chara_band_by_age(df)
    # plot_music_band_by_region(df)
    # plot_chara_band_by_region(df, show_all=False)
    # plot_music_band_by_gender(df, show_all=False)
    # plot_chara_band_by_gender(df)


def plot_music_band_by_age(df):
    _plot_band_by_age(df, BANDS_MUSIC, "Favorite Bands (Music) By Age Group")


def plot_chara_band_by_age(df):
    _plot_band_by_age(df, BANDS_CHARA, "Favorite Bands (Characters) By Age Group")


def plot_music_band_by_region(df, show_all=True):
    _plot_band_by_region(df, BANDS_MUSIC, "Favorite Bands (Music) By Region", show_all=show_all)


def plot_chara_band_by_region(df, show_all=True):
    _plot_band_by_region(df, BANDS_CHARA, "Favorite Bands (Characters) By Region", show_all=show_all)


def plot_music_band_by_gender(df):
    _plot_band_by_gender(df, BANDS_MUSIC, "Favorite Bands (Music) By Gender")


def plot_chara_band_by_gender(df):
    _plot_band_by_gender(df, BANDS_CHARA, "Favorite Bands (Characters) By Gender")


def _plot_band_by_age(df, band_col, plot_title):
    df = _filter_age(df)
    counts, counts_norm = _group_counts_for_answer(df, stat_col=AGE, answer_col=band_col, answer_values=ALL_BANDS)
    _plot_group_counts_for_answer(counts, counts_norm, plot_title=plot_title, sort=sort_ages, transpose=True,
                                  x_title="Band", y_title="Percent in Age Group With Band Listed")


def _plot_band_by_region(df, band_col, plot_title, show_all):
    def data_sort(c, c_norm):
        if show_all:
            return sort_regions(c, c_norm, data_has_all=True)
        else:
            return sort_regions(c, c_norm, data_has_all=False)

    df.replace(to_replace="North Asia and Central Asia", value="North/Central Asia", inplace=True)
    df = _filter_region(df, show_all)
    counts, counts_norm = _group_counts_for_answer(df, stat_col=REGION, answer_col=band_col, answer_values=ALL_BANDS)
    _plot_group_counts_for_answer(counts, counts_norm, plot_title=plot_title, sort=data_sort, transpose=True,
                                  x_title="Band", y_title="Percent in Region With Band Listed")


def _plot_band_by_gender(df, band_col, plot_title):
    df = _filter_gender(df)
    counts, counts_norm = _group_counts_for_answer(df, stat_col=GENDER, answer_col=band_col, answer_values=ALL_BANDS)
    _plot_group_counts_for_answer(counts, counts_norm, plot_title=plot_title, transpose=True,
                                  x_title="Band", y_title="Percent in Gender With Band Listed")


def _group_counts_for_answer(df, stat_col, answer_col, answer_values):
    """
    For each statistical group (e.g. people from Oceania),
    make new DataFrames consisting of number of respondents that gave an
    answer to the question represented by answer_col (e.g. favorite characters).
    Each column is an answer value, and each row is a group in the statistic.

    The elements of answer_values should not be substrings of each other,
    or that will match false-positives.

    E.g. stat_col=AGE, answer_col=BAND_MUSIC will give you tables with
    different ages in the rows and different bands in the columns, where
    each cell tells you how many people in that age listed that band as a
    favorite music-wise.

    :param df: DataFrame
    :param stat_col: String; column name
    :param answer_col: String; column name
    :param answer_values: List of Strings; all legal values for answer column
    :return: two DataFrames, one with raw counts and one with percentages in group
    """
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


def _plot_group_counts_for_answer(counts, counts_normalized, plot_title, x_title, y_title, sort=None, transpose=False):
    if sort:
        counts, counts_normalized = sort(counts, counts_normalized)
    if transpose:
        counts, counts_normalized = counts.transpose(), counts_normalized.transpose()

    ax = counts_normalized.plot(kind="bar", colormap="viridis", rot=0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))  # add tick every 5%
    plt.subplots_adjust(wspace=0.2)
    ax.set_title(plot_title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    _label_bars_with_raw_values(ax, counts)
    plt.show()


def sort_ages(counts, counts_normalized):
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


def sort_regions(counts, counts_normalized, data_has_all=True):
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


def draw_heatmap(df):
    likes_pp = df[df[BANDS_MUSIC] == "Poppin'Party"]
    likes_pp = _filter_gender(likes_pp)
    pp_counts = pd.crosstab(likes_pp[GENDER], likes_pp[REGION], normalize="columns")
    print(pp_counts)

    plt.title("Poppin'Party as Favourite Band")
    ax = sns.heatmap(pp_counts, annot=True, cmap="coolwarm")
    _fix_heatmap()
    ax.vlines(list(range(len(pp_counts.columns))), *ax.get_ylim())  # add lines to separate columns
    plt.show()


def _filter_gender(df):
    res = df[df[GENDER] != NO_RESPONSE]
    res = res.dropna(subset=[GENDER])  # remove if gender is NaN
    return res


def _filter_age(df):
    res = df[df[AGE] != NO_RESPONSE]
    res = res.dropna(subset=[AGE])  # remove if age is NaN
    return res


def _filter_region(df, keep_all_legal=True):
    """
    :param keep_all_legal: Whether to keep regions with low sample sizes or not
    """
    if not keep_all_legal:
        keep_list = ["North America", "Southeast Asia", "Europe", "South America", "Oceania"]
        df = df[df[REGION].isin(keep_list)]
    return df.dropna(subset=[REGION])  # remove if region is NaN


def _fix_heatmap():
    """
    Fixes Seaborn bug that crops top/bottom of heatmap on show
    https://github.com/mwaskom/seaborn/issues/1773#issuecomment-546466986
    """
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values


def _label_bars_with_raw_values(ax, band_region_counts):
    for i, p in enumerate(ax.patches):
        age_index = i // len(band_region_counts.index)
        band_index = i % len(band_region_counts.index)
        _age = band_region_counts.columns[age_index]
        _band = band_region_counts.index[band_index]
        raw_value = band_region_counts.loc[_band][_age]
        ax.annotate(str(raw_value), (p.get_x(), p.get_height() * 1.005), size="xx-small")


main()
