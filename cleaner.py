import pandas as pd

from constants import *


class DataCleaner:
    """
    Cleans DataFrames of unneeded data, invalid responses, etc.
    """

    @staticmethod
    def prepare_data_frame(tsv_path):
        df = pd.read_table(tsv_path)[
            [REGION,
             GENDER,
             AGE,
             BANDS_MUSIC,
             BANDS_CHARA,
             CHARACTERS,
             CHARACTER_REASONS,
             SONGS_ORIGINAL,
             SONGS_COVER,
             FRANCHISE_PARTICIPATION,
             PLAY_STYLE,
             OTHER_GAMES_IDOL,
             OTHER_GAMES_RHYTHM]
        ]  # Filter out unneeded data

        return df

    @staticmethod
    def filter_gender(
            df
    ):
        res = df[df[GENDER] != NO_RESPONSE]
        res = res.dropna(subset=[GENDER])  # remove if gender is NaN
        return res

    @staticmethod
    def filter_age(
            df
    ):
        res = df[df[AGE] != NO_RESPONSE]
        res = res.dropna(subset=[AGE])  # remove if age is NaN
        return res

    @staticmethod
    def filter_region(
            df,
            keep_all_legal=True
    ):
        """
        :param df: DataFrame
        :param keep_all_legal: Whether to keep regions with low sample sizes or not
        """
        if not keep_all_legal:
            keep_list = ["North America", "Southeast Asia", "Europe", "South America", "Oceania"]
            df = df[df[REGION].isin(keep_list)]
        return df.dropna(subset=[REGION])  # remove if region is NaN
