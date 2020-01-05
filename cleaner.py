import pandas as pd

from constants import *


class DataCleaner:
    """
    Cleans DataFrames of unneeded data, invalid responses, etc.
    """

    @staticmethod
    def prepare_data_frame(
            tsv_path
    ):
        df = pd.read_table(tsv_path)[
            [REGION,
             GENDER,
             AGE,
             BANDS_MUSIC,
             BANDS_CHARA,
             CHARACTERS,
             CHARACTER_REASONS,
             CHARACTER_POPIPA,
             CHARACTER_AFTERGLOW,
             CHARACTER_GURIGURI,
             CHARACTER_HHW,
             CHARACTER_PASUPARE,
             CHARACTER_RAS,
             CHARACTER_ROSELIA,
             SONGS_ORIGINAL,
             SONGS_COVER,
             FRANCHISE_PARTICIPATION,
             PLAY_STYLE,
             OTHER_GAMES_IDOL,
             OTHER_GAMES_RHYTHM]
        ]  # Filter out unneeded data

        return df

    @staticmethod
    def filter_invalids(
            df,
            column
    ):
        """
        General method for removing rows with invalid values in a column.
        :param df: DataFrame
        :param column: name of column to check
        """
        return df.dropna(subset=[column])  # remove if NaN

    @classmethod
    def filter_gender(
            cls,
            df
    ):
        res = df[df[GENDER] != NO_RESPONSE]
        return cls.filter_invalids(res, GENDER)

    @classmethod
    def filter_age(
            cls,
            df
    ):
        res = df[df[AGE] != NO_RESPONSE]
        return cls.filter_invalids(res, AGE)

    @classmethod
    def filter_region(
            cls,
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
        return cls.filter_invalids(df, REGION)
