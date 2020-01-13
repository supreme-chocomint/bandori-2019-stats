import pandas as pd

from constants import *


class ResponseParser:
    """
    Parses the response of a person into usable answers.
    """

    @staticmethod
    def unique_answers(df, column):
        """
        Finds all unique answers from all responses.
        Main use case is to break up multi-answer responses into individual answers,
        but this method works on single-answer responses as well (it just does unnecessary work).
        Substrings assumed to be individual answers if comma-separated (after removing round brackets and their
        contents).

        e.g. if response is "Europe (includes Russia), North America [NA] (includes Mexico, Central America,
                Caribbean)", then "Europe" and "North America [NA]" are the two individual answers of the response,
                and will be included in the returned array.

        :return: Array
        """
        # remove round brackets' contents (https://stackoverflow.com/a/40621332)
        df[column].replace(r"\([^()]*\)", "", regex=True, inplace=True)
        answers = df[column].str.split(",", expand=True)  # split up answers
        return answers.stack().str.strip().unique().tolist()  # make into Series, clean, and get all unique


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
             JP_SERVER,
             FRANCHISE_PARTICIPATION,
             SEIYUU,
             PLAY_STYLE,
             OTHER_GAMES_IDOL,
             OTHER_GAMES_RHYTHM]
        ]  # Filter out unneeded data

        df = df.replace(to_replace="North Asia and Central Asia", value="North/Central Asia")  # makes plotting nicer
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
        res = df[df[column] != NO_RESPONSE]
        return res.dropna(subset=[column])  # remove if NaN

    @classmethod
    def filter_gender(
            cls,
            df
    ):
        return cls.filter_invalids(df, GENDER)

    @classmethod
    def filter_age(
            cls,
            df
    ):
        return cls.filter_invalids(df, AGE)

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
