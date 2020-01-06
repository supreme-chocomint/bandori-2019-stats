from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

from constants import *
from cleaner import DataCleaner


class AssociationMiner:
    """
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
    """

    def __init__(
            self,
            tsv_path
    ):
        self.df = DataCleaner.prepare_data_frame(tsv_path)

    def mine(
            self,
            columns,
            column_values,
            min_frequency=0.01  # ~25 responses
    ):
        """
        Generic function to mine rules from responses.
        :param columns: List of column names to consider.
        :param column_values: List of column values each column can have (one list per column).
        :param min_frequency: threshold frequency for itemset to be considered "frequent"
        :return: Rules
        """
        raw_itemsets = self._generate_frequent_itemsets(columns, column_values, min_frequency)
        return self._generate_association_rules(raw_itemsets)

    def mine_favorite_characters(self):
        """
        Mines for rules regarding all favorite characters.
        :return Rules
        """
        return self.mine([CHARACTERS], [ALL_CHARACTERS])

    def mine_favorite_band_members(self):
        """
        Mines for rules regarding favorite character in each band.
        :return Rules
        """
        return self.mine(
            [CHARACTER_POPIPA,
             CHARACTER_AFTERGLOW,
             CHARACTER_GURIGURI,
             CHARACTER_HHW,
             CHARACTER_PASUPARE,
             CHARACTER_ROSELIA],
            [[ALL_CHARACTERS] * 7][0]  # list of 7 CHARACTER lists
        )

    def mine_favorite_character_reasons(
            self,
            antecedent="all"
    ):
        """
        Mines for rules involving favorite characters and reasons for liking those characters.
        :param antecedent: "character", "reason", or "all"
        :return: Rules
        """
        if antecedent not in ["all", "character", "reason"]:
            raise ValueError("invalid antecedent argument: must be 'all', 'character', or 'reason'")

        rules = self.mine([CHARACTERS, CHARACTER_REASONS], [ALL_CHARACTERS, ALL_CHARACTER_REASONS])
        rules = Rules(rules.search(one_of=ALL_CHARACTER_REASONS))

        if antecedent == "all":
            return rules
        elif antecedent == "character":
            return Rules(rules.search(one_of=ALL_CHARACTERS, location="antecedents"))
        elif antecedent == "reason":
            return Rules(rules.search(one_of=ALL_CHARACTER_REASONS, location="antecedents"))

    def mine_age_favorite_characters(self):
        """
        Mines for rules that predict age from favorite characters.
        The reverse doesn't seem to predict anything useful (the predictions are just popular characters).
        Since this is predicting age, the predictions are overwhelmingly 20-24yrs and 14-19yrs, since confidence
        must be >30%, and less common age groups wouldn't make this threshold.
        """
        age_values = DataCleaner.filter_age(self.df)[AGE].unique().tolist()
        table = self.mine(
            [CHARACTERS, AGE], [ALL_CHARACTERS, age_values]
        ).search(
            one_of=age_values,
            location="consequents"
        )
        return Rules(table)

    def mine_gender_favorite_characters(self):
        """
        Mines for rules that predict gender from favorite characters.
        """
        gender_values = DataCleaner.filter_gender(self.df)[GENDER].unique().tolist()
        table = self.mine(
            [CHARACTERS, GENDER], [ALL_CHARACTERS, gender_values]
        ).search(
            one_of=gender_values,
            location="consequents"
        )
        return Rules(table)

    def _generate_frequent_itemsets(
            self,
            columns,
            column_values,
            min_frequency
    ):
        """
        Uses the values of columns to generate frequent itemsets for association rule mining.
        :param columns: List of column names to use
        :param column_values: List; each element is itself a list, holding the legal values of the column
        :param min_frequency: threshold frequency for set to be considered "frequent"
        :return DataFrame
        """
        lists = self._reduce(self.df, columns, column_values)
        one_hot_df = self._transform_to_one_hot(lists)
        return self._find_sets(one_hot_df, min_frequency=min_frequency)

    def _generate_association_rules(
            self,
            itemsets
    ):
        """
        Uses frequent itemsets to generate rules with 30% confidence, 1 antecedent, and sorted by lift.
        If confidence is too high, rules are too specific to individual people, as opinions vary quite a bit.
        :param itemsets: DataFrame
        :return: Rules
        """
        rules = self._find_rules(itemsets, metric="confidence", metric_threshold=0.3)
        rules.organize(max_antecedents=1, sort_by=["lift"], sort_ascending=[False])
        return rules

    @staticmethod
    def _find_sets(
            one_hot_df,
            min_frequency
    ):
        """
        Finds frequent itemsets.
        FPGrowth algorithm used instead of Apriori because it can handle min_frequency=0.
        :param min_frequency: Float; threshold occurrence for a set to be considered "frequent"
        :return DataFrame
        """
        itemsets = fpgrowth(one_hot_df, min_support=min_frequency, use_colnames=True)
        return itemsets.sort_values(by=["support"], ascending=False)

    @staticmethod
    def _filter_itemsets(
            itemsets,
            remove_single_items=True
    ):
        """
        Filters itemsets based on flags.
        :param itemsets: DataFrame
        :param remove_single_items: Bool; whether to remove sets with one item or not
        :return DataFrame
        """
        # Remove all sets with one item, because those are just the most popular choices
        if remove_single_items:
            itemsets = itemsets.copy()
            itemsets["length"] = itemsets["itemsets"].apply(lambda x: len(x))
            return itemsets[itemsets["length"] > 1]
        else:
            return itemsets

    @staticmethod
    def _find_rules(
            itemsets,
            metric,
            metric_threshold
    ):
        """
        Uses itemsets attribute to find rules.
        :param metric: "confidence" or "lift"
        :param metric_threshold: Float, [0, 1]
        :return Rules
        """
        return Rules(association_rules(itemsets, metric=metric, min_threshold=metric_threshold))

    @staticmethod
    def _reduce(
            df,
            column_list,
            column_values_list
    ):
        """
        Reduces a DataFrame to lists, where each list holds the values of the columns listed in column_list.
        :param df: DataFrame
        :param column_list: A list of columns to _reduce to
        :param column_values_list: A list parallel to column_list that lists values to look for in each column
        :return List of Lists
        """
        lists = []  # each element is a row in DataFrame
        for col_i, column in enumerate(column_list):
            df = DataCleaner.filter_invalids(df, column)
            for list_i, column_value in df[column].items():

                # Find all values in multi-response
                legal_values = column_values_list[col_i]
                found_values = []
                for v in legal_values:
                    if v in column_value:
                        found_values.append(v)

                # Merge them into list of sets
                try:
                    lists[list_i].extend(found_values)
                except IndexError:
                    # Haven't made this set yet
                    lists.append(found_values)

        return lists

    def _transform_to_one_hot(
            self,
            itemset_list
    ):
        """
        Converts itemset list into one-hot encoded DataFrame,
        which is required for frequent itemset mining.
        :param itemset_list: A list of lists
        :return DataFrame
        """
        encoder = TransactionEncoder()
        array = encoder.fit(itemset_list).transform(itemset_list)
        df = pd.DataFrame(array)

        # rename columns
        columns = self._parse_columns(encoder.columns_)
        df.rename(columns={k: v for k, v in enumerate(columns)}, inplace=True)

        return df

    @staticmethod
    def _parse_columns(
            columns
    ):
        """
        Remove quotes in column names, because Pandas doesn't like them
        """
        res = []
        for column in columns:
            res.append(column.replace('"', ''))
        return res


class Rules:
    """
    Represents a set of association rules.
    """

    def __init__(
            self,
            df
    ):
        """
        :param df: DataFrame; original table that is always retained
        """
        self._df = df
        self._organized_df = None
        self._sort_by = ["lift"]
        self._sort_ascending = [False]

    @property
    def table(self):
        """
        :return: DataFrame
        """
        return self._df

    def search(
            self,
            one_of,
            location="all",
            use_organized=True
    ):
        """
        Filters out rules that don't match search condition.
        E.g. if one_of=["Chisato", "Hina"], all rules with "Chisato" or "Hina" in antecedents or consequents
        will be returned.

        :param one_of: List; each element is search term, with entire list being a disjunction/OR
        :param location: "antecedents", "consequents", or "all"; where to look for search terms
        :param use_organized: Bool; whether to use organized table or not
        :return: DataFrame with results
        """
        if location not in ["all", "antecedents", "consequents"]:
            raise ValueError("invalid location argument: must be 'all', 'antecedents', or 'consequents'")

        if use_organized and self._organized_df is not None:
            rules = self._organized_df.copy()
        else:
            rules = self._df.copy()

        res = None
        filter_partials = []

        for term in one_of:

            # Do filtering/search of term at specified locations
            if location == "all":
                filter_partials.append(rules[rules["antecedents"].astype(str).str.contains(term)])
                filter_partials.append(rules[rules["consequents"].astype(str).str.contains(term)])
            else:
                filter_partials.append(rules[rules[location].astype(str).str.contains(term)])

            # Union partial filter results to get final result
            if res is not None:
                filter_partials.append(res)
            res = pd.concat(filter_partials).drop_duplicates()
            filter_partials = []

        # Resort with original sort order
        return res.sort_values(by=self._sort_by, ascending=self._sort_ascending)

    def organize(
            self,
            min_antecedents=1,
            min_consequents=1,
            min_rule_length=2,
            max_antecedents=None,
            max_consequents=None,
            max_rule_length=None,
            sort_by=None,
            sort_ascending=None,
    ):
        """
        Filter and sort own DataFrame table, with the intent of making data more readable.
        The new table is saved to its own attribute, while the original is retained.

        :param min_antecedents: Int; min to keep
        :param min_consequents: Int; min to keep
        :param min_rule_length: Int; minimum length of antecedents + consequents to keep
        :param max_antecedents: Int or None; max to keep
        :param max_consequents: Int or None; max to keep
        :param max_rule_length: Int or None; maximum length of antecedents + consequents to keep
        :param sort_by: List of Strings; column names
        :param sort_ascending: List of Bool; parallel to _sort_by and determines sort order of corresponding column
        """

        if sort_by is None:
            sort_by = ["confidence"]
        if sort_ascending is None:
            sort_ascending = ["False"]

        rules = self._df.copy()
        rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
        rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
        rules["rule_len"] = rules.apply(lambda row: row.antecedent_len + row.consequent_len, axis=1)

        # Filter
        filtered = rules[
            (rules["antecedent_len"] >= min_antecedents) &
            (rules["consequent_len"] >= min_consequents) &
            (rules["rule_len"] >= min_rule_length)
            ]
        filtered = filtered[filtered["antecedent_len"] <= max_antecedents] if max_antecedents else filtered
        filtered = filtered[filtered["consequent_len"] <= max_consequents] if max_consequents else filtered
        filtered = filtered[filtered["rule_len"] <= max_rule_length] if max_rule_length else filtered

        # Sort
        self._organized_df = filtered.sort_values(by=sort_by, ascending=sort_ascending)
        self._sort_by = sort_by
        self._sort_ascending = sort_ascending
