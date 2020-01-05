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

    def mine_favorite_characters(self):
        """
        Mines for rules regarding all favorite characters.
        Resulting rules are restricted to those with at most 2 consequents (to avoid too-specific data)
        and confidence of at least 30% (making confidence too high makes rules too specific to individual people).
        Result sorted by antecedent length (since single-item predictors probably more interesting) and lift.
        :return:
        """
        raw_itemsets = self._generate_frequent_itemsets([CHARACTERS], [ALL_CHARACTERS])
        raw_rules = self._find_rules(raw_itemsets, metric="confidence", metric_threshold=0.3)
        return self._organize_rules(
            raw_rules, max_consequents=2, sort_by=["antecedent_len", "lift"], sort_ascending=[True, False]
        )

    def mine_favorite_band_members(self):
        """
        Mines for rules regarding favorite character in each band.
        All confidences are >30% and rules are sorted by lift.
        """
        raw_itemsets = self._generate_frequent_itemsets(
            [CHARACTER_POPIPA,
             CHARACTER_AFTERGLOW,
             CHARACTER_GURIGURI,
             CHARACTER_HHW,
             CHARACTER_PASUPARE,
             CHARACTER_RAS,
             CHARACTER_ROSELIA],
            [[ALL_CHARACTERS] * 7][0]  # list of 7 CHARACTER lists
        )
        raw_rules = self._find_rules(raw_itemsets, metric="confidence", metric_threshold=0.3)
        return self._organize_rules(raw_rules, sort_by=["lift"], sort_ascending=[False])

    def _generate_frequent_itemsets(
            self,
            columns,
            column_values,
            min_frequency=0.01
    ):
        """
        Uses the values of columns to generate frequent itemsets for association rule mining.
        :param columns: List of column names to use
        :param column_values: List; each element is itself a list, holding the legal values of the column
        :param min_frequency: threshold frequency for set to be considered "frequent"
        """
        lists = self._reduce(self.df, columns, column_values)
        one_hot_df = self._transform_to_one_hot(lists)
        return self._find_sets(one_hot_df, min_frequency=min_frequency)

    @staticmethod
    def _find_sets(
            one_hot_df,
            min_frequency=0.01  # ~25 responses
    ):
        """
        Finds frequent itemsets.
        FPGrowth algorithm used instead of Apriori because it can handle min_frequency=0.
        :param min_frequency: Float; threshold occurrence for a set to be considered "frequent"
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
        """
        # Remove all sets with one item, because those are just the most popular choices
        if remove_single_items:
            itemsets = itemsets.copy()
            itemsets["length"] = itemsets["itemsets"].apply(lambda x: len(x))
            return itemsets[itemsets["length"] > 1]

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
        """
        return association_rules(itemsets, metric=metric, min_threshold=metric_threshold)

    @staticmethod
    def _organize_rules(
            rules,
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
        Filter and sort.
        :param rules: DataFrame
        :param min_antecedents: Int; min to keep
        :param min_consequents: Int; min to keep
        :param min_rule_length: Int; minimum length of antecedents + consequents to keep
        :param max_antecedents: Int or None; max to keep
        :param max_consequents: Int or None; max to keep
        :param max_rule_length: Int or None; maximum length of antecedents + consequents to keep
        :param sort_by: List of Strings; column names
        :param sort_ascending: List of Bool; parallel to sort_by and determines sort order of corresponding column
        """

        if sort_by is None:
            sort_by = ["confidence"]
        if sort_ascending is None:
            sort_ascending = ["False"]

        # Make columns for applying filters and sorts
        rules = rules.copy()
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
        return filtered.sort_values(by=sort_by, ascending=sort_ascending)

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
