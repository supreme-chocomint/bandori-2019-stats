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
        self.lists = self._reduce(DataCleaner.prepare_data_frame(tsv_path), [CHARACTERS], [ALL_CHARACTERS])
        self.one_hot_df = self._transform_to_one_hot(self.lists)
        self.itemsets = None
        self.rules = None

    def find_sets(
            self,
            min_frequency=0.01,  # ~25 responses
            remove_single_items=True
    ):
        """
        Finds frequent itemsets.
        The original result of the search is saved as an attribute, and a filtered version
        is returned by default. To get the original, turn off the argument flags or
        access it directly as an object attribute.
        Original result is saved because it is required for rule finding.

        FPGrowth algorithm used instead of Apriori because it can handle min_frequency=0.

        :param min_frequency: Float; threshold occurrence for a set to be considered "frequent"
        :param remove_single_items: Bool; whether to remove sets with one item or not
        """
        self.itemsets = fpgrowth(self.one_hot_df, min_support=min_frequency, use_colnames=True)

        # Remove all sets with one item, because those are just the most popular choices
        if remove_single_items:
            itemsets = self.itemsets.copy()
            itemsets["length"] = itemsets["itemsets"].apply(lambda x: len(x))
            return itemsets[itemsets["length"] > 1].sort_values(by=["support"], ascending=False)
        else:
            return self.itemsets

    def find_rules(
            self,
            metric,
            metric_threshold,
            min_antecedents=1,
            min_consequents=1,
            min_rule_length=2,
            max_antecedents=None,
            max_consequents=None,
            max_rule_length=None,
            sort_by=None,
            sort_ascending=None
    ):
        """
        Uses itemsets attribute to find rules, and optionally filter those rules.
        :param metric: "confidence" or "lift"
        :param metric_threshold: Float, [0, 1]
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

        self.rules = association_rules(self.itemsets, metric=metric, min_threshold=metric_threshold)

        # Make columns for applying filters and sorts
        rules = self.rules.copy()
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
        Remove quotes in column names, because Pandas doesn't like it
        :param columns:
        :return:
        """
        res = []
        for column in columns:
            res.append(column.replace('"', ''))
        return res
