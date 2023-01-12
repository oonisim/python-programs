"""
ML functions based on pandas
"""
import logging
from typing import (
    Tuple,
)

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import (
    StratifiedShuffleSplit
)


# --------------------------------------------------------------------------------
# python logger
# TODO: have common utility to get logger
# --------------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
def stratified_shuffle_split_into_train_test(
        dataframe: pd.DataFrame,
        column_name: str,
        n_splits: int = 1, 
        test_size: float = 0.2,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Shuffle and split data set into train and test data sets in the way
    the proportions among strata of the specified column data is retained.

    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

    Args:
        dataframe: pandas dataframe to shuffle/split
        column_name: column, the data of which to retain  the strata
        n_splits: number of k-fold splits. 1 to just shuffle
        test_size: float between 0.0 and 1.0 as the proportion of the test dataset,
                   or int to  represents the absolute number of test samples.
        random_state: Pass an int for reproducible output across multiple function calls
    Returns: (train, test) of pandas dataframes
    """
    assert len(dataframe) > 0, "invalid dataframe wit no data"
    assert column_name in dataframe.columns, "invalid column name"

    train_data_set: pd.DataFrame
    test_data_set: pd.DataFrame

    split = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=42
    )
    for train_index, test_index in split.split(dataframe, dataframe[column_name]):
        train_data_set = dataframe.loc[train_index]
        test_data_set = dataframe.loc[test_index]

    logger.debug(
        "strata [\n%s\n]", test_data_set[column_name].value_counts() / len(test_data_set)
    )
    return train_data_set, test_data_set
