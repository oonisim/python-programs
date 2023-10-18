"""
ML functions based on pandas
"""
import logging
from typing import (
    List,
    Dict,
    Tuple,
)

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import (
    StratifiedShuffleSplit
)
from sklearn.metrics import (
    mean_squared_error
)
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.model_selection import (
    cross_val_score,
    RepeatedKFold,
    GridSearchCV
)
from sklearn.linear_model import (
    Ridge
)
from sklearn.compose import (
    ColumnTransformer
)
from sklearn.pipeline import (
    Pipeline
)


# --------------------------------------------------------------------------------
# python logger
# TODO: have common utility to get logger
# --------------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Data processing
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


# --------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------
def normalize_numeric_categorical_columns(
        dataframe: pd.DataFrame,
        numeric_columns: List[str] = None,
        categorical_columns: List[str] = None,
        normalize_or_standardize: str = "n"
):
    """Normalize the numeric columns and one-hot-encode categorical columns.
    Return (transformed dataframe, fitted pipeline) so that the same normalization
    can be executed with the fitted pipeline.

    Use MinMaxScaler when normalize_or_standardize is N or n, otherwise StandardScaler
    for numerica columns.

    numeric_columns = [
        'age',
        'height',
        'weight',
    ]
    categorical_columns = [
        'nationality',
        'ethnicity',
        'gender',
        'occupation',
    ]

    Args:
        dataframe: dataframe to transform
        numeric_columns: list of numeric column names
        categorical_columns: list of categorical column names
        normalize_or_standardize: specify normalize or standardize

    Returns: (transformed, pipeline)
    """
    assert set(categorical_columns).issubset(set(dataframe.columns))
    assert set(numeric_columns).issubset(set(dataframe.columns))

    if normalize_or_standardize.lower().startswith('n'):
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    numeric_pipeline = Pipeline([
        ('scaler', scaler),
    ])
    categorical_pipeline = Pipeline([
        ('one_hot_encoder', OneHotEncoder()),
    ])
    pipeline = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("category", categorical_pipeline, categorical_columns),
        ],
        remainder='passthrough'
    )

    transformed: pd.DataFrame = pipeline.fit_transform(dataframe)
    return transformed, pipeline
