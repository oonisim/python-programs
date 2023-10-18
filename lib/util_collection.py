"""Module for collection (list, set, dict) utilities
"""
from typing import (
    List,
    Callable,
    Any,
)


# --------------------------------------------------------------------------------
# Sort:
# sorted(collection, key=func) sorts the collection according to the values from
# the key function. func will get each row in the collection.
# See:
# https://docs.python.org/3/library/functions.html#sorted
# https://stackoverflow.com/questions/4110665
#
# Alternative is jq bindings e.g. https://github.com/doloopwhile/pyjq.
# --------------------------------------------------------------------------------
def sort_list_of_records_at_nth_element(
        x: List[List[Any]], position: int, f: Callable = lambda x: x, reverse: bool = False
) -> List[List[Any]]:
    """Sort a list of records (record is another list) with i-th element of the record.

    Use e.g. sort a list of record with elements (liability, balance, credit) by balance.
    x = [
        # balance, credit, loan
        [1, 3, 5],
        [5, 3, 1],
        [1, 2, 3]
    ]
    sorted(x, key=lambda row: row[2])
    -----
    [
        [5, 3, 1],
        [1, 2, 3],
        [1, 3, 5]
    ]

    Args:
        x: List of records
        position: i-th position in the record to sort with
        f: function to convert the n-th element
        reverse: flag to reverse sort
    Returns: List of records sorted with the element at the position
    """
    assert isinstance(x, list) and len(x) > 0 and isinstance(x[0], list), "Invalid x"
    assert 0 <= position < len(x[0]), \
        "invalid position [%s] for list length [%s]" % (position, len(x[0]))

    # --------------------------------------------------------------------------------
    # 'f' corresponds with e.g. 'int' function below.
    # https://stackoverflow.com/a/17555237
    # in this method, the integers are lexicographically compared.
    # Thus, '5' will be larger than '20'. If an integer comparison is to be made,
    # key=lambda x: int(x[3]) should be used.
    # --------------------------------------------------------------------------------
    x.sort(key=lambda record: f(record[position]), reverse=reverse)
    return x


