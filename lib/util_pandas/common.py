"""

"""
import logging
import pandas as pd


def stream(df: pd.DataFrame, num: int):
    """Stream num rows as a batch from the dataframe
    Args:
        df: dataframe to stream
        num: number of rows to stream as a batch
    Yields: A dataframe with num rows
    """
    assert num > 0
    assert len(df) > 0
    logging.debug(f"split(): splitting {len(df)} df into {num} assignments.")

    # Total size of the df
    total = len(df)

    # Each assignment has 'quota' size which can be zero if total < number of assignments.
    quota = int(total / num)

    # Left over after each assignment takes its 'quota'
    residual = total % num

    start: int = 0
    while start < total:
        # As long as residual is there, each assignment has (quota + 1) as its df.
        if residual > 0:
            size = quota + 1
            residual -= 1
        else:
            size = quota

        end: int = start + size
        yield df[start: min(end, total)]

        start = end
        end += size
