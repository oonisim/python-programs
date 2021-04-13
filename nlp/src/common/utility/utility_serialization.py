import os
import pickle
from . utility_file import (
    is_path_creatable
)


def serialize(path: str, state: object):
    assert is_path_creatable(path), f"Cannot create {path}."
    with open(path, 'wb') as f:
        pickle.dump(state, f)


def deserialize(path: str):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            state = pickle.load(f)

        assert state is not None
        return state
    else:
        raise RuntimeError(f"Path {path} does not exist.")
