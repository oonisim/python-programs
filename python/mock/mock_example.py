"""Module to test mock"""
import argparse
import json
import logging
import pathlib
import sys
from typing import (
    Dict,
    Any
)
from unittest import (
    mock
)


def get_args():
    """Get command line arguments"""
    try:
        parser = argparse.ArgumentParser(description=__name__)
        parser.add_argument(
            '-m', '--model_name', type=str, required=True,
            help='name of the model'
        )
        parser.add_argument(
            '-d', '--data-dir', type=str, required=True,
            help='path to the data directory'
        )
        _args, _ = parser.parse_known_args(sys.argv[1:])
        _args = vars(_args)
        return _args

    except SystemExit as error:
        logging.fatal("argparse failed due to %s", error)
        raise error


def main(args: Dict[str, Any]):
    print(json.dumps(args, indent=4, default=str))


if __name__ == "__main__":
    script_name: str = pathlib.Path(__file__).name
    mock_args = [
        script_name,
        "-m", 'sentence-transformers/gtr-t5-large',
        '-d', '/tmp/model',
    ]
    with mock.patch('sys.argv', mock_args):
        main(get_args())




