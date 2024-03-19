"""Module to download Huggingface model and tokenizer"""
import argparse
import json
import logging
import sys
from typing import (
    Dict,
    Any
)
from unittest import mock


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
        parser.add_argument(
            '-u', '--uid', type=int, required=False,
            help='user id', default=-1
        )
        parser.add_argument(
            '-l', '--log_level', type=int, required=False,
            choices=[10, 20, 30, 40], default=logging.INFO,
            help='specify the logging level (10 for INFO)',
        )
        parser.add_argument(
            '--initialize', required=False, action='store_true',
            help='initialize flag.'
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
    DEBUG: bool = False
    if DEBUG:
        args = [
            "dummy_script_name",
            "-d",
            "path/to/dir",
            # "-u", "100",
            "-m",
            "model_name"
        ]
        try:
            with mock.patch("sys.argv", args):
                print(sys.argv[1:])
                main(get_args())
        except SystemExit as error:
            print(error)
            pass
    else:
        main(get_args())
