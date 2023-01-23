"""Common function module
"""
import argparse
from typing import (
    Dict,
    Any,
)

from util_file import (
    is_file,
    is_dir,
    is_path_creatable
)

# --------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------
ARG_LOG_LEVEL = "log_level"
ARG_SOURCE_DIR = "source_directory"
ARG_SOURCE_FILE = "source_filename"
ARG_TARGET_DIR = "target_directory"
ARG_TARGET_FILE = "target_filename"


def parse_commandline_arguments() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='ETL program')
    parser.add_argument(
        '-s', '--source-directory', type=str, required=True,
        help='path to source image directory'
    )
    parser.add_argument(
        '-f', '--source-filename', type=str, required=False,
        help='source filename'
    )
    parser.add_argument(
        '-d', '--target-directory', type=str, required=True,
        help='path to destination directory to safe the transformed images'
    )
    parser.add_argument(
        '-t', '--target-filename', type=str, required=False,
        help='target filename'
    )
    parser.add_argument(
        '-l', '--log-level', type=int, choices=[10, 20, 30, 40], required=False,
        help='specify the logging level (10 for INFO)',
    )
    args = vars(parser.parse_args())
    return args


def process_commandline_arguments(args: Dict[str, Any]) -> Dict[str, Any]:
    """Process command line arguments
    Args:
        args: dictionary of command line arguments
    Returns: args
    Raises: RuntimeError for invalid conditions e.g. directory does not exit.
    """
    source_directory: str = args[ARG_SOURCE_DIR]
    if not is_dir(source_directory):
        raise RuntimeError(f"source directory [{source_directory}] does not exist.")

    target_directory: str = args[ARG_TARGET_DIR]
    if not (is_dir(target_directory) or is_path_creatable(target_directory)) or is_file(target_directory):
        raise RuntimeError(f"target directory [{target_directory}] is file or cannot create.")

    return args
