"""

"""
from util_file import (
    get_file_suffix,
    list_files_in_directory,
    get_dir_name,
    get_path_to_this_py_script,
)


def test_get_file_suffix():
    """
    Objective:
        Verify the function returns the last suffix only if exists.

    Expected:
        1. path/to/dir/a.b.c returns .c
        2. path/to/dir/a returns '' (empty string)
    """
    # --------------------------------------------------------------------------------
    # Test condition #1: path/to/dir/a.b.c returns .c
    # --------------------------------------------------------------------------------
    path01 = "path/to/dir/file.tar.gz"
    assert get_file_suffix(path01) == ".gz"

    # --------------------------------------------------------------------------------
    # Test condition #1: path/to/dir/a returns '' (empty string)
    # --------------------------------------------------------------------------------
    path02 = "path/to/dir/file"
    assert get_file_suffix(path02) == ""

    path03 = "path/to/dir/"
    assert get_file_suffix(path03) == ""


def test_list_files_in_directory():
    """
    Objective:
        Verify the function returns a list of file matching the pattern.

    Expected:
        1. (path, "*.py") returns the .py file only
    """
    # --------------------------------------------------------------------------------
    # Test condition #1. (path, "*.py") returns the .py file only
    # --------------------------------------------------------------------------------
    path: str = get_dir_name(get_path_to_this_py_script())
    for file in list_files_in_directory(path=path, pattern=r'*\.py'):
        assert get_file_suffix(file) == ".py"

    pattern = r'\#\$\.\%'
    for file in list_files_in_directory(path=path, pattern=pattern):
        assert False, f"should not match the pattern [{pattern}]"

