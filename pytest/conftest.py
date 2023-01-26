def pytest_addoption(parser):
    parser.addoption(
        "--my-option", action="store", default="default_value_01", help="my option: default_value_01"
    )
    parser.addoption(
        "--data-directory", action="store", default="default_directory", help="path to data directory"
    )
