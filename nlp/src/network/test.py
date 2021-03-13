import logging


def test_log_message(caplog):
    logger = logging.getLogger("test_log_message")

    caplog.set_level(logging.DEBUG, logger="test_log_message")
    logger.debug("foobar")
