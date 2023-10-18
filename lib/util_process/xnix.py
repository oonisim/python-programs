"""
*NIX process utility
"""
import os
import signal
import logging
from subprocess import (
    Popen,
    PIPE
)
from typing import (
    List,
    Optional
)

from util_logging import (
    get_logger
)

# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------
_logger: logging.Logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------
def preexec_func_sig_ign():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_command(command_line_string: str):
    """
    Run command as a process
    """
    name: str = "run_command()"
    _logger.debug("%s: command line string [%s]", name, command_line_string)

    command_line: List[str] = command_line_string.split()
    proc: Optional = None
    pid: int
    stdout: str
    stderr: str
    exit_code: int

    try:
        # See https://docs.python.org/3/library/subprocess.html#popen-constructor for
        # the kwargs for Popen constructor.
        # We open a pipe to stdin so that the program can die when the pipe is broken
        popen_kwargs = {
            "stdin": PIPE,
            "stdout": PIPE,
            "env": dict(os.environ),
            "preexec_fn": preexec_func_sig_ign,
            "universal_newlines": True,
        }

        # On exit, standard file descriptors are closed, and the process is waited for.
        with Popen(command_line, **popen_kwargs) as proc:
            pid: int = proc.pid
            stdout, stderr = proc.communicate()
            exit_code = proc.returncode

        _logger.debug(
            "%s: pid [%s], exit code [%s], stdout [%s], stderr [%s]",
            pid, exit_code, stdout, stderr
        )
        return pid, exit_code, stdout, stderr

    except OSError as exception:
        _logger.error("%s: failed due to [%s]", name, exception)
        raise RuntimeError(f"{name} failed to run {command_line_string}") from exception

    finally:
        if proc:
            proc.kill()


# --------------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------------
def test_run_command_success():
    """
    Objective:
        Verify the ls command is executed as a process successfully.

    Expected:
        1. exit_code is 0, pid > 0
        2. no stderr output
    """
    pid, exit_code, stdout, stderr = run_command("ls -l -a")

    # --------------------------------------------------------------------------------
    # Test condition #1
    # --------------------------------------------------------------------------------
    assert pid > 0, f"expected pid > 0 got [{pid}]"
    assert exit_code == 0, f"expected exit code 0 got [{exit_code}]"

    # --------------------------------------------------------------------------------
    # Test condition #2
    # --------------------------------------------------------------------------------
    assert not stderr, f"expected empty stderr, got [{stderr}]"


def test_run_command_fail():
    """
    Objective:
        Verify the invalid command fails to run.

    Expected:
        1. exit_code is not 0
    """

    try:
        pid, exit_code, stdout, stderr = run_command("hoge tako ika")
        assert False, \
            f"expected failure, got pid {pid}, exit code {exit_code}, stdout {stdout}, stderr {stderr}"
    except RuntimeError as exception:
        pass
