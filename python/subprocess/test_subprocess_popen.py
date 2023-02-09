import os
import pathlib
import signal
from subprocess import (
    Popen,
    PIPE
)
from typing import (
    List,
    Optional
)


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def preexec_func():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_command_line():
    """
    """
    command_line: List[str] = [f'./echo_command_line_arguments', 'tako', 'ika', 'bin']
    env: dict = dict(os.environ)
    proc: Optional = None

    print(f"\nrun command line {command_line}")
    try:
        # See https://docs.python.org/3/library/subprocess.html#popen-constructor for
        # the kwargs for Popen constructor.
        # We open a pipe to stdin so that the program can die when the pipe is broken
        popen_kwargs = {
            "stdin": PIPE,
            "stdout": PIPE,
            "env": env,
            "preexec_fn": preexec_func,
            "universal_newlines": True,
        }

        # On exit, standard file descriptors are closed, and the process is waited for.
        with Popen(command_line, **popen_kwargs) as proc:
            stdout, stderr = proc.communicate()

            print("-"*80)
            print(f"pyspark started with pid {proc.pid}")
            print(f"stdout [{stdout.rstrip()}]")
            print(f"exit code [{proc.returncode}]")

    finally:
        if proc:
            proc.kill()


if __name__ == "__main__":
    os.chdir(str(pathlib.Path(__file__).parent))
    run_command_line()
