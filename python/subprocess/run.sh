#!/usr/bin/env bash
gcc echo_command_line_arguments.c -o echo_command_line_arguments
python3 test_subprocess_popen.py
