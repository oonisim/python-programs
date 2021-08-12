#!/usr/bin/python
"""
Requirements:
Parse the csv files 'data.csv'. Print out the:
a) row (game) count
b) average of "Game Length"
Skip malformed game data
Example input
"Game Number", "Game Length"
1, 30
2, 29
3, 31
4, 16
5, 24
6, 29
7, 28
8, 117
from thousands to millions of rows
"""
import csv
import numpy


def _to_Num(s):
    try:
        return int(s)
    except Exception as e:
        try:
            return float(s)
        except ValueError:
            return 0


def _is_ints(s):
    return [_to_Num(x) for x in s]


def stream(path):
    with open(path, 'rt+') as f:
        # Check if there is a header row
        has_header = csv.Sniffer().has_header(f.read(1024))

        # Reset the FP
        f.seek(0)
        lines = csv.reader(f, skipinitialspace=True)

        # Skip header if exists
        if has_header:
            next(lines)

        # yield rows
        for row in lines:
            if len(row) == 2:
                yield _is_ints(row)


avg_length = 0
count = 0

for game_number, game_length in stream('data.csv'):
    try:
        x = _to_Num(game_length)
        count += 1
        avg_length += (x - avg_length) / count
    except Exception as e:  # Could not cast to int. Row is malformed.
        print(e)
        continue

print(f"Games processed: {count}")  # Skip the header
print(f"Average game length: {avg_length}")
print("DONE")