import argparse

parser = argparse.ArgumentParser(description='argparse test program')
parser.add_argument('-y', '--year', type=int, required=False, help='specify the target year')
parser.add_argument('-q', '--qtr', type=int, choices=[1, 2, 3, 4], required=False, help='specify the target quarter', )
parser.add_argument(
    '-l', '--log_level', type=int, choices=[10, 20, 30, 40], required=False,
    help='specify the logging level (10 for INFO)',
)
parser.add_argument(
    '-s', '--source-directory', type=str, required=True,
    help='path to source image directory'
)

args, unknownargs = parser.parse_known_args()
args = vars(args)
print(type(args))

print(unknownargs)


print(args.items())
print(f"--year is {args['year']}")
print(f"--qrt is {args['qtr']}")
print(f"--log-level is {args['log_level']}")

print(f"--source-directory is {args['source_directory']}")
print(f"args.get('log_legvel', None): {args.get('log_legvel', None)}")