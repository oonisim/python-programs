import argparse

parser = argparse.ArgumentParser(description='argparse test program')
parser.add_argument(
    '--list', type=str.upper, nargs='+',
    help="list argument"
)
parser.add_argument(
    '-b', '--boolean', action="store_true",
    help='boolean argument'
)
args = vars(parser.parse_args())
for k, v in args.items():
    print(f"{k}:{v}")
