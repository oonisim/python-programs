import argparse


def parse_args():
    # --------------------------------------------------------------------------------
    # https://docs.python.org/dev/library/argparse.html#dest
    # --------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="model_dir")
    args = parser.parse_args()
    return args


args = parse_args()
for key, value in vars(args).items():
    print(f"{key}:{value}")
