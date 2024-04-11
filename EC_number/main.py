import argparse
from utils import *
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    Seed_everything(seed=args.seed)
    predict(args=args, seed=args.seed)
