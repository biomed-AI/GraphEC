import argparse
from utils import *
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--fasta", type=str, default='./EC_number/data/example.fa')
    args = parser.parse_args()
    Seed_everything(seed=args.seed)
    predict(args=args, seed=args.seed)
