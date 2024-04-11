import os
import sys
import argparse
from model import *
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    Seed_everything(seed=args.seed)
    model_class = GraphEC_AS
    predict(model_class, args)
