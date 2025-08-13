import os
import sys
import argparse
from model import *
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    Seed_everything(seed=args.seed)
    model_class = GraphEC_AS
    torch.cuda.set_device(args.gpu)
    predict(model_class, args)
