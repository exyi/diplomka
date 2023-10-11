#!/usr/bin/env python3

import os, sys, time
from model import csv_loader, hyperparameters

def init_argparser(parser):
    parser.add_argument('--train_set', type=str, help='Path to tfrecord file with training data. The *.tfrecord.meta.json must be in the same directory')
    parser.add_argument('--val_set', type=str, help='Path to tfrecord file with validation data. The *.tfrecord.meta.json must be in the same directory')
    hyperparameters.add_parser_args(parser, hyperparameters.Hyperparams)

def main_argv(argv: list[str]):
    import argparse
    parser = argparse.ArgumentParser(description='Train NtC network using PyTorch ignite')
    init_argparser(parser)
    args = parser.parse_args(argv)
    main_args(args)

def main_args(args):
    p = hyperparameters.Hyperparams.from_args(args)

    from . import training, torchutils
    print(f"device: {torchutils.device}    logdir: {args.logdir}")
    print(p)
    training.train(args.train_set, args.val_set, p, args.logdir)

if __name__ == '__main__':
    print("Use the launch.py script using `torch-train` subcommand.")
    exit(1)
