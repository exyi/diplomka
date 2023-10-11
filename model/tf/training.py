import argparse, sys
from model import hyperparameters

def init_argparser(parser):
    parser.add_argument('--load_model', type=str, help='Start from the specified model. Other provided hyperparameters might be ignored if they affect the model architecture, not the training process')
    parser.add_argument('--train_set', type=str, help='Path to tfrecord file with training data. The *.tfrecord.meta.json must be in the same directory')
    parser.add_argument('--val_set', type=str, help='Path to tfrecord file with validation data. The *.tfrecord.meta.json must be in the same directory')
    parser.add_argument('--eager', action="store_true", help='Run in eager mode', default=False)
    parser.add_argument('--profile', type=str, help='Run tensorflow profiler. The value specified for which batches the profiler should be run (for example 10,20 for 10..20)', default=False)
    parser.add_argument('--fp16', action="store_true", help='Run in (mixed) Float16 mode. By default, float32 is used', default=False)
    parser.add_argument('--bfp16', action="store_true", help='Run in (mixed) BFloat16 mode', default=False)
    hyperparameters.add_parser_args(parser)

help = """Train NtCnet using tensorflow"""

usage = """
"""

def main_args(args):
    from . import training_core
    training_core.main_args(args)

def main(argv):
    parser = argparse.ArgumentParser()
    init_argparser(parser)
    args = parser.parse_args(argv)
    main_args(args)

if __name__ == '__main__':
    main(sys.argv[1:])
