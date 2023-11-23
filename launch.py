#!/usr/bin/env python3

import argparse, os, sys, time
os.environ["NTCNET_INTERNAL_NO_HEAVY_IMPORTS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import model.tf.training as training_tf
import model.torch.training_args as training_torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default=None, help='Path for saving Tensorboard logs and other outputs')
parser.add_argument('--logname', type=str, default=None, help='Name of subdirectory to create in logdir')

subparsers = parser.add_subparsers(dest="command")
training_tf.init_argparser(subparsers.add_parser("tf-train", help=training_tf.help, usage=training_tf.usage))
torch_train_cmd = subparsers.add_parser("torch-train", help="Train using PyTorch")
training_torch.init_argparser(torch_train_cmd)
# pass any additional arguments to the script
# torch_train_cmd.add_argument('torch_args', nargs=argparse.REMAINDER)

test_cmd = subparsers.add_parser("test", help="Run unit tests")
test_cmd.add_argument('test_args', nargs=argparse.REMAINDER)

def main(argv):
    args = parser.parse_args(argv)

    logdir = args.logdir
    if logdir is None:
        if args.command == 'torch-train':
            logdir = "pt-logs"
        else:
            logdir = "tf-logs"
    subdir = os.path.exists(logdir) or bool(args.logname)
    if subdir:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.join(logdir, f"{timestamp}_{args.logname or 'test'}")
        args.logdir = logdir
    os.makedirs(logdir, exist_ok=True)
    import model.utils
    model.utils.set_logdir(logdir)

    os.environ["NTCNET_INTERNAL_NO_HEAVY_IMPORTS"] = ""
    if args.command == "tf-train":
        training_tf.main_args(args)
    elif args.command == "torch-train":
        training_torch.main_args(args)
    elif args.command == "test":
        import unittest
        import model.torch.tests
        unittest.main(module=model.torch.tests, argv=[ argv[0], *trimdoubledash(args.test_args) ])
    else:
        parser.print_help()
        sys.exit(1)

def trimdoubledash(argv):
    if len(argv) > 0 and argv[0] == "--":
        return argv[1:]
    return argv
if __name__ == "__main__":
    main(sys.argv[1:])
