#!/usr/bin/env python3

import argparse, os, sys
os.environ["NTCNET_INTERNAL_NO_HEAVY_IMPORTS"] = "1"

import model.tf.training as training_tf

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")
training_tf.init_argparser(subparsers.add_parser("tf-train", help=training_tf.help, usage=training_tf.usage))
torch = subparsers.add_parser("torch-train", help="Train using PyTorch")
# pass any additional arguments to the script
torch.add_argument('torch_args', nargs=argparse.REMAINDER)

def main(argv):
    args = parser.parse_args(argv)

    if hasattr(args, "logdir") and args.logdir is not None:
        import model.utils
        model.utils.set_logdir(args.logdir)

    os.environ["NTCNET_INTERNAL_NO_HEAVY_IMPORTS"] = ""
    if args.command == "tf-train":
        training_tf.main_args(args)
    elif args.command == "torch-train":
        from model.torch import training as training_torch_core
        print("Running pytorch trainer with args:", args.torch_args)
        training_torch_core.main(trimdoubledash(args.torch_args))
    else:
        parser.print_help()
        sys.exit(1)

def trimdoubledash(argv):
    if len(argv) > 0 and argv[0] == "--":
        return argv[1:]
    return argv
if __name__ == "__main__":
    main(sys.argv[1:])
