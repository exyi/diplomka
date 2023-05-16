#!/usr/bin/env python3

import argparse
import os, sys, math
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", help="gpu count", default=1, type=int)
parser.add_argument("--gpu_mem", help="minimal gpu memory", default="2gb", type=str)
parser.add_argument("--gpu_cap", help="minimal gpu capability (cuda61, cuda75, ...)", default=None, type=str)
parser.add_argument("--fp16", help="use fp16", action="store_true")
parser.add_argument("--walltime", help="walltime in hours", default=8, type=float)
parser.add_argument("--cpu", help="cpu count", default=6, type=int)
parser.add_argument("--training_set", help="training set to use", default=None, type=str)
parser.add_argument("--val_set", help="validation set to use", default=None, type=str)
parser.add_argument("--script", help="script to run", default="scripts/metajobs/training.tf.py")
parser.add_argument("name", help="job name", default="bjob")
parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments to script")

args = parser.parse_args()

variables = { "TBLOG": args.name }
if args.training_set is not None:
	variables["TRAINING_SET"] = args.training_set
if args.val_set is not None:
	variables["VAL_SET"] = args.val_set
lflags = [ "select=1", f"ncpus={args.cpu}", f"scratch_local=12gb", f"mem=16gb" ]
qsubargs = [
	"-N", f"ntcnet-{os.path.basename(args.script)}-{args.name}",
]
if args.gpu > 0:
	if args.walltime > 24:
		qsubargs += [ "-q", f"gpu_long" ]
	else:
		qsubargs += [ "-q", f"gpu" ]
	lflags += [ f"ngpus={args.gpu}", f"gpu_mem={args.gpu_mem}" ]
	if args.gpu_cap is not None:
		lflags += [ f"gpu_cap={args.gpu_cap}" ]
	elif args.fp16:
		lflags += [ f"gpu_cap=cuda70" ]

qsubargs += ["-l", ":".join(lflags)]

qsubargs += [ "-l", f"walltime={math.floor(args.walltime)}:{math.floor(args.walltime % 1 * 60):02}:00" ]

qsubargs += [ "-v", ",".join([ f"{k}='{v}'" for k, v in variables.items() ]) ]

print("qsub", *qsubargs, args.script, *args.args)
subprocess.run(["qsub", *qsubargs, "--", args.script, *args.args])