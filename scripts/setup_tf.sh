# Optional argument: output directory, otherwise it's $SCRATCH

directory="$1"
if test -z "$directory"; then
	directory="$SCRATCH"
fi
if which nvidia-smi; then
	CUDA=1
else
	CUDA=0
fi

module add zstd/1.5.2-gcc-10.2.1-jj6bl5j
module add python/3.10.4-gcc-8.3.0-ovkjwzd
if test $CUDA -eq 1; then
	module add cuda/cuda-11.2.0-intel-19.0.4-tn4edsz
fi

pushd "$SCRATCH"

tar -I zstd -xf /storage/brno2/home/exyi/tf-venv.tar.zst

VENV_DIR="$SCRATCH/tf-venv"
real_python="$(which python3.10)"
rm "$VENV_DIR/bin/python3.10"
ln -s "$real_python" "$VENV_DIR/bin/python3.10"
py="$VENV_DIR/bin/python"

if test $CUDA -eq 1; then
	tar -I xz -xf /storage/praha1/home/exyi/cudnnlibs/cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz
	NVIDIA_DIR="$(dirname "$(dirname "$(which nvcc)")")"
	export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVIDIA_DIR"
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SCRATCH/cudnn-linux-x86_64-8.9.1.23_cuda11-archive/lib
	export CPATH=$CPATH:$SCRATCH/cudnn-linux-x86_64-8.9.1.23_cuda11-archive/include
fi

popd
