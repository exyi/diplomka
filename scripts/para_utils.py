import multiprocessing.managers
import multiprocessing.shared_memory
import os
import re
import threading
import multiprocessing.pool
from typing import Any, Callable, Generic, Iterable, TypeVar
import numpy as np

T = TypeVar('T')
S = TypeVar('S')

class MockPool:
    class ApplyResult(Generic[T]):
        def __init__(self, x: T) -> None:
            self.x = x
        def get(self, timeout=None) -> T:
            return self.x
        def ready(self) -> bool:
            return True
        def successful(self) -> bool:
            return True
        
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

    @staticmethod
    def make_process_pool(n: int):
        if n == 1:
            return MockPool()
        else:
            return multiprocessing.pool.Pool(n)
    @staticmethod
    def make_thread_pool(n: int):
        if n == 1:
            return MockPool()
        else:
            return multiprocessing.pool.ThreadPool(n)

    def apply_async(self, f, args, callback=None, error_callback=None):
        r = f(*args)
        if callback is not None:
            callback(r)
        return MockPool.ApplyResult(r)
    
    def map(self, func: Callable[[S], T], iterable: Iterable[S], chunksize=None) -> list[T]:
        return [func(x) for x in iterable]

    def map_async(self, func: Callable[[S], T], iterable: Iterable[S], chunksize=None):
        return MockPool.ApplyResult(self.map(func, iterable, chunksize))
    
    def starmap(self, func: Callable[..., T], iterable: Iterable[Iterable[Any]], chunksize: int | None = None) -> list[T]:
        return [func(*x) for x in iterable]
    
    def starmap_async(self, func: Callable[..., T], iterable: Iterable[Iterable[Any]], chunksize: int | None = None):
        return MockPool.ApplyResult(self.starmap(func, iterable, chunksize))

def batched_map(pool: MockPool | multiprocessing.pool.Pool | multiprocessing.pool.ThreadPool | None, f, arrays: np.ndarray | tuple[np.ndarray, ...], more_args: tuple, batch_size:int = 100, axis=0):
    if not isinstance(arrays, tuple):
        arrays = (arrays,)
    assert len(set([a.shape[axis] for a in arrays])) == 1, "All arrays must have the same length along the specified"

    if pool is None or arrays[0].shape[axis] <= batch_size * 2:
        return [ f(*arrays, *more_args) ]

    results = []
    batches = [
        np.array_split(a, a.shape[0] // batch_size, axis=axis)
        for a in arrays
    ]
    for batch in zip(*batches):
        results.append(pool.apply_async(f, (*batch, *more_args)))
    return [ r.get() for r in results ]


def _make_shmem(size, mgr: multiprocessing.managers.SharedMemoryManager | None = None):
    if mgr is None:
        return multiprocessing.shared_memory.SharedMemory(create=True, size=size)
    else:
        return mgr.SharedMemory(size=size)


class SharedMemoryInstance:
    def __init__(self, mem: multiprocessing.shared_memory.SharedMemory, own: bool, dim, dtype: np.dtype):
        self.mem = mem
        self.dim = dim
        self.dtype = dtype
        self.own = own
        self.array = np.ndarray(dim, dtype=dtype, buffer=mem.buf)
        self.handle = SharedMemoryHandle(mem.name, dim, dtype)

    @staticmethod
    def create_arr(array: np.ndarray, mgr: multiprocessing.managers.SharedMemoryManager | None = None):
        size = array.size * array.itemsize
        mem = _make_shmem(size, mgr)
        arr2 = np.ndarray(array.shape, dtype=array.dtype, buffer=mem.buf)
        arr2[...] = array
        return SharedMemoryInstance(mem, True, array.shape, array.dtype)
    
    @staticmethod
    def create_zeros(shape, dtype, mgr: multiprocessing.managers.SharedMemoryManager | None = None):
        size = np.prod(shape) * np.dtype(dtype).itemsize
        mem = _make_shmem(size, mgr)
        arr = np.ndarray(shape, dtype=dtype, buffer=mem.buf)
        arr[...] = 0
        return SharedMemoryInstance(mem, True, shape, dtype)

    def __enter__(self):
        return self.array

    def __exit__(self, *args):
        self.array = None
        if self.own:
            self.mem.unlink()
        self.mem.close()


class SharedMemoryHandle:
    def __init__(self, name, dim, dtype):
        self.name = name
        self.dim = dim
        self.dtype = dtype
    
    def instantiate(self):
        mem = multiprocessing.shared_memory.SharedMemory(name=self.name)
        return SharedMemoryInstance(mem, False, self.dim, self.dtype)

def parse_thread_count(x: str):
    def t_round(x):
        if x <= 0:
            raise ValueError(f"Thread count cannot be negative ({x})")
        if x <= 1:
            return 1
        return round(x)
    x = x.strip()
    cpus = len(os.sched_getaffinity(0)) # returns the number of assigned cores from taskset, while os.cpu_count() returns all cores on the machine (funny way how to run out of memory in MetaCentrum...)
    if x == "0" or x == "all":
        return cpus
    if re.fullmatch(r"\d+", x):
        return int(x)
    if re.fullmatch(r"[+]\d+", x):
        return cpus + int(x[1:])
    if re.fullmatch(r"-\d+", x):
        return cpus - int(x[1:])
    if re.fullmatch(r"\d+([.]\d+)?%", x):
        return t_round(cpus * float(x[:-1]) / 100)
    raise ValueError(f"Cannot parse thread count '{x}'!")

    
