import os
import re

class MockPool:
    class ApplyResult:
        def __init__(self, x) -> None:
            self.x = x
        def get(self):
            return self.x
    def apply_async(self, f, args, callback=None):
        r = f(*args)
        if callback is not None:
            callback(r)
        return MockPool.ApplyResult(r)

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

    
