
import time
from typing import Callable, Dict, Literal, Optional, TypeVar, Union

module_load_time = time.time()

TResult = TypeVar("TResult")
def retry_on_error(f: Callable[[], TResult], max_retries = 10, sleep_time = 0.1, max_wait_time: Optional[float] = None) -> TResult:
    if max_wait_time is None:
        process_runtime = time.time() - module_load_time
        if process_runtime < 10:
            max_wait_time = 0.3 # don't slow down initial failure
            max_retries = min(4, max_retries)
        elif process_runtime < 120:
            max_wait_time = 10.0
        else:
            max_wait_time = 60.0

    total_wait_time = 0
    for i in range(max_retries):
        try:
            return f()
        except Exception as e:
            if i == max_retries - 1:
                raise e

            print(f"Retrying error: {e}")
            if i == 0:
                continue
            sleep_time = min(sleep_time, (max_wait_time - total_wait_time) / (max_retries - i))
            if sleep_time > 0:
                time.sleep(sleep_time)
                total_wait_time += sleep_time
                sleep_time *= 2

    assert False

