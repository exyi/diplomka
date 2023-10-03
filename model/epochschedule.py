
from typing import Any, Optional, TypeVar, Union
import math, itertools
import numpy as np

TPr = TypeVar('TPr', bound=Union[int, float, bool, str])
def parse_epoch_schedule(input_str: str, num_epochs: int, tt: type[TPr] = int) -> list[tuple[int, TPr]]:
    splits = [ s.split('*') for s in input_str.split(";") ]
    non_x_sum = sum([ int(epochs) for epochs, _ in splits if epochs != "x" ])
    x_count = sum([ 1 for epochs, _ in splits if epochs == "x" ])
    x_min = (num_epochs - non_x_sum) // x_count
    x_mod = (num_epochs - non_x_sum) % x_count
    if x_min < 0:
        raise ValueError(f"Can't fit {num_epochs} epochs into {input_str}")
    x = [ x_min + (1 if i < x_mod else 0) for i in range(x_count) ]
    x.reverse()
    return [ ((int(epochs) if epochs != "x" else x.pop()), tt(value)) for epochs, value in splits ]

T = TypeVar('T')
def schedule_to_list(schedule: list[tuple[int, T]]) -> list[T]:
    return list(itertools.chain.from_iterable([ [ value ] * epochs for epochs, value in schedule ]))

def schedule_from_list(schedule: list[T]) -> list[tuple[int, T]]:
    def core():
        last_value: T = object() # type: ignore
        count = 0
        for value in schedule:
            if value != last_value:
                if count > 0:
                    yield count, last_value
                last_value = value
                count = 1
            else:
                count += 1
        if count > 0:
            yield count, last_value
    return list(core())

def get_batch_size_from_maxlen(seq_len_schedule: list[tuple[int, int]], base_batch_size: int, max_batch_size: Optional[int] = None):
    base_seq_len = min([ seq_len for _, seq_len in seq_len_schedule ])
    def core():
        for (epochs, seq_len) in seq_len_schedule:
            batch_size = math.ceil(base_batch_size * base_seq_len / seq_len)
            if max_batch_size is not None and batch_size > max_batch_size:
                batch_size = max_batch_size
            yield epochs, batch_size
    return list(core())

def get_step_count(batch_size_schedule: list[tuple[int, int]], ds_cardinality) -> int:
    return sum([ epochs * math.ceil(ds_cardinality / batch_size) for epochs, batch_size in batch_size_schedule ])
