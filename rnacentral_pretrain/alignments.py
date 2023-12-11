import os, math, sys, argparse, re, subprocess, tempfile, dataclasses, inspect, time, itertools, json
import concurrent.futures as f
from typing import Any, Optional
import numpy as np
import polars as pl
import asyncio as aio

threadpool = None

@dataclasses.dataclass
class InputRNA:
    id: str
    sequence: str
    structure: Optional[str]


@dataclasses.dataclass
class AlignerOptions:
    viennaRNA: bool = False

def al_vienna_distance(a: InputRNA, b: InputRNA, cpu_affinity: Optional[list[int]]) -> Optional[float]:
    """
    RNAdistance from viennaRNA package.
    Only looks at RNA secondary structure
    """

    if a.id == b.id:
        print(f'RNAdistance called for the same sequence {a.id}')
        return 0.0

    if a.structure is None or b.structure is None:
        return None
    
    if len(a.structure) > 4000 or len(b.structure) > 4000:
        # RNAdistance uses fixed-sized buffer and fails for length > 4000
        return None

    p = subprocess.Popen(['RNAdistance'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if cpu_affinity is not None:
        os.sched_setaffinity(p.pid, cpu_affinity)
    stdout, stderr = p.communicate(input=f'{a.structure}\n{b.structure}\n'.encode())

    if p.returncode != 0:
        print(stderr.decode())
        raise RuntimeError(f'RNAdistance failed for {a.id} and {b.id}')
    m = re.match(r'f: (\d+[.]?\d*)', stdout.decode())
    if m is None:
        print(stdout.decode())
        print(f'RNAdistance has no output for {a.id}[{len(a.structure)}] and {b.id}[{len(a.structure)}]')
        return None
    return float(m.group(1))

def load_epoch(input, epoch_size, max_length):
    s_time = time.time()
    df = pl.scan_parquet(input)
    df = df.filter(pl.col('len') <= max_length)
    df = df.filter(pl.col('len') == pl.col("ss").struct.field("secondary_structure").str.len_chars())
    count = int(df.select(pl.count('*')).collect()[0, 0])
    if count > max_length:
        sample = np.random.choice(count, epoch_size, replace=False)
        sample_bitmap = np.zeros(count, dtype=np.bool_)
        sample_bitmap[sample] = True
        df = df.with_columns(pl.Series('sample', sample_bitmap, dtype=pl.Boolean)).filter(pl.col('sample'))
    df = df.select(
        pl.col("upi"),
        pl.coalesce(pl.col("seq_short"), pl.col("seq_long")).alias("seq"),
        pl.col("ss").struct.field("secondary_structure").alias("ss"),
    )
    df = df.collect()
    df = df.sample(fraction=1, shuffle=True)
    print(f'Loaded {len(df)} / {count} rows in {time.time() - s_time:.2f}s')
    return df

async def analyze_batch(batch: pl.DataFrame, opt: AlignerOptions):
    assert threadpool is not None
    sequences = [
        InputRNA(upi, seq, ss)
        for upi, seq, ss in batch.select('upi', 'seq', 'ss').iter_rows()
    ]

    pairs = []
    for i in range(len(sequences)):
        for j in range(i):
            pairs.append((i, j))
    result: dict[str, list[Any]] = { }

    if opt.viennaRNA:
        result["vienna_distance"] = [ threadpool.submit(al_vienna_distance, sequences[i], sequences[j], None) for i, j in pairs ]

    futures = list(itertools.chain.from_iterable(result.values()))
    for future in futures:
        await aio.wrap_future(future)

    def get_matrix(list: list[Any]):
        assert all(x.done() for x in list), f"not all futures are done: {list}"
        m = [ [ None ] * i for i in range(len(sequences)) ]
        for x, (i, j) in zip(list, pairs):
            m[max(i, j)][min(i, j)] = x.result()
        return m
    return {
        "ids": [ s.id for s in sequences ],
        **{
            k: get_matrix(v)
            for k, v in result.items()
        }
    }

def write_result(out_file, r):
    with open(out_file, 'a') as f:
        f.write(json.dumps(r, indent=None))
        f.write('\n')

async def iterate_interleaved_parallel(iter, f):
    prev = None
    loop = aio.get_running_loop()
    for i, x in enumerate(iter):
        next = aio.run_coroutine_threadsafe(f(x, i), loop)
        if prev is not None:
            await aio.wrap_future(prev)
        prev = next
    if prev is not None:
        await aio.wrap_future(prev)

async def run_batches(df: pl.DataFrame, opt: AlignerOptions, batch_size: int, out_file: str):
    start_time = time.time()
    last_time = time.time()
    batch_times = []
    async def f(batch, i):
        r = await analyze_batch(batch, opt)
        write_result(out_file, r)

        nonlocal last_time
        this_time = time.time()
        batch_times.append(this_time - last_time)
        print(f'Batch {i} done in {this_time - last_time:.2f}s, mean={np.mean(batch_times[-30:])}\r', end='', flush=True)
        last_time = this_time
    batches = df.iter_slices(batch_size)
    await iterate_interleaved_parallel(batches, f)

    print(f"Epoch finished, mean batch time: {np.mean(batch_times):.2f}s, total time: {(time.time() - start_time)/60:.2f} min")

def main(args):
    threads = args.threads
    if threads == 0:
        threads = os.cpu_count()
    print(f'{threads=}')
    global threadpool
    assert threadpool is None
    threadpool = f.ThreadPoolExecutor(threads)

    opt = AlignerOptions(viennaRNA=args.al_vienna_distance)

    while True:
        df = load_epoch(args.input, args.epoch_size, args.max_length)
        aio.run(run_batches(df, opt, args.batch_size, args.output))

def main_v(argv):
    parser = argparse.ArgumentParser(description='Aligns random structures from the specified parquet files')
    parser.add_argument('--input', required=True, type=str, help='Input parquet file with RNA sequences and secondary structures')
    parser.add_argument('--output', required=True, type=str, help='Output JSON lines with alignment scores')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for alignment')
    parser.add_argument('--epoch_size', type=int, default=28800, help='Batch size for loading')
    parser.add_argument('--max_length', type=int, default=6000, help='Maximum length of the sequence, other sequences are ignored')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use')

    parser.add_argument('--al_vienna_distance', action='store_true', help=inspect.getdoc(al_vienna_distance))
    args = parser.parse_args(argv)
    main(args)

if __name__ == '__main__':
    main_v(sys.argv[1:])
