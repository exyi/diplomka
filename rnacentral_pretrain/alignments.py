import hashlib
import os, math, sys, argparse, re, subprocess, tempfile, dataclasses, inspect, time, itertools, json, threading
import concurrent.futures as f
from typing import Any, Iterable, Optional, Sequence, Union
import typing
import numpy as np
import polars as pl
import asyncio as aio

threadpool = None
affinity_cpus_lock = threading.Lock()
affinity_cpus: Optional[dict[int, Optional[Union['type[ProcessSentinel]', subprocess.Popen]]]] = None

process_run_options = {
    'niceness': None
}

def set_niceness(niceness: Optional[int]):
    if niceness is not None:
        x = os.nice(0)
        os.nice(niceness - x)

def process_preexec(niceness, cpu):
    def core():
        set_niceness(niceness)

        if cpu is not None:
            os.sched_setaffinity(os.getpid(), [cpu])
    return core

class ProcessSentinel:
    returncode: Optional[int] = None

def proc_run(args, cwd=None, stdin=None):
    cpu = None
    if affinity_cpus is not None:
        with affinity_cpus_lock:
            for cpu, p2 in affinity_cpus.items():
                if p2 is None or p2.returncode is not None:
                    affinity_cpus[cpu] = ProcessSentinel
                    break

    want_preexec = process_run_options['niceness'] is not None or cpu is not None
    try:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, stdin=stdin, preexec_fn=(process_preexec(process_run_options['niceness'], cpu) if want_preexec else None))
        if cpu is not None:
            assert affinity_cpus is not None
            with affinity_cpus_lock:
                affinity_cpus[cpu] = p
        return p
    except:
        if cpu is not None:
            assert affinity_cpus is not None
            with affinity_cpus_lock:
                affinity_cpus[cpu] = None
        raise

@dataclasses.dataclass
class InputRNA:
    id: str
    sequence: str
    structure: Optional[str]


@dataclasses.dataclass
class AlignerOptions:
    viennaRNA: bool = False
    needle: bool = False
    rnalign2d: bool = False
    xmers: tuple[int, ...] = tuple()
    write_batches: Optional[str] = None

def program_executable(program: str) -> list[str]:
    if os.environ.get(f'RNC_AL_OVERRIDE_{program.upper()}', ''):
        return os.environ[f'RNC_AL_OVERRIDE_{program.upper()}'].split()
    return [ program ]

def write_seq_struct_fasta(batch: list[InputRNA], file: typing.TextIO, uppercase=False):
    for rna in batch:
        sequence = rna.sequence if not uppercase else rna.sequence.upper()
        file.write(f'>{rna.id}\n{sequence}\n{rna.structure}\n')

def al_locarna_distance(a: InputRNA, b: InputRNA, cpu_affinity: Optional[list[int]]) -> dict[str, float]:
    """
    broken POS
    """
    raise NotImplementedError()

def al_vienna_distance(a: InputRNA, b: InputRNA, cpu_affinity: Optional[list[int]]) -> dict[str, float]:
    """
    RNAdistance from viennaRNA package.
    Only looks at RNA secondary structure
    """

    if a.id == b.id:
        print(f'RNAdistance called for the same sequence {a.id}')
        return {'vienna_distance': 0.0}

    if a.structure is None or b.structure is None:
        return {}
    
    if len(a.structure) > 4000 or len(b.structure) > 4000:
        # RNAdistance uses fixed-sized buffer and fails for length > 4000
        return {}

    p = proc_run([*program_executable('RNAdistance')], stdin=subprocess.PIPE)
    
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
        return {}
    return {'vienna_distance': float(m.group(1)) }

def get_pairs(len: int):
    pairs = []
    for i in range(len):
        for j in range(i):
            pairs.append((i, j))
    return pairs


def al_needle_distance(batch: list[InputRNA], cpu_affinity: Optional[list[int]]) -> dict[str, list[float]]:
    """
    Use needleall to score all pairs of sequences in the batch.
    Secondary structure is ignored.
    """
    with tempfile.TemporaryDirectory(prefix="rna_alignments_needle") as dir:
        with open(os.path.join(dir, 'input.fasta'), 'w') as f:
            for i, rna in enumerate(batch):
                f.write(f'>my_sequence_grgr_{i}\n{rna.sequence}\n')

        p = proc_run([*program_executable('needleall'), '-asequence', 'input.fasta', '-bsequence', 'input.fasta', '-gapopen', '5', '-gapextend', '0.3', '-aformat', 'srspair', '-minscore', '-10', '-datafile', 'EDNAFULL', '-outfile', 'output.needleall'], cwd=dir)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print(stderr.decode())
            raise RuntimeError(f'needleall failed for {batch[0].id} and {batch[1].id}')
        if not os.path.exists(os.path.join(dir, 'output.needleall')):
            print(stderr.decode())
            print(stdout.decode())
            raise RuntimeError(f'needleall failed for {batch[0].id} and {batch[1].id}')
        score = np.zeros((len(batch), len(batch)))
        length = -np.ones((len(batch), len(batch)))
        identity = np.zeros((len(batch), len(batch)))
        similarity = np.zeros((len(batch), len(batch)))
        gaps = np.zeros((len(batch), len(batch)))

        i, j = None, None
        with open(os.path.join(dir, 'output.needleall')) as f:
            for line in f.readlines():
                if m:=re.match(r'# 1: my_sequence_grgr_(\d+)', line):
                    i = int(m.group(1))
                elif m:=re.match(r'# 2: my_sequence_grgr_(\d+)', line):
                    j = int(m.group(1))
                elif m:=re.match(r'# Length: +(\d+)', line):
                    assert length[i, j] == -1
                    length[i, j] = int(m.group(1))
                elif m:=re.match(r'# Identity: +(\d+)/(\d+) +\((\d+([.]\d+)?)%\)', line):
                    assert identity[i, j] == 0
                    identity[i, j] = float(m.group(3)) / 100
                elif m:=re.match(r'# Similarity: +(\d+)/(\d+) +\((\d+([.]\d+)?)%\)', line):
                    assert similarity[i, j] == 0
                    similarity[i, j] = float(m.group(3)) / 100
                elif m:=re.match(r'# Gaps: +(\d+)/(\d+) +\((\d+([.]\d+)?)%\)', line):
                    assert gaps[i, j] == 0
                    gaps[i, j] = float(m.group(3)) / 100
                elif m:=re.match(r'# Score: +(\d+([.]\d+)?)', line):
                    assert score[i, j] == 0
                    score[i, j] = float(m.group(1))

        pairs = get_pairs(len(batch))
        assert np.all(np.array([ float(length[i, j]) for i, j in pairs ]) >= 0), f'length={length}'
        return {
            "needle_length": [ float(length[i, j]) for i, j in pairs ],
            "needle_identity": [ float(identity[i, j]) for i, j in pairs ],
            "needle_similarity": [ float(similarity[i, j]) for i, j in pairs ],
            "needle_gaps": [ float(gaps[i, j]) for i, j in pairs ],
            "needle_score": [ float(score[i, j]) for i, j in pairs ],
        }
    
def al_rnalign2d_distance(a: InputRNA, b: InputRNA, cpu_affinity: Optional[list[int]]) -> dict[str, float]:
    """
    """
    with tempfile.TemporaryDirectory(prefix="rna_alignments_ra2d") as dir:
        with open(os.path.join(dir, 'input.fasta'), 'w') as f:
            write_seq_struct_fasta([a, b], f)
        p = proc_run([*program_executable('rnalign2d'), "-i", os.path.join(dir, "input.fasta"), "-o", os.path.join(dir, "output.fasta")], cwd=dir)
        stdout, stderr = p.communicate()
        if not os.path.exists(os.path.join(dir, 'output.fasta')) or p.returncode != 0:
            print(stderr.decode())
            print(stdout.decode())
            raise RuntimeError(f'rnalign2d failed for {a.id} and {b.id}')
        with open(os.path.join(dir, 'output.fasta')) as f:
            alignment = []
            lines = list(f.readlines())
            line_i = 0
            while line_i < len(lines):
                if lines[line_i].strip() == "":
                    line_i += 1
                    continue

                assert lines[line_i].startswith('>')
                assert "(" not in lines[line_i+1]
                assert "A" not in lines[line_i+2]
                alignment.append((
                    lines[line_i].strip()[1:].strip(),
                    lines[line_i+1].strip(),
                    lines[line_i+2].strip(),
                ))
                line_i += 3
        assert len(alignment) == 2, f"could not read alignment of {a.id} and {b.id}"

        (_, seq1, struct1), (_, seq2, struct2) = alignment
        assert len(seq1) == len(seq2), f"alignment of {a.id} and {b.id} has different lengths (Seq, '{seq1}' vs '{seq2}')"
        assert len(struct1) == len(struct2), f"alignment of {a.id} and {b.id} has different lengths (SS, '{struct1}' vs '{struct2}')"
        assert len(seq1) == len(struct1), f"alignment of {a.id} and {b.id} has different lengths (SS vs Seq, '{seq1}' vs '{struct1}')"
        seq_mismatch = 0
        struct_mismatch = 0
        double_mismatch = 0
        gap = 0
        for i in range(len(seq1)):
            if seq1[i] != seq2[i]:
                seq_mismatch += 1
            if struct1[i] != struct2[i]:
                struct_mismatch += 1
            if seq1[i] != seq2[i] and struct1[i] != struct2[i]:
                double_mismatch += 1
            if seq1[i] == "-" or seq2[i] == "-":
                gap += 1

        return {
            'rnalign2d_seq_dist': seq_mismatch / len(seq1),
            'rnalign2d_ss_dist': struct_mismatch / len(seq1),
            'rnalign2d_comb_dist': double_mismatch / len(seq1),
            'rnalign2d_gap_dist': gap / len(seq1),
        }


def common_xmers(batch: list[InputRNA], x = 6) -> dict[str, list[float]]:
    """
    Calculates number of common x-mers between all pairs of sequences in the batch.
    MAFFT uses something similar with x=6
    """
    xmers = [
        set(rna.sequence[i:i+x] for i in range(len(rna.sequence) - x + 1))
        for rna in batch
    ]
    pairs = get_pairs(len(batch))
    return {
        f'common_{x}mers': [ len(xmers[i] & xmers[j]) for i, j in pairs ]
    }

def load_epoch(input: list[str], epoch_size, max_length):
    s_time = time.time()
    # load 2 partitions and then shuffle it
    if len(input) <= 2:
        partitions = input
    else:
        partitions = list(np.random.choice(input, 2, replace=False))
    df = pl.concat([
        pl.scan_parquet(partition, cache=False, low_memory=True, hive_partitioning=False)
        for partition in partitions
    ])
    df = df.filter(pl.col('len') <= max_length)
    df = df.filter(pl.col('len') == pl.col("ss").struct.field("secondary_structure").str.len_chars())
    count = int(df.select(pl.count('*')).collect(streaming=True)[0, 0])
    print(f"data_count={count}")
    if count > epoch_size:
        sample = np.random.choice(count, epoch_size, replace=False)
        sample_bitmap = np.zeros(count, dtype=np.bool_)
        sample_bitmap[sample] = True
        df = df.with_columns(pl.Series('sample', sample_bitmap, dtype=pl.Boolean)).filter(pl.col('sample'))
    df = df.select(
        pl.col("upi"),
        pl.coalesce(pl.col("seq_short"), pl.col("seq_long")).alias("seq"),
        pl.col("ss").struct.field("secondary_structure").alias("ss"),
    )
    df = df.unique('upi')
    df = df.collect(streaming=True)
    df = df.sample(fraction=1, shuffle=True)
    print(f'Loaded {len(df)} / {count} rows in {time.time() - s_time:.2f}s')
    return df

async def analyze_batch(batch: pl.DataFrame, opt: AlignerOptions):
    assert threadpool is not None
    sequences = [
        InputRNA(upi, seq, ss)
        for upi, seq, ss in batch.select('upi', 'seq', 'ss').iter_rows()
    ]

    pairs = get_pairs(len(sequences))
    result: dict[str, list[f.Future[dict[str, float]]]] = { }
    big_results: list[f.Future[dict[str, list[float]]]] = []

    if opt.viennaRNA:
        result["vienna"] = [ threadpool.submit(al_vienna_distance, sequences[i], sequences[j], None) for i, j in pairs ]
    if opt.rnalign2d:
        result['rnalign2d'] = [ threadpool.submit(al_rnalign2d_distance, sequences[i], sequences[j], None) for i, j in pairs ]
    if opt.needle:
        big_results.append(
            threadpool.submit(al_needle_distance, sequences, None)
        )
    for x in opt.xmers:
        big_results.append(
            threadpool.submit(common_xmers, sequences, x)
        )
    if opt.write_batches:
        os.makedirs(opt.write_batches, exist_ok=True)
        batch_id_sha = hashlib.sha256('||'.join([ s.id for s in sequences ]).encode()).hexdigest()[0:16]
        with open(os.path.join(opt.write_batches, f'batch_{batch_id_sha}.fasta'), 'w') as file:
            write_seq_struct_fasta(sequences, file)

    futures = list(itertools.chain.from_iterable(result.values()))
    for future in futures:
        await aio.wrap_future(future)
    for future in big_results:
        await aio.wrap_future(future)

    T = typing.TypeVar('T')
    def unwrap_future(future: typing.Union[T, f.Future[T]], assert_done = False) -> T:
        if isinstance(future, f.Future):
            if assert_done:
                assert future.done(), f"future is not done: {future}"
            return future.result()
        return future

    def get_matrix(l: Iterable[Optional[float]]) -> list[list[Optional[float]]]:
        m: list[list[Optional[float]]] = [ [ None ] * i for i in range(len(sequences)) ]
        for x, (i, j) in zip(l, pairs):
            m[max(i, j)][min(i, j)] = x
        return m
    
    def get_matrixdict(l: list[dict[str, float]]) -> dict[str, list[list[Optional[float]]]]:
        keys: set[str] = set()
        for d in l:
            keys.update(d.keys())
        result_keys = list(sorted(keys))
        return {
            k: get_matrix([ d.get(k, None) for d in l ])
            for k in result_keys
        }
    
    result_dict: dict[str, Any] = {
        "ids": [ s.id for s in sequences ]
    }
    for k, v in result.items():
        result_dict.update(get_matrixdict([ unwrap_future(v, assert_done=True) for v in result[k] ]))
    for k, v in itertools.chain(*[ unwrap_future(r).items() for r in big_results ]):
        result_dict[k] = v

    return result_dict

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
    set_niceness(args.niceness)
    process_run_options['niceness'] = args.niceness

    threads = args.threads
    if args.cpu_affinity is not None:
        affinity_cpus = {
            cpu: None
            for cpu in args.cpu_affinity
        }
        threads = threads or len(affinity_cpus)
    if threads == 0:
        threads = os.cpu_count()
    print(f'{threads=}')
    global threadpool
    assert threadpool is None
    threadpool = f.ThreadPoolExecutor(threads)

    opt = AlignerOptions(viennaRNA=args.al_vienna_distance, needle=args.al_needle, rnalign2d=args.al_rnalign2d, xmers=tuple(args.al_common_xmers or []), write_batches=args.write_batches)

    while True:
        df = load_epoch(args.input, args.epoch_size, args.max_length)
        aio.run(run_batches(df, opt, args.batch_size, args.output))

def main_v(argv):
    parser = argparse.ArgumentParser(description='Aligns random structures from the specified parquet files')
    parser.add_argument('--input', required=True, type=str, nargs="+", help='Input parquet file with RNA sequences and secondary structures')
    parser.add_argument('--output', required=True, type=str, help='Output JSON lines with alignment scores')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for alignment')
    parser.add_argument('--epoch_size', type=int, default=28800, help='Batch size for loading')
    parser.add_argument('--max_length', type=int, default=6000, help='Maximum length of the sequence, other sequences are ignored')
    parser.add_argument('--threads', type=int, default=None, help='Number of threads to use')
    parser.add_argument('--niceness', type=int, default=None, help='Process niceness')
    parser.add_argument('--cpu_affinity', type=int, nargs="+", default=None, help='')

    parser.add_argument('--al_vienna_distance', action='store_true', help=inspect.getdoc(al_vienna_distance))
    parser.add_argument('--al_needle', action='store_true', help=inspect.getdoc(al_needle_distance))
    parser.add_argument('--al_common_xmers', type=int, nargs="+", help=inspect.getdoc(common_xmers))
    parser.add_argument('--al_rnalign2d', action='store_true', help=inspect.getdoc(al_rnalign2d_distance))

    parser.add_argument('--write_batches', type=str, help='Write batches to the specified directory')
    args = parser.parse_args(argv)
    main(args)

if __name__ == '__main__':
    main_v(sys.argv[1:])
