from collections import defaultdict
import collections.abc
import dataclasses, typing, os, time, math, re
from typing import Any, Optional
import polars as pl
import numpy as np
import fm
import torch
import faiss
import concurrent.futures as f
import sqlite3

threadpool = f.ThreadPoolExecutor(max_workers=math.ceil((os.cpu_count() or 4 )/2))

@dataclasses.dataclass
class FmResult:
    embeddings: list[torch.Tensor]
    logits: list[torch.Tensor]
    correct_logits: list[torch.Tensor]
    seq_predicted: list[str]

def load_embedding(model_location: Optional[str], device: torch.device):
    model, alphabet = fm.pretrained.rna_fm_t12(model_location)
    model = model.half()
    model = model.eval()
    model = model.to(device)
    # for x in model.layers:
    #     x.self_attn = torch.compile(x.self_attn)
    batch_converter = alphabet.get_batch_converter()

    class RnaFmEmbedding:
        fm_alphabet = alphabet
        start_padding = int(alphabet.prepend_bos)
        end_padding = int(alphabet.append_eos)

        def compute_embeddings(self, batch: list[str]) -> FmResult:
            _, _, batch_t = batch_converter([ (f"label{i}", x) for i, x in enumerate(batch) ])
            batch_t = batch_t.to(device)
            # print(batch_t.shape)
            # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with torch.no_grad():
            #         with torch.profiler.record_function("model_inferrence"):
            #             results = model(batch_t.to(device), repr_layers=[12])
            # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            with torch.no_grad():
                results = model(batch_t, repr_layers=[12])

            embeddings = results["representations"][12].cpu()
            assert embeddings.dtype == torch.float16
            logits = results["logits"].cpu()
            padding = self.start_padding + self.end_padding
            seq_predicted = [
                torch.argmax(logits[i, 0:(len(seq) + padding), :], dim=1).cpu()
                for i, seq in enumerate(batch)
            ]
            seq_predicted = [
                "".join(alphabet.all_toks[s] for s in ss)
                for ss in seq_predicted
            ]
            correct_logits = [
                # This but using vectorization: torch.tensor([ logits[i, ii, batch_t[i, ii]] for ii in range(len(seq)) ])
                torch.gather(logits[i, 0:(len(seq) + padding), :], 1, batch_t[i, 0:(len(seq) + padding)].to(logits.device).unsqueeze(1)).squeeze(1)
                for i, seq in enumerate(batch)
            ]
            return FmResult(
                [ embeddings[i, 0:(len(seq) + padding), :] for i, seq in enumerate(batch)],
                [ logits[i, 0:(len(seq) + padding), :] for i, seq in enumerate(batch)],
                correct_logits,
                seq_predicted
            )
        
    return RnaFmEmbedding()

def create_embedding_index(
    batches: typing.Iterator[pl.DataFrame],
    output_directory: str,
    batch_size: int = 16,
    max_seq_len: int = 1022,
    embedding_path: Optional[str] = None,
    gpu: bool = False,
    db_path: Optional[str] = None,
    start_index: int = 0
):
    db = sqlite3.connect(db_path, check_same_thread=False) if db_path is not None else None
    device = torch.device("cuda") if gpu else torch.device("cpu")
    embedding = load_embedding(embedding_path, device)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    last_job: Optional[f.Future] = None

    for megabatch_id, megabatch in enumerate(batches, start=start_index):

        assert sqlite3.threadsafety > 0
        if db is None:
            batch_db = sqlite3.connect(":memory:", check_same_thread=False)
            batch_db.execute(sql["create_embeddings"])
        else:
            batch_db = db

        outfile = os.path.join(output_directory, f"megabatch_{megabatch_id}.parquet")
        if os.path.exists(outfile):
            print(f"Skipping batch {megabatch_id}, file already exists")
            continue
        assert megabatch is not None, f"Batch {megabatch_id} is None"

        start_time = time.time()

        split_sequences = []
        for i, (id, seq) in enumerate(megabatch[["upi", "seq"]].iter_rows()):
            if len(seq) <= max_seq_len:
                split_sequences.append((id, i, None, seq))
            else:
                for j in range(0, len(seq), 512):
                    split_sequences.append((id, i, j, seq[j:(j + max_seq_len)]))

        # cols = defaultdict(lambda: [])
        batch_times = []
        for i in range(0, len(split_sequences), batch_size):
            batch_start_time = time.time()
            batch = split_sequences[i:(i + batch_size)]

            result = embedding.compute_embeddings([ x[3] for x in batch ])

            batch_times.append(time.time() - batch_start_time)

            insert_rows = []
            for j, (upi, _, seq_offset, seq_slice), emb, logits, correct_logits, seq_predicted in zip(range(len(batch)), batch, result.embeddings, result.logits, result.correct_logits, result.seq_predicted):
                # upi, deconflict_id, batch_id, batch_index, seq_offset, seq_slice, embedding_avg, embedding_fst, embedding_last, embedding_mid, logits_max, logits_correct, seq_predicted
                insert_rows.append((
                    upi,
                    0,
                    megabatch_id,
                    j,
                    seq_offset or 0,
                    seq_slice,
                    f16_bytes(emb.mean(dim=0)),
                    f16_bytes(emb[0, :]),
                    f16_bytes(emb[-1, :]),
                    f16_bytes(emb[len(emb) // 2, :]),
                    f16_bytes(torch.max(logits, dim=1)[0]),
                    f16_bytes(correct_logits),
                    seq_predicted
                ))
            batch_db.executemany(sql["insert_emb"] + " ON CONFLICT (upi, deconflict_id, seq_offset) DO NOTHING", insert_rows)
            batch_db.commit()

            # cols["embedding_avg"].extend([ e.mean(dim=0).cpu().numpy() for e in result.embeddings ])
            # cols["embedding_fst"].extend([ e[0, :].cpu().numpy() for e in result.embeddings ])
            # cols["embedding_last"].extend([ e[-1, :].cpu().numpy() for e in result.embeddings ])
            # cols["embedding_mid"].extend([ e[len(e) // 2, :].cpu().numpy() for e in result.embeddings ])
            # cols["logits_max"].extend([ torch.max(e, dim=1)[0].cpu().numpy() for e in result.logits ])
            # cols["logits_correct"].extend([ e.cpu().numpy() for e in result.correct_logits ])
            # cols["seq_predicted"].extend(result.seq_predicted)
            del result

            print(f"Batch {megabatch_id:5}:{i:5}/{len(split_sequences)} (seq {batch[0][1]}, {(time.time() - batch_start_time) * 1000:.1f}ms) \r", end="")

        print(f"Batch {megabatch_id:5}:{len(split_sequences):5}/{len(split_sequences)} (seq {len(megabatch)}, {np.median(batch_times) * 1000:.1f}ms)   ")

        if last_job is not None:
            if not last_job.done():
                print("waiting for last write job to finish...")
                last_job.result()
            if last_job.exception() is not None:
                print(f"last job failed: {last_job.exception()}")

        last_job = threadpool.submit(write_results, outfile, megabatch_id, batch_db, db is None, split_sequences, start_time)
        last_job.result()

def write_results(outfile, batch_id: int, db: sqlite3.Connection, close_db: bool, split_sequences, start_time):
    schema: defaultdict[str, pl.PolarsDataType] = defaultdict(lambda: pl.List(inner=pl.Float32))
    schema["seq_predicted"] = pl.Utf8

    # pad and concat numpy embeddings
    # batch_df = pl.DataFrame({
    #     "upi": pl.Series([ x[0] for x in split_sequences ], dtype=pl.Utf8),
    #     "seq_offset": pl.Series([ x[2] for x in split_sequences ], dtype=pl.Int32),
    #     "seq_slice": pl.Series([ x[3] for x in split_sequences ], dtype=pl.Utf8),
    #     **{k: pl.Series(v, dtype=schema[k]) for k, v in cols.items()}
    # })
    batch_df = pl.read_database(f"SELECT * from embeddings WHERE batch_id = {int(batch_id)}", db)
    decode = lambda col: list(f16_decode(col).astype(np.float32))
    batch_df = batch_df.select([
        "upi",
        "seq_offset",
        "seq_slice",
        pl.col("embedding_avg").map_elements(decode, return_dtype=pl.List(inner=pl.Float32)),
        pl.col("embedding_fst").map_elements(decode, return_dtype=pl.List(inner=pl.Float32)),
        pl.col("embedding_last").map_elements(decode, return_dtype=pl.List(inner=pl.Float32)),
        pl.col("embedding_mid").map_elements(decode, return_dtype=pl.List(inner=pl.Float32)),
        pl.col("logits_max").map_elements(decode, return_dtype=pl.List(inner=pl.Float32)),
        pl.col("logits_correct").map_elements(decode, return_dtype=pl.List(inner=pl.Float32)),
        "seq_predicted"
    ])
    batch_df.write_parquet(outfile)
    print(f"Wrote {outfile} in {time.time() - start_time:.2f}s")

    if close_db:
        db.close()

def load_batches(input_file: str, batch_size, skip_batches: set[int], db_path: Optional[str]) -> typing.Iterable[pl.DataFrame]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False) if db_path is not None else None
    pl.disable_string_cache()
    lf = pl.scan_parquet(input_file, cache=False, low_memory=True)
    lf = lf.filter(pl.col("len") < 6000)
    lf = lf.select(
        pl.col("id"),
        pl.col("upi"),
        pl.coalesce(pl.col("seq_short"), pl.col("seq_long")).alias("seq")
    )
    lf = lf.filter(pl.col("seq").is_not_null())

    def load_slice(batch_i) -> tuple[bool, pl.DataFrame]:
        if batch_i in skip_batches:
            return True, None # type: ignore
        print("loading slice")
        df = lf.slice(batch_i * batch_size, length=batch_size).collect(streaming=True)
        print("loaded slice")
        if len(df) == 0:
            return False, df
        if db is not None:
            already_done = db.execute(f"SELECT upi FROM embeddings WHERE upi IN ({','.join(['?'] * len(df))})", df["upi"].to_list()).fetchall()
            already_done = set(x[0] for x in already_done)
            if len(already_done) > 0:
                print(f"Skipping {len(already_done)} already calculated sequences in batch {batch_i}")
                df = df.filter(pl.col("upi").is_in(already_done).not_())
        else:
            print("WARNING: no deconfliction DB")
        return True, df

    class Result(collections.abc.Iterable):
        def __iter__(self) -> typing.Iterator[pl.DataFrame]:
            cont, batch = load_slice(0)
            i = 0
            while cont:
                i += 1
                next_batch = threadpool.submit(load_slice, i)
                yield batch
                cont, batch = next_batch.result()

    return Result()

sql = {
    "create_batches": """
        CREATE TABLE IF NOT EXISTS batches (
            batch_i INTEGER,
            count INTEGER
        )
        """,
    "create_embeddings": """
        CREATE TABLE IF NOT EXISTS embeddings (
            upi TEXT NOT NULL,
            deconflict_id INTEGER NOT NULL,
            batch_id INTEGER NOT NULL,
            batch_index INTEGER NOT NULL,
            seq_offset INTEGER,
            seq_slice TEXT,
            embedding_avg BLOB,
            embedding_fst BLOB,
            embedding_last BLOB,
            embedding_mid BLOB,
            logits_max BLOB,
            logits_correct BLOB,
            seq_predicted TEXT,
            PRIMARY KEY (upi, deconflict_id, seq_offset)
        )""",
    "insert_emb": """
            INSERT INTO embeddings
                ( upi, deconflict_id, batch_id, batch_index, seq_offset, seq_slice, embedding_avg, embedding_fst, embedding_last, embedding_mid, logits_max, logits_correct, seq_predicted)
            VALUES (?,             ?,        ?,            ?,          ?,         ?,            ?,             ?,              ?,             ?,          ?,              ?, ?)
            """
}

def f16_bytes(col):
    if isinstance(col, torch.Tensor):
        col = col.half().cpu().numpy()
    col = np.array(col, dtype=np.float16)
    return col.tobytes()

def f16_decode(col):
    return np.frombuffer(col, dtype=np.float16)

def regen_sqlite_index(directory: str, sqlite_path: str):
    batches = [ (f, int(m.group(1))) for f in os.listdir(directory) if (m := re.match(r"megabatch_(\d+)\.parquet", f)) ]

    db = sqlite3.connect(sqlite_path)
    db.execute(sql["create_batches"])
    db.execute(sql["create_embeddings"])
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS embeddings_upi_ix ON embeddings (upi, deconflict_id, seq_offset)")
    db.execute("CREATE INDEX IF NOT EXISTS embeddings_batch_ix ON embeddings (batch_id, batch_index)")

    for batch, batch_id in batches:
        if db.execute("SELECT count(*) FROM batches WHERE batch_i = ?", (batch_id,)).fetchone()[0] > 0:
            print(f"Skipping batch {batch}")
            continue

        df = pl.read_parquet(os.path.join(directory, batch))
        opt_col = lambda col: pl.col(col) if col in df.columns else pl.lit(None).alias(col)

        def rows():
            for i, (upi, seq_offset, seq_slice, embedding_avg, embedding_fst, embedding_last, embedding_mid, logits_max, logits_correct, seq_predicted) in enumerate(df.select(
                pl.col("upi"),
                pl.col("seq_offset"),
                pl.col("seq_slice"),
                pl.col("embedding_avg"),
                opt_col("embedding_fst"),
                opt_col("embedding_last"),
                opt_col("embedding_mid"),
                opt_col("logits_max"),
                opt_col("logits_correct"),
                opt_col("seq_predicted")
            ).iter_rows()):
                yield (upi, 0, batch_id, i, (seq_offset or 0), seq_slice, f16_bytes(embedding_avg), f16_bytes(embedding_fst), f16_bytes(embedding_last), f16_bytes(embedding_mid), f16_bytes(logits_max), f16_bytes(logits_correct), seq_predicted)
        db.executemany(sql["insert_emb"] + " ON CONFLICT (upi, deconflict_id, seq_offset) DO NOTHING", rows())
        db.execute("INSERT INTO batches (batch_i, count) VALUES (?, ?)", (batch_id, len(df)))
        db.commit()
        print(f"Inserted batch {batch}")

    db.close()

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", required=True)
    # parser.add_argument("--indexes", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--db")
    parser.add_argument("--regen_db", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args(argv)

    if args.regen_db:
        regen_sqlite_index(args.output_dir, args.db)
        return

    existing_results = {
        int(m.group(1)) - args.start_index
        for file in os.listdir(args.output_dir)
        if (m := re.match(r"megabatch_(\d+)\.parquet", file)) and int(m.group(1)) >= args.start_index
    }

    batches = load_batches(args.input, batch_size=10000, skip_batches=existing_results, db_path=args.db)
    create_embedding_index(
        iter(batches),
        args.output_dir,
        gpu=args.gpu,
        db_path=args.db,
        start_index=args.start_index
    )

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
