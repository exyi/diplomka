from typing import Any, Callable, List
import math
import numpy as np
import csv_loader

def clamp(v, min_v, max_v):
    return np.minimum(max_v, np.maximum(min_v, v))

def get_ntc_weight(strategy: str, ntc_list: List[str] = csv_loader.ntcs):
    min_frequency = 100
    max_frequency = max(csv_loader.ntc_frequencies.values())
    clip_min = 0.1
    clip_max = 5.0
    if strategy == "one":
        return np.ones(len(ntc_list), dtype=np.float32)
    elif strategy == "flat":
        return np.array([
                0 if k == "[UNK]" else
                0.01 if k == "NANT" else
                1
                for k in ntc_list
            ], dtype=np.float32)
    elif strategy == "clip-linear":
        return np.array([
                0 if k == "[UNK]" else
                0.01 if k == "NANT" else
                clamp(5 / (csv_loader.ntc_frequencies[k] / 20_000), clip_min, clip_max)
                for k in ntc_list
            ], dtype=np.float32)
    elif strategy == "linear":
        return np.array([
                0 if k == "[UNK]" else
                0.01 if k == "NANT" else
                clamp(max_frequency / csv_loader.ntc_frequencies[k] * 0.03, 0.0, 7.0)
                for k in ntc_list], dtype=np.float32)
    elif strategy == "almostlinear":
        return np.array([
                0 if k == "[UNK]" else
                0.01 if k == "NANT" else
                clamp((max_frequency / csv_loader.ntc_frequencies[k] * 0.01) ** 0.7, 0.0, 7.0)
                for k in ntc_list], dtype=np.float32)
    elif strategy == "clip-sqrt":
        return np.array([
                0 if k == "[UNK]" else
                0.01 if k == "NANT" else
                clamp(5 / math.sqrt(csv_loader.ntc_frequencies[k] / min_frequency), clip_min, clip_max)
                for k in ntc_list
            ], dtype=np.float32)
    elif strategy == "sqrtB":
        return np.array([
                0 if k == "[UNK]" else
                0.01 if k == "NANT" else
                clamp(math.sqrt(max_frequency / csv_loader.ntc_frequencies[k]) * 0.01, 0.0, 20.0)
                for k in ntc_list
            ], dtype=np.float32)
    elif strategy == "sqrtB-clip":
        return np.array([
                0 if k == "[UNK]" else
                0.01 if k == "NANT" else
                clamp(math.sqrt(max_frequency / csv_loader.ntc_frequencies[k]) * 0.03, 0.1, 3.0)
                for k in ntc_list
            ], dtype=np.float32)

    elif strategy == "log":
        return np.array([
                0 if k == "[UNK]" else
                0.01 if k == "NANT" else
                5.0 / math.log(5.0 + csv_loader.ntc_frequencies[k] / min_frequency)
                for k in ntc_list
            ], dtype=np.float32)
        
    elif strategy == "ignore-AAs":
        return np.array([
                0 if k == "[UNK]" else
                0.01 if k == "NANT" else
                0.01 if k in ["AA00", "AA08", "AA04"] else
                1
                for k in ntc_list
            ], dtype=np.float32)
    else:
        raise ValueError(f"Unknown NtC weights strategy: {strategy}")

def ntc_based_sample_weighter(strategy: str, ntc_list, backend) -> Callable[[Any], np.ndarray]:
    ntc_weights = get_ntc_weight(strategy, ntc_list)
    def sample_weighter(x):
        if isinstance(x, dict):
            ntcs = x["NtC"]
        else:
            ntcs = x
        return backend.gather(ntc_weights, ntcs)
    return sample_weighter

def antihelix_weighter(ntc_list: List[str], backend, helix_weight = 0.1, not_helix_weight = 1):
    aa00 = ntc_list.index("AA00")
    aa08 = ntc_list.index("AA08")
    aa04 = ntc_list.index("AA04")
    aa03 = ntc_list.index("AA03")
    aa09 = ntc_list.index("AA09")
    def weighter(x):
        ntcs = x["NtC"]
        pairing = x["pairing_is_canonical"]
        pairing = pairing[..., 1:] & pairing[..., :-1]
        probably_helix = pairing & (
            (ntcs == aa00) |
            (ntcs == aa08) |
            (ntcs == aa04) |
            (ntcs == aa03) |
            (ntcs == aa09)
        )
        return backend.where(probably_helix, helix_weight, not_helix_weight)
    return weighter

def combined_weighter(b, *ws):
    def weighter(x):
        return b.reduce_prod([w(x) for w in ws], axis=0)
    return weighter

def get_weighter(strategy, backend, ntc_list: List[str]):
    s = strategy.split("+")
    w = ntc_based_sample_weighter(s[0], ntc_list, backend)
    if len(s) == 1:
        return w
    if s[1] == "helix":
        w = combined_weighter(backend, w, antihelix_weighter(ntc_list, backend))
    return w
