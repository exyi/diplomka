from typing import Any, Callable, Optional
import torch, torch.utils.data, torch.utils.data.datapipes, torch.utils.data.datapipes.datapipe
# import torchdata
# import torchdata.dataloader2
# import torchdata.datapipes as torchpipes
import numpy as np

from model import dataset_tf, hyperparameters, sample_weight
from .torchutils import TensorDict, pad_nested

class IteratorWrapper(torch.utils.data.datapipes.datapipe.IterDataPipe):
    def __init__(self, len, mkiterator):
        super().__init__()
        self.len = len
        self.mkiterator = mkiterator
    def __iter__(self):
        return self.mkiterator()
    def __len__(self):
        return self.len


class NtcDatasetWrapper:
    def __init__(self, mkloader: Callable[[], dataset_tf.NtcDatasetLoader]):
        self._mkloader = mkloader

    def get_loader(self):
        return self._mkloader()
    
    def get_data(self, max_len: Optional[int], batch_size: Optional[int], shuffle: Optional[int]) -> torch.utils.data.datapipes.datapipe.IterDataPipe:
        def mkiterator():
            return self._mkloader().get_data(max_len=max_len, shuffle=shuffle, batch=None).as_numpy_iterator()
        pipe = IteratorWrapper(self._mkloader().cardinality or -1, mkiterator)
        if batch_size is not None:
            pipe = pipe.batch(batch_size).collate(collate_fn)
            # torch.utils.data.datapipes.iter.Collator
        else:
            raise ValueError("batch_size must be specified")
        return pipe


    def __len__(self):
        return self._mkloader().cardinality or -1

# def tf_to_torch(tfdata: dataset_tf.tf.data.Dataset):
#     return NtcDatasetWrapper(tfdata.as_numpy_iterator, len=len(tfdata))

def collate_fn(batch: list[tuple[TensorDict, TensorDict, float]]):
    """
    Converts list of items to a single batch
    Adds "lengths" field to the result
    """
    batch.sort(key=lambda x: x[0]["sequence"].shape[0], reverse=True)
    ## get sequence lengths
    lengths = torch.LongTensor([ t[0]["sequence"].shape[0] for t in batch ])
    result_in = dict()
    for key in batch[0][0].keys():
        example = batch[0][0][key]
        if isinstance(example, np.ndarray):
            result_in[key] = pad_nested(torch.nested.nested_tensor([ t[0][key] for t in batch ]))
    result_out = dict()
    for key in batch[0][1].keys():
        result_out[key] = pad_nested(torch.nested.nested_tensor([ t[1][key] for t in batch ]))
    result_in["lengths"] = lengths
    result_out["lengths"] = lengths
    result_out["sample_weight"] = pad_nested(torch.nested.nested_tensor([t[2] for t in batch]))
    return result_in, result_out

def make_loader1(pipe: torch.utils.data.datapipes.datapipe.IterDataPipe):
    loader = torch.utils.data.DataLoader(pipe, num_workers=0, collate_fn=lambda x: x[0])
    return loader

## TODO: fix out of process dataset loading
# def make_loader2(pipe: torch.utils.data.datapipes.datapipe.IterDataPipe):
#     loader = torchdata.dataloader2.DataLoader2(pipe, reading_service=torchdata.dataloader2.MultiProcessingReadingService(num_workers=0))

#     return loader

class Datasets:
    def __init__(self, validation_ds: torch.utils.data.DataLoader, val_size, train_data: NtcDatasetWrapper, train_size) -> None:
        self._validation_ds = validation_ds
        self.train_size = train_size
        self._train_data = train_data
        self.val_size = val_size
        pass

    def get_validation_ds(self):
        return self._validation_ds
    
    def get_train_ds(self, max_len, batch_size):
        return make_loader1(self._train_data.get_data(max_len=max_len, shuffle=20_000, batch_size=batch_size))
    
def make_test_val(train_files: list[str], val_files: list[str], p: hyperparameters.Hyperparams):
    def mkloader(files, ntc_rmsd_threshold):
        return lambda: dataset_tf.NtcDatasetLoader(files, features=p.outputs, ntc_rmsd_threshold=ntc_rmsd_threshold).set_sample_weighter(sample_weight.get_weighter(p.sample_weight, dataset_tf.tf, list(dataset_tf.NtcDatasetLoader.NTCS)))

    val_base = NtcDatasetWrapper(mkloader(val_files, 0))
    train_base = NtcDatasetWrapper(mkloader(train_files, p.nearest_ntc_threshold))
    return Datasets(make_loader1(val_base.get_data(None, p.batch_size, None)), len(val_base), train_base, len(train_base))

