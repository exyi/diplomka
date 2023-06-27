import torch
import fm
import onnx


model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, repr_layers):
        super().__init__()
        self.model = model
        self.repr_layers = repr_layers

    def forward(self, batch_tokens):
        results = self.model(batch_tokens, repr_layers=[self.repr_layers])
        return results["representations"][self.repr_layers]

def export_onnx_model(seq_len, batch_size, output_file):

    sample_batch = torch.randint(2, 10, (batch_size, seq_len))
    sample_batch[0, :] = alphabet.cls_idx
    sample_batch[-2, :] = alphabet.padding_idx
    sample_batch[-1, :] = alphabet.eos_idx
    wrapper = ModelWrapper(model, 12)

    torch.onnx.export(wrapper, sample_batch, output_file, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size", 1: "seq_len"}})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-file", type=str, default="rna-fm.onnx")
    args = parser.parse_args()

    export_onnx_model(args.seq_len, args.batch_size, args.output_file)

