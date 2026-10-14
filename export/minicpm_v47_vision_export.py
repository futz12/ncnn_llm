"""Streaming MiniCPM-V 4.7 SigLIP/merger export using standard ncnn layers.

Single-image graphs deliberately leave spatial window packing to the runner.
Weights are fetched one tensor at a time, never through AutoModel/state_dict.
All emitted weights are float32, including conversion from BF16 checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

VISION = "model.vision_tower"


class SafeTensorSource:
    """Lazy indexed checkpoint access; only the current tensor is materialized."""
    def __init__(self, model_dir):
        from safetensors import safe_open
        self.safe_open = safe_open
        root = Path(model_dir)
        self.files = {}
        index = root / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
            self.files = {k: root / v for k, v in mapping.items()}
        else:
            for path in sorted(root.glob("*.safetensors")):
                with safe_open(str(path), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in self.files:
                            raise ValueError(f"Duplicate checkpoint tensor: {key}")
                        self.files[key] = path
        if not self.files:
            raise FileNotFoundError(f"No safetensors checkpoint in {root}")

    def get(self, key):
        if key not in self.files:
            raise KeyError(f"Missing MiniCPM-V checkpoint tensor: {key}")
        with self.safe_open(str(self.files[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key).float().numpy()


class Graph:
    def __init__(self):
        self.layers = []
        self.weights = []  # descriptors only, not arrays
        self.add("Input", [], "in0")

    def add(self, kind, inputs, outputs, params=""):
        if isinstance(outputs, str):
            outputs = [outputs]
        name = f"{kind.lower()}_{len(self.layers)}"
        self.layers.append(" ".join([kind, name, str(len(inputs)), str(len(outputs)),
                                     *inputs, *outputs, params]).rstrip())
        return outputs[0]

    def weight(self, key, shape, tagged=False):
        self.weights.append((key, tuple(shape), tagged))

    def ln(self, x, name, width, eps):
        out = self.add("LayerNorm", [x], f"v{len(self.layers)}", f"0={width} 1={eps:.8e} 2=1")
        self.weight(name + ".weight", (width,))
        self.weight(name + ".bias", (width,))
        return out

    def linear(self, x, name, inp, out):
        y = self.add("Gemm", [x], f"v{len(self.layers)}",
                     f"3=1 5=1 6=1 8={out} 9={inp} 10=4")
        self.weight(name + ".weight", (out, inp), True)
        self.weight(name + ".bias", (out,), True)
        return y

    def attention_residual(self, x, name, width, heads, eps):
        tag = len(self.layers)
        residual, branch = f"res{tag}", f"branch{tag}"
        self.add("Split", [x], [residual, branch])
        norm = self.ln(branch, name + ".layer_norm1", width, eps)
        projections = [f"qsrc{tag}", f"ksrc{tag}", f"vsrc{tag}"]
        self.add("Split", [norm], projections)
        qkv = []
        for letter, source in zip("qkv", projections):
            projected = self.linear(source, name + f".self_attn.{letter}_proj", width, width)
            shaped = self.add("Reshape", [projected], f"r{letter}{tag}",
                              f"0={width // heads} 1={heads} 2=-1")
            qkv.append(self.add("Permute", [shaped], f"p{letter}{tag}", "0=2"))
        # ncnn parses decimal fractions with uint32 accumulators and scans
        # at most 15 characters. Long mantissas silently change this scale.
        attn = self.add("SDPA", qkv, f"attn{tag}", f"6={1 / math.sqrt(width // heads):.8e}")
        perm = self.add("Permute", [attn], f"perm{tag}", "0=2")
        flat = self.add("Reshape", [perm], f"flat{tag}", f"0={width} 1=-1")
        projected = self.linear(flat, name + ".self_attn.out_proj", width, width)
        return self.add("BinaryOp", [residual, projected], f"attnres{tag}", "0=0")

    def block(self, x, name, width, intermediate, heads, eps, tanh=True):
        x = self.attention_residual(x, name, width, heads, eps)
        tag = len(self.layers)
        residual, branch = f"mlpres{tag}", f"mlpbranch{tag}"
        self.add("Split", [x], [residual, branch])
        norm = self.ln(branch, name + ".layer_norm2", width, eps)
        y = self.linear(norm, name + ".mlp.fc1", width, intermediate)
        y = self.add("GELU", [y], f"gelu{tag}", f"0={int(tanh)}")
        y = self.linear(y, name + ".mlp.fc2", intermediate, width)
        return self.add("BinaryOp", [residual, y], f"mlpout{tag}", "0=0")

    def finish(self, x):
        self.add("Noop", [x], "out0")
        return self

    def param(self):
        blobs = set()
        for layer in self.layers:
            p = layer.split()
            blobs.update(p[4:4 + int(p[2]) + int(p[3])])
        return f"7767517\n{len(self.layers)} {len(blobs)}\n" + "\n".join(self.layers) + "\n"

    def write(self, source, directory, stem):
        directory = Path(directory)
        param, binary = directory / (stem + ".ncnn.param"), directory / (stem + ".ncnn.bin")
        # Write into a temporary name so an interrupted conversion is not loadable.
        tmp = binary.with_suffix(binary.suffix + ".tmp")
        try:
            with tmp.open("wb") as f:
                for key, shape, tagged in self.weights:
                    arr = source.get(key)
                    if tuple(arr.shape) != shape:
                        raise ValueError(f"{key}: expected {shape}, got {arr.shape}")
                    if tagged:
                        f.write(b"\0\0\0\0")
                    np.asarray(arr, dtype="<f4", order="C").tofile(f)
                    del arr
            tmp.replace(binary)
            param.write_text(self.param())
        finally:
            if tmp.exists():
                tmp.unlink()
        return {"param": param.name, "bin": binary.name}


def build_graphs(config):
    vc = config["vision_config"]
    width = vc["hidden_size"]
    intermediate = vc["intermediate_size"]
    heads = vc["num_attention_heads"]
    depth = vc["num_hidden_layers"]
    insert = config.get("insert_layer_id", vc.get("insert_layer_id", 6))
    eps = vc.get("layer_norm_eps", 1e-6)
    patch = vc.get("patch_size", 14)
    channels = vc.get("num_channels", 3)
    out_width = config["text_config"]["hidden_size"]
    if width % heads or not 0 <= insert < depth:
        raise ValueError("Invalid attention dimensions or insertion layer")
    if list(vc.get("window_kernel_size", [2, 2])) != [2, 2] or list(config.get("merge_kernel_size", [2, 2])) != [2, 2]:
        raise ValueError("Only 2x2 window and spatial mergers are supported")
    if config.get("merger_times", 1) != 1:
        raise ValueError("Only merger_times=1 is supported")
    if vc.get("hidden_act", "gelu_pytorch_tanh") not in ("gelu_pytorch_tanh", "gelu"):
        raise ValueError("Unsupported vision activation")
    graphs = {}
    g = Graph()
    x = g.add("Convolution", ["in0"], "patches", f"0={width} 1={patch} 3={patch} 5=1 6={width * channels * patch * patch}")
    g.weight(VISION + ".embeddings.patch_embedding.weight", (width, channels, patch, patch), True)
    g.weight(VISION + ".embeddings.patch_embedding.bias", (width,))
    x = g.add("Reshape", [x], "patch_flat", f"0=-1 1={width}")
    x = g.add("Permute", [x], "patch_rows", "0=1")
    graphs["patch_embed"] = g.finish(x)
    for name, indices in (("encoder_pre", range(insert + 1)), ("encoder_post", range(insert + 1, depth))):
        g = Graph()
        x = "in0"
        for i in indices:
            x = g.block(x, VISION + f".encoder.layers.{i}", width, intermediate, heads, eps,
                        vc.get("hidden_act", "gelu_pytorch_tanh") == "gelu_pytorch_tanh")
        if name == "encoder_post":
            x = g.ln(x, VISION + ".post_layernorm", width, eps)
        graphs[name] = g.finish(x)
    g = Graph()
    x = g.attention_residual("in0", VISION + ".vit_merger", width, heads, eps)
    graphs["window_attn"] = g.finish(x)
    for component, base, hidden, output, epsilon, tanh in (
        ("window_mlp", VISION + ".vit_merger", vc.get("window_intermediate_size", intermediate * 4), width, eps, True),
        ("projector", "model.merger.mlp.0", width * 4, out_width, 1e-6, False),
    ):
        g = Graph()
        x = g.ln("in0", base + ".pre_norm", width * 4, epsilon)
        x = g.linear(x, base + ".linear_1", width * 4, hidden)
        x = g.add("GELU", [x], "activated", f"0={int(tanh)}")
        x = g.linear(x, base + ".linear_2", hidden, output)
        graphs[component] = g.finish(x)
    return graphs


def export_minicpm_v47_vision(model_dir, output_dir, prefix="", source=None):
    config = json.loads((Path(model_dir) / "config.json").read_text())
    if config.get("model_type") != "minicpmv4_7":
        raise ValueError("Expected model_type=minicpmv4_7")
    graphs = build_graphs(config)
    source = source or SafeTensorSource(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vc = config["vision_config"]
    width = vc["hidden_size"]
    grid = vc.get("image_size", 980) // vc.get("patch_size", 14)
    manifest = {
        "type": "minicpm_v47", "format_version": 1,
        "components": {}, "hidden_size": width,
        "output_hidden_size": config["text_config"]["hidden_size"],
        "patch_size": vc.get("patch_size", 14), "position_grid_size": grid,
        "insert_layer_id": config.get("insert_layer_id", vc.get("insert_layer_id", 6)),
        "num_hidden_layers": vc["num_hidden_layers"], "window_kernel_size": [2, 2],
        "merge_kernel_size": [2, 2], "downsample_mode": "16x", "weight_dtype": "float32",
        "scope": "single_image_thumbnail",
    }
    for name, graph in graphs.items():
        manifest["components"][name] = graph.write(source, output, prefix + "vision_" + name)
    pos = source.get(VISION + ".embeddings.position_embedding.weight")
    if tuple(pos.shape) != (grid * grid, width):
        raise ValueError(f"Invalid position embeddings shape {pos.shape}")
    pos_name = prefix + "vision_position.f32"
    np.asarray(pos, dtype="<f4", order="C").tofile(output / pos_name)
    manifest["position_embeddings"] = {"file": pos_name, "rows": grid * grid, "cols": width}
    (output / (prefix + "minicpm_v47_vision.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    export_minicpm_v47_vision(args.model_dir, args.output_dir, args.prefix)
