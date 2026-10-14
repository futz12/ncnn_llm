# Copyright (c) 2026 ncnn_llm authors. Apache-2.0.
"""Memory-bounded MiniCPM language conversion: streamed tables, decoder only."""
import gc
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import numpy as np
import torch
from transformers import AutoConfig

from minicpm_v47_vision_export import SafeTensorSource
from qwen35_export import (Qwen35Decoder, PortableGatedNorm, _shape_args,
                           _rename_cache_io, torch_chunk_gated_delta_rule)


def stream_table(source, key, shape, binary, rows_per_chunk=4096):
    """Write one tagged FP32 tensor without allocating the vocabulary table."""
    binary = Path(binary)
    temporary = binary.with_suffix(binary.suffix + ".tmp")
    try:
        with source.safe_open(str(source.files[key]), framework="pt", device="cpu") as checkpoint:
            table = checkpoint.get_slice(key)
            if tuple(table.get_shape()) != tuple(shape):
                raise ValueError(f"checkpoint shape mismatch for {key}")
            with temporary.open("wb") as output:
                output.write(struct.pack("<I", 0))
                for start in range(0, shape[0], rows_per_chunk):
                    block = table[start:start + rows_per_chunk].float().numpy()
                    output.write(np.asarray(block, dtype="<f4").tobytes())
                    del block
        temporary.replace(binary)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_tables(model_dir, output_dir, cfg, prefix):
    source = SafeTensorSource(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab, hidden = int(cfg.vocab_size), int(cfg.hidden_size)
    layers = {
        "embed_token": ("model.language_model.embed_tokens.weight",
                        f"Embed embed 1 1 in0 out0 0={hidden} 1={vocab} 2=0 3={hidden * vocab}"),
        "proj_out": ("lm_head.weight",
                     f"Gemm head 1 1 in0 out0 2=0 3=1 4=0 5=1 6=1 7=0 8={vocab} 9={hidden} 10=-1"),
    }
    for name, (key, layer) in layers.items():
        stem = output_dir / f"{prefix}_{name}.ncnn"
        binary, param = Path(str(stem) + ".bin"), Path(str(stem) + ".param")
        # File existence/size does not establish checkpoint identity. Always
        # regenerate rather than mixing stale language and new vision weights.
        print(f"[stream] {name}: {vocab} x {hidden}", flush=True)
        stream_table(source, key, (vocab, hidden), binary)
        param.write_text("7767517\n2 2\nInput input 0 1 in0\n" + layer + "\n")


def export_language_components(model_dir, output_dir, prefix=None):
    from minicpm_v47_export import load_language_only
    root = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    cfg = root.text_config
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    prefix = prefix or destination.name
    export_tables(model_dir, output_dir, cfg, prefix)
    param = destination / f"{prefix}_decoder.ncnn.param"
    binary = destination / f"{prefix}_decoder.ncnn.bin"
    print("[load] decoder only, without vocabulary embedding/output tables", flush=True)
    model = load_language_only(model_dir, root, component="decoder")
    language = model.model.language_model
    for layer in language.layers:
        if getattr(layer, "layer_type", getattr(layer, "block_type", None)) == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            layer.linear_attn.recurrent_gated_delta_rule = torch_chunk_gated_delta_rule
            layer.linear_attn.norm = PortableGatedNorm(layer.linear_attn.norm)
    decoder = Qwen35Decoder(model).eval()
    seq, hidden = 8, int(cfg.hidden_size)
    torch.manual_seed(0)
    ids = torch.randint(0, min(int(cfg.vocab_size), 50000), (seq,))
    source = SafeTensorSource(model_dir)
    key = "model.language_model.embed_tokens.weight"
    with source.safe_open(str(source.files[key]), framework="pt", device="cpu") as checkpoint:
        table = checkpoint.get_slice(key)
        hidden_in = torch.cat([table[int(token):int(token) + 1].float() for token in ids]).unsqueeze(0)
    mask = torch.triu(torch.ones(seq, seq) * float("-inf"), diagonal=1)[None, None]
    rope_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    pos = torch.arange(seq).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = language.rotary_emb(hidden_in, pos)
    cos, sin = cos[..., :rope_dim // 2], sin[..., :rope_dim // 2]
    linear = sum(kind == "linear_attention" for kind in cfg.layer_types)
    conv_dim = int(cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
                   + cfg.linear_value_head_dim * cfg.linear_num_value_heads)
    conv_kernel = int(cfg.linear_conv_kernel_dim)
    heads, key_dim, value_dim = int(cfg.linear_num_value_heads), int(cfg.linear_key_head_dim), int(cfg.linear_value_head_dim)
    conv_states = [torch.zeros(1, conv_dim, conv_kernel) for _ in range(linear)]
    gdr_states = [torch.zeros(1, heads, key_dim, value_dim) for _ in range(linear)]
    args = (hidden_in, mask, cos, sin, *conv_states, *gdr_states)
    with torch.no_grad():
        reference = language(inputs_embeds=hidden_in, use_cache=False).last_hidden_state
        wrapped = decoder(*args)
        error = float((wrapped[0] - reference).abs().max())
        print(f"[check] trained decoder wrapper vs HF max_abs={error:.7g}", flush=True)
        torch.testing.assert_close(wrapped[0], reference, atol=3e-3, rtol=3e-3)
    np.savez(destination / "_check_ref.npz", in0=hidden_in.numpy().squeeze(0),
             in1=mask.numpy().squeeze(0).squeeze(0), in2=cos.numpy().squeeze(0),
             in3=sin.numpy().squeeze(0), target=wrapped[0].numpy().squeeze(0))
    # /tmp may be tmpfs. Multi-gigabyte traces there would consume RAM again,
    # defeating component-wise loading even after the tensors are released.
    with tempfile.TemporaryDirectory(prefix=".minicpm_decoder_", dir=destination) as temporary:
        ts = Path(temporary) / "decoder.ts"
        torch.jit.trace(decoder, args, check_trace=False).save(str(ts))
        del decoder, language, model, layer, reference, wrapped
        gc.collect()
        shapes = [",".join(_shape_args(length, hidden, rope_dim // 2,
                     conv_dim, conv_kernel, heads, key_dim, value_dim, linear)) for length in (seq, 16)]
        temporary_param, temporary_bin = Path(temporary) / "decoder.param", Path(temporary) / "decoder.bin"
        # The wheel's CLI wrapper imports Torch again. Invoke its bundled
        # executable directly when installed, avoiding a second Python runtime.
        import importlib.util
        import os
        spec = importlib.util.find_spec("pnnx")
        executable = Path(spec.origin).parent / ("pnnx.exe" if os.name == "nt" else "pnnx") if spec and spec.origin else None
        command = [str(executable) if executable and executable.is_file() else "pnnx", str(ts), f"inputshape={shapes[0]}", f"inputshape2={shapes[1]}",
                   "fp16=0", "device=cpu", "moduleop=qwen35_gdr.ShortConv,qwen35_gdr.GatedDeltaRule",
                   f"ncnnparam={temporary_param}", f"ncnnbin={temporary_bin}"]
        print("[convert] decoder TorchScript to ncnn", flush=True)
        for key, suffix in (("pnnxparam", ".pnnx.param"), ("pnnxbin", ".pnnx.bin"),
                            ("pnnxpy", "_pnnx.py"), ("pnnxonnx", ".pnnx.onnx"),
                            ("ncnnpy", "_ncnn.py")):
            command.append(f"{key}={Path(temporary) / ('decoder' + suffix)}")
        subprocess.run(command, check=True, cwd=temporary)
        _rename_cache_io(str(temporary_param), linear, linear)
        shutil.copyfile(temporary_bin, binary)
        shutil.copyfile(temporary_param, param)
    print("[done] memory-bounded language components", flush=True)
