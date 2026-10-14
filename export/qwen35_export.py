# Copyright (c) 2026 ncnn_llm authors. All rights reserved.
# Use of this source code is governed by a Apache License 2.0.

"""Export Qwen3.5 text decoders to ncnn.

The text tower mixes regular attention with Qwen3.5's gated delta rule.  The
regular attention is exported as ncnn SDPA and rewritten to KV-cache form;
linear-attention layers are kept as the project's ShortConv/GatedDeltaRule
custom operators.  The command deliberately starts from fp32 HF weights and
only then creates BF16 and block-INT8 copies::

  python export/qwen35_export.py -m models/qwen3.5_2b \
      -o assets/qwen3.5_2b --keep-fp32

The same command works for a 4B/9B checkpoint once its local HF directory is
available.  Vision export is intentionally separate: the decoder assets made
here are usable for text chat and keep the model graph small enough to verify
before adding the much larger vision tower.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import add_kvcache  # noqa: E402
import bf16_convert  # noqa: E402
import quantize_model  # noqa: E402
from llama_export import write_tokenizer  # noqa: E402
from qwen35_vision_export import export_qwen35_vision  # noqa: E402
from qwen35_gdr import GatedDeltaRule, ShortConv, torch_chunk_gated_delta_rule  # noqa: E402


def _rms(x, weight, eps):
    y = x
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    # Qwen3.5 RMSNorm stores an offset from one, unlike Llama RMSNorm.
    return y * (1.0 + weight)


def _gated_rms(x, gate, weight, eps):
    y = x
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y * weight * F.silu(gate)


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _apply_rope(q, k, cos_half, sin_half, rotary_dim, head_dim=None):
    # Runtime cos/sin inputs contain half-width values.  HF's Qwen3.5 rotary
    # embedding duplicates them across the rotated width before applying
    # rotate_half, while the remaining head dimensions pass through.
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
    if rotary_dim == head_dim:
        # A full-width rotary head has no pass-through tail. Emitting its
        # zero-width Crop/Concat is invalid in ncnn even though Torch accepts it.
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class Qwen35Embed(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.embed_tokens = language_model.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class Qwen35LMHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, hidden):
        return self.lm_head(hidden)


class PortableGatedNorm(nn.Module):
    """CPU reference for the optional FLA/Triton fused gated RMSNorm."""

    def __init__(self, source):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.eps = float(getattr(source, "variance_epsilon", getattr(source, "eps", 1e-6)))

    def forward(self, x, gate):
        return _gated_rms(x, gate, self.weight, self.eps)


class Qwen35Decoder(nn.Module):
    """Full-sequence decoder with explicit linear-attention states.

    Full attention states are intentionally absent here.  ``add_kvcache``
    rewrites every SDPA node after pnnx, which gives the runtime the same cache
    contract as the existing Qwen3.5 0.8B asset.  Conv/GDR states are explicit
    inputs and outputs because they are custom ncnn operators rather than SDPA.
    """

    def __init__(self, model):
        super().__init__()
        # Keep one canonical module path. Rebinding individual layers under a
        # second attribute makes pnnx emit duplicate Gemm inputs.
        self.language_model = model.model.language_model
        self.cfg = self.language_model.config
        self.num_layers = len(self.language_model.layers)
        self.num_q = int(self.cfg.num_attention_heads)
        self.num_kv = int(self.cfg.num_key_value_heads)
        self.hidden_size = int(self.cfg.hidden_size)
        self.head_dim = int(self.cfg.head_dim)
        self.rope_dim = int(self.head_dim * float(self.cfg.partial_rotary_factor))
        self.eps = float(self.cfg.rms_norm_eps)
        self.sconv_indices = []
        self.gdr_indices = []
        sconv = []
        gdr = []
        for layer in self.language_model.layers:
            if getattr(layer, "layer_type", getattr(layer, "block_type", None)) == "linear_attention":
                la = layer.linear_attn
                self.sconv_indices.append(len(sconv))
                self.gdr_indices.append(len(gdr))
                sconv.append(ShortConv(la.conv1d.weight, la.conv_kernel_size))
                gdr.append(GatedDeltaRule(la.A_log, la.dt_bias, la.head_k_dim,
                                          la.head_v_dim, la.num_k_heads,
                                          la.num_v_heads, self.eps))
            else:
                self.sconv_indices.append(-1)
                self.gdr_indices.append(-1)
        self.sconv_ops = nn.ModuleList(sconv)
        self.gdr_ops = nn.ModuleList(gdr)

    def _full_attention(self, x, layer, mask, cos_half, sin_half):
        attn = layer.self_attn
        bsz, seq_len, _ = x.shape
        q = attn.q_proj(x).view(bsz, seq_len, self.num_q, self.head_dim * 2)
        q, gate = q.chunk(2, dim=-1)
        gate = gate.reshape(bsz, seq_len, -1)
        k = attn.k_proj(x).view(bsz, seq_len, self.num_kv, self.head_dim)
        v = attn.v_proj(x).view(bsz, seq_len, self.num_kv, self.head_dim)
        # Inline Qwen3.5 RMSNorm here.  Calling the HF module introduces
        # dtype/device ``aten::to`` nodes even though this export is FP32.
        q = _rms(q, attn.q_norm.weight, attn.q_norm.eps).transpose(1, 2)
        k = _rms(k, attn.k_norm.weight, attn.k_norm.eps).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = _apply_rope(q, k, cos_half, sin_half, self.rope_dim, self.head_dim)
        if self.num_kv != self.num_q:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=attn.scaling, enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=attn.scaling)
        out = out.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        out = out * torch.sigmoid(gate)
        return attn.o_proj(out)

    def _linear_attention(self, x, layer, conv_state, gdr_state, sconv, gdr):
        la = layer.linear_attn
        bsz, seq_len, _ = x.shape
        mixed = la.in_proj_qkv(x).transpose(1, 2)
        z = la.in_proj_z(x).reshape(bsz, seq_len, -1, la.head_v_dim)
        b = la.in_proj_b(x)
        a = la.in_proj_a(x)
        mixed, new_conv = sconv(la.conv1d.weight, mixed, conv_state)
        mixed = mixed.transpose(1, 2)
        query, key, value = torch.split(mixed, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
        query = query.reshape(bsz, seq_len, -1, la.head_k_dim)
        key = key.reshape(bsz, seq_len, -1, la.head_k_dim)
        value = value.reshape(bsz, seq_len, -1, la.head_v_dim)
        # GatedDeltaRule performs GQA expansion itself. Keeping the query/key
        # tensors at their native key-head count is required for models whose
        # linear attention uses a 16 -> 32 head ratio (Qwen3.5 4B/9B).
        core, new_gdr = gdr(la.A_log, la.dt_bias, b, a, query, key, value, gdr_state)
        core = core.reshape(-1, la.head_v_dim)
        z = z.reshape(-1, la.head_v_dim)
        core = _gated_rms(core, z, la.norm.weight, la.layer_norm_epsilon)
        core = core.reshape(bsz, seq_len, -1)
        return la.out_proj(core), new_conv, new_gdr

    def forward(self, hidden, attn_mask, cos_half, sin_half, *states):
        # Keep the state interface variadic so the same exporter handles the
        # 18 linear layers in 2B and the 24 linear layers in 4B/9B.
        state_count = len(self.sconv_ops)
        conv_states = states[:state_count]
        gdr_states = states[state_count:state_count * 2]
        conv_outs = []
        gdr_outs = []
        si = gi = 0
        x = hidden
        for li, layer in enumerate(self.language_model.layers):
            residual = x
            x = _rms(x, layer.input_layernorm.weight, self.eps)
            if getattr(layer, "layer_type", getattr(layer, "block_type", None)) == "linear_attention":
                x, new_conv, new_gdr = self._linear_attention(
                    x, layer, conv_states[si], gdr_states[gi],
                    self.sconv_ops[si], self.gdr_ops[gi])
                conv_outs.append(new_conv)
                gdr_outs.append(new_gdr)
                si += 1
                gi += 1
            else:
                x = self._full_attention(x, layer, attn_mask, cos_half, sin_half)
            x = residual + x
            residual = x
            x = _rms(x, layer.post_attention_layernorm.weight, self.eps)
            mlp = layer.mlp
            x = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
            x = residual + x
        x = _rms(x, self.language_model.norm.weight, self.eps)
        return (x, *conv_outs, *gdr_outs)


def _rename_cache_io(param_path, conv_count, gdr_count):
    """Collapse pnnx's one-input-per-state form into runtime cache groups."""
    with open(param_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if len(lines) < 2 or lines[0].strip() != "7767517":
        raise RuntimeError(f"not an ncnn param file: {param_path}")

    replacements = {}
    for i in range(conv_count):
        replacements[f"in{4 + i}"] = f"cache_conv{i}"
        replacements[f"out{1 + i}"] = f"out_cache_conv{i}"
    for i in range(gdr_count):
        replacements[f"in{4 + conv_count + i}"] = f"cache_gdr{i}"
        replacements[f"out{1 + conv_count + i}"] = f"out_cache_gdr{i}"

    # pnnx can serialize the GQA inputs as query,value,key when value has
    # more heads than key. The runtime contract is always query,key,value.
    reshape_heads = {}
    reshape_src = {}
    slice_pos = {}
    for line in lines[2:]:
        fields = line.split()
        if len(fields) < 7 or fields[0] != "Reshape":
            continue
        try:
            bottom_count = int(fields[2])
            top_count = int(fields[3])
            outputs = fields[4 + bottom_count:4 + bottom_count + top_count]
            attrs = fields[4 + bottom_count + top_count:]
            head_attr = next(x for x in attrs if x.startswith("1="))
            reshape_heads[outputs[0]] = int(float(head_attr[2:]))
            reshape_src[outputs[0]] = fields[4]  # Reshape 的输入 blob
        except (ValueError, StopIteration, IndexError):
            continue

    # Slice 层：top blob -> 切片序号（q=0,k=1,v=2 的位置契约）
    for line in lines[2:]:
        fields = line.split()
        if len(fields) < 7 or fields[0] != "Slice":
            continue
        try:
            bottom_count = int(fields[2])
            top_count = int(fields[3])
            outputs = fields[4 + bottom_count:4 + bottom_count + top_count]
            for idx, name in enumerate(outputs):
                slice_pos[name] = idx
        except (ValueError, IndexError):
            continue

    body = []
    for line in lines[2:]:
        fields = line.split()
        if fields and fields[0] == "Input" and fields[1] in replacements:
            continue
        # Make the rewrite idempotent when an already rewritten parameter
        # file is reused. Older exports may contain duplicate grouped cache
        # Input lines from a second pass.
        if fields and fields[0] == "Input" and fields[1] in ("conv_cache", "gdr_cache"):
            continue
        if fields and fields[0] == "qwen35_gdr.ShortConv":
            fields[0] = "ShortConv"
        elif fields and fields[0] == "qwen35_gdr.GatedDeltaRule":
            fields[0] = "GatedDeltaRule"
        if fields and fields[0] in ("GatedDeltaRule", "qwen35_gdr.GatedDeltaRule"):
            bottom_count = int(fields[2])
            bottom_ids = fields[4:4 + bottom_count]
            if bottom_count >= 8:
                q_heads = reshape_heads.get(bottom_ids[4])
                k_heads = reshape_heads.get(bottom_ids[5])
                v_heads = reshape_heads.get(bottom_ids[6])
                if q_heads is not None and q_heads == v_heads and k_heads != q_heads:
                    bottom_ids[5], bottom_ids[6] = bottom_ids[6], bottom_ids[5]
                    fields[4:4 + bottom_count] = bottom_ids
                elif (q_heads is not None and k_heads is not None and v_heads is not None
                      and q_heads == k_heads == v_heads):
                    # 头数全等时头数启发式失效（如 qwen3.5-2b: 16/16/16）：
                    # 回退到 Slice 位置溯源 —— q/k/v 必须按切片顺序 0/1/2 排列。
                    # reshape_src[blob] = 其输入 blob；slice_pos[blob] = 它在 Slice tops 中的序号
                    pos = [slice_pos.get(reshape_src.get(b, ""), -1) for b in bottom_ids[4:7]]
                    if all(p >= 0 for p in pos) and pos != sorted(pos):
                        order = sorted(range(3), key=lambda i: pos[i])
                        qkv = [bottom_ids[4 + i] for i in order]
                        bottom_ids[4:7] = qkv
                        fields[4:4 + bottom_count] = bottom_ids
        fields = [replacements.get(x, x) for x in fields]
        # Keep the custom-op contract as (weight, mixed_qkv, state). pnnx
        # can emit the state before mixed_qkv, which makes prefill consume an
        # empty state as its sequence input and can crash the runtime.
        if (fields and fields[0] == "ShortConv" and len(fields) >= 9
                and fields[5].startswith("cache_conv")
                and not fields[6].startswith("cache_conv")):
            fields[5], fields[6] = fields[6], fields[5]
        body.append(" ".join(fields) if fields else line)

    def grouped(name, blobs):
        return "Input %-24s 0 %d %s" % (name, len(blobs), " ".join(blobs))

    first_input = next((i for i, line in enumerate(body)
                        if line.split() and line.split()[0] == "Input"), 0)
    grouped_inputs = [
        grouped("conv_cache", [f"cache_conv{i}" for i in range(conv_count)]),
        grouped("gdr_cache", [f"cache_gdr{i}" for i in range(gdr_count)]),
    ]
    body[first_input:first_input] = grouped_inputs

    def noutputs(line):
        f = line.split()
        return int(f[3]) if len(f) >= 4 else 0

    nonempty = [line for line in body if line.strip()]
    blobs = sum(noutputs(line) for line in nonempty)
    out = [lines[0], f"{len(nonempty)} {blobs}", *body]
    with open(param_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")


def _graph_stats(path):
    types = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().splitlines()[2:]:
            fields = line.split()
            if fields:
                types[fields[0]] += 1
    return types


def _shape_args(seq_len, hidden, rope_half, conv_dim, conv_kernel, heads,
                gdr_k_dim, gdr_v_dim, state_count):
    shapes = [
        f"[1,{seq_len},{hidden}]f32",
        f"[1,1,{seq_len},{seq_len}]f32",
        f"[1,{seq_len},{rope_half}]f32",
        f"[1,{seq_len},{rope_half}]f32",
    ]
    shapes += [f"[1,{conv_dim},{conv_kernel}]f32"] * state_count
    shapes += [f"[1,{heads},{gdr_k_dim},{gdr_v_dim}]f32"] * state_count
    return shapes


def export(model_dir, output_dir, int8_dir="", prefix="", keep_fp32=True,
           do_bf16=True, do_int8=True, cross_check=True, bits=8,
           vision_exporter=None):
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    os.makedirs(output_dir, exist_ok=True)
    prefix = prefix or os.path.basename(output_dir.rstrip("/\\"))
    root_cfg = AutoConfig.from_pretrained(model_dir)
    tie_word_embeddings = getattr(root_cfg, "tie_word_embeddings", False)
    cfg = root_cfg.text_config if hasattr(root_cfg, "text_config") else root_cfg
    hidden = int(cfg.hidden_size)
    layers = int(cfg.num_hidden_layers)
    full_layers = sum(x == "full_attention" for x in cfg.layer_types)
    linear_layers = sum(x == "linear_attention" for x in cfg.layer_types)
    head_dim = int(cfg.head_dim)
    rope_dim = int(head_dim * float(cfg.partial_rotary_factor))
    rope_theta = float(cfg.rope_parameters.get("rope_theta", 10000000.0))
    conv_dim = int(cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
                   + cfg.linear_value_head_dim * cfg.linear_num_value_heads)
    conv_kernel = int(cfg.linear_conv_kernel_dim)
    gdr_k_dim = int(cfg.linear_key_head_dim)
    gdr_heads = int(cfg.linear_num_value_heads)
    gdr_v_dim = int(cfg.linear_value_head_dim)
    paths = {
        "embed_param": os.path.join(output_dir, f"{prefix}_embed_token.ncnn.param"),
        "embed_bin": os.path.join(output_dir, f"{prefix}_embed_token.ncnn.bin"),
        "dec_param": os.path.join(output_dir, f"{prefix}_decoder.ncnn.param"),
        "dec_bin": os.path.join(output_dir, f"{prefix}_decoder.ncnn.bin"),
        "head_param": os.path.join(output_dir, f"{prefix}_proj_out.ncnn.param"),
        "head_bin": os.path.join(output_dir, f"{prefix}_proj_out.ncnn.bin"),
    }
    tok = write_tokenizer(model_dir, output_dir)
    ref_cache = os.path.join(output_dir, "_check_ref.npz")
    have_fp32 = all(os.path.isfile(p) and os.path.getsize(p) for p in paths.values())
    if not have_fp32:
        print(f"=== Loading {model_dir} in fp32 ===", flush=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
        # Transformers may dispatch to FLA/Triton even for CPU tensors when
        # those optional packages are installed. Replace those two kernels in
        # the reference copy so validation remains portable and deterministic.
        for layer in model.model.language_model.layers:
            if getattr(layer, "layer_type", getattr(layer, "block_type", None)) == "linear_attention":
                layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                layer.linear_attn.recurrent_gated_delta_rule = torch_chunk_gated_delta_rule
                layer.linear_attn.norm = PortableGatedNorm(layer.linear_attn.norm)
        embed = Qwen35Embed(model.model.language_model).eval()
        decoder = Qwen35Decoder(model).eval()
        head = Qwen35LMHead(model).eval()
        seq = 8
        torch.manual_seed(0)
        ids = torch.randint(0, min(int(cfg.vocab_size), 50000), (1, seq), dtype=torch.long)
        hidden_in = embed(ids)
        mask = torch.triu(torch.ones(seq, seq) * float("-inf"), diagonal=1)[None, None]
        pos = torch.arange(seq, dtype=torch.long).view(1, -1)
        # transformers>=5.10: qwen3_5 rotary_emb 走 mrope 路径，要求 (3, bs, seq)
        pos3 = pos.unsqueeze(0).expand(3, -1, -1)
        cos_full, sin_full = model.model.language_model.rotary_emb(hidden_in, pos3)
        cos = cos_full[..., :rope_dim // 2]
        sin = sin_full[..., :rope_dim // 2]
        conv_states = [torch.zeros(1, conv_dim, conv_kernel) for _ in range(linear_layers)]
        gdr_states = [torch.zeros(1, gdr_heads, gdr_k_dim, gdr_v_dim)
                      for _ in range(linear_layers)]
        dec_args = (hidden_in, mask, cos, sin, *conv_states, *gdr_states)
        with torch.no_grad():
            ref_hidden = model.model.language_model(inputs_embeds=hidden_in,
                                                    use_cache=False).last_hidden_state
            wrapped = decoder(*dec_args)
            err = float((wrapped[0] - ref_hidden).abs().max())
        print(f"[check] decoder wrapper vs HF: max|d|={err:.3e}", flush=True)
        if err > 3e-3:
            raise RuntimeError(f"Qwen3.5 wrapper mismatch: {err}")
        with tempfile.TemporaryDirectory(prefix="ncnn_qwen35_") as td:
            ts_embed = os.path.join(td, "embed.ts")
            ts_dec = os.path.join(td, "decoder.ts")
            ts_head = os.path.join(td, "head.ts")
            torch.jit.trace(embed, (ids,), check_trace=False).save(ts_embed)
            torch.jit.trace(decoder, dec_args, check_trace=False).save(ts_dec)
            torch.jit.trace(head, (wrapped[0],), check_trace=False).save(ts_head)
            shape1 = _shape_args(seq, hidden, rope_dim // 2, conv_dim,
                                 conv_kernel, gdr_heads, gdr_k_dim, gdr_v_dim,
                                 linear_layers)
            shape2 = _shape_args(16, hidden, rope_dim // 2, conv_dim,
                                 conv_kernel, gdr_heads, gdr_k_dim, gdr_v_dim,
                                 linear_layers)
            cmds = [
                f'pnnx "{ts_embed}" inputshape=[1,{seq}]i64 inputshape2=[1,16]i64 fp16=0 device=cpu ncnnparam="{paths["embed_param"]}" ncnnbin="{paths["embed_bin"]}"',
                f'pnnx "{ts_dec}" inputshape={",".join(shape1)} inputshape2={",".join(shape2)} fp16=0 device=cpu moduleop=qwen35_gdr.ShortConv,qwen35_gdr.GatedDeltaRule ncnnparam="{paths["dec_param"]}" ncnnbin="{paths["dec_bin"]}"',
                f'pnnx "{ts_head}" inputshape=[1,{seq},{hidden}]f32 fp16=0 device=cpu ncnnparam="{paths["head_param"]}" ncnnbin="{paths["head_bin"]}"',
            ]
            for cmd in cmds:
                print("[run]", cmd[:240] + ("..." if len(cmd) > 240 else ""), flush=True)
                subprocess.run(cmd, shell=True, check=True)
        _rename_cache_io(paths["dec_param"], linear_layers, linear_layers)
        np.savez(ref_cache,
                 in0=hidden_in.detach().numpy().squeeze(0),
                 in1=mask.detach().numpy().squeeze(0).squeeze(0),
                 in2=cos.detach().numpy().squeeze(0),
                 in3=sin.detach().numpy().squeeze(0),
                 target=wrapped[0].detach().numpy().squeeze(0))
        del model
    else:
        print("=== Reusing existing fp32 graph ===", flush=True)

    stats = _graph_stats(paths["dec_param"])
    print(f"[graph] {dict(stats)}", flush=True)
    custom_sc = stats.get("ShortConv", 0) + stats.get("qwen35_gdr.ShortConv", 0)
    custom_gdr = stats.get("GatedDeltaRule", 0) + stats.get("qwen35_gdr.GatedDeltaRule", 0)
    if custom_gdr != linear_layers or custom_sc != linear_layers:
        raise RuntimeError("custom Qwen3.5 operators were expanded by pnnx")
    if stats.get("SDPA", 0) != full_layers:
        raise RuntimeError(f"expected {full_layers} SDPA layers, found {stats.get('SDPA', 0)}")
    if cross_check and os.path.isfile(ref_cache):
        print("[check] ncnn execution skipped: the stock Python ncnn binding "
              "does not register Qwen3.5 custom operators", flush=True)

    add_kvcache.add_kvcache_to_param(paths["dec_param"], scale=None)
    proj_out_bin = os.path.basename(paths["embed_bin"]) if tie_word_embeddings else os.path.basename(paths["head_bin"])
    if tie_word_embeddings and os.path.isfile(paths["head_bin"]) and paths["head_bin"] != paths["embed_bin"]:
        try:
            os.remove(paths["head_bin"])
        except Exception:
            pass

    vision_setting = {"type": "close"}
    if hasattr(root_cfg, "vision_config") and root_cfg.vision_config:
        export_vision = vision_exporter or export_qwen35_vision
        v_cfg = export_vision(model_dir, output_dir, prefix=f"{prefix}_", do_bf16=False)
        if v_cfg:
            vision_setting = v_cfg

    model_json = {
        "type": "qwen3.5",
        "params": {
            "embed_token_param": os.path.basename(paths["embed_param"]),
            "embed_token_bin": os.path.basename(paths["embed_bin"]),
            "decoder_param": os.path.basename(paths["dec_param"]),
            "decoder_bin": os.path.basename(paths["dec_bin"]),
            "proj_out_param": os.path.basename(paths["head_param"]),
            "proj_out_bin": proj_out_bin,
        },
        "tokenizer": {
            "type": "bbpe" if tok["use_byte_encoder"] else "bpe",
            "vocab_file": "vocab.txt", "merges_file": "merges.txt",
            "bos": tok["bos"], "eos": tok["eos"], "pad": tok["pad"],
            "eos_ids": tok["eos_ids"],
            "unk": tok["unk"], "additional_special_tokens": tok["added"],
        },
        "setting": {
            "attn_cnt": full_layers, "sconv_cnt": linear_layers,
            "gdr_cnt": linear_layers,
            "rope": {"type": "RoPE", "rope_head_dim": rope_dim,
                     "rope_theta": rope_theta},
            "vision": vision_setting,
            "functions": {"type": "tool_call", "tool_call_id": "<tool_call>",
                          "tool_call_end_id": "</tool_call>"},
        },
    }
    with open(os.path.join(output_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2, ensure_ascii=False)

    if os.path.isfile(ref_cache):
        try:
            os.remove(ref_cache)
        except Exception:
            pass

    if keep_fp32:
        fp32_dir = output_dir + "_fp32"
        os.makedirs(fp32_dir, exist_ok=True)
        all_files = list(model_json["params"].values()) + ["vocab.txt", "merges.txt", "model.json"]
        if model_json.get("setting", {}).get("vision", {}).get("type") != "close":
            v = model_json["setting"]["vision"]
            for k in ["vision_embed_patch_param", "vision_embed_patch_bin",
                      "vision_embed_pos_param", "vision_embed_pos_bin",
                      "vision_encoder_param", "vision_encoder_bin"]:
                if k in v:
                    all_files.append(v[k])
        for name in all_files:
            src = os.path.join(output_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(fp32_dir, name))

    if do_bf16:
        tool = bf16_convert.find_ncnnoptimize()
        if not tool:
            raise RuntimeError("ncnnoptimize not found; build with NCNN_LLM_ENABLE_TOOLS=ON")
        bf16_convert.convert_model_dir(output_dir, tool, flag=bf16_convert.BF16)
    if do_int8:
        tool = quantize_model.find_ncnnllm2int()
        if not tool:
            raise RuntimeError("ncnnllm2int not found; build with NCNN_LLM_ENABLE_TOOLS=ON")
        int8_dir = int8_dir or output_dir.rstrip("/\\") + f"_int{bits}"
        quantize_model.quantize_model_dir(output_dir, int8_dir, tool,
                                          bits=bits, block=64, method="minmax")
    print(f"[success] {'BF16' if do_bf16 else 'FP32'}: {output_dir}")
    if do_int8:
        print(f"[success] INT{bits}: {int8_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", "-m", required=True)
    ap.add_argument("--output-dir", "-o", required=True)
    ap.add_argument("--int8-dir", default="")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--bits", type=int, default=8, choices=[4, 6, 8])
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--no-cross-check", action="store_true")
    ap.add_argument("--no-keep-fp32", action="store_true")
    args = ap.parse_args()
    export(args.model_dir, args.output_dir, args.int8_dir, args.prefix,
           keep_fp32=not args.no_keep_fp32, do_bf16=not args.no_bf16,
           do_int8=not args.no_int8, cross_check=not args.no_cross_check,
           bits=args.bits)


if __name__ == "__main__":
    main()
