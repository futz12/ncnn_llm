# Fix: Xiaomi-OCR-0 deployment fixes (GEMM / reshape / token alignment)

## Summary

This PR collects the fixes found while deploying Xiaomi-OCR-0 (Qwen3.5-VL hybrid) on ncnn_llm.
All fixes are verified against HuggingFace reference outputs, layer by layer.

**5 bug fixes** — 3 at the inference-kernel/export level, 2 at the application level:

| # | Fix | Level | Impact |
|---|---|---|---|
| 1 | int8 GEMM TILE_N mismatch | kernel | vision `proj_0` maxerr 2.638 → 7.6e-6 |
| 2 | RMSNormGated reshape `0d`→`0c` | graph/export | L0 hidden error 0.08 → 0.0008 (100×) |
| 3 | image token id `memset` corruption | memory | token id `-117901064` → correct |
| 4 | thinking template incomplete | precision | sequence aligned with HF processor |
| 5 | last-token row after image injection | logic | first token `ar` → `MR` |

Plus: `--no-bf16` / `--force-naive` CLI options, MSVC STL compat shims, and
`transformers>=5.10` compatibility in the export script.

---

## 1. int8 GEMM TILE_N mismatch (kernel)

**Symptom**: decoder output garbled; int8 model unusable.

**Root cause**: `gemm_BT_x86` in ncnn's `gemm_x86.cpp` selects `TILE_N` differently
between weight pre-packing (`M=0` → `M<=1` branch → `TILE_N=192`) and forward
(`M=5916` → `M>1` branch → `TILE_N=384`). The packed `BT_data` is split by 192
but read with a 384 stride, so columns 192-767 read the wrong channel.

**Fix**: infer the real `TILE_N` from the pre-packed `BT_data`:
```cpp
if (BT.w > 0 && TILE_K > 0 && BT.w % TILE_K == 0) {
    int bttile_n = BT.w / TILE_K;
    if (bttile_n < TILE_N) TILE_N = bttile_n;
}
```

**Verification**: `proj_0` maxerr 2.638 → **7.6e-6**; vision output std 0.666 → **0.1207** (HF 0.1209).

> Note: the ncnn submodule is pinned to `dcf9478e` and the 9-line fix is provided
> as `ncnn_gemm_tile_n_fix.patch` (see `docs/xiaomi-ocr-deploy-tutorial.md` §5.3).

---

## 2. RMSNormGated reshape `0d` → `0c` (graph/export)

**Symptom**: 24-layer hidden state error 0.43 (HF bf16 only 0.06); logits fully off.

**Root cause**: pnnx emits `"128,*(16,0d)"` for the GDR output reshape. `0d`
resolves to **0** for 3D ncnn Mats (depth unused), collapsing the sequence dim
to a single slice that broadcasts against the full `silu(gate)` tensor.

**Fix**: `0d` → `0c` (sequence length), preserving the per-token layout.
Applied automatically in `export/qwen35_export_xiaomi_ocr_0.py` via `_fix_gdr_reshape()`.

**Verification**: L0 error 0.08 → **0.0008** (100×); all 24 layers aligned with HF.

---

## 3. image token id `memset` corruption (memory)

**Root cause**: `memset(ptr, image_pad_id, h*sizeof(int))` fills **one byte**
(`248056 & 0xFF = 0xF8`) repeated → `0xF8F8F8F8` = `-117901064`.

**Fix**: per-int loop assignment in `inject_image_embeds` (`src/utils/rope_embed.cpp`).

---

## 4. thinking template incomplete (precision)

**Root cause**: `--enable-thinking` emitted `\n` (2 tokens);
HF processor default is `\n\n` (4 tokens).

**Fix**: complete the template in `src/utils/prompt.cpp`.

---

## 5. last-token row after image injection (logic)

**Root cause**: `total_prompt_tokens` computed **before** `inject_image_embeds`
(=22 text tokens), so `row_range(total_prompt_tokens-1, 1)` read row 21
instead of the sequence-final row 1499. lm_head received the wrong hidden state.

**Fix**: move the computation to after injection (=1500) in `src/ncnn_llm_gpt.cpp`.

**Verification**: first token argmax 277 (`ar`) → **17896 (`MR`)**, matching HF.

---

## Additional changes

- **`--no-bf16` / `--force-naive` CLI options** — disable BF16 storage or force
  naive GDR/ShortConv kernels for debugging and precision comparison.
- **MSVC STL compat shims** — provide `__std_max_element_f_` / `__std_min_element_4i`
  for objects compiled against an older STL.
- **transformers>=5.10 compatibility** — handle `partial_rotary_factor` nested
  in `rope_parameters` (fallback to top-level attribute for older versions).
- **Vision resize alignment** — `ceil` → `std::rint` (round-half-to-even) and
  `ncnn_mat_resize` → `ncnn_mat_resize_bicubic` to match HF `smart_resize`.

---

## Verification

All 24 decoder layers aligned with HF (bf16 precision):

| Layer | ncnn vs HF mean_abs | Before |
|---|---|---|
| L0 | 0.0008 | 0.08 |
| L12 | 0.0037 | 0.27 |
| L23 | 0.0147 | 0.43 |

End-to-end OCR (4 test images, all correct):

| Image | Output |
|---|---|
| receipt | `MR. D.I.Y. SDN BHD` |
| qianziwen | `行书千文` |
| dense-equations | `We then define the following variables:` |
| handwrite-formula | `(b) we got Friedmann equation:` |

---

## Files changed

- `src/ncnn_llm_gpt.cpp` — vision resize, token alignment, MSVC shims
- `src/utils/prompt.cpp` — thinking template
- `src/utils/rope_embed.cpp` — image token id fix
- `export/qwen35_export_xiaomi_ocr_0.py` — transformers compat + GDR reshape fix
- `examples/llm_ncnn_run/` — `--no-bf16` / `--force-naive` options
- `ncnn_gemm_tile_n_fix.patch` — 9-line ncnn TILE_N fix
- `ncnn` — submodule pinned to `dcf9478e`
- `docs/` — deployment tutorial, debug case study, learning path, notes

---

## Test plan

- [x] `cmake --build build --config Release --target llm_ncnn_run` compiles clean
- [x] int8 model OCR output matches HF (4 images)
- [x] fp32 model OCR output matches HF (4 images)
- [x] Layer-by-layer hidden state comparison vs HF (24 layers)
- [x] vision feature / mrope cache comparison vs HF
