# Golden bytes for sparkforge's FP4 repack, from FlashInfer's own prep functions.
# _e4m3_to_s0e5m3 and _cute_dsl_pack_fp4_weight are copied verbatim from
# flashinfer@db2373490172a31dbbdc489ef82fe174e6d88f06 flashinfer/gemm/gemm_bf16_fp4_cute_dsl.py
# (Copyright (c) 2026 by FlashInfer team, Apache License 2.0).
import json, torch

def _e4m3_to_s0e5m3(sf_u8):
    f16 = sf_u8.contiguous().view(torch.float8_e4m3fn).to(torch.float16)
    bits = f16.view(torch.int16).to(torch.int32) & 0xFFFF
    return ((bits >> 7) & 0xFF).to(torch.uint8)

_CUTE_DSL_PACK_TILE_K = 16
_CUTE_DSL_PACK_TILE_N = 64
_CUTE_DSL_PACK_INTS_PER_TILE = 128

def _cute_dsl_pack_fp4_weight(b):
    if b.dtype != torch.uint8:
        b = b.view(torch.uint8)
    k_half, n = b.shape
    k = k_half * 2
    device = b.device
    k_tiles = k // _CUTE_DSL_PACK_TILE_K
    n_tiles = n // _CUTE_DSL_PACK_TILE_N
    k_half_per_tile = _CUTE_DSL_PACK_TILE_K // 2
    u32_pos = torch.arange(_CUTE_DSL_PACK_INTS_PER_TILE, device=device, dtype=torch.long)
    u32_idx_local = u32_pos % 2
    lane = (u32_pos // 2) % 32
    n_warp_idx = u32_pos // 64
    tc_col = lane // 4
    tc_row_half = lane % 4
    base_n = n_warp_idx * 8 + tc_col
    byte_k_half_offset = torch.tensor([0, 4, 0, 4], device=device, dtype=torch.long)
    n_offset_stack = torch.tensor([[0, 0, 16, 16], [32, 32, 48, 48]], device=device, dtype=torch.long)
    byte_n_offset = n_offset_stack[u32_idx_local]
    k_half_in_tile = tc_row_half[:, None] + byte_k_half_offset[None, :]
    n_in_tile = base_n[:, None] + byte_n_offset
    within_idx = (k_half_in_tile * _CUTE_DSL_PACK_TILE_N + n_in_tile).reshape(-1)
    tile_bytes = (b.reshape(k_tiles, k_half_per_tile, n_tiles, _CUTE_DSL_PACK_TILE_N)
                  .permute(0, 2, 1, 3).reshape(k_tiles, n_tiles, k_half_per_tile * _CUTE_DSL_PACK_TILE_N))
    gathered = tile_bytes[:, :, within_idx].reshape(k_tiles, n_tiles, _CUTE_DSL_PACK_INTS_PER_TILE, 4)
    return gathered.view(torch.int32).reshape(k_tiles, n_tiles * _CUTE_DSL_PACK_INTS_PER_TILE)

g = torch.Generator().manual_seed(767)
cases = []
for n, k in [(128, 64), (64, 32)]:
    w = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)          # checkpoint layout [N, K/2]
    # Positive E4M3 scale bytes: normals, a few subnormals (exponent 0) and zero.
    sf = torch.randint(0x08, 0x7f, (n, k // 16), generator=g, dtype=torch.uint8)
    sf.view(-1)[:3] = torch.tensor([0x00, 0x01, 0x07], dtype=torch.uint8)
    packed = _cute_dsl_pack_fp4_weight(w.t().contiguous())                            # as _prepare_cute_dsl_sm12x
    sf_out = _e4m3_to_s0e5m3(sf.t().contiguous())                                     # linear SF: no unswizzle
    cases.append({"n": n, "k": k, "weight": w.numpy().tobytes().hex(), "scale": sf.numpy().tobytes().hex(),
                  "packed": packed.contiguous().view(torch.uint8).numpy().tobytes().hex(),
                  "scale_s0e5m3": sf_out.numpy().tobytes().hex()})
print(json.dumps({"source": "flashinfer@db2373490172a31dbbdc489ef82fe174e6d88f06 gemm_bf16_fp4_cute_dsl.py (_cute_dsl_pack_fp4_weight, _e4m3_to_s0e5m3)",
                  "torch": torch.__version__, "cases": cases}))
