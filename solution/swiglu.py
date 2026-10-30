"""Fused Triton kernels for Qwen2 training on a T4 (sm_75).

Kernels in this file
--------------------
* SwiGLU  (silu(gate) * up), fwd + bwd      -- the original example kernels, kept as-is
                                               (int64 offsets added for very large tensors).
* RMSNorm (Qwen2RMSNorm), fwd + bwd         -- OWN kernel. One pass over the row instead of the
                                               ~6 elementwise PyTorch ops (+ fp32 copies) of the
                                               HF implementation; backward also fuses the weight
                                               gradient (per-program partial sums, no atomics).
* RoPE    (apply_rotary_pos_emb), fwd + bwd -- OWN kernel. q and k rotated in place of the
                                               slice / neg / cat / mul / add chain of HF.

Everything is mathematically identical to the HF reference. Accumulation is in fp32, and the
"cast normalized x to input dtype, then multiply by weight" order of Qwen2RMSNorm is preserved.
Anything unusual (non-CUDA tensor, explicit position_ids, odd shapes, other activation, ...)
falls back to the original HF code path.

Entry point: patch_qwen2_mlp() (the name the example used). It installs the MLP patch and,
while PATCH_EXTRAS is True, the RMSNorm and RoPE patches as well. patch_all() does the same.
Importing this module has no side effects.
"""

import warnings

import torch
import triton
import triton.language as tl

BLOCK = 1024
PATCH_EXTRAS = True  # set to False to patch only the MLP (as in the original example)


# ---------------------------------------------------------------------------------------------
# SwiGLU: silu(gate) * up  (memory bound, already close to the roofline; kept from the example)
# ---------------------------------------------------------------------------------------------
@triton.jit
def example_swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def example_swiglu_bwd_kernel(grad_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements,
                              BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    dup = grad * gate * sig
    dgate = grad * up * sig * (1.0 + gate * (1.0 - sig))
    tl.store(dgate_ptr + offsets, dgate.to(dgate_ptr.dtype.element_ty), mask=mask)
    tl.store(dup_ptr + offsets, dup.to(dup_ptr.dtype.element_ty), mask=mask)


class SwiGLU(torch.autograd.Function):
    """silu(gate) * up in one launch each way instead of three PyTorch ops."""

    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        grad = grad.to(gate.dtype).contiguous()
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](grad, gate, up, dgate, dup, n,
                                                           BLOCK=BLOCK)
        return dgate, dup


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)


# ---------------------------------------------------------------------------------------------
# RMSNorm (own kernel)
#   y = weight * ( (x * rstd).to(x.dtype) ),   rstd = 1/sqrt(mean(x^2) + eps)
#   g = dy * weight
#   dx = rstd * (g - xhat * mean(g * xhat)),   dweight = sum_rows(dy * xhat)
# ---------------------------------------------------------------------------------------------
@triton.jit
def _rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, stride_x, stride_y, N, eps,
                        BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    # HF: weight * hidden_states.to(input_dtype)  -> round to the input dtype first
    xhat = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride_y + cols, (xhat * w).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr,
                        stride_dy, stride_x, stride_dx, n_rows, N, rows_per_prog,
                        BLOCK_N: tl.constexpr, COMPUTE_DW: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK_N], dtype=tl.float32)
    row_start = pid * rows_per_prog
    row_end = tl.minimum(row_start + rows_per_prog, n_rows)
    for row in range(row_start, row_end):
        r = row.to(tl.int64)
        dy = tl.load(dy_ptr + r * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + r * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + r)
        xhat = x * rstd
        dyw = dy * w
        c = tl.sum(dyw * xhat, axis=0) / N
        dx = rstd * (dyw - xhat * c)
        tl.store(dx_ptr + r * stride_dx + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        if COMPUTE_DW:
            dw += dy * xhat
    if COMPUTE_DW:
        tl.store(dw_ptr + pid * N + cols, dw, mask=mask)


_SM_COUNT = {}


def _num_sms(device):
    if device.type != "cuda":
        return 8
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        N = shape[-1]
        x2 = x.reshape(-1, N)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        weight = weight.contiguous()
        n_rows = x2.shape[0]
        out_dtype = torch.promote_types(x.dtype, weight.dtype)  # same as `weight * x` in HF
        y = torch.empty((n_rows, N), dtype=out_dtype, device=x.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        block_n = triton.next_power_of_2(N)
        num_warps = min(max(block_n // 256, 1), 8)
        _rmsnorm_fwd_kernel[(n_rows,)](x2, weight, y, rstd, x2.stride(0), y.stride(0), N, eps,
                                       BLOCK_N=block_n, num_warps=num_warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        n_rows, N = x2.shape
        dy2 = dy.reshape(-1, N)
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dx = torch.empty((n_rows, N), dtype=x2.dtype, device=x2.device)
        need_dw = ctx.needs_input_grad[1]  # False for frozen weights (e.g. LoRA): skip the work
        n_progs = max(1, min(n_rows, _num_sms(x2.device) * 4))
        rows_per_prog = triton.cdiv(n_rows, n_progs)
        n_progs = triton.cdiv(n_rows, rows_per_prog)
        dw_part = (torch.empty((n_progs, N), dtype=torch.float32, device=x2.device)
                   if need_dw else dx)  # dummy pointer when unused
        block_n = triton.next_power_of_2(N)
        num_warps = min(max(block_n // 256, 1), 8)
        _rmsnorm_bwd_kernel[(n_progs,)](dy2, x2, weight, rstd, dx, dw_part,
                                        dy2.stride(0), x2.stride(0), dx.stride(0),
                                        n_rows, N, rows_per_prog,
                                        BLOCK_N=block_n, COMPUTE_DW=need_dw,
                                        num_warps=num_warps)
        dw = dw_part.sum(0).to(weight.dtype) if need_dw else None
        return dx.view(ctx.shape), dw, None


# ---------------------------------------------------------------------------------------------
# RoPE (own kernel)
#   HF: y = x * cos + rotate_half(x) * sin,   rotate_half(x) = cat(-x2, x1)
#   forward : y1 = x1*c1 - x2*s1,  y2 = x2*c2 + x1*s2
#   backward: dx1 = dy1*c1 + dy2*s2,  dx2 = dy2*c2 - dy1*s1   (transpose of the rotation)
#   One program per (batch, position); all heads of that position are handled as a 2D tile.
# ---------------------------------------------------------------------------------------------
@triton.jit
def _rope_kernel(x_ptr, out_ptr, cos_ptr, sin_ptr, n_heads, seq_len,
                 sx_b, sx_h, sx_t, so_b, so_h, so_t,
                 HALF: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
                 BACKWARD: tl.constexpr):
    pid = tl.program_id(0)
    b = (pid // seq_len).to(tl.int64)
    t = (pid % seq_len).to(tl.int64)
    d = tl.arange(0, BLOCK_D)
    h = tl.arange(0, BLOCK_H)
    dmask = d < HALF
    mask = (h < n_heads)[:, None] & dmask[None, :]

    cs_base = pid.to(tl.int64) * (2 * HALF)
    c1 = tl.load(cos_ptr + cs_base + d, mask=dmask, other=0.0).to(tl.float32)[None, :]
    c2 = tl.load(cos_ptr + cs_base + HALF + d, mask=dmask, other=0.0).to(tl.float32)[None, :]
    s1 = tl.load(sin_ptr + cs_base + d, mask=dmask, other=0.0).to(tl.float32)[None, :]
    s2 = tl.load(sin_ptr + cs_base + HALF + d, mask=dmask, other=0.0).to(tl.float32)[None, :]

    h64 = h[:, None].to(tl.int64)
    x_off = b * sx_b + t * sx_t + h64 * sx_h + d[None, :]
    o_off = b * so_b + t * so_t + h64 * so_h + d[None, :]
    x1 = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + x_off + HALF, mask=mask, other=0.0).to(tl.float32)

    if BACKWARD:
        y1 = x1 * c1 + x2 * s2
        y2 = x2 * c2 - x1 * s1
    else:
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
    tl.store(out_ptr + o_off, y1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + o_off + HALF, y2.to(out_ptr.dtype.element_ty), mask=mask)


def _rope_apply(x, cos, sin, backward):
    """x: [B, H, T, D] (any strides, last stride 1); cos/sin: contiguous [B, T, D]."""
    B, H, T, D = x.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    out = torch.empty_like(x)
    half = D // 2
    block_d = triton.next_power_of_2(half)
    block_h = triton.next_power_of_2(H)
    num_warps = 4 if block_h * block_d >= 2048 else 2
    _rope_kernel[(B * T,)](x, out, cos, sin, H, T,
                           x.stride(0), x.stride(1), x.stride(2),
                           out.stride(0), out.stride(1), out.stride(2),
                           HALF=half, BLOCK_H=block_h, BLOCK_D=block_d, BACKWARD=backward,
                           num_warps=num_warps)
    return out


class RoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        cos = cos.contiguous()
        sin = sin.contiguous()
        ctx.save_for_backward(cos, sin)
        return _rope_apply(q, cos, sin, False), _rope_apply(k, cos, sin, False)

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        return _rope_apply(dq, cos, sin, True), _rope_apply(dk, cos, sin, True), None, None


# ---------------------------------------------------------------------------------------------
# Patches (class / module-level, so they work before or after Soup builds the model)
# ---------------------------------------------------------------------------------------------
def patch_qwen2_mlp_only():
    """Send Qwen2MLP.forward through the fused SwiGLU kernel."""
    from transformers.models.qwen2 import modeling_qwen2

    if getattr(modeling_qwen2.Qwen2MLP, "_fused_swiglu_patched", False):
        return
    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda:
            return original(self, x)
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = forward
    modeling_qwen2.Qwen2MLP._fused_swiglu_patched = True


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused RMSNorm kernel."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls, "_fused_rmsnorm_patched", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        n = hidden_states.shape[-1]
        if (not hidden_states.is_cuda or hidden_states.numel() == 0 or n > 16384
                or self.weight.dim() != 1 or self.weight.shape[0] != n
                or self.weight.device != hidden_states.device):
            return original(self, hidden_states)
        return RMSNormFn.apply(hidden_states, self.weight, self.variance_epsilon)

    cls.forward = forward
    cls._fused_rmsnorm_patched = True


def patch_qwen2_rope():
    """Replace modeling_qwen2.apply_rotary_pos_emb (looked up as a global on every call)."""
    from transformers.models.qwen2 import modeling_qwen2

    if getattr(modeling_qwen2, "_fused_rope_patched", False):
        return
    original = modeling_qwen2.apply_rotary_pos_emb

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        ok = (
            position_ids is None and unsqueeze_dim == 1
            and q.is_cuda and q.dim() == 4 and k.dim() == 4 and cos.dim() == 3
            and cos.shape == sin.shape and k.shape[0] == q.shape[0] and k.shape[2] == q.shape[2]
            and cos.shape[0] == q.shape[0] and cos.shape[1] == q.shape[2]
            and cos.shape[2] == q.shape[3] == k.shape[3] and q.shape[3] % 2 == 0
            and q.dtype == k.dtype == cos.dtype == sin.dtype
            and q.numel() > 0 and k.numel() > 0
            and not cos.requires_grad and not sin.requires_grad
        )
        if not ok:
            return original(q, k, cos, sin, position_ids=position_ids,
                            unsqueeze_dim=unsqueeze_dim)
        return RoPEFn.apply(q, k, cos, sin)

    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
    modeling_qwen2._fused_rope_patched = True


def patch_all():
    patch_qwen2_mlp_only()
    if PATCH_EXTRAS:
        for fn in (patch_qwen2_rmsnorm, patch_qwen2_rope):
            try:
                fn()
            except Exception as exc:  # never break the run because an optional patch failed
                warnings.warn(f"{fn.__name__} skipped: {exc!r}")


def patch_qwen2_mlp():
    """Entry point kept under the example's name: installs every patch in this file."""
    patch_all()
