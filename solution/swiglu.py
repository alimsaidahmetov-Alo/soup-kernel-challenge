"""Highly optimized Triton MLP Activation, RMSNorm & RoPE kernels for Qwen2.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def qwen2_mlp_act_fwd(
    gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    out = gate * sig * up

    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def qwen2_mlp_act_bwd(
    grad_ptr, gate_ptr, up_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    gs = gate * sig
    dup = grad * gs
    dgate = grad * up * (sig + gs * (1.0 - sig))

    # In-place запись напрямую в gate_ptr и up_ptr
    tl.store(gate_ptr + offsets, dgate.to(gate_ptr.dtype.element_ty), mask=mask)
    tl.store(up_ptr + offsets, dup.to(up_ptr.dtype.element_ty), mask=mask)


@triton.jit
def qwen2_rmsnorm_fwd(
    x_ptr, w_ptr, out_ptr, rstd_ptr, stride, N: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)

    out = x * rstd * w
    tl.store(out_ptr + row * stride + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def qwen2_rmsnorm_bwd(
    dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_partial_ptr,
    stride, N: tl.constexpr, BLOCK: tl.constexpr, HAS_DW: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    dy = tl.load(dy_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    x_hat = x * rstd
    wdy = w * dy
    c1 = tl.sum(x_hat * wdy, axis=0) / N
    dx = (wdy - x_hat * c1) * rstd

    tl.store(dx_ptr + row * stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)
    if HAS_DW:
        tl.store(
            dw_partial_ptr + row * stride + cols,
            (dy * x_hat).to(dw_partial_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def qwen2_rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_xb, stride_xh, stride_xs,
    stride_cb, stride_cs,
    H: tl.constexpr, S: tl.constexpr, HALF_D: tl.constexpr,
    BLOCK_D: tl.constexpr, BACKWARD: tl.constexpr
):
    pid = tl.program_id(0)
    s_idx = pid % S
    rem = pid // S
    h_idx = rem % H
    b_idx = rem // H

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < HALF_D

    x_base = x_ptr + b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs
    out_base = out_ptr + b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs
    cs_base = b_idx * stride_cb + s_idx * stride_cs

    x1 = tl.load(x_base + d_offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_base + HALF_D + d_offs, mask=mask, other=0.0).to(tl.float32)

    cos1 = tl.load(cos_ptr + cs_base + d_offs, mask=mask, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_ptr + cs_base + d_offs, mask=mask, other=0.0).to(tl.float32)
    cos2 = tl.load(cos_ptr + cs_base + HALF_D + d_offs, mask=mask, other=0.0).to(tl.float32)
    sin2 = tl.load(sin_ptr + cs_base + HALF_D + d_offs, mask=mask, other=0.0).to(tl.float32)

    if not BACKWARD:
        o1 = x1 * cos1 - x2 * sin1
        o2 = x2 * cos2 + x1 * sin2
    else:
        o1 = x1 * cos1 + x2 * sin2
        o2 = x2 * cos2 - x1 * sin1

    tl.store(out_base + d_offs, o1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_base + HALF_D + d_offs, o2.to(out_ptr.dtype.element_ty), mask=mask)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if not gate.is_contiguous():
            gate = gate.contiguous()
        if not up.is_contiguous() or up.dtype != gate.dtype:
            up = up.to(gate.dtype).contiguous()

        out = torch.empty_like(gate)
        n = gate.numel()
        BLOCK = 1024

        qwen2_mlp_act_fwd[(triton.cdiv(n, BLOCK),)](
            gate, up, out, n, BLOCK=BLOCK, num_warps=4
        )
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        if not grad.is_contiguous() or grad.dtype != gate.dtype:
            grad = grad.to(gate.dtype).contiguous()

        n = gate.numel()
        BLOCK = 1024

        qwen2_mlp_act_bwd[(triton.cdiv(n, BLOCK),)](
            grad, gate, up, n, BLOCK=BLOCK, num_warps=4
        )
        return gate, up


class FastRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, orig_shape[-1])
        M, N = x_2d.shape
        BLOCK = 1 << (N - 1).bit_length()

        out = torch.empty_like(x_2d)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        qwen2_rmsnorm_fwd[(M,)](
            x_2d, weight, out, rstd, N, N=N, eps=eps, BLOCK=BLOCK, num_warps=4
        )
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.orig_shape = orig_shape
        ctx.BLOCK = BLOCK
        return out.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, rstd = ctx.saved_tensors
        M, N = x_2d.shape
        dy_2d = dy.contiguous().view(M, N)

        has_dw = ctx.needs_input_grad[1]
        if has_dw:
            dw_partial = torch.empty((M, N), dtype=weight.dtype, device=x_2d.device)
        else:
            dw_partial = dy_2d

        # Записываем dx прямо поверх dy_2d (экономим выделение памяти под dx!)
        qwen2_rmsnorm_bwd[(M,)](
            dy_2d, x_2d, weight, rstd, dy_2d, dw_partial,
            N, N=N, BLOCK=ctx.BLOCK, HAS_DW=has_dw, num_warps=4
        )
        dw = dw_partial.sum(dim=0, dtype=weight.dtype) if has_dw else None
        return dy_2d.view(ctx.orig_shape), dw, None


def _launch_rope(x, cos, sin, out, backward=False):
    B, H, S, D = x.shape
    HALF_D = D // 2
    BLOCK_D = 1 << (HALF_D - 1).bit_length()
    stride_cb = cos.stride(0) if cos.shape[0] > 1 else 0
    stride_cs = cos.stride(1)
    qwen2_rope_kernel[(B * H * S,)](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2),
        stride_cb, stride_cs,
        H=H, S=S, HALF_D=HALF_D, BLOCK_D=BLOCK_D, BACKWARD=backward, num_warps=2
    )


class FastRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        cos_2d = cos.squeeze(1) if cos.ndim == 4 else (cos if cos.ndim == 3 else cos.unsqueeze(0))
        sin_2d = sin.squeeze(1) if sin.ndim == 4 else (sin if sin.ndim == 3 else sin.unsqueeze(0))
        cos_2d = cos_2d.contiguous()
        sin_2d = sin_2d.contiguous()

        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        _launch_rope(q, cos_2d, sin_2d, q_out, backward=False)
        _launch_rope(k, cos_2d, sin_2d, k_out, backward=False)
        ctx.save_for_backward(cos_2d, sin_2d)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        cos_2d, sin_2d = ctx.saved_tensors
        dq_out = torch.empty_like(dq)
        dk_out = torch.empty_like(dk)
        _launch_rope(dq, cos_2d, sin_2d, dq_out, backward=True)
        _launch_rope(dk, cos_2d, sin_2d, dk_out, backward=True)
        return dq_out, dk_out, None, None


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)


def patch_qwen2_mlp():
    """Интегрируем кастомные ядра SwiGLU, RMSNorm и RoPE в модель Qwen2."""
    from transformers.models.qwen2 import modeling_qwen2

    orig_mlp = modeling_qwen2.Qwen2MLP.forward

    def mlp_forward(self, x):
        if not x.is_cuda:
            return orig_mlp(self, x)
        return self.down_proj(SwiGLU.apply(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = mlp_forward

    if hasattr(modeling_qwen2, "Qwen2RMSNorm"):
        orig_norm = modeling_qwen2.Qwen2RMSNorm.forward

        def norm_forward(self, x):
            if not x.is_cuda:
                return orig_norm(self, x)
            return FastRMSNorm.apply(x, self.weight, self.variance_epsilon)

        modeling_qwen2.Qwen2RMSNorm.forward = norm_forward

    if hasattr(modeling_qwen2, "apply_rotary_pos_emb"):
        orig_rope = modeling_qwen2.apply_rotary_pos_emb

        def fast_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            if not q.is_cuda or unsqueeze_dim != 1 or (q.shape[-1] % 2 != 0):
                return orig_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
            return FastRoPE.apply(q, k, cos, sin)

        modeling_qwen2.apply_rotary_pos_emb = fast_apply_rotary_pos_emb
