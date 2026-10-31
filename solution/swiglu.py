"""Highly optimized Triton MLP Activation & RMSNorm kernels for Qwen2.
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

    # In-place запись напрямую в gate_ptr и up_ptr (экономия памяти и времени!)
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
    stride, N: tl.constexpr, BLOCK: tl.constexpr
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
    tl.store(dw_partial_ptr + row * stride + cols, dy * x_hat, mask=mask)


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

        # Модифицируем gate и up на месте без выделения новой памяти
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
        BLOCK = triton.next_power_of_2(N)

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

        dx = torch.empty_like(x_2d)
        dw_partial = torch.empty((M, N), dtype=torch.float32, device=x_2d.device)

        qwen2_rmsnorm_bwd[(M,)](
            dy_2d, x_2d, weight, rstd, dx, dw_partial, N, N=N, BLOCK=ctx.BLOCK, num_warps=4
        )
        dw = dw_partial.sum(dim=0).to(weight.dtype)
        return dx.view(ctx.orig_shape), dw, None


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)


def patch_qwen2_mlp():
    """Интегрируем кастомные ядра SwiGLU и RMSNorm в модель Qwen2."""
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
