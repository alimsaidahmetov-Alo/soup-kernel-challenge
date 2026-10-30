"""Custom optimized SwiGLU kernel for Qwen2 MLP in Triton.
"""

import torch
import triton
import triton.language as tl

BLOCK = 2048


@triton.jit
def my_custom_swiglu_fwd_kernel(
    gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    out = gate * tl.sigmoid(gate) * up
    
    tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def my_custom_swiglu_bwd_kernel(
    grad_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        
        my_custom_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](
            gate, up, out, n, BLOCK=BLOCK
        )
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        grad = grad.to(gate.dtype).contiguous()
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        
        my_custom_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](
            grad, gate, up, dgate, dup, n, BLOCK=BLOCK
        )
        return dgate, dup


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)


def patch_qwen2_mlp():
    """Интегрируем кастомное ядро в модель Qwen2."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = forward
