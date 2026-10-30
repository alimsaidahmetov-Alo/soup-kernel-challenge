"""
Optimized Triton Kernels & Model Patches for Soup Kernel Challenge (Decentrathon 6.0).
Target: Qwen2.5-1.5B-Instruct on Kaggle T4 (sm_75, FP16, 4-bit NF4, LoRA).

Credits & Algorithmic Ideas (as required by Section 8 & 11.1 of the Spec):
- Online Softmax & Fused Cross-Entropy: Milakov & Gimelshein (2018), FlashAttention (Dao et al., 2022)
- Fused RMSNorm & RoPE vectorization concepts: inspired by Triton official tutorials & Liger/Unsloth papers (implemented from scratch).
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ==============================================================================
# 1. СОБСТВЕННЫЙ КЕРНЕЛ №1: Fused RMSNorm (Forward + Backward)
# Убирает аллокации промежуточных FP32 тензоров (56 вызовов на каждый шаг)
# ==============================================================================

@triton.jit
def soup_rmsnorm_fwd_kernel(
    x_ptr, w_ptr, y_ptr, rstd_ptr,
    stride_x, N_COLS: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    x_row_ptr = x_ptr + row_idx * stride_x
    y_row_ptr = y_ptr + row_idx * stride_x

    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N_COLS
    rstd = tl.math.rsqrt(var + eps)
    tl.store(rstd_ptr + row_idx, rstd)

    y = (x * rstd) * w
    tl.store(y_row_ptr + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def soup_rmsnorm_bwd_kernel(
    dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr,
    stride_x, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

     offset = row_idx * stride_x
    dy = tl.load(dy_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)

    x_hat = x * rstd
    wdy = w * dy
    c1 = tl.sum(x_hat * wdy, axis=0) / N_COLS
    dx = (wdy - x_hat * c1) * rstd

    tl.store(dx_ptr + offset + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class FastRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x_2d = x.contiguous().view(-1, orig_shape[-1])
        M, N = x_2d.shape

        y = torch.empty_like(x_2d)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        BLOCK_SIZE = triton.next_power_of_2(N)

        soup_rmsnorm_fwd_kernel[(M,)](
            x_2d, weight, y, rstd,
            x_2d.stride(0), N_COLS=N, eps=eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.orig_shape = orig_shape
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, rstd = ctx.saved_tensors
        M, N = x_2d.shape
        dy_2d = dy.contiguous().view(M, N)
        dx = torch.empty_like(x_2d)

        soup_rmsnorm_bwd_kernel[(M,)](
            dy_2d, x_2d, weight, rstd, dx,
            x_2d.stride(0), N_COLS=N, BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=4
        )
        return dx.view(ctx.orig_shape), None, None


# ==============================================================================
# 2. СОБСТВЕННЫЙ КЕРНЕЛ №2: Fused RoPE (Rotary Position Embedding)
# Заменяет 8+ мелких ядер PyTorch (slice, neg, cat, mul, add) на 1 проход
# ==============================================================================

@triton.jit
def soup_rope_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    out_q_ptr, out_k_ptr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_cs,
    N_QH: tl.constexpr, N_KH: tl.constexpr,
    HALF_DIM: tl.constexpr, BLOCK_D: tl.constexpr, IS_BWD: tl.constexpr
):
    pid_bs = tl.program_id(0)
    seq_len = tl.num_programs(1)
    pid_s = tl.program_id(1)
    pid_b = pid_bs

    d_offs = tl.arange(0, BLOCK_D)
    mask_d = d_offs < HALF_DIM

    cos_offset = (pid_b * seq_len + pid_s) * stride_cs
    cos1 = tl.load(cos_ptr + cos_offset + d_offs, mask=mask_d, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_ptr + cos_offset + d_offs, mask=mask_d, other=0.0).to(tl.float32)

    if IS_BWD:
        sin1 = -sin1

    # Обработка голов Query
    for h in range(N_QH):
        q_base = pid_b * stride_qb + h * stride_qh + pid_s * stride_qs
        q1 = tl.load(q_ptr + q_base + d_offs, mask=mask_d, other=0.0).to(tl.float32)
        q2 = tl.load(q_ptr + q_base + HALF_DIM + d_offs, mask=mask_d, other=0.0).to(tl.float32)

        oq1 = q1 * cos1 - q2 * sin1
        oq2 = q2 * cos1 + q1 * sin1

        tl.store(out_q_ptr + q_base + d_offs, oq1.to(out_q_ptr.dtype.element_ty), mask=mask_d)
        tl.store(out_q_ptr + q_base + HALF_DIM + d_offs, oq2.to(out_q_ptr.dtype.element_ty), mask=mask_d)

    # Обработка голов Key (GQA: меньше голов, чем у Q)
    for h in range(N_KH):
        k_base = pid_b * stride_kb + h * stride_kh + pid_s * stride_ks
        k1 = tl.load(k_ptr + k_base + d_offs, mask=mask_d, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + k_base + HALF_DIM + d_offs, mask=mask_d, other=0.0).to(tl.float32)

        ok1 = k1 * cos1 - k2 * sin1
        ok2 = k2 * cos1 + k1 * sin1

        tl.store(out_k_ptr + k_base + d_offs, ok1.to(out_k_ptr.dtype.element_ty), mask=mask_d)
        tl.store(out_k_ptr + k_base + HALF_DIM + d_offs, ok2.to(out_k_ptr.dtype.element_ty), mask=mask_d)


class FastRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        # q: [B, QH, S, D], k: [B, KH, S, D]
        B, QH, S, D = q.shape
        KH = k.shape[1]
        HALF_D = D // 2
        BLOCK_D = triton.next_power_of_2(HALF_D)

        cos_c = cos.contiguous().view(-1, D)
        sin_c = sin.contiguous().view(-1, D)
        if cos_c.shape[0] == S and B > 1:
            cos_c = cos_c.unsqueeze(0).expand(B, S, D).contiguous().view(-1, D)
            sin_c = sin_c.unsqueeze(0).expand(B, S, D).contiguous().view(-1, D)

        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)

        soup_rope_kernel[(B, S)](
            q, k, cos_c, sin_c, out_q, out_k,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            cos_c.stride(0),
            N_QH=QH, N_KH=KH, HALF_DIM=HALF_D, BLOCK_D=BLOCK_D, IS_BWD=False,
            num_warps=2
        )
        ctx.save_for_backward(cos_c, sin_c)
        ctx.shapes = (B, QH, KH, S, HALF_D, BLOCK_D)
        return out_q, out_k

    @staticmethod
    def backward(ctx, dq, dk):
        cos_c, sin_c = ctx.saved_tensors
        B, QH, KH, S, HALF_D, BLOCK_D = ctx.shapes
        dq = dq.contiguous()
        dk = dk.contiguous()
        out_dq = torch.empty_like(dq)
        out_dk = torch.empty_like(dk)

        soup_rope_kernel[(B, S)](
            dq, dk, cos_c, sin_c, out_dq, out_dk,
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            cos_c.stride(0),
            N_QH=QH, N_KH=KH, HALF_DIM=HALF_D, BLOCK_D=BLOCK_D, IS_BWD=True,
            num_warps=2
        )
        return out_dq, out_dk, None, None


# ==============================================================================
# 3. СОБСТВЕННЫЙ КЕРНЕЛ №3: Fused MLP Activation + Direct Out-Projection
# Включает оптимизированную векторизацию и экономит полосу пропускания T4
# ==============================================================================

@triton.jit
def soup_fast_geglu_swiglu_fwd(
    gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    g = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    val = (g * tl.sigmoid(g)) * u
    tl.store(out_ptr + offsets, val.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def soup_fast_geglu_swiglu_bwd_inplace(
    grad_ptr, gate_ptr, up_ptr, n_elements, BLOCK: tl.constexpr
):
    """Перезаписывает gate -> dgate и up -> dup IN-PLACE, экономя 2 тензора [B, S, 8960] в пике!"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    dy = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(g)
    silu = g * sig
    dup = dy * silu
    dgate = dy * u * (sig + silu * (1.0 - sig))

    tl.store(gate_ptr + offsets, dgate.to(gate_ptr.dtype.element_ty), mask=mask)
    tl.store(up_ptr + offsets, dup.to(up_ptr.dtype.element_ty), mask=mask)


class InplaceSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        BLOCK = 2048
        soup_fast_geglu_swiglu_fwd[(triton.cdiv(n, BLOCK),)](
            gate, up, out, n, BLOCK=BLOCK, num_warps=8
        )
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        grad = grad.to(gate.dtype).contiguous()
        n = gate.numel()
        BLOCK = 2048
        # In-place модификация сохраненных gate и up экономит ~70 МБ на каждый слой в backward
        soup_fast_geglu_swiglu_bwd_inplace[(triton.cdiv(n, BLOCK),)](
            grad, gate, up, n, BLOCK=BLOCK, num_warps=8
        )
        return gate, up


# ==============================================================================
# 4. СОБСТВЕННЫЙ КЕРНЕЛ №4: Fused Cross-Entropy over 152K Vocabulary (In-Place Grad)
# Решает главную проблему из п. 2 и 9 ТЗ: убирает материализацию FP32 логитов
# и считает loss + градиент прямо в буфере FP16 логитов!
# ==============================================================================

@triton.jit
def soup_cross_entropy_fwd_bwd_kernel(
    logits_ptr, labels_ptr, losses_ptr,
    stride_logits, VOCAB_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr, IGNORE_INDEX: tl.constexpr
):
    row_idx = tl.program_id(0)
    label = tl.load(labels_ptr + row_idx)

    if label == IGNORE_INDEX:
        tl.store(losses_ptr + row_idx, 0.0)
        return

    row_logits_ptr = logits_ptr + row_idx * stride_logits

    # Проход 1: Online Softmax (поиск максимума и суммы экспонент без выделения памяти)
    m_i = -float("inf")
    d_i = 0.0

    for start_n in range(0, VOCAB_SIZE, BLOCK_SIZE):
        cols = start_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < VOCAB_SIZE
        x = tl.load(row_logits_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)

        block_max = tl.max(x, axis=0)
        m_new = tl.maximum(m_i, block_max)
        d_i = d_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    lse = m_i + tl.log(d_i)
    target_logit = tl.load(row_logits_ptr + label).to(tl.float32)
    loss = lse - target_logit
    tl.store(losses_ptr + row_idx, loss)


@triton.jit
def soup_cross_entropy_bwd_inplace_kernel(
    logits_ptr, labels_ptr, dloss_ptr,
    stride_logits, inv_num_valid,
    VOCAB_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr, IGNORE_INDEX: tl.constexpr
):
    row_idx = tl.program_id(0)
    label = tl.load(labels_ptr + row_idx)
    row_logits_ptr = logits_ptr + row_idx * stride_logits

    if label == IGNORE_INDEX:
        for start_n in range(0, VOCAB_SIZE, BLOCK_SIZE):
            cols = start_n + tl.arange(0, BLOCK_SIZE)
            mask = cols < VOCAB_SIZE
            tl.store(row_logits_ptr + cols, tl.zeros([BLOCK_SIZE], dtype=logits_ptr.dtype.element_ty), mask=mask)
        return

    m_i = -float("inf")
    d_i = 0.0
    for start_n in range(0, VOCAB_SIZE, BLOCK_SIZE):
        cols = start_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < VOCAB_SIZE
        x = tl.load(row_logits_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        m_new = tl.maximum(m_i, block_max)
        d_i = d_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    lse = m_i + tl.log(d_i)
    scale = tl.load(dloss_ptr).to(tl.float32) * inv_num_valid

    # Записываем градиент прямо поверх входных logits (экономит 1.25 ГБ памяти!)
    for start_n in range(0, VOCAB_SIZE, BLOCK_SIZE):
        cols = start_n + tl.arange(0, BLOCK_SIZE)
        mask = cols < VOCAB_SIZE
        x = tl.load(row_logits_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        probs = tl.exp(x - lse)
        grad = tl.where(cols == label, probs - 1.0, probs) * scale
        tl.store(row_logits_ptr + cols, grad.to(logits_ptr.dtype.element_ty), mask=mask)


class FastCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index=-100):
        # logits: [N, V] в FP16! Не переводим в FP32 на хосте!
        N, V = logits.shape
        logits = logits.contiguous()
        labels = labels.contiguous()

        losses = torch.empty(N, dtype=torch.float32, device=logits.device)
        valid_mask = (labels != ignore_index)
        num_valid = valid_mask.sum().clamp(min=1).item()

        BLOCK_SIZE = 4096
        soup_cross_entropy_fwd_bwd_kernel[(N,)](
            logits, labels, losses,
            logits.stride(0), VOCAB_SIZE=V, BLOCK_SIZE=BLOCK_SIZE,
            IGNORE_INDEX=ignore_index, num_warps=8
        )

        ctx.save_for_backward(logits, labels)
        ctx.num_valid = num_valid
        ctx.ignore_index = ignore_index
        return losses.sum() / num_valid

    @staticmethod
    def backward(ctx, dloss):
        logits, labels = ctx.saved_tensors
        N, V = logits.shape
        dloss = dloss.contiguous()
        inv_num_valid = 1.0 / float(ctx.num_valid)

        BLOCK_SIZE = 4096
        soup_cross_entropy_bwd_inplace_kernel[(N,)](
            logits, labels, dloss,
            logits.stride(0), inv_num_valid,
            VOCAB_SIZE=V, BLOCK_SIZE=BLOCK_SIZE,
            IGNORE_INDEX=ctx.ignore_index, num_warps=8
        )
        return logits, None, None


# ==============================================================================
# 5. ЕДИНАЯ ТОЧКА ПОДКЛЮЧЕНИЯ К SOUP: apply()
# Вызывается автоматически харнессом перед soup train
# ==============================================================================

def apply():
    """Подключает все Triton-кернелы к моделям Qwen2 и пайплайну обучения Soup."""
    from transformers.models.qwen2 import modeling_qwen2

    # 1. Патч Qwen2RMSNorm
    orig_rmsnorm_fwd = modeling_qwen2.Qwen2RMSNorm.forward

    def fast_rmsnorm_forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return orig_rmsnorm_fwd(self, hidden_states)
        return FastRMSNormFunc.apply(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = fast_rmsnorm_forward

    # 2. Патч Qwen2MLP (In-place backward SwiGLU)
    orig_mlp_fwd = modeling_qwen2.Qwen2MLP.forward

    def fast_mlp_forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda:
            return orig_mlp_fwd(self, x)
        return self.down_proj(InplaceSwiGLU.apply(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = fast_mlp_forward

    # 3. Патч RoPE (apply_rotary_pos_emb)
    orig_rope = modeling_qwen2.apply_rotary_pos_emb

    def fast_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if not q.is_cuda or unsqueeze_dim != 1:
            return orig_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
        return FastRoPEFunc.apply(q, k, cos, sin)

    modeling_qwen2.apply_rotary_pos_emb = fast_apply_rotary_pos_emb

    # 4. Патч CrossEntropyLoss для Qwen2ForCausalLM (убирает .float() на 152K словаре!)
    orig_ce_forward = nn.CrossEntropyLoss.forward

    def fast_ce_forward(self, input, target):
        if (
            input.is_cuda
            and input.ndim == 2
            and input.shape[-1] > 32000  # срабатывает на словаре 151936
            and self.reduction == "mean"
            and self.weight is None
            and self.label_smoothing == 0.0
        ):
            return FastCrossEntropyFunction.apply(input, target, self.ignore_index)
        return orig_ce_forward(self, input, target)

    nn.CrossEntropyLoss.forward = fast_ce_forward

    # 5. Патч Qwen2ForCausalLM.forward: предотвращаем принудительный перевод logits в FP32
    # В стандартном transformers: logits = logits.float() выделяет 1.25 ГБ перед CrossEntropy!
    orig_causal_fwd = modeling_qwen2.Qwen2ForCausalLM.forward

    def fast_causal_lm_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = outputs[0]
        # Оставляем logits в родном FP16 (экономия 1.25 ГБ памяти сразу!)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = FastCrossEntropyFunction.apply(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                -100,
            )

        if not return_dict and return_dict is not None:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    modeling_qwen2.Qwen2ForCausalLM.forward = fast_causal_lm_forward
