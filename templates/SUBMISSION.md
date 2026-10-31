# Soup Kernel Challenge

Make `soup train` faster and lighter on a free Kaggle T4 with your own Triton kernels.
Case of [Soup](https://github.com/MakazhanAlpamys/Soup) at Decentrathon 6.0,
October 30 12:00 – October 31 23:59 (Astana time, UTC+5; the deadline is October 31 18:59 UTC).
Teams of up to 4, one result per team. AI agents are allowed.

## Links / Ссылки на материалы

* **GitHub Repository:** `<вставьте_ссылку_на_репозиторий>`
* **Kaggle Notebook:** [`<вставьте_ссылку_на_ноутбук>`](https://www.kaggle.com/code/alimsaidahmetov/notebookad310d557e)
* **Video:** `<вставьте_ссылку_на_видео>`
* **Presentation:** `<вставьте_ссылку_на_презентацию>`

## Team / Команда

* **Амина**: Frontend-разработчик, AI/Prompt-инженер
* **Алим**: Project Manager, UI/UX Дизайнер, Frontend-разработчик
* **Дария**: Product Manager
* **Аиша**: Backend-разработчик

## Our Results

| Metric | Base Soup 0.77.0 | Solution | Result / Ratio |
|---|---|---|---|
| Speedup | 1.000x | 1.012x[cite: 1] | Counts as 1.000 (margin < 5%)[cite: 1] |
| Memory (base/solution) | 1.000x | 1.158x[cite: 1] | **1.158x** (x0.863 of base peak)[cite: 1] |
| Quality (eval loss vs base) | baseline | -0.000%[cite: 1] | PASS (within 0.5% limit)[cite: 1] |
| Checks status | — | 10 / 10 passed[cite: 1] | **ALL CHECKS PASSED**[cite: 1] |
| Custom kernels runtime | — | 7.4%[cite: 1] | PASS (>= 5% requirement)[cite: 1] |

Tested kernels[cite: 1]:
- `qwen2_mlp_act_fwd` / `qwen2_mlp_act_bwd` (`solution/swiglu.py`)[cite: 1]
- `qwen2_rmsnorm_fwd` / `qwen2_rmsnorm_bwd` (`solution/swiglu.py`)[cite: 1]

## The task

`soup train` with `soup-cli==0.77.0` fine-tunes **Qwen2.5-1.5B-Instruct** (4-bit NF4 weights,
LoRA r=16) on 256 Kazakh instructions: batch 4, up to 512 tokens, one epoch (64 steps), on one
of the two GPUs Kaggle gives. Public rows: 0–255 for training, 40000–40255 for your own eval; the
jury trains and evaluates on other, hidden rows. Make this same run take less time or less GPU memory, without
changing what it trains. Your own `@triton.jit` kernels must do real work in it.

Base Soup on a Kaggle T4 (our measurement; yours comes from `harness/run_task.py`):

| | Base Soup 0.77.0 |
|---|---|
| Whole `soup train` command | about 79–83 s (56–61 s of it are the 64 training steps) |
| Peak GPU memory (nvidia-smi) | about 14,350 MiB of 15,360 |
| Held-out loss: no adapter → after training | 2.271 → 2.050 (−9.7%) |

Where the GPU time of a training step goes (our torch.profiler run of base Soup): matrix multiplies about
36%, element-wise ops and dtype casts about 44%, softmax / cross-entropy over the 151,936-token
vocabulary about 7%, attention about 5%, NF4 dequantization about 5%. For scale: Unsloth trains a
similar LoRA on the same data and GPU in about 2,400 MiB (with its own gradient offloading and
LoRA on all seven projections).

## Quick start on Kaggle

1. Work in the GitHub repository that the hackathon platform created for your team: work in other
   repositories is not checked. If it is empty or holds only a README, copy this starter kit into
   it (the **Use this template** button only makes a new repository and cannot fill an existing
   one):

   ```bash
   git clone <your platform repository> && cd <its folder>
   git fetch [https://github.com/MakazhanAlpamys/soup-kernel-challenge.git](https://github.com/MakazhanAlpamys/soup-kernel-challenge.git) main
   git checkout FETCH_HEAD -- .
   git commit -m "Soup Kernel Challenge starter kit"
   git push origin HEAD
