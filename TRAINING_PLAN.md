# Fine-Tuning Plan — Personal AI Assistant

**Handoff document.** Written on the laptop (RTX 3060 6 GB), intended to be executed on
the lab machine (RTX Titan). Companion to `PLAN.md`, which describes the agent itself —
copy both files over.

---

## 0. What this fine-tune is and is not for

Read this before writing any training code.

**Fine-tuning will not make the model know your documents.** Facts absorbed into weights
come back confidently wrong, with no citation and no way to correct them short of
retraining. Personal knowledge belongs in RAG, where it can be updated, cited, and
verified. This is not a limitation to work around — it is the correct division of labour.

**What fine-tuning is genuinely for here:**

1. **Reliable tool calling.** The single biggest win. Base 8B models emit malformed tool
   arguments, forget to call `search_docs`, and invent tools. A few hundred good traces
   fixes this decisively.
2. **Domain reasoning style.** Physics/math problem setup — declaring knowns, choosing
   the right approach, routing computation to `calc` instead of doing arithmetic in-head.
3. **Output conventions.** How you want derivations laid out, units handled, LaTeX
   formatted, citations rendered.

**Consequence for sequencing:** do not train first. The best training data is logged
traces from the working RAG agent, with failures corrected by hand. Phase 1 and 2 of
`PLAN.md` must exist before this document's Step 3 can produce anything worth training on.

---

## 1. Step 1 — Identify the GPU before choosing anything

"RTX Titan" is ambiguous and the answer changes the entire training configuration.
Run this first on the lab machine:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print(torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))"
```

Branch on **compute capability**:

| Capability | Card | dtype | Flash Attention 2 | Notes |
|---|---|---|---|---|
| `(7, 5)` | **Titan RTX** (Turing, 24 GB) | **fp16 only** | **Not supported** | Most likely case. See warnings below. |
| `(7, 0)` | Titan V (Volta, 12 GB) | fp16 only | Not supported | Only 12 GB — treat VRAM budget as half. |
| `(8, 6)` | RTX A6000 / 3090-class | bf16 | Supported | Best case, use bf16 throughout. |
| `(8, 9)`+ | Ada / Hopper | bf16 | Supported | Best case. |

### If it is Turing `(7, 5)` — three things that will otherwise waste a day

- **No bfloat16.** Turing has no native bf16. Set `bf16=False, fp16=True`. Copying a
  modern training config that uses bf16 will either error or silently run in emulation
  at terrible speed. fp16 needs loss scaling — the HF `Trainer` handles this, but watch
  for NaN losses and drop the LR if they appear.
- **No Flash Attention 2.** FA2 requires Ampere (sm_80) or newer. Use PyTorch SDPA
  (`attn_implementation="sdpa"`) or xformers instead. Installing `flash-attn` will
  compile for twenty minutes and then fail at runtime.
- **Skip `torch.compile`.** Unreliable on Turing; the speedup isn't worth the debugging.

4-bit NF4 via bitsandbytes *does* work on Turing (it's what Colab T4s run), so QLoRA is
fine.

---

## 2. Step 2 — Choose base model and method

### Decide the deployment target first

This determines model size, and it is the most common place people waste a training run:

- **Deploying back to the 6 GB laptop** → train **8B**. A 14B fine-tune cannot be served
  on a 3060 Mobile at usable speed, no matter how good it is.
- **Serving from the lab machine** → **14B** is affordable and meaningfully better at
  multi-step math.

Default recommendation: **`Qwen3-8B`**, unless the lab machine is the permanent host.
Strong math, native tool-calling format, and it matches the model `PLAN.md` targets.

### Method: QLoRA

| Setting | Value | Rationale |
|---|---|---|
| Quantization | 4-bit NF4, double quant | 8B → ~5.5 GB base, leaves headroom on 24 GB |
| LoRA rank `r` | 32 | 16 is enough for pure style; 32 for behaviour change |
| `lora_alpha` | 64 | Keep at `2 × r` |
| `lora_dropout` | 0.05 | |
| Target modules | `q,k,v,o,gate,up,down_proj` | All linear layers — not just attention |
| Seq length | 4096 | Must fit a system prompt + retrieved chunks + tool trace |
| Optimizer | `paged_adamw_8bit` | `bitsandbytes` is already in the laptop venv |
| Grad checkpointing | on | Trades ~20% speed for large VRAM savings |
| LR | 1e-4 to 2e-4, cosine, 5% warmup | Standard LoRA range |
| Epochs | 2–3 | More overfits a small personal dataset fast |
| Effective batch | 16 (e.g. bs 2 × grad accum 8) | |

**Framework:** `unsloth` (~2× faster, lower VRAM, works on Turing) with HF `trl`
`SFTTrainer` as the fallback if Unsloth's install fights the driver. `axolotl` is a
reasonable third option if you prefer YAML configs over Python.

Full fine-tuning is not worth it: 24 GB cannot hold 8B weights plus Adam states, and
LoRA reaches ~95% of the quality for this kind of behavioural task anyway.

---

## 3. Step 3 — Build the dataset (this is 80% of the work)

**500–2000 excellent examples beat 50,000 scraped ones.** Plan to spend far more time
here than on training, which is a couple of hours of GPU time.

### Where examples come from, in priority order

1. **Logged agent traces (best).** Run the phase-2 RAG agent from `PLAN.md` normally for
   a week or two. Log every turn: system prompt, user message, retrieved chunks, tool
   calls, tool results, final answer. Then **hand-correct the failures** — fix the
   malformed tool call, fix the wrong tool choice, rewrite the bad answer. Training on
   corrected real traces is far more effective than synthetic data because the input
   distribution matches deployment exactly.
2. **Synthetic tool-call traces.** For coverage of tools that rarely fire, generate
   scenarios with a larger model, then review every one by hand. Cover the hard cases:
   multi-step calls, a tool returning an error, a tool returning nothing relevant, and
   questions that need *no* tool at all (over-calling is a real failure mode).
3. **Domain Q&A from your own material.** Physics/math problems worked in your preferred
   style — showing the `calc` hand-off rather than doing arithmetic inline.

### Format

Match the base model's chat template exactly, including tool-call serialisation — a
mismatch between training format and inference format silently destroys tool calling.
Verify with `tokenizer.apply_chat_template(...)` and read the raw string output.

```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "tool_calls": [{"function": {"name": "search_docs", "arguments": "{...}"}}]},
  {"role": "tool", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

### Hygiene, all of it mandatory

- **Mask the loss to assistant turns only.** Do not train on user text, system prompts,
  or tool outputs. `train_on_responses_only` in Unsloth, or `DataCollatorForCompletionOnlyLM`.
  Getting this wrong is the most common silent failure in SFT.
- **Decontaminate against the eval set** from `PLAN.md` §5. Any overlap makes every
  subsequent measurement meaningless.
- **Deduplicate** near-identical examples — personal corpora are highly repetitive.
- **Hold out 10%** for validation.
- **Balance the mix.** Roughly: 50% tool-calling traces, 30% domain reasoning, 20%
  no-tool-needed conversation. That last slice is what stops the tuned model calling
  `search_docs` when you say hello.

---

## 4. Step 4 — Train

```bash
uv venv --python 3.11 .venv-train && source .venv-train/bin/activate
uv pip install "unsloth[cu121-torch240]" trl peft accelerate bitsandbytes datasets
```
Pin the CUDA/torch variant to the lab machine's driver — check `nvidia-smi` output first.
Use a **separate venv** from the agent's; training and serving dependency trees conflict.

Run order:
1. **Smoke test** — 10 examples, 1 epoch, confirm loss decreases and nothing NaNs.
2. **Overfit test** — train on 20 examples until loss ≈ 0. If it can't overfit 20
   examples, the data format or loss masking is broken. Do not skip this.
3. **Full run** — log to TensorBoard or W&B; watch train vs. val loss diverge.

Expect roughly 1–3 hours for 1500 examples × 3 epochs at 4096 seq len on a Titan RTX.
Checkpoint every epoch so you can compare and roll back.

**Signs it went wrong:** val loss rising while train loss falls (overfit — fewer epochs
or lower `r`); loss plateauing near zero almost immediately (loss mask broken, it's
learning to copy); NaN loss (fp16 scaling — lower the LR).

---

## 5. Step 5 — Evaluate against the base model

Never ship a fine-tune you haven't measured. Same golden set from `PLAN.md` §5, run
**base vs. tuned** head to head:

| Metric | What it catches |
|---|---|
| Tool-call validity rate | The main thing you're training for |
| Correct tool selection | Including correctly calling *nothing* |
| Answer correctness (1–5, manual) | End-to-end quality |
| Retrieval hit rate @5 | Should be *unchanged* — it's not part of the model |
| General-knowledge spot check (~20 off-domain questions) | **Catastrophic forgetting** |

That last row matters. A narrow personal dataset can degrade general ability badly while
every domain metric improves. If it forgets, lower `r`, cut epochs, or mix in a few
hundred general instruction examples.

If the tuned model doesn't clearly beat base on tool-call validity, the fix is almost
always more and better data — not more epochs.

---

## 6. Step 6 — Export to Ollama

```bash
# 1. Merge the LoRA adapter into the base weights (fp16)
python -c "
from peft import AutoPeftModelForCausalLM
m = AutoPeftModelForCausalLM.from_pretrained('out/checkpoint-final', torch_dtype='float16')
m.merge_and_unload().save_pretrained('merged/')
"

# 2. Convert to GGUF, then quantize
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
python convert_hf_to_gguf.py ../merged --outfile ../assistant-f16.gguf --outtype f16
./llama-quantize ../assistant-f16.gguf ../assistant-q4_k_m.gguf Q4_K_M
```

Then a `Modelfile`:

```
FROM ./assistant-q4_k_m.gguf
PARAMETER num_ctx 8192
PARAMETER temperature 0.3
SYSTEM """<the exact system prompt used in training>"""
```
```bash
ollama create personal-assistant -f Modelfile
```

Two things to get right:
- **The chat template must match training.** If Ollama's default template for the base
  architecture differs from what you trained on, tool calling breaks in ways that look
  like a bad fine-tune. Set `TEMPLATE` explicitly in the Modelfile if they differ.
- **Q4_K_M is the laptop target** (~5 GB, fits 6 GB VRAM). Keep the f16 GGUF — if the
  lab machine is the host, serve Q6_K or Q8_0 instead for better quality.

Copy the final `.gguf` back to the laptop and re-run the eval set there. Quantization
costs a little quality; confirm it's still ahead of base after the round trip.

---

## 7. Order of operations

```
PLAN.md phase 1  →  RAG + eval set working
PLAN.md phase 2  →  agent loop with tools, logging every trace
        ↓          (run it for 1–2 weeks, collect real usage)
Step 3           →  correct the logged failures into a dataset   ← the real work
Step 1, 2        →  lab machine: detect GPU, pick config
Step 4, 5        →  train, evaluate vs. base
Step 6           →  GGUF → Ollama → back to the laptop
```

Skipping straight to Step 4 with a synthetic dataset is possible and will produce a
model. It will also produce no measurable improvement, because there'll be nothing to
measure it against and the training distribution won't match how you actually use it.
