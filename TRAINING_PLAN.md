# Fine-Tuning Plan — Personal AI Assistant

Executed on the workstation `cgu-ubuntu` (TITAN RTX, 24 GB) — the same machine that serves
the model and hosts development. Companion to `PLAN.md`, which describes the agent itself.

*This was originally written as a handoff document from a 6 GB laptop. That machine is no
longer in the picture; the hardware questions it hedged on are now answered below.*

---

## 0. What this fine-tune is and is not for

Read this before writing any training code.

**Fine-tuning will not make the model know your documents.** Facts absorbed into weights
come back confidently wrong, with no citation and no way to correct them short of
retraining. Personal knowledge belongs in RAG, where it can be updated, cited, and
verified. This is not a limitation to work around — it is the correct division of labour.

**What fine-tuning is genuinely for here:**

1. ~~**Reliable tool calling.**~~ **This was the single biggest win when the plan targeted
   a 2024-era 8B**, which emitted malformed arguments, forgot to call `search_docs`, and
   invented tools. It largely is not any more — see the reassessment below.
2. **Domain reasoning style.** Physics/math problem setup — declaring knowns, choosing
   the right approach, routing computation to `calc` instead of doing arithmetic in-head.
3. **Output conventions.** How you want derivations laid out, units handled, LaTeX
   formatted, citations rendered.

### Reassessed 2026-08-04: this fine-tune may not be worth doing

The models in `PLAN.md` §3 score **0.685 on BFCL-V4** and **94–95% on TAU2**. That is a
categorically different starting point from the base model this document was written
against, and it removes most of reason 1 — the reason that justified the whole exercise.

Reasons 2 and 3 are real but much smaller, and both can be attacked with prompting first,
at zero GPU cost and with no risk of catastrophic forgetting.

**So: treat training as optional-until-proven-necessary, not as a planned milestone.** The
gate is the `PLAN.md` §10 eval set. Run it against the base model; if tool-call validity and
correct tool selection are already high, the honest answer is that you do not need this
document and should spend the time on retrieval quality instead — which is where the
remaining errors will actually be.

### The gate has now fired once, against training. Measured 2026-08-04.

The phase-0 bake-off (`PLAN.md` §3) scored the served model at **21/22 correct tool
selections, zero malformed arguments, and zero over-calls** across 22 scenarios — including
all four "call nothing" checks, which §3 below flags as the failure mode most worth training
against. Reason 1 is not merely diminished; on this evidence it is absent.

That is 22 scripted scenarios, not the real eval set, so it is indicative rather than
conclusive — the `PLAN.md` §10 golden set is still the gate of record and it does not exist
yet. But the burden of proof has moved. **Do not start this document's Step 3 without a
measured failure from the §10 set naming what training is supposed to fix.** Reasons 2 and 3
(reasoning style, output conventions) remain open and should be attacked with prompting
first, at zero GPU cost.

Keep logging traces regardless — that instruction is unchanged and unconditional.

Keep logging traces regardless. They cost nothing to collect, they are the only thing that
makes training *possible* later, and per `PLAN.md` §2 they are part of the durable state you
carry between machines.

**Consequence for sequencing:** do not train first. The best training data is logged
traces from the working RAG agent, with failures corrected by hand. Phase 1 and 2 of
`PLAN.md` must exist before this document's Step 3 can produce anything worth training on.

---

## 1. Step 1 — The GPU is identified. It is Turing.

Measured 2026-08-04:

```
NVIDIA TITAN RTX, 24576 MiB, driver 595.58.03, CUDA toolkit 13.2
```

**TITAN RTX = Turing = sm_75.** The ambiguity this section used to hedge on is resolved,
and the constraints below are facts, not contingencies. Re-confirm the capability tuple
once `torch` is installed, but expect `(7, 5)`:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print(torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))"
```

### Three things that will otherwise waste a day

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

**Do not confuse this with Ollama serving.** `OLLAMA_FLASH_ATTENTION=1` in `PLAN.md` §3 is
a llama.cpp kernel flag that works on Turing. The sm_75 restrictions above apply to the
PyTorch training stack only.

---

## 2. Step 2 — Choose base model and method

### You cannot fine-tune what you serve. Train the 9B.

This is the constraint that decides everything else, and it was measured, not assumed:

**A 27B QLoRA fits 24 GB only at sequence length 2048**, sitting at 22–23 GB even with
Unsloth and gradient checkpointing. Any longer sequence or larger batch OOMs mid-run. And
that figure is for Ampere/Ada — **this card is Turing with no FA2**, so SDPA will use more
memory, not less. The 35B-A3B MoE is worse still: all experts stay resident, so ~20–22 GB
of 4-bit weights leaves nothing for activations.

Step 3 requires **4096**, because a training example must hold a system prompt, retrieved
chunks and a full tool trace. Training at 2048 would truncate the RAG context the model is
being taught to use — which defeats the purpose more thoroughly than not training at all.

**So the serve target and the train target are different models, deliberately:**

| Role | Model | Why |
|---|---|---|
| Served (`PLAN.md` §3) | `qwen3.6:27b-mtp-q4_K_M` | Best quality that fits 24 GB at inference. Settled by the phase-0 bake-off, 2026-08-04. |
| **Fine-tuned (this document)** | **`Qwen3.5-9B`** | The largest that trains at 4096 on 24 GB. BFCL-V4 0.661 — only 0.024 behind the 27B. |

This is not a compromise so much as a clarification of what the tuned model is *for*: it is
the fast path and the offline fallback (`PLAN.md` §3), where a tuned 9B plausibly beats an
untuned one by more than the gap to the 27B. The big model stays off-the-shelf.

**Do not size the training run for the Jetson either.** The 9B is a good fit for an
Orin-class board by coincidence, not by design — confirm the board per `PLAN.md` §2 before
assuming it runs there.

### Method: QLoRA

The numbers below target a 9B on 24 GB, which is comfortable — the headroom is deliberate,
because Turing without FA2 spends more memory on attention than published recipes assume.
If you OOM, cut seq length last: it is the one setting that changes what the model learns.

| Setting | Value | Rationale |
|---|---|---|
| Quantization | 4-bit NF4, double quant | 9B → ~6 GB base, leaves real headroom on 24 GB |
| LoRA rank `r` | 32 | 16 is enough for pure style; 32 for behaviour change |
| `lora_alpha` | 64 | Keep at `2 × r` |
| `lora_dropout` | 0.05 | |
| Target modules | `q,k,v,o,gate,up,down_proj` | All linear layers — not just attention |
| Seq length | 4096 | Must fit a system prompt + retrieved chunks + tool trace |
| Optimizer | `paged_adamw_8bit` | `bitsandbytes` works on Turing |
| Grad checkpointing | on | Trades ~20% speed for large VRAM savings |
| LR | 1e-4 to 2e-4, cosine, 5% warmup | Standard LoRA range |
| Epochs | 2–3 | More overfits a small personal dataset fast |
| Effective batch | 16 (bs 2 × grad accum 8) | At 9B/4096 this fits; drop to bs 1 × 16 if it does not |

**Framework:** `unsloth` (~2× faster, lower VRAM, works on Turing) with HF `trl`
`SFTTrainer` as the fallback if Unsloth's install fights the driver. `axolotl` is a
reasonable third option if you prefer YAML configs over Python.

Full fine-tuning is not worth it: 24 GB cannot hold 9B weights plus Adam states — it is
not close — and LoRA reaches ~95% of the quality for this kind of behavioural task anyway.

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
- **Decontaminate against the eval set** from `PLAN.md` §10. Any overlap makes every
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
uv pip install unsloth trl peft accelerate bitsandbytes datasets
```

**The `cu121-torch240` extra this section used to pin is stale — do not copy it.** This
machine has CUDA toolkit 13.2 and driver 595.58.03, far newer than that wheel targets.
Resolve the correct variant at install time:

```bash
nvidia-smi --query-gpu=driver_version --format=csv   # driver ceiling
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Install plain `unsloth` first and let it resolve, and only pin an extra if that fails. The
driver is backward-compatible with older CUDA wheels, so a `cu124`/`cu126` torch build is
fine and is the safer choice — the newest toolkit is not the one with the best-tested
bitsandbytes and Unsloth support on Turing.

Python 3.11 is deliberate and differs from the system 3.12.3 — `uv` will fetch it. Use a
**separate venv** from the agent's; training and serving dependency trees conflict.

Run order:
1. **Smoke test** — 10 examples, 1 epoch, confirm loss decreases and nothing NaNs.
2. **Overfit test** — train on 20 examples until loss ≈ 0. If it can't overfit 20
   examples, the data format or loss masking is broken. Do not skip this.
3. **Full run** — log to TensorBoard or W&B; watch train vs. val loss diverge.

Expect roughly **3–8 hours** for 1500 examples × 3 epochs at 4096 seq len — the older 1–3 h
figure assumed FA2; Turing without it is substantially slower even at 9B. Checkpoint every
epoch so you can compare and roll back. Disk is not a concern (207 GB free), but each 9B
merge is ~18 GB in fp16 — clean up intermediates.

**Signs it went wrong:** val loss rising while train loss falls (overfit — fewer epochs
or lower `r`); loss plateauing near zero almost immediately (loss mask broken, it's
learning to copy); NaN loss (fp16 scaling — lower the LR).

---

## 5. Step 5 — Evaluate against the base model

Never ship a fine-tune you haven't measured. Same golden set from `PLAN.md` §10, run
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
- **Do not over-quantize a 9B.** It is small enough that quality loss bites harder than on
  a 27B, and you are not short of VRAM. **Q6_K or Q8_0 is the right default here** —
  Q4_K_M only if this adapter is destined for the Jetson. Keep the f16 GGUF (~18 GB) so you
  can re-quantize without re-merging.

Remember the tuned 9B is the *fast path and fallback*, not the primary — it coexists with
the served 27B/35B from `PLAN.md` §3 rather than replacing it. Budget VRAM for whichever
combination you intend to keep resident, and set `OLLAMA_KEEP_ALIVE` accordingly.

Re-run the eval set against the quantized GGUF here, and compare it to **both** the base 9B
*and* the served primary — if the tuned 9B does not beat the untuned primary on your golden
set, it earns a place only as the offline fallback. Quantization costs a little quality;
confirm it survives the round trip, and confirm TTFT still meets the `PLAN.md` §5 budget,
which is measured end-to-end from the Jetson, not from localhost.

---

## 7. Order of operations

```
PLAN.md phase 0  →  install Ollama + deps (nothing is installed yet)
PLAN.md phase 1  →  RAG + eval set working
PLAN.md phase 2  →  agent loop with tools, logging every trace
        ↓          (run it for 1–2 weeks, collect real usage)
Step 3           →  correct the logged failures into a dataset   ← the real work
Step 1, 2        →  GPU already identified (TITAN RTX, sm_75); pick config
Step 4, 5        →  train, evaluate vs. base
Step 6           →  GGUF → Ollama, served from this same machine
```

Step 1 is now nearly free — the hardware is known. The gate is entirely Step 3, and Step 3
is gated on `PLAN.md` phases 0–2 actually existing and having been *used*.

Skipping straight to Step 4 with a synthetic dataset is possible and will produce a
model. It will also produce no measurable improvement, because there'll be nothing to
measure it against and the training distribution won't match how you actually use it.
