# NemotronCascade2-MoE-ExpertFingerprinting-vLLM-Capture

Per-token expert routing capture for `nvidia/Nemotron-Cascade-2-30B-A3B` during vLLM inference. Records which experts are selected, their routing weights, and full router logits across all 23 MoE layers — no vLLM fork required.

Built for expert fingerprinting, specialization mapping, and domain-specific activation profiling of MoE architectures.

## Our Aim

- Provide a lightweight capture tool that hooks into vLLM's existing `BaseRouter.capture_fn` API to record token-level expert routing decisions during inference, without modifying vLLM source code.

- Output flat, analysis-ready parquet files with per-token, per-layer expert activations — enabling downstream expert fingerprinting, domain specialization analysis, and activation profiling across workloads.

- Support user-defined prompt sets with topic grouping, so users can profile expert activation patterns across custom domains and workloads relevant to their deployment.

## What It Captures

For every token processed (prompt + generated), across all 23 MoE layers:

| Signal | Shape | Description |
|--------|-------|-------------|
| `expert_id_0..5` | `[num_tokens, 6]` int32 | Which 6 experts (out of 128) were selected |
| `expert_weight_0..5` | `[num_tokens, 6]` float32 | Sigmoid routing weight per selected expert |
| `router_logit_0..127` | `[num_tokens, 128]` float32 | Raw gate scores for ALL 128 experts |

Full score visibility — not just the top-6 winners, but the entire expert score landscape including near-misses.

## Installation

```bash
git clone https://github.com/AmanPriyanshu/NemotronCascade2-MoE-ExpertFingerprinting-vLLM-Capture.git
cd NemotronCascade2-MoE-ExpertFingerprinting-vLLM-Capture
pip install .
```

### Requirements

- 8x H100 80GB (or equivalent ~640GB VRAM for BF16 tp=8)
- Python 3.12+, CUDA 12.8+

Pinned dependencies: `vllm==0.19.0`, `transformers==4.57.6`, `torch==2.10.0`, `numpy==2.2.6`, `pyarrow>=19.0.0`

> **Note:** Hooks into vLLM internals (`BaseRouter.capture_fn`, monkey-patched `select_experts`, monkey-patched `NemotronHMoE.forward`). Pinned to vLLM 0.19.0 — other versions may break.

## Usage

```bash
# Single prompt (default)
vllm-expert-capture

# 5 prompts, sequential mode (per-prompt isolation)
vllm-expert-capture --num-prompts 5 --mode sequential --output-dir activations

# Batch mode (max throughput, aggregated activations)
vllm-expert-capture --num-prompts 5 --mode batch --output-dir activations

# Custom prompts with topic grouping
vllm-expert-capture --prompts my_prompts.json --mode sequential
```

### Custom Prompts Format

```json
[
  {
    "topic": "math",
    "name": "arithmetic",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 25 * 37? Show your reasoning."}
    ]
  },
  {
    "topic": "code",
    "name": "python_func",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a function to find the LCS of two strings."}
    ]
  }
]
```

### Modes

- **`sequential`** (default) — Prompts run one at a time. Each prompt's activations are isolated with correct `prompt_index` and `prompt_name` attribution. Use this for per-prompt/per-topic expert profiling.
- **`batch`** — All prompts in one `llm.generate()` call. Max throughput but activations are interleaved across prompts (no per-prompt attribution). Use this for aggregate corpus-level analysis.

## Output Format

Single flat **parquet** file with zstd compression + JSON metadata sidecar.

### Parquet Schema (145 fixed columns)

One row per (token, layer) pair:

| Column | Type | Description |
|--------|------|-------------|
| `prompt_index` | int32 | Prompt index (0-indexed, -1 for batch aggregated) |
| `prompt_name` | string | Prompt label |
| `token_position` | int32 | Position within captured sequence |
| `token_id` | int32 | Vocabulary token ID |
| `layer_index` | int32 | MoE layer index in the 52-layer model |
| `expert_id_0` .. `expert_id_5` | int32 | 6 selected expert IDs (top-6 routing) |
| `expert_weight_0` .. `expert_weight_5` | float32 | Sigmoid routing weights for selected experts |
| `router_logit_0` .. `router_logit_127` | float32 | Raw gate scores for all 128 experts |

### Output Structure

```
activations/
├── activations.parquet    # Single merged parquet (all prompts)
└── metadata.json          # Config, token_ids, per-layer summaries
```

### Reading the Output

```python
import pyarrow.parquet as pq

table = pq.read_table("activations/activations.parquet")
df = table.to_pandas()

# Filter by prompt
math_df = df[df.prompt_name == "arithmetic"]

# Filter by layer
layer_10 = df[df.layer_index == 10]

# Get expert IDs for token 42 at layer 10
row = df[(df.token_position == 42) & (df.layer_index == 10)].iloc[0]
expert_ids = [row[f"expert_id_{k}"] for k in range(6)]
expert_weights = [row[f"expert_weight_{k}"] for k in range(6)]
router_logits = [row[f"router_logit_{k}"] for k in range(128)]
```

## Architecture Reference

```
Nemotron-Cascade-2-30B-A3B (52 layers):
MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME

M = Mamba-2 (dense)              — 22 layers
E = MoE FFN (128 experts, top-6) — 23 layers  <- captured
* = Attention GQA anchor (dense) — 7 layers

MoE layer indices: [1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51]

Routing: GroupedTopKRouter with sigmoid scoring
  128 routed experts, 8 expert groups
  Top-2 groups then top-6 overall
  Sigmoid scores (independent, not normalized)
```

## Repository Structure

```
NemotronCascade2-MoE-ExpertFingerprinting-vLLM-Capture/
├── pyproject.toml
├── README.md
├── LICENSE
└── src/vllm_expert_capture/
    ├── __init__.py
    ├── capture.py          # Model loading, prompt execution, parquet writing
    └── worker_hooks.py     # Hook functions for vLLM worker processes
```

`worker_hooks.py` is separated from `capture.py` because vLLM's `apply_model()` uses pickle to send functions to worker processes. Functions in `__main__` can't be pickled across process boundaries — they must be in an importable module.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{priyanshu2025nemotronexpertcapture,
  title={NemotronCascade2-MoE-ExpertFingerprinting-vLLM-Capture: Token-Level Expert
         Routing Capture for Mixture-of-Experts Models},
  author={Priyanshu, Aman and Vijay, Supriti},
  year={2025},
  howpublished={\url{https://github.com/AmanPriyanshu/NemotronCascade2-MoE-ExpertFingerprinting-vLLM-Capture}}
}
```

## Authors

- [Aman Priyanshu](https://amanpriyanshu.github.io/)
- [Supriti Vijay](https://supritivijay.github.io/)

## License

Apache-2.0
