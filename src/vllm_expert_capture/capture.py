"""
capture.py — Record per-token expert routing decisions from Nemotron-Cascade-2-30B-A3B.

Hooks into vLLM's built-in capture_fn on each MoE layer's GroupedTopKRouter
to record topk_ids, topk_weights, and raw router_logits for every token
across all 23 MoE layers. No vLLM fork required.

Output: single flat parquet + JSON metadata sidecar.

Parquet schema (145 fixed columns, one row per token x layer):
  prompt_index      int32     which prompt (0-indexed)
  prompt_name       string    prompt label
  token_position    int32     position within this prompt's captured sequence
  token_id          int32     vocabulary token ID
  layer_index       int32     MoE layer index in the 52-layer model
  expert_id_0..5    int32     6 selected expert IDs (top-6 routing)
  expert_weight_0..5 float32  sigmoid routing weights for selected experts
  router_logit_0..127 float32 raw gate scores for all 128 experts

Modes:
  sequential (default) — prompts run one at a time, per-prompt isolation
  batch — all prompts in one llm.generate() call, aggregated (no per-prompt split)

Requires: VLLM_ALLOW_INSECURE_SERIALIZATION=1 (set automatically)
"""

import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

from vllm_expert_capture.worker_hooks import (
    install_hooks,
    start_recording,
    stop_recording,
    get_activations,
)

MODEL_ID = "nvidia/Nemotron-Cascade-2-30B-A3B"
NUM_EXPERTS = 128
TOP_K = 6
NUM_MOE_LAYERS = 23

DEFAULT_PROMPTS = [
    {
        "name": "math",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 25 * 37? Show your reasoning step by step."},
        ],
    },
    {
        "name": "code",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a Python function to find the longest common subsequence of two strings. Include type hints."},
        ],
    },
    {
        "name": "tool_use",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to a search tool."},
            {"role": "user", "content": "What are the latest developments in quantum error correction? Search for recent papers and summarize."},
        ],
    },
    {
        "name": "structured_io",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Parse into JSON {name, age, occupation, hobbies}: John Smith is a 34-year-old software engineer who enjoys rock climbing, photography, and cooking."},
        ],
    },
    {
        "name": "planning",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Outline a step-by-step plan for building a CI/CD pipeline for a Python monorepo with 5 services using GitHub Actions."},
        ],
    },
]


def _raw_to_arrays(raw_data, moe_indices):
    """Convert raw activation dict to concatenated numpy arrays.

    Returns (token_pos, layer_idx, ids, weights, logits) all as 1D/2D numpy arrays
    with one entry per (token, layer).
    """
    chunk_token_pos = []
    chunk_layer_idx = []
    chunk_ids = []
    chunk_weights = []
    chunk_logits = []

    for layer_idx in moe_indices:
        data = raw_data.get(layer_idx)
        if data is None:
            continue

        ids_arr = data["ids"].astype(np.int32)
        num_tokens = ids_arr.shape[0]

        chunk_token_pos.append(np.arange(num_tokens, dtype=np.int32))
        chunk_layer_idx.append(np.full(num_tokens, layer_idx, dtype=np.int32))
        chunk_ids.append(ids_arr)

        if data["weights"] is not None:
            chunk_weights.append(data["weights"].astype(np.float32))
        else:
            chunk_weights.append(np.zeros((num_tokens, TOP_K), dtype=np.float32))

        logits_arr = data.get("router_logits")
        if logits_arr is not None:
            chunk_logits.append(logits_arr.astype(np.float32))
        else:
            chunk_logits.append(np.zeros((num_tokens, NUM_EXPERTS), dtype=np.float32))

    return (
        np.concatenate(chunk_token_pos),
        np.concatenate(chunk_layer_idx),
        np.concatenate(chunk_ids),
        np.concatenate(chunk_weights),
        np.concatenate(chunk_logits),
    )


def build_table(prompt_index, prompt_name, token_ids, token_pos, layer_idx,
                ids, weights, logits):
    """Build a pyarrow table from numpy arrays for one prompt (or aggregated batch)."""
    n = len(token_pos)
    token_ids_arr = np.array(token_ids, dtype=np.int32)

    # Map token positions to token IDs (captured may include prompt tokens beyond generated)
    mapped_token_id = np.where(
        token_pos < len(token_ids_arr),
        token_ids_arr[np.minimum(token_pos, len(token_ids_arr) - 1)],
        np.int32(-1),
    )

    columns = {
        "prompt_index": pa.array(np.full(n, prompt_index, dtype=np.int32)),
        "prompt_name": pa.array([prompt_name] * n, type=pa.string()),
        "token_position": pa.array(token_pos),
        "token_id": pa.array(mapped_token_id),
        "layer_index": pa.array(layer_idx),
    }
    for k in range(TOP_K):
        columns[f"expert_id_{k}"] = pa.array(ids[:, k])
    for k in range(TOP_K):
        columns[f"expert_weight_{k}"] = pa.array(weights[:, k])
    for k in range(NUM_EXPERTS):
        columns[f"router_logit_{k}"] = pa.array(logits[:, k])

    return pa.table(columns)


def summarize_activations(raw_data):
    """Compute per-layer expert activation counts and mean weights."""
    summary = {}
    for layer_idx, data in raw_data.items():
        ids = data["ids"]
        weights = data["weights"]

        expert_counts = np.zeros(NUM_EXPERTS, dtype=np.int64)
        expert_weight_sums = np.zeros(NUM_EXPERTS, dtype=np.float64)

        for k in range(ids.shape[1]):
            np.add.at(expert_counts, ids[:, k], 1)
            if weights is not None:
                np.add.at(expert_weight_sums, ids[:, k], weights[:, k])

        active_mask = expert_counts > 0
        mean_weights = np.zeros(NUM_EXPERTS, dtype=np.float64)
        if weights is not None:
            mean_weights[active_mask] = expert_weight_sums[active_mask] / expert_counts[active_mask]

        summary[layer_idx] = {
            "total_tokens": int(ids.shape[0]),
            "unique_experts": int(active_mask.sum()),
            "expert_counts": expert_counts.tolist(),
            "expert_mean_weights": mean_weights.tolist(),
        }
    return summary


def print_layer_summary(summary, moe_indices, label, num_tokens):
    """Pretty-print per-layer activation summary."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {label}  --  {num_tokens} tokens")
    print(f"{sep}")
    print(f"  {'Layer':>6} | {'Tokens':>7} | {'Active':>6}/128 | Top-5 experts (count)")
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*10}-+-{'-'*40}")

    for layer_idx in moe_indices:
        data = summary.get(layer_idx)
        if data is None:
            print(f"  {layer_idx:>6} | {'---':>7} | {'---':>10} | no data")
            continue
        counts = np.array(data["expert_counts"])
        top5_idx = np.argsort(counts)[-5:][::-1]
        top5_str = ", ".join(f"e{i}={counts[i]}" for i in top5_idx if counts[i] > 0)
        print(f"  {layer_idx:>6} | {data['total_tokens']:>7} | {data['unique_experts']:>6}/128 | {top5_str}")


def run_sequential(llm, tokenizer, params, prompts, moe_indices):
    """Run prompts one at a time. Returns list of (table, meta) per prompt."""
    tables = []
    all_meta = []

    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/{len(prompts)}: {prompt['name']} ---")

        llm.apply_model(start_recording)

        t0 = time.time()
        text = tokenizer.apply_chat_template(
            prompt["messages"], tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        outputs = llm.generate([text], params)
        elapsed = time.time() - t0

        llm.apply_model(stop_recording)

        activation_results = llm.apply_model(get_activations)
        raw_data = next((r for r in activation_results if r is not None), None)

        out = outputs[0].outputs[0]
        num_tokens = len(out.token_ids)
        token_ids = list(out.token_ids)

        print(f"  {num_tokens} tokens in {elapsed:.1f}s")
        print(f"  Output: {out.text[:200]}...")

        summary = {}
        if raw_data is None or len(raw_data) == 0:
            print("  WARNING: No activation data captured!")
        else:
            summary = summarize_activations(raw_data)
            print_layer_summary(summary, moe_indices, prompt["name"], num_tokens)

            for layer_idx, data in raw_data.items():
                captured = data["ids"].shape[0]
                if captured != num_tokens:
                    print(f"  NOTE: Layer {layer_idx} captured {captured} tokens "
                          f"(generated {num_tokens}) -- includes prompt tokens")

            token_pos, layer_idx, ids, weights, logits = _raw_to_arrays(raw_data, moe_indices)
            table = build_table(i, prompt["name"], token_ids, token_pos, layer_idx,
                                ids, weights, logits)
            tables.append(table)
            print(f"  Table: {len(table)} rows, {table.num_columns} cols")

        all_meta.append({
            "name": prompt["name"],
            "index": i,
            "num_tokens_generated": num_tokens,
            "token_ids": token_ids,
            "text_preview": out.text[:500],
            "elapsed_seconds": round(elapsed, 2),
            "per_layer_summary": {str(k): v for k, v in summary.items()},
        })

    return tables, all_meta


def run_batch(llm, tokenizer, params, prompts, moe_indices):
    """Run all prompts in a single llm.generate() call. Returns (table, meta)."""
    print(f"\n--- Batch mode: {len(prompts)} prompts ---")

    texts = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            prompt["messages"], tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        texts.append(text)

    llm.apply_model(start_recording)
    t0 = time.time()
    outputs = llm.generate(texts, params)
    elapsed = time.time() - t0
    llm.apply_model(stop_recording)

    activation_results = llm.apply_model(get_activations)
    raw_data = next((r for r in activation_results if r is not None), None)

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  {total_tokens} total tokens in {elapsed:.1f}s "
          f"({total_tokens/elapsed:.0f} tok/s)")

    all_token_ids = []
    for o in outputs:
        all_token_ids.extend(list(o.outputs[0].token_ids))

    table = None
    summary = {}
    if raw_data is None or len(raw_data) == 0:
        print("  WARNING: No activation data captured!")
    else:
        for layer_idx, data in raw_data.items():
            print(f"  Layer {layer_idx}: {data['ids'].shape[0]} tokens captured")
            break

        summary = summarize_activations(raw_data)
        print_layer_summary(summary, moe_indices, f"BATCH ({len(prompts)} prompts)", total_tokens)

        token_pos, layer_idx_arr, ids, weights, logits = _raw_to_arrays(raw_data, moe_indices)
        table = build_table(-1, "batch_aggregated", all_token_ids, token_pos,
                            layer_idx_arr, ids, weights, logits)
        print(f"  Table: {len(table)} rows, {table.num_columns} cols")

    per_prompt = []
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        out = output.outputs[0]
        per_prompt.append({
            "name": prompt["name"],
            "index": i,
            "num_tokens_generated": len(out.token_ids),
            "token_ids": list(out.token_ids),
            "text_preview": out.text[:500],
        })

    batch_meta = {
        "mode": "batch",
        "num_prompts": len(prompts),
        "total_tokens_generated": total_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(total_tokens / elapsed, 1),
        "per_prompt": per_prompt,
        "aggregated_summary": {str(k): v for k, v in summary.items()},
    }

    return table, batch_meta


def main():
    parser = argparse.ArgumentParser(
        description="Capture expert activations from Nemotron-Cascade-2-30B-A3B")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Path to JSON file with custom prompts")
    parser.add_argument("--output-dir", type=str, default="activations",
                        help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--tp", type=int, default=8,
                        help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max model context length")
    parser.add_argument("--num-prompts", type=int, default=1,
                        help="Number of default prompts to run")
    parser.add_argument("--mode", choices=["sequential", "batch"],
                        default="sequential",
                        help="sequential: one at a time. batch: all at once")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)
    else:
        prompts = DEFAULT_PROMPTS[:args.num_prompts]

    sep = "=" * 72
    print(sep)
    print("  Expert Activation Capture")
    print(f"  Model: {MODEL_ID}")
    print(f"  Mode: {args.mode}")
    print(f"  Prompts: {len(prompts)}, Max tokens: {args.max_tokens}, TP: {args.tp}")
    print(f"  Output: {args.output_dir}/")
    print(sep)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"Loading model (tp={args.tp}, enforce_eager=True)...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("\nRegistering capture hooks on MoE layers...")
    results = llm.apply_model(install_hooks)
    moe_indices = results[0]
    print(f"  Hooked {len(moe_indices)} MoE layers: {moe_indices}")
    assert len(moe_indices) == NUM_MOE_LAYERS, \
        f"Expected {NUM_MOE_LAYERS} MoE layers, found {len(moe_indices)}"

    params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=args.max_tokens)

    # --- Run and build merged parquet ---
    if args.mode == "sequential":
        tables, prompts_meta = run_sequential(llm, tokenizer, params, prompts, moe_indices)
        if tables:
            merged = pa.concat_tables(tables)
        else:
            merged = None
        metadata = {
            "model": MODEL_ID,
            "mode": "sequential",
            "config": {
                "num_experts": NUM_EXPERTS,
                "top_k": TOP_K,
                "num_moe_layers": NUM_MOE_LAYERS,
                "moe_layer_indices": moe_indices,
            },
            "prompts": prompts_meta,
        }
        total_tokens = sum(p["num_tokens_generated"] for p in prompts_meta)

    elif args.mode == "batch":
        merged, batch_meta = run_batch(llm, tokenizer, params, prompts, moe_indices)
        metadata = {
            "model": MODEL_ID,
            "mode": "batch",
            "config": {
                "num_experts": NUM_EXPERTS,
                "top_k": TOP_K,
                "num_moe_layers": NUM_MOE_LAYERS,
                "moe_layer_indices": moe_indices,
            },
            "batch": batch_meta,
        }
        total_tokens = batch_meta["total_tokens_generated"]

    # --- Write single parquet ---
    if merged is not None:
        pq_path = os.path.join(args.output_dir, "activations.parquet")
        pq.write_table(merged, pq_path, compression="zstd")
        pq_size = os.path.getsize(pq_path) / 1024 / 1024
        print(f"\nSaved {pq_path} ({pq_size:.1f} MB, {len(merged)} rows, {merged.num_columns} cols)")
    else:
        print("\nWARNING: No activation data to save!")

    # --- Write metadata sidecar ---
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    meta_size = os.path.getsize(meta_path) / 1024
    print(f"Saved {meta_path} ({meta_size:.1f} KB)")

    # --- List output ---
    print(f"\nOutput files in {args.output_dir}/:")
    for fname in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, fname)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            print(f"  {fname:40s} {size / 1024 / 1024:.1f} MB")
        else:
            print(f"  {fname:40s} {size / 1024:.1f} KB")

    # --- Global expert utilization summary ---
    print(f"\n{sep}")
    print("  GLOBAL EXPERT UTILIZATION")
    print(sep)

    if args.mode == "sequential":
        global_counts = np.zeros((len(moe_indices), NUM_EXPERTS), dtype=np.int64)
        for r in prompts_meta:
            for i, li in enumerate(moe_indices):
                data = r["per_layer_summary"].get(str(li))
                if data:
                    global_counts[i] += np.array(data["expert_counts"])
    else:
        global_counts = np.zeros((len(moe_indices), NUM_EXPERTS), dtype=np.int64)
        for i, li in enumerate(moe_indices):
            data = batch_meta["aggregated_summary"].get(str(li))
            if data:
                global_counts[i] = np.array(data["expert_counts"])

    for i, layer_idx in enumerate(moe_indices):
        counts = global_counts[i]
        active = (counts > 0).sum()
        total = counts.sum()
        top3 = np.argsort(counts)[-3:][::-1]
        bot3 = np.argsort(counts)[:3]
        print(f"  Layer {layer_idx:>2}: {active:>3}/128 active, {total:>6} selections  "
              f"top=[{','.join(f'{e}:{counts[e]}' for e in top3)}]  "
              f"bot=[{','.join(f'{e}:{counts[e]}' for e in bot3)}]")

    print(f"\n  Total tokens: {total_tokens}")
    print(f"  Output dir: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
