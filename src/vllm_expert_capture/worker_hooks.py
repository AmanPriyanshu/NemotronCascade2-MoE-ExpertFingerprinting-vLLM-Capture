"""
worker_hooks.py — Functions that run inside vLLM worker processes.

These must be in a separate importable module (not __main__) so that
pickle can resolve them across process boundaries. vLLM's EngineCore
runs in a separate process and uses pickle to send functions via
collective_rpc / apply_model.

Captures per-token, per-layer:
  - topk_ids: which 6 experts were selected [num_tokens, 6]
  - topk_weights: sigmoid routing weights for selected experts [num_tokens, 6]
  - router_logits: raw logits for ALL 128 experts [num_tokens, 128]
"""

import types

import numpy as np
import torch


def install_hooks(model):
    """Install capture hooks on all MoE layers. Runs in each worker process."""
    import vllm.model_executor.models.nemotron_h as _nemotron_h
    from vllm.distributed import get_tensor_model_parallel_rank

    tp_rank = get_tensor_model_parallel_rank()
    prefix = f"[TP{tp_rank}]"
    print(f"{prefix} install_hooks called", flush=True)

    if not hasattr(_nemotron_h, '_capture_store'):
        _nemotron_h._capture_store = {
            "recording": False,
            "data": {},
        }
        print(f"{prefix} Created new _capture_store", flush=True)
    else:
        print(f"{prefix} Reusing existing _capture_store", flush=True)

    store = _nemotron_h._capture_store
    store["recording"] = False
    store["data"] = {}

    moe_indices = []
    total_layers = len(model.model.layers)
    print(f"{prefix} Scanning {total_layers} layers for MoE...", flush=True)

    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer, _nemotron_h.NemotronHMoEDecoderLayer):
            store["data"][idx] = []

            # -- Hook 1: capture_fn on router for topk_ids --
            def _make_capture(layer_idx, s, p):
                call_count = [0]
                def capture(topk_ids):
                    call_count[0] += 1
                    if call_count[0] <= 3:
                        print(f"{p} capture_fn fired: layer={layer_idx}, "
                              f"recording={s['recording']}, "
                              f"topk_ids.shape={topk_ids.shape}, "
                              f"call #{call_count[0]}", flush=True)
                    if s["recording"]:
                        s["data"][layer_idx].append({
                            "ids": topk_ids.detach().cpu().to(torch.int32).numpy(),
                        })
                        if call_count[0] <= 3:
                            print(f"{p} capture_fn STORED ids for layer {layer_idx}, "
                                  f"entries now: {len(s['data'][layer_idx])}", flush=True)
                return capture

            router = layer.mixer.experts.router
            router.set_capture_fn(_make_capture(idx, store, prefix))
            print(f"{prefix} Layer {idx}: set capture_fn on router "
                  f"(type={type(router).__name__})", flush=True)

            # -- Hook 2: monkey-patch select_experts for topk_weights --
            original_select = type(router).select_experts

            def _make_patched_select(layer_idx, s, orig, p):
                call_count = [0]
                def patched(router_self, hidden_states, router_logits):
                    call_count[0] += 1
                    topk_weights, topk_ids = orig(router_self, hidden_states, router_logits)
                    if call_count[0] <= 3:
                        print(f"{p} select_experts fired: layer={layer_idx}, "
                              f"recording={s['recording']}, "
                              f"weights.shape={topk_weights.shape}, "
                              f"call #{call_count[0]}", flush=True)
                    if s["recording"]:
                        entries = s["data"].get(layer_idx, [])
                        if entries and "weights" not in entries[-1]:
                            entries[-1]["weights"] = topk_weights.detach().cpu().numpy()
                    return topk_weights, topk_ids
                return patched

            router.select_experts = types.MethodType(
                _make_patched_select(idx, store, original_select, prefix), router
            )

            # -- Hook 3: monkey-patch NemotronHMoE.forward for raw router_logits --
            moe_module = layer.mixer
            original_forward = type(moe_module).forward

            def _make_patched_forward(layer_idx, s, orig_fwd, p):
                call_count = [0]
                def patched_forward(moe_self, hidden_states):
                    call_count[0] += 1
                    if call_count[0] <= 3:
                        print(f"{p} MoE.forward fired: layer={layer_idx}, "
                              f"recording={s['recording']}, "
                              f"hidden.shape={hidden_states.shape}, "
                              f"call #{call_count[0]}", flush=True)

                    if s["recording"]:
                        num_tokens, hidden_dim = hidden_states.shape
                        h = hidden_states.view(-1, hidden_dim)
                        router_logits, _ = moe_self.gate(h)
                        s.setdefault("_pending_logits", {})[layer_idx] = \
                            router_logits.detach().cpu().numpy()
                        if call_count[0] <= 3:
                            print(f"{p} captured router_logits.shape="
                                  f"{router_logits.shape} for layer {layer_idx}", flush=True)

                    result = orig_fwd(moe_self, hidden_states)

                    if s["recording"]:
                        entries = s["data"].get(layer_idx, [])
                        pending = s.get("_pending_logits", {}).pop(layer_idx, None)
                        if entries and pending is not None:
                            entries[-1]["router_logits"] = pending
                            if call_count[0] <= 3:
                                print(f"{p} attached router_logits to entry for "
                                      f"layer {layer_idx}", flush=True)

                    return result
                return patched_forward

            moe_module.forward = types.MethodType(
                _make_patched_forward(idx, store, original_forward, prefix), moe_module
            )

            moe_indices.append(idx)

    print(f"{prefix} Installed hooks on {len(moe_indices)} MoE layers: {moe_indices}", flush=True)
    return moe_indices


def start_recording(model):
    """Enable recording in the worker process."""
    import vllm.model_executor.models.nemotron_h as _nemotron_h
    from vllm.distributed import get_tensor_model_parallel_rank

    tp_rank = get_tensor_model_parallel_rank()
    store = _nemotron_h._capture_store
    store["recording"] = True
    for k in store["data"]:
        store["data"][k] = []
    store["_pending_logits"] = {}
    print(f"[TP{tp_rank}] start_recording: recording={store['recording']}, "
          f"data keys={list(store['data'].keys())}", flush=True)
    return True


def stop_recording(model):
    """Disable recording in the worker process."""
    import vllm.model_executor.models.nemotron_h as _nemotron_h
    from vllm.distributed import get_tensor_model_parallel_rank

    tp_rank = get_tensor_model_parallel_rank()
    store = _nemotron_h._capture_store
    store["recording"] = False

    total_entries = sum(len(v) for v in store["data"].values())
    non_empty = sum(1 for v in store["data"].values() if len(v) > 0)
    print(f"[TP{tp_rank}] stop_recording: {total_entries} total entries "
          f"across {non_empty}/{len(store['data'])} layers", flush=True)

    for layer_idx, entries in store["data"].items():
        if entries:
            e = entries[0]
            print(f"[TP{tp_rank}] Layer {layer_idx} sample entry: "
                  f"ids={'shape='+str(e['ids'].shape) if 'ids' in e else 'MISSING'}, "
                  f"weights={'shape='+str(e['weights'].shape) if 'weights' in e else 'MISSING'}, "
                  f"logits={'shape='+str(e['router_logits'].shape) if 'router_logits' in e else 'MISSING'}",
                  flush=True)
            break
    return True


def get_activations(model):
    """Retrieve captured activation data from the worker process."""
    from vllm.distributed import get_tensor_model_parallel_rank

    tp_rank = get_tensor_model_parallel_rank()
    if tp_rank != 0:
        print(f"[TP{tp_rank}] get_activations: skipping (not rank 0)", flush=True)
        return None

    import vllm.model_executor.models.nemotron_h as _nemotron_h
    store = _nemotron_h._capture_store

    result = {}
    for layer_idx, entries in store["data"].items():
        if not entries:
            continue

        ids_list = [e["ids"] for e in entries if "ids" in e]
        weights_list = [e["weights"] for e in entries if "weights" in e]
        logits_list = [e["router_logits"] for e in entries if "router_logits" in e]

        result[layer_idx] = {
            "ids": np.concatenate(ids_list, axis=0) if ids_list else None,
            "weights": np.concatenate(weights_list, axis=0) if weights_list else None,
            "router_logits": np.concatenate(logits_list, axis=0) if logits_list else None,
        }

    print(f"[TP{tp_rank}] get_activations: returning data for "
          f"{len(result)} layers", flush=True)
    if result:
        first_key = next(iter(result))
        d = result[first_key]
        print(f"[TP{tp_rank}] Sample layer {first_key}: "
              f"ids={d['ids'].shape if d['ids'] is not None else None}, "
              f"weights={d['weights'].shape if d['weights'] is not None else None}, "
              f"logits={d['router_logits'].shape if d['router_logits'] is not None else None}",
              flush=True)
    return result
