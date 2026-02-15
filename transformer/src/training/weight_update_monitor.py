# training/weight_update_monitor.py
#
# Monitors:
#   1) Gradient health (after backward, preferably after clipping).
#   2) Actual parameter update health (after optimizer.step) using sampled Δw.
#
# Properties:
#   - No full parameter clones.
#   - No full tensor diffs.
#   - Only scalar reductions + small sampled gathers.
#   - Supports multiple optimizer param groups (per-parameter LR).
#
# Intended call order per optimizer step:
#   loss.backward()
#   clip_grad_norm_()   # recommended
#   grad = monitor.check_gradients(model, optimizer)
#
#   optimizer.step()
#   upd = monitor.check_updates(model, optimizer)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GradientDiagnostics:
    """Gradient diagnostics for one parameter tensor."""
    grad_norm_l2: Optional[float]      # ||grad||_2
    grad_max_abs: Optional[float]      # max|grad|
    grad_mean_abs: Optional[float]     # mean|grad|
    learning_rate: Optional[float]     # per-param LR from optimizer
    is_vanishing: bool                 # grad_norm_l2 <= threshold
    is_exploding: bool                 # grad_norm_l2 >= threshold


@dataclass
class UpdateDiagnostics:
    """Update diagnostics for one parameter tensor (sample-based)."""
    delta_w_sample_norm_l2: float      # ||Δw_sample||_2
    delta_w_sample_max_abs: float      # max|Δw_sample|
    w_sample_norm_l2: float            # ||w_sample||_2
    update_ratio: float                # ||Δw||/(||w||+ε)
    frozen_steps: int                  # consecutive frozen count
    is_frozen: bool                    # frozen_steps >= patience
    learning_rate: Optional[float]     # per-param LR from optimizer


class WeightUpdateMonitor:
    """Monitor gradients and actual parameter updates without full clones."""

    def __init__(
        self,
        sample_size: int = 1024,
        sample_seed: int = 1234,
        vanishing_grad_threshold: float = 1e-7,
        exploding_grad_threshold: float = 1e2,
        frozen_update_ratio_threshold: float = 1e-12,
        frozen_patience_steps: int = 3,
        epsilon: float = 1e-12,
    ) -> None:
        self._sample_size = int(sample_size)
        self._sample_seed = int(sample_seed)
        self._epsilon = float(epsilon)

        self._vanishing_grad_threshold = float(vanishing_grad_threshold)
        self._exploding_grad_threshold = float(exploding_grad_threshold)

        self._frozen_update_ratio_threshold = float(frozen_update_ratio_threshold)
        self._frozen_patience_steps = int(frozen_patience_steps)

        self._sample_index_cache: Dict[str, torch.Tensor] = {}
        self._previous_sample_cache: Dict[str, torch.Tensor] = {}
        self._frozen_steps: Dict[str, int] = {}

    def reset(self) -> None:
        """Clears state that depends on prior steps (use after loading checkpoints)."""
        self._previous_sample_cache.clear()
        self._frozen_steps.clear()
        # Keep _sample_index_cache (stable sampling by name/shape).

    @staticmethod
    def _stable_hash_32(text: str) -> int:
        """Stable 32-bit hash (FNV-1a). Avoids Python hash randomization."""
        h = 2166136261
        for b in text.encode("utf-8"):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    @staticmethod
    def _learning_rate_by_param_id(
        optimizer: torch.optim.Optimizer,
    ) -> Dict[int, float]:
        """Builds a map from id(param) to LR (supports multiple param groups)."""
        lr_map: Dict[int, float] = {}
        for group in optimizer.param_groups:
            group_lr = float(group.get("lr", 0.0))
            for p in group["params"]:
                lr_map[id(p)] = group_lr
        return lr_map

    @torch.no_grad()
    def _get_sample_indices(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """Returns cached deterministic sample indices for a parameter tensor."""
        if name in self._sample_index_cache:
            return self._sample_index_cache[name]

        n = param.numel()
        k = min(self._sample_size, n)

        gen = torch.Generator(device=param.device)
        seed = (self._sample_seed + self._stable_hash_32(name) + n) % (2**31 - 1)
        gen.manual_seed(seed)

        idx = torch.randint(0, n, (k,), generator=gen, device=param.device)
        self._sample_index_cache[name] = idx
        return idx

    @staticmethod
    @torch.no_grad()
    def _sample_param_values(param: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gathers a small sample of parameter values (no full clone)."""
        flat = param.detach().flatten()
        return flat.index_select(0, indices)

    @torch.no_grad()
    def check_gradients(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, GradientDiagnostics]:
        """Computes gradient diagnostics after backward (ideally after clipping)."""
        lr_map = self._learning_rate_by_param_id(optimizer)
        out: Dict[str, GradientDiagnostics] = {}

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            lr = lr_map.get(id(p))

            if p.grad is None:
                out[name] = GradientDiagnostics(
                    grad_norm_l2=None,
                    grad_max_abs=None,
                    grad_mean_abs=None,
                    learning_rate=lr,
                    is_vanishing=False,
                    is_exploding=False,
                )
                continue

            g = p.grad.detach()
            g_norm = float(g.norm(p=2).item())
            g_max = float(g.abs().max().item())
            g_mean = float(g.abs().mean().item())

            out[name] = GradientDiagnostics(
                grad_norm_l2=g_norm,
                grad_max_abs=g_max,
                grad_mean_abs=g_mean,
                learning_rate=lr,
                is_vanishing=(g_norm <= self._vanishing_grad_threshold),
                is_exploding=(g_norm >= self._exploding_grad_threshold),
            )

        return out

    @torch.no_grad()
    def check_updates(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, UpdateDiagnostics]:
        """Computes update diagnostics after optimizer.step() using sampled Δw.

        Note: Only monitors parameters that received gradients in the backward pass.
        Parameters without gradients (e.g., unused cross-attention in decoder-only models)
        are automatically skipped to avoid false "frozen" warnings.
        """
        lr_map = self._learning_rate_by_param_id(optimizer)
        out: Dict[str, UpdateDiagnostics] = {}

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # Skip parameters that didn't receive gradients (not in computational graph)
            # This filters out unused modules like cross-attention in decoder-only LMs
            if p.grad is None or p.grad.abs().sum().item() == 0:
                continue

            lr = lr_map.get(id(p))

            idx = self._get_sample_indices(name, p)
            cur = self._sample_param_values(p, idx)

            w_norm = float(cur.norm(p=2).item())

            if name not in self._previous_sample_cache:
                self._previous_sample_cache[name] = cur.clone()
                self._frozen_steps[name] = 0
                out[name] = UpdateDiagnostics(
                    delta_w_sample_norm_l2=0.0,
                    delta_w_sample_max_abs=0.0,
                    w_sample_norm_l2=w_norm,
                    update_ratio=0.0,
                    frozen_steps=0,
                    is_frozen=False,
                    learning_rate=lr,
                )
                continue

            prev = self._previous_sample_cache[name]
            dw = cur - prev

            dw_norm = float(dw.norm(p=2).item())
            dw_max = float(dw.abs().max().item())

            update_ratio = dw_norm / (w_norm + self._epsilon)
            frozen_now = update_ratio <= self._frozen_update_ratio_threshold

            if frozen_now:
                self._frozen_steps[name] = self._frozen_steps.get(name, 0) + 1
            else:
                self._frozen_steps[name] = 0

            frozen_steps = self._frozen_steps[name]
            is_frozen = frozen_steps >= self._frozen_patience_steps

            out[name] = UpdateDiagnostics(
                delta_w_sample_norm_l2=dw_norm,
                delta_w_sample_max_abs=dw_max,
                w_sample_norm_l2=w_norm,
                update_ratio=update_ratio,
                frozen_steps=frozen_steps,
                is_frozen=is_frozen,
                learning_rate=lr,
            )

            self._previous_sample_cache[name] = cur.clone()

        return out

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """Simple percentile on Python floats (p in [0,1])."""
        if not values:
            return 0.0
        xs = sorted(values)
        n = len(xs)
        k = max(0, min(n - 1, math.ceil(p * n) - 1))
        return float(xs[k])

    @classmethod
    def aggregate_gradient_stats(
        cls,
        gradients: Dict[str, GradientDiagnostics],
    ) -> Dict[str, float]:
        norms = [g.grad_norm_l2 for g in gradients.values() if g.grad_norm_l2 is not None]
        if not norms:
            return {"count": 0.0}

        xs = sorted(float(v) for v in norms)
        n = len(xs)
        median = xs[n // 2]

        return {
            "count": float(n),
            "median": float(median),
            "p95": cls._percentile(xs, 0.95),
            "min": float(xs[0]),
            "max": float(xs[-1]),
            "vanishing_count": float(sum(1 for g in gradients.values() if g.is_vanishing)),
            "exploding_count": float(sum(1 for g in gradients.values() if g.is_exploding)),
        }

    @classmethod
    def aggregate_update_stats(
        cls,
        updates: Dict[str, UpdateDiagnostics],
    ) -> Dict[str, float]:
        ratios = [u.update_ratio for u in updates.values()]
        if not ratios:
            return {"count": 0.0}

        xs = sorted(float(v) for v in ratios)
        n = len(xs)
        median = xs[n // 2]

        return {
            "count": float(n),
            "median": float(median),
            "p95": cls._percentile(xs, 0.95),
            "min": float(xs[0]),
            "max": float(xs[-1]),
            "frozen_count": float(sum(1 for u in updates.values() if u.is_frozen)),
        }

    @staticmethod
    def top_k_largest_gradients(
        gradients: Dict[str, GradientDiagnostics],
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        items = [(n, g.grad_norm_l2) for n, g in gradients.items() if g.grad_norm_l2 is not None]
        items.sort(key=lambda t: t[1], reverse=True)
        return [(n, float(v)) for n, v in items[:k]]

    @staticmethod
    def top_k_smallest_updates(
        updates: Dict[str, UpdateDiagnostics],
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        items = [(n, u.update_ratio) for n, u in updates.items()]
        items.sort(key=lambda t: t[1])
        return [(n, float(v)) for n, v in items[:k]]
