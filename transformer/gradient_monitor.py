"""Gradient flow monitoring utility for deep neural networks.

This module provides GradientGainMonitor, a diagnostic tool for analyzing
gradient flow through transformer blocks or any sequential architecture with
residual connections.

Typical usage example:
    with GradientGainMonitor(model.blocks) as monitor:
        loss.backward()
        print(monitor.report())
"""
import torch
import math
from typing import List, Optional, Union, Tuple


class GradientGainMonitor:
    r"""Monitors gradient norms and gains across network blocks.

    Records gradient magnitudes for each block output and computes per-block
    gradient gains to diagnose vanishing/exploding gradient problems.

    Gradient gain measures how gradient magnitude changes between blocks:

    .. math::
        \gamma_i = \frac{\|\frac{\partial L}{\partial h_{i-1}}\|}{
                         \|\frac{\partial L}{\partial h_i}\|}

    We define gamma_i as previous_norm / current_norm (backward gain).

    Interpretation:
        - gamma > 1: Gradient amplifies (grows) going backward
        - gamma near 1: Gradient preserved (often healthy for skip connections)
        - gamma < 1: Gradient dampens (shrinks)
        - gamma near 0: Severe vanishing gradient

    For L blocks, records L norms and computes L-1 gain values.

    Note: This tool *detects* gradient flow patterns; it does not prevent
    vanishing/exploding gradients. Use detected patterns to inform
    architectural or training decisions.

    Attributes:
        blocks: List of nn.Module blocks being monitored.
        norm_type: Type of metric used ('l2', 'l1', 'linf', or 'mean').
        strict_single_backward: If True, raises error on multiple backward
            calls without reset(). If False, accumulates/updates norms.
    """

    def __init__(
        self,
        blocks: Union[List[torch.nn.Module], torch.nn.ModuleList, torch.nn.Sequential],
        norm_type: str = 'l2',
        strict_single_backward: bool = True,
        eps: float = 1e-30
    ):
        """Initializes the gradient gain monitor.

        Args:
            blocks: List, ModuleList, or Sequential of nn.Module blocks to
                monitor. Each block should be a layer or transformer block
                whose output gradients you want to track.
            norm_type: Type of metric to compute. Options are:
                - 'l2' (default): Euclidean norm
                - 'l1': Manhattan norm
                - 'linf': Maximum absolute value
                - 'mean': Mean absolute value (not a norm mathematically)
            strict_single_backward: If True, raises error if backward() is
                called multiple times without reset(). If False, overwrites
                previous measurements (useful for gradient accumulation).
            eps: Epsilon for numerical stability when checking for zero
                gradients. Default 1e-30.

        Raises:
            ValueError: If norm_type is not one of the supported types.
        """
        self.blocks = list(blocks)
        self.norm_type = norm_type
        self.strict_single_backward = strict_single_backward
        self.eps = eps

        # Validate norm_type early
        valid_norm_types = {'l2', 'l1', 'linf', 'mean'}
        if norm_type not in valid_norm_types:
            raise ValueError(
                f"norm_type must be one of {valid_norm_types}, "
                f"got '{norm_type}'"
            )

        # Initialize storage and tracking
        self._forward_hook_handles = []
        self._recorded_gradient_norms: List[Optional[float]] = (
            [None] * len(self.blocks)
        )
        self._backward_has_been_called = False

        # Register forward hooks on each block
        # These will attach gradient hooks during forward pass
        for block_index, block_module in enumerate(self.blocks):
            forward_hook = self._create_forward_hook_for_block(
                block_index
            )
            hook_handle = block_module.register_forward_hook(
                forward_hook
            )
            self._forward_hook_handles.append(hook_handle)

    def _create_forward_hook_for_block(self, block_index: int):
        """Creates forward hook that attaches gradient hook to output.

        Factory pattern captures block_index in closure for each hook.

        Args:
            block_index: Index of the block (0 to L-1).

        Returns:
            Forward hook function.
        """
        def forward_hook_function(
            module: torch.nn.Module,
            module_inputs: tuple,
            module_output: Union[torch.Tensor, tuple, list, dict]
        ):
            """Attaches gradient hook to block output during forward pass."""
            # Extract main output tensor from various formats
            if isinstance(module_output, torch.Tensor):
                output_tensor = module_output
            elif isinstance(module_output, dict):
                # Handle dict outputs (common in HuggingFace transformers)
                # Try common keys
                if 'hidden_states' in module_output:
                    output_tensor = module_output['hidden_states']
                elif 'last_hidden_state' in module_output:
                    output_tensor = module_output['last_hidden_state']
                elif len(module_output) > 0:
                    # Use first value
                    output_tensor = next(iter(module_output.values()))
                else:
                    raise TypeError(
                        f"Block {block_index} output dict is empty"
                    )
            elif (isinstance(module_output, (tuple, list)) and
                  len(module_output) > 0):
                # Handle blocks that return (output, aux_data)
                first_output = module_output[0]
                if isinstance(first_output, torch.Tensor):
                    output_tensor = first_output
                else:
                    raise TypeError(
                        f"Block {block_index} output[0] is "
                        f"{type(first_output)}, expected Tensor"
                    )
            else:
                raise TypeError(
                    f"Block {block_index} output is "
                    f"{type(module_output)}, expected Tensor, tuple, "
                    f"list, or dict"
                )

            # Only attach hook if tensor requires grad
            if not output_tensor.requires_grad:
                return

            # Register gradient hook on the output tensor
            # This hook will fire during backward() to record norm
            gradient_hook_function = (
                self._create_gradient_hook_for_block(block_index)
            )
            output_tensor.register_hook(gradient_hook_function)

        return forward_hook_function

    def _create_gradient_hook_for_block(self, block_index: int):
        r"""Creates gradient hook that records norm during backward.

        Args:
            block_index: Index of the block (0 to L-1).

        Returns:
            Gradient hook function.
        """
        def gradient_hook_function(gradient_tensor: torch.Tensor):
            """Records gradient norm when backward pass reaches this block."""
            # Detach to prevent graph retention and ensure safety
            gradient_tensor = gradient_tensor.detach()

            current_recorded_norm = (
                self._recorded_gradient_norms[block_index]
            )

            # Handle multiple backward calls based on strictness setting
            if current_recorded_norm is not None:
                if self.strict_single_backward:
                    raise RuntimeError(
                        f"Gradient hook fired twice for block {block_index}. "
                        f"This likely means you called backward() multiple "
                        f"times (e.g., gradient accumulation, retain_graph=True, "
                        f"or checkpointing). Either call monitor.reset() between "
                        f"backward passes, or create the monitor with "
                        f"strict_single_backward=False."
                    )
                # Non-strict mode: overwrite with latest measurement

            # Mark that backward has been called
            self._backward_has_been_called = True

            # Compute and store gradient norm
            computed_norm = self._compute_gradient_norm(gradient_tensor)
            self._recorded_gradient_norms[block_index] = computed_norm

        return gradient_hook_function

    def _compute_gradient_norm(
        self,
        gradient_tensor: torch.Tensor
    ) -> float:
        """Computes gradient norm/metric based on configured norm_type.

        Args:
            gradient_tensor: Gradient tensor to compute norm of (should be detached).

        Returns:
            Scalar norm value.
        """
        if self.norm_type == 'l2':
            # Euclidean norm (most common)
            return gradient_tensor.norm(p=2).item()
        elif self.norm_type == 'l1':
            # Manhattan norm (less sensitive to outliers)
            return gradient_tensor.norm(p=1).item()
        elif self.norm_type == 'linf':
            # Maximum absolute value (detects spikes)
            return gradient_tensor.abs().max().item()
        elif self.norm_type == 'mean':
            # Mean absolute value (not a norm, but useful metric)
            return gradient_tensor.abs().mean().item()
        else:
            raise ValueError(
                f"Unknown norm_type: {self.norm_type}. "
                f"Use 'l2', 'l1', 'linf', or 'mean'."
            )

    def reset(self):
        """Resets recorded norms for next backward pass.

        Call between backward() passes when doing gradient accumulation
        or multiple losses.
        """
        self._recorded_gradient_norms = [None] * len(self.blocks)
        self._backward_has_been_called = False

    def norms(self) -> List[Optional[float]]:
        r"""Returns gradient norms for each block output.

        Returns:
            List of length L (number of blocks). Each element is:
                - float: gradient norm for that block
                - None: if block did not receive gradients

        Raises:
            RuntimeError: If backward() has not been called yet.
        """
        if not self._backward_has_been_called:
            raise RuntimeError(
                "No gradients recorded. Did you call backward()? "
                "Gradient hooks only fire during the backward pass."
            )
        return self._recorded_gradient_norms

    def gains(self) -> List[Optional[float]]:
        r"""Returns gradient gain ratios between consecutive blocks.

        Gain ratio measures gradient magnitude change:

        .. math::
            \gamma_i = \frac{\|\frac{\partial L}{\partial h_{i-1}}\|}{
                             \|\frac{\partial L}{\partial h_i}\|}

        We define gamma_i as previous_norm / current_norm (backward gain).

        Interpretation:
            - gamma > 1: Gradient amplifies going backward
            - gamma near 1: Gradient preserved (often healthy)
            - gamma < 1: Gradient dampens
            - gamma near 0: Severe vanishing
            - gamma = inf: Zero gradient denominator (complete vanishing)

        Returns:
            List of length L-1. Each element is:
                - float: gain value
                - float('inf'): if denominator < eps
                - None: if either norm is None

        Raises:
            RuntimeError: If backward() has not been called yet.
        """
        if not self._backward_has_been_called:
            raise RuntimeError(
                "No gradients recorded. Did you call backward()?"
            )

        recorded_norms = self._recorded_gradient_norms
        computed_gains = []

        # Compute gain for each transition between consecutive blocks
        for current_block_index in range(1, len(recorded_norms)):
            previous_block_norm = recorded_norms[current_block_index - 1]
            current_block_norm = recorded_norms[current_block_index]

            # Handle missing data
            if previous_block_norm is None or current_block_norm is None:
                computed_gains.append(None)
            # Handle zero/near-zero denominator (use eps for stability)
            elif current_block_norm < self.eps:
                computed_gains.append(float('inf'))
            # Normal case: compute ratio
            else:
                gain_ratio = previous_block_norm / current_block_norm
                computed_gains.append(gain_ratio)

        return computed_gains

    def log_gains(self) -> List[Optional[float]]:
        r"""Returns log-space gradient gains for numerical stability.

        Log-gains are more stable than raw gains for visualization:

        .. math::
            \log \gamma_i = \log \|\frac{\partial L}{\partial h_{i-1}}\| -
                           \log \|\frac{\partial L}{\partial h_i}\|

        Interpretation:
            - log_gamma > 0: Gradient amplifies
            - log_gamma = 0: Gradient preserved
            - log_gamma < 0: Gradient dampens

        Returns:
            List of length L-1. Each element is:
                - float: log gain value
                - float('inf'): if denominator < eps
                - None: if either norm is None

        Raises:
            RuntimeError: If backward() has not been called yet.
        """
        if not self._backward_has_been_called:
            raise RuntimeError(
                "No gradients recorded. Did you call backward()?"
            )

        recorded_norms = self._recorded_gradient_norms
        computed_log_gains = []

        for current_block_index in range(1, len(recorded_norms)):
            previous_block_norm = recorded_norms[current_block_index - 1]
            current_block_norm = recorded_norms[current_block_index]

            if previous_block_norm is None or current_block_norm is None:
                computed_log_gains.append(None)
            elif current_block_norm < self.eps or previous_block_norm < self.eps:
                computed_log_gains.append(float('inf'))
            else:
                # log(a/b) = log(a) - log(b)
                log_gain = math.log(previous_block_norm) - math.log(current_block_norm)
                computed_log_gains.append(log_gain)

        return computed_log_gains

    def summary_stats(self) -> dict:
        """Returns summary statistics about gradient flow.

        Returns:
            Dictionary containing:
                - 'mean_gain': Average gain across transitions
                - 'min_gain': Minimum gain value
                - 'max_gain': Maximum gain value
                - 'mean_log_gain': Average log-gain
                - 'num_amplifying': Count of amplifying transitions (gamma > 2.0)
                - 'num_damping': Count of damping transitions (gamma < 0.5)
                - 'num_healthy': Count of healthy transitions (0.5 <= gamma <= 2.0)
                - 'num_vanishing': Count of vanishing transitions (gamma = inf)
                - 'num_missing': Count of transitions with missing data

        Raises:
            RuntimeError: If backward() has not been called yet.
        """
        computed_gains = self.gains()
        computed_log_gains = self.log_gains()

        # Filter valid gains (not None, not inf)
        valid_gains = [g for g in computed_gains
                       if g is not None and g != float('inf')]
        valid_log_gains = [lg for lg in computed_log_gains
                           if lg is not None and lg != float('inf')]

        # Count categories
        num_amplifying = sum(1 for g in computed_gains
                            if g is not None and g != float('inf') and g > 2.0)
        num_damping = sum(1 for g in computed_gains
                         if g is not None and g != float('inf') and g < 0.5)
        num_healthy = sum(1 for g in computed_gains
                         if g is not None and g != float('inf') and 0.5 <= g <= 2.0)
        num_vanishing = sum(1 for g in computed_gains if g == float('inf'))
        num_missing = sum(1 for g in computed_gains if g is None)

        return {
            'mean_gain': sum(valid_gains) / len(valid_gains) if valid_gains else None,
            'min_gain': min(valid_gains) if valid_gains else None,
            'max_gain': max(valid_gains) if valid_gains else None,
            'mean_log_gain': sum(valid_log_gains) / len(valid_log_gains) if valid_log_gains else None,
            'num_amplifying': num_amplifying,
            'num_damping': num_damping,
            'num_healthy': num_healthy,
            'num_vanishing': num_vanishing,
            'num_missing': num_missing,
        }

    def report(self) -> str:
        """Generates human-readable gradient flow report.

        Returns:
            Formatted string with norms, gains, and summary statistics.
        """
        if not self._backward_has_been_called:
            return "No gradients recorded yet. Call backward() first."

        recorded_norms = self._recorded_gradient_norms
        computed_gains = self.gains()
        computed_log_gains = self.log_gains()
        stats = self.summary_stats()

        # Build report header
        report_lines = [
            "Gradient Flow Analysis",
            "=" * 60,
            f"Metric type: {self.norm_type}",
            f"Strict mode: {self.strict_single_backward}",
            ""
        ]

        # Report gradient norms for each block
        report_lines.append("Gradient Norms:")
        for block_index, norm_value in enumerate(recorded_norms):
            if norm_value is None:
                formatted_norm = "None"
            else:
                formatted_norm = f"{norm_value:.4e}"
            report_lines.append(
                f"  Block {block_index:2d}: {formatted_norm}"
            )

        # Report gradient gains between blocks
        report_lines.append("")
        report_lines.append("Gradient Gains (ratio of consecutive norms):")

        for transition_index, (gain_value, log_gain) in enumerate(
            zip(computed_gains, computed_log_gains)
        ):
            from_block = transition_index
            to_block = transition_index + 1

            # Format gain with status indicator (heuristic thresholds)
            if gain_value is None:
                formatted_gain = "None"
                formatted_log = "None"
                status_indicator = ""
            elif gain_value == float('inf'):
                formatted_gain = "inf"
                formatted_log = "inf"
                status_indicator = " [VANISHING]"
            elif gain_value > 2.0:
                formatted_gain = f"{gain_value:.4f}"
                formatted_log = f"{log_gain:.4f}"
                status_indicator = " [AMPLIFYING]"
            elif gain_value < 0.5:
                formatted_gain = f"{gain_value:.4f}"
                formatted_log = f"{log_gain:.4f}"
                status_indicator = " [DAMPING]"
            else:
                formatted_gain = f"{gain_value:.4f}"
                formatted_log = f"{log_gain:.4f}"
                status_indicator = " [Healthy]"

            report_lines.append(
                f"  Transition {from_block:2d} to {to_block:2d}: "
                f"gamma={formatted_gain}, log_gamma={formatted_log}{status_indicator}"
            )

        # Add summary statistics
        report_lines.extend([
            "",
            "Summary Statistics:",
            f"  Mean gain: {stats['mean_gain']:.4f}" if stats['mean_gain'] is not None else "  Mean gain: N/A",
            f"  Min gain:  {stats['min_gain']:.4f}" if stats['min_gain'] is not None else "  Min gain: N/A",
            f"  Max gain:  {stats['max_gain']:.4f}" if stats['max_gain'] is not None else "  Max gain: N/A",
            f"  Mean log-gain: {stats['mean_log_gain']:.4f}" if stats['mean_log_gain'] is not None else "  Mean log-gain: N/A",
            "",
            "Transition Categories:",
            f"  Healthy (0.5 <= gamma <= 2.0): {stats['num_healthy']}",
            f"  Amplifying (gamma > 2.0):      {stats['num_amplifying']}",
            f"  Damping (gamma < 0.5):         {stats['num_damping']}",
            f"  Vanishing (gamma = inf):       {stats['num_vanishing']}",
            f"  Missing data:                  {stats['num_missing']}",
        ])

        return "\n".join(report_lines)

    def close(self):
        """Removes all forward hooks and cleans up resources.

        Call when done monitoring to prevent memory leaks.
        Automatically called when using context manager.
        """
        for hook_handle in self._forward_hook_handles:
            hook_handle.remove()
        self._forward_hook_handles.clear()

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Context manager exit. Ensures cleanup even on exception."""
        self.close()
        return False  # Do not suppress exceptions

    def __len__(self):
        """Returns the number of blocks being monitored."""
        return len(self.blocks)
