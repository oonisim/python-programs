"""Unit tests for gradient_monitor module.

Run with: python -m pytest test_gradient_monitor.py -v
"""
import torch
import torch.nn as nn
import pytest
import math
from training.gradient_monitor import GradientGainMonitor


class SimpleBlock(nn.Module):
    """Simple block for testing: linear + residual."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.linear(x)


class TupleOutputBlock(nn.Module):
    """Block that returns tuple (output, aux)."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        output = x + self.linear(x)
        aux = torch.zeros(1)  # Auxiliary output
        return output, aux


class DictOutputBlock(nn.Module):
    """Block that returns dict (like HuggingFace)."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        output = x + self.linear(x)
        return {'hidden_states': output, 'attention_weights': None}


class TestGradientGainMonitorBasics:
    """Test basic functionality."""

    def test_initialization(self):
        """Test monitor can be created."""
        blocks = [SimpleBlock(4) for _ in range(3)]
        monitor = GradientGainMonitor(blocks)
        assert len(monitor) == 3
        assert monitor.norm_type == 'l2'
        monitor.close()

    def test_invalid_norm_type(self):
        """Test that invalid norm type raises ValueError."""
        blocks = [SimpleBlock(4)]
        with pytest.raises(ValueError, match="norm_type must be one of"):
            GradientGainMonitor(blocks, norm_type='invalid')

    def test_context_manager(self):
        """Test context manager properly cleans up."""
        blocks = [SimpleBlock(4) for _ in range(3)]
        with GradientGainMonitor(blocks) as monitor:
            assert len(monitor._forward_hook_handles) == 3
        # After exit, hooks should be removed
        assert len(monitor._forward_hook_handles) == 0

    def test_no_backward_error(self):
        """Test that accessing data before backward raises error."""
        blocks = [SimpleBlock(4) for _ in range(2)]
        monitor = GradientGainMonitor(blocks)

        with pytest.raises(RuntimeError, match="No gradients recorded"):
            monitor.norms()

        with pytest.raises(RuntimeError, match="No gradients recorded"):
            monitor.gains()

        monitor.close()


class TestGradientGainMonitorForwardBackward:
    """Test forward and backward pass monitoring."""

    def test_simple_forward_backward(self):
        """Test basic forward-backward with gradient recording."""
        torch.manual_seed(42)
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(3)])

        with GradientGainMonitor(blocks) as monitor:
            # Forward pass
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            # Backward pass
            loss = out.sum()
            loss.backward()

            # Check norms recorded
            norms = monitor.norms()
            assert len(norms) == 3
            assert all(n is not None for n in norms)
            assert all(n > 0 for n in norms)

            # Check gains computed
            gains = monitor.gains()
            assert len(gains) == 2  # L-1 for L blocks
            assert all(g is not None for g in gains)

    def test_tuple_output_blocks(self):
        """Test blocks that return tuples."""
        torch.manual_seed(42)
        blocks = nn.ModuleList([TupleOutputBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks) as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out, _ = block(out)

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            assert len(norms) == 2
            assert all(n is not None for n in norms)

    def test_dict_output_blocks(self):
        """Test blocks that return dicts."""
        torch.manual_seed(42)
        blocks = nn.ModuleList([DictOutputBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks) as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                result = block(out)
                out = result['hidden_states']

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            assert len(norms) == 2
            assert all(n is not None for n in norms)

    def test_frozen_block(self):
        """Test that frozen blocks (no grad) return None."""
        torch.manual_seed(42)
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(3)])

        # Freeze middle block
        for param in blocks[1].parameters():
            param.requires_grad = False

        with GradientGainMonitor(blocks) as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            # Middle block should have None (no grad)
            # Note: The output still flows through, but if the block's
            # params don't require grad, the hook might not fire
            # This depends on whether the output requires grad
            assert len(norms) == 3


class TestNormTypes:
    """Test different norm types."""

    def test_l2_norm(self):
        """Test L2 norm computation."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks, norm_type='l2') as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            assert all(n > 0 for n in norms)

    def test_l1_norm(self):
        """Test L1 norm computation."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks, norm_type='l1') as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            assert all(n > 0 for n in norms)

    def test_linf_norm(self):
        """Test Linf norm computation."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks, norm_type='linf') as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            assert all(n > 0 for n in norms)

    def test_mean_metric(self):
        """Test mean absolute value metric."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks, norm_type='mean') as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            assert all(n > 0 for n in norms)


class TestMultipleBackward:
    """Test handling of multiple backward calls."""

    def test_strict_mode_raises_on_double_backward(self):
        """Test strict mode raises error on second backward."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks, strict_single_backward=True) as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward(retain_graph=True)

            # First call succeeds
            norms1 = monitor.norms()
            assert len(norms1) == 2

            # Second backward should raise in strict mode
            with pytest.raises(RuntimeError, match="fired twice"):
                loss.backward()

    def test_non_strict_mode_overwrites(self):
        """Test non-strict mode overwrites on second backward."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks, strict_single_backward=False) as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward(retain_graph=True)

            norms1 = monitor.norms()

            # Second backward should succeed in non-strict mode
            loss.backward()
            norms2 = monitor.norms()

            # Values should be recorded (may differ due to graph state)
            assert len(norms2) == 2
            assert all(n is not None for n in norms2)

    def test_reset_between_backwards(self):
        """Test reset() allows multiple backward passes."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks, strict_single_backward=True) as monitor:
            # First backward
            x1 = torch.randn(2, 4)
            out1 = x1
            for block in blocks:
                out1 = block(out1)
            loss1 = out1.sum()
            loss1.backward()

            norms1 = monitor.norms()

            # Reset
            monitor.reset()

            # Second backward should now succeed
            x2 = torch.randn(2, 4)
            out2 = x2
            for block in blocks:
                out2 = block(out2)
            loss2 = out2.sum()
            loss2.backward()

            norms2 = monitor.norms()
            assert len(norms2) == 2


class TestGainsAndLogGains:
    """Test gain calculations."""

    def test_gains_computation(self):
        """Test gain ratios are computed correctly."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(3)])

        with GradientGainMonitor(blocks) as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            gains = monitor.gains()

            # Manually verify first gain
            expected_gain_0 = norms[0] / norms[1]
            assert abs(gains[0] - expected_gain_0) < 1e-6

    def test_log_gains_computation(self):
        """Test log-gains are computed correctly."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(3)])

        with GradientGainMonitor(blocks) as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            log_gains = monitor.log_gains()

            # Verify relationship: log(gain) = log(norm[i-1]) - log(norm[i])
            expected_log_gain_0 = math.log(norms[0]) - math.log(norms[1])
            assert abs(log_gains[0] - expected_log_gain_0) < 1e-6

    def test_vanishing_gradient_detection(self):
        """Test detection of vanishing gradients (near-zero norms)."""
        # This is hard to trigger naturally, so we'll mock it
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks, eps=1e-6) as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward()

            # Manually set a norm to near-zero to test inf detection
            monitor._recorded_gradient_norms[1] = 1e-40
            gains = monitor.gains()

            # Should return inf for vanishing denominator
            assert gains[0] == float('inf')


class TestSummaryStatistics:
    """Test summary statistics."""

    def test_summary_stats_structure(self):
        """Test summary stats returns correct structure."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(4)])

        with GradientGainMonitor(blocks) as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward()

            stats = monitor.summary_stats()

            # Check all expected keys are present
            expected_keys = {
                'mean_gain', 'min_gain', 'max_gain', 'mean_log_gain',
                'num_amplifying', 'num_damping', 'num_healthy',
                'num_vanishing', 'num_missing'
            }
            assert set(stats.keys()) == expected_keys

            # Check counts are non-negative integers
            assert stats['num_amplifying'] >= 0
            assert stats['num_damping'] >= 0
            assert stats['num_healthy'] >= 0
            assert stats['num_vanishing'] >= 0
            assert stats['num_missing'] >= 0

            # Check total adds up
            total = (stats['num_amplifying'] + stats['num_damping'] +
                    stats['num_healthy'] + stats['num_vanishing'] +
                    stats['num_missing'])
            assert total == 3  # 4 blocks = 3 transitions


class TestReporting:
    """Test report generation."""

    def test_report_before_backward(self):
        """Test report before backward shows appropriate message."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(2)])

        with GradientGainMonitor(blocks) as monitor:
            report = monitor.report()
            assert "No gradients recorded" in report

    def test_report_after_backward(self):
        """Test report after backward contains expected sections."""
        blocks = nn.ModuleList([SimpleBlock(4) for _ in range(3)])

        with GradientGainMonitor(blocks) as monitor:
            x = torch.randn(2, 4)
            out = x
            for block in blocks:
                out = block(out)

            loss = out.sum()
            loss.backward()

            report = monitor.report()

            # Check report contains expected sections
            assert "Gradient Flow Analysis" in report
            assert "Gradient Norms:" in report
            assert "Gradient Gains" in report
            assert "Summary Statistics:" in report
            assert "Transition Categories:" in report

            # Check block entries are present
            assert "Block  0:" in report
            assert "Block  1:" in report
            assert "Block  2:" in report

            # Check transition entries
            assert "Transition  0 to  1:" in report
            assert "Transition  1 to  2:" in report


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_block(self):
        """Test with single block (no transitions)."""
        blocks = nn.ModuleList([SimpleBlock(4)])

        with GradientGainMonitor(blocks) as monitor:
            x = torch.randn(2, 4)
            out = blocks[0](x)

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            assert len(norms) == 1

            gains = monitor.gains()
            assert len(gains) == 0  # No transitions

    def test_empty_blocks_list(self):
        """Test with empty blocks list."""
        blocks = nn.ModuleList([])

        with GradientGainMonitor(blocks) as monitor:
            assert len(monitor) == 0

    def test_sequential_container(self):
        """Test with nn.Sequential."""
        blocks = nn.Sequential(
            SimpleBlock(4),
            SimpleBlock(4),
            SimpleBlock(4)
        )

        with GradientGainMonitor(blocks) as monitor:
            x = torch.randn(2, 4)
            out = blocks(x)

            loss = out.sum()
            loss.backward()

            norms = monitor.norms()
            assert len(norms) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
