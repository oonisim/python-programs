"""Examples demonstrating GradientGainMonitor usage.

This script shows various use cases for monitoring gradient flow in neural networks.
Run with: python example_gradient_monitor.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from gradient_monitor import GradientGainMonitor


# =============================================================================
# Example Models
# =============================================================================

class TransformerBlock(nn.Module):
    """Simplified transformer block with Pre-LN architecture."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Pre-LN: norm before attention
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        # Pre-LN: norm before MLP
        x = x + self.mlp(self.ln2(x))
        return x


class ResidualBlock(nn.Module):
    """Simple residual block."""
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.layers(x)


# =============================================================================
# Example 1: Basic Usage
# =============================================================================

def example_1_basic_usage():
    """Basic usage: monitoring a simple model."""
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)

    # Create a simple model with 4 residual blocks
    torch.manual_seed(42)
    dim = 64
    blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(4)])
    model = nn.Sequential(*blocks)

    # Create some dummy data
    batch_size = 8
    x = torch.randn(batch_size, dim)
    target = torch.randn(batch_size, dim)

    # Monitor gradient flow
    with GradientGainMonitor(blocks) as monitor:
        # Forward pass
        output = model(x)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        loss.backward()

        # Print report
        print(monitor.report())
        print()


# =============================================================================
# Example 2: Comparing Different Architectures
# =============================================================================

def example_2_comparing_architectures():
    """Compare gradient flow in different architectures."""
    print("=" * 70)
    print("Example 2: Comparing Pre-LN vs Post-LN Transformer")
    print("=" * 70)

    torch.manual_seed(42)
    dim = 64
    seq_len = 16

    # Create Pre-LN transformer
    pre_ln_blocks = nn.ModuleList([TransformerBlock(dim) for _ in range(4)])

    # Create data
    x = torch.randn(4, seq_len, dim)
    target = torch.randn(4, seq_len, dim)

    print("\nPre-LN Transformer Gradient Flow:")
    print("-" * 70)
    with GradientGainMonitor(pre_ln_blocks) as monitor:
        out = x
        for block in pre_ln_blocks:
            out = block(out)

        loss = nn.MSELoss()(out, target)
        loss.backward()

        print(monitor.report())
        print()

        # Get summary
        stats = monitor.summary_stats()
        print(f"Pre-LN Mean Gain: {stats['mean_gain']:.4f}")
        print(f"Pre-LN Healthy Transitions: {stats['num_healthy']}/{len(monitor)-1}")
        print()


# =============================================================================
# Example 3: Different Norm Types
# =============================================================================

def example_3_different_norm_types():
    """Compare different norm types."""
    print("=" * 70)
    print("Example 3: Different Norm Types")
    print("=" * 70)

    torch.manual_seed(42)
    dim = 64
    blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(3)])

    x = torch.randn(4, dim)
    target = torch.randn(4, dim)

    norm_types = ['l2', 'l1', 'linf', 'mean']

    for norm_type in norm_types:
        print(f"\nUsing {norm_type} norm:")
        print("-" * 70)

        with GradientGainMonitor(blocks, norm_type=norm_type) as monitor:
            model = nn.Sequential(*blocks)
            output = model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()

            norms = monitor.norms()
            print(f"Norms: {[f'{n:.4e}' for n in norms]}")

            gains = monitor.gains()
            print(f"Gains: {[f'{g:.4f}' for g in gains]}")
            print()


# =============================================================================
# Example 4: Training Loop with Monitoring
# =============================================================================

def example_4_training_loop():
    """Monitor gradient flow during training."""
    print("=" * 70)
    print("Example 4: Monitoring During Training")
    print("=" * 70)

    torch.manual_seed(42)
    dim = 32
    blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(4)])
    model = nn.Sequential(*blocks)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate some dummy data
    train_data = [(torch.randn(8, dim), torch.randn(8, dim)) for _ in range(10)]

    print("\nTraining with gradient monitoring every 3 steps:")
    print("-" * 70)

    with GradientGainMonitor(blocks, strict_single_backward=False) as monitor:
        for step, (x, target) in enumerate(train_data):
            optimizer.zero_grad()

            output = model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()

            # Monitor gradient flow every 3 steps
            if step % 3 == 0:
                stats = monitor.summary_stats()
                print(f"\nStep {step}:")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Mean gain: {stats['mean_gain']:.4f}")
                print(f"  Healthy: {stats['num_healthy']}, "
                      f"Amplifying: {stats['num_amplifying']}, "
                      f"Damping: {stats['num_damping']}")

                # Check for problematic gradient flow
                if stats['num_vanishing'] > 0:
                    print(f"  WARNING: {stats['num_vanishing']} vanishing transitions detected!")
                if stats['num_amplifying'] > 2:
                    print(f"  WARNING: Many amplifying transitions - check for exploding gradients")

            optimizer.step()
            monitor.reset()  # Reset for next iteration

    print("\n")


# =============================================================================
# Example 5: Gradient Accumulation
# =============================================================================

def example_5_gradient_accumulation():
    """Monitor gradient flow with gradient accumulation."""
    print("=" * 70)
    print("Example 5: Gradient Accumulation")
    print("=" * 70)

    torch.manual_seed(42)
    dim = 32
    blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(3)])
    model = nn.Sequential(*blocks)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simulate gradient accumulation over 4 micro-batches
    accumulation_steps = 4
    micro_batches = [(torch.randn(2, dim), torch.randn(2, dim))
                     for _ in range(accumulation_steps)]

    print(f"\nAccumulating gradients over {accumulation_steps} micro-batches:")
    print("-" * 70)

    optimizer.zero_grad()

    with GradientGainMonitor(blocks, strict_single_backward=False) as monitor:
        for i, (x, target) in enumerate(micro_batches):
            output = model(x)
            loss = nn.MSELoss()(output, target) / accumulation_steps
            loss.backward()

            # Monitor each micro-batch
            stats = monitor.summary_stats()
            print(f"\nMicro-batch {i+1}:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Mean gain: {stats['mean_gain']:.4f}")

            monitor.reset()  # Reset between accumulation steps

        optimizer.step()

    print("\n")


# =============================================================================
# Example 6: Detecting Gradient Problems
# =============================================================================

def example_6_detecting_problems():
    """Demonstrate detection of gradient flow problems."""
    print("=" * 70)
    print("Example 6: Detecting Gradient Flow Problems")
    print("=" * 70)

    torch.manual_seed(42)
    dim = 64

    # Create a "problematic" deep network (no residuals)
    class DeepBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Tanh(),  # Tanh can cause vanishing gradients in deep nets
                nn.Linear(dim, dim),
                nn.Tanh()
            )

        def forward(self, x):
            return self.layers(x)  # No residual connection!

    print("\n1. Deep network WITHOUT residual connections:")
    print("-" * 70)

    deep_blocks = nn.ModuleList([DeepBlock(dim) for _ in range(8)])

    with GradientGainMonitor(deep_blocks) as monitor:
        x = torch.randn(4, dim)
        target = torch.randn(4, dim)

        out = x
        for block in deep_blocks:
            out = block(out)

        loss = nn.MSELoss()(out, target)
        loss.backward()

        stats = monitor.summary_stats()
        print(f"Mean gain: {stats['mean_gain']:.4f}")
        print(f"Min gain: {stats['min_gain']:.4f}")
        print(f"Max gain: {stats['max_gain']:.4f}")
        print(f"Damping transitions: {stats['num_damping']}")

        if stats['num_damping'] > len(deep_blocks) // 2:
            print("\n⚠️  WARNING: Many damping transitions detected!")
            print("   This suggests gradient vanishing. Consider:")
            print("   - Adding residual connections")
            print("   - Using Pre-LN architecture")
            print("   - Reducing network depth")
            print("   - Adjusting initialization")

    print("\n2. Same network WITH residual connections:")
    print("-" * 70)

    residual_blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(8)])

    with GradientGainMonitor(residual_blocks) as monitor:
        out = x
        for block in residual_blocks:
            out = block(out)

        loss = nn.MSELoss()(out, target)
        loss.backward()

        stats = monitor.summary_stats()
        print(f"Mean gain: {stats['mean_gain']:.4f}")
        print(f"Min gain: {stats['min_gain']:.4f}")
        print(f"Max gain: {stats['max_gain']:.4f}")
        print(f"Healthy transitions: {stats['num_healthy']}")

        if stats['num_healthy'] > len(residual_blocks) // 2:
            print("\n✓ Healthy gradient flow detected!")
            print("  Residual connections are helping maintain gradient magnitude.")

    print()


# =============================================================================
# Example 7: Log-Gains for Visualization
# =============================================================================

def example_7_log_gains():
    """Use log-gains for more stable analysis."""
    print("=" * 70)
    print("Example 7: Using Log-Gains")
    print("=" * 70)

    torch.manual_seed(42)
    dim = 64
    blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(5)])

    x = torch.randn(4, dim)
    target = torch.randn(4, dim)

    with GradientGainMonitor(blocks) as monitor:
        model = nn.Sequential(*blocks)
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        gains = monitor.gains()
        log_gains = monitor.log_gains()

        print("\nGradient Gains (raw and log-space):")
        print("-" * 70)
        print(f"{'Transition':<15} {'Raw Gain':<15} {'Log Gain':<15} {'Interpretation'}")
        print("-" * 70)

        for i, (gain, log_gain) in enumerate(zip(gains, log_gains)):
            if log_gain is not None and log_gain != float('inf'):
                if log_gain > 0.1:
                    interp = "Amplifying"
                elif log_gain < -0.1:
                    interp = "Damping"
                else:
                    interp = "Preserved"

                print(f"{i} -> {i+1:<10} {gain:<15.4f} {log_gain:<15.4f} {interp}")

        print("\nNote: Log-gains are more stable for analysis and visualization.")
        print("      log_gain ≈ 0 means gradient is well-preserved.")
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "    GradientGainMonitor Examples".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    examples = [
        example_1_basic_usage,
        example_2_comparing_architectures,
        example_3_different_norm_types,
        example_4_training_loop,
        example_5_gradient_accumulation,
        example_6_detecting_problems,
        example_7_log_gains,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠️  Example {i} failed with error: {e}\n")

        if i < len(examples):
            input("Press Enter to continue to next example...")
            print("\n")

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
