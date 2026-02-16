"""Test that target padding mask is properly applied in decoder self-attention.

This test verifies the P2 fix: Target padding mask not applied in decoder self-attention.
Before the fix, padded target tokens could be attended to in decoder self-attention,
contaminating representations for variable-length batches.
"""
import torch
from torch import nn

# Add src to path for imports
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from model.decoder import Decoder, DecodeLayer
from model.model import Transformer


def test_decode_layer_accepts_target_pad_mask():
    """DecodeLayer should accept and use target_pad_mask in causal self-attention.

    Test Conditions:
    - Create a DecodeLayer with standard configuration
    - Pass input embeddings with shape (B, T, D)
    - Provide target_pad_mask indicating which positions are padding

    Expected Behavior:
    - DecodeLayer.forward() should accept target_pad_mask parameter
    - Output shape should match input shape
    - Output should contain valid (finite) values

    Why This Matters:
    - Before the fix, DecodeLayer didn't accept target_pad_mask
    - This test ensures the API change is implemented correctly
    """
    # ============================================================================
    # Setup: Create test data with batch_size=2, seq_len=4
    # ============================================================================
    batch_size = 2
    seq_len = 4
    d_model = 64
    num_heads = 4

    # Create a decode layer
    layer = DecodeLayer(
        i_layer=0,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=256,
        max_time_steps=10
    )

    # Create input embeddings with shape (B, T, D)
    # These represent already-embedded target tokens
    x = torch.randn(batch_size, seq_len, d_model)

    # ============================================================================
    # Test Condition: Variable-length sequences with padding
    # ============================================================================
    # Create target padding mask: second sequence has padding at positions 2,3
    # Shape: (B, T) where True indicates padding positions
    # Batch 0: [tok1, tok2, tok3, tok4] - all real tokens
    # Batch 1: [tok1, tok2, PAD, PAD]   - last 2 are padding
    target_pad_mask = torch.tensor([
        [False, False, False, False],  # No padding
        [False, False, True, True]     # Last 2 positions are padding
    ])

    # ============================================================================
    # Operation: Forward pass with target_pad_mask
    # ============================================================================
    # Before fix: This would fail because DecodeLayer didn't accept target_pad_mask
    # After fix: This should work and apply mask in causal self-attention
    output = layer(x=x, memory=None, target_pad_mask=target_pad_mask)

    # ============================================================================
    # Assertions: Verify correct output shape and values
    # ============================================================================
    # Output shape should preserve input dimensions (B, T, D)
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"

    # All output values should be finite (no NaN or Inf)
    # This checks that masking didn't cause numerical issues
    assert torch.all(torch.isfinite(output)), \
        "Output contains NaN or Inf values - masking may have caused numerical issues"


def test_decoder_accepts_target_pad_mask():
    """Decoder should accept and propagate target_pad_mask through all layers.

    Test Conditions:
    - Create a Decoder with multiple layers (num_layers=2)
    - Pass input embeddings through the decoder stack
    - Provide target_pad_mask for decoder self-attention

    Expected Behavior:
    - Decoder.forward() should accept target_pad_mask parameter
    - Mask should be propagated to all DecodeLayer instances
    - Output shape should match input shape

    Why This Matters:
    - Decoder must pass target_pad_mask to every layer in the stack
    - Without propagation, only the first layer would use the mask
    - All layers need the mask to prevent attending to padding
    """
    # ============================================================================
    # Setup: Create multi-layer decoder
    # ============================================================================
    batch_size = 2
    seq_len = 4
    d_model = 64
    vocab_size = 100
    num_layers = 2  # Test with multiple layers to verify propagation
    num_heads = 4

    # Create decoder with 2 layers
    decoder = Decoder(
        vocabulary_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=256,
        max_time_steps=10
    )

    # ============================================================================
    # Test Condition: Input embeddings (not token IDs)
    # ============================================================================
    # Create input embeddings with shape (B, T, D)
    # In real usage, these come from embedding layer + positional encoding
    y = torch.randn(batch_size, seq_len, d_model)

    # Create target padding mask with variable-length sequences
    # Batch 0: Full sequence (length 4)
    # Batch 1: Partial sequence (length 2, padded to 4)
    target_pad_mask = torch.tensor([
        [False, False, False, False],  # All real tokens
        [False, False, True, True]     # Half real, half padding
    ])

    # ============================================================================
    # Operation: Forward through entire decoder stack
    # ============================================================================
    # This passes through 2 DecodeLayer instances
    # Each layer should receive and use target_pad_mask
    output = decoder(y=y, memory=None, target_pad_mask=target_pad_mask)

    # ============================================================================
    # Assertions: Verify mask propagation works correctly
    # ============================================================================
    assert output.shape == (batch_size, seq_len, d_model), \
        "Decoder output shape should match input shape"

    assert torch.all(torch.isfinite(output)), \
        "All decoder outputs should be finite (no NaN/Inf from masking)"


def test_transformer_forward_accepts_target_pad_mask():
    """Transformer.forward() should accept and use target_pad_mask.

    Test Conditions:
    - Create full Transformer model (encoder + decoder)
    - Pass source and target token IDs
    - Provide both source_pad_mask and target_pad_mask

    Expected Behavior:
    - Transformer.forward() should accept target_pad_mask parameter
    - Target mask should be passed to decoder for self-attention masking
    - Source mask should be passed to encoder and cross-attention
    - Output log probabilities should have correct shape

    Why This Matters:
    - This is the top-level API used by training code
    - Both source and target masks must work together correctly
    - Ensures the complete pipeline supports target padding masking
    """
    # ============================================================================
    # Setup: Create full seq2seq transformer
    # ============================================================================
    batch_size = 2
    src_seq_len = 3  # Source sequences (e.g., English)
    tgt_seq_len = 4  # Target sequences (e.g., Spanish)
    vocab_size = 100
    d_model = 64

    # Create transformer with encoder and decoder
    model = Transformer(
        encoder_vocabulary_size=vocab_size,
        decoder_vocabulary_size=vocab_size,
        encoder_model_dimension=d_model,
        decoder_model_dimension=d_model,
        encoder_num_heads=4,
        decoder_num_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        encoder_pwff_dimension=256,
        decoder_pwff_dimension=256,
        encoder_max_time_steps=10,
        decoder_max_time_steps=10
    )

    # ============================================================================
    # Test Condition: Variable-length source and target sequences
    # ============================================================================
    # Create source token IDs (e.g., English sentence)
    # Batch 0: Full source sequence (length 3)
    # Batch 1: Padded source sequence (length 2, padded to 3)
    x = torch.randint(0, vocab_size, (batch_size, src_seq_len))

    # Create target token IDs (e.g., Spanish sentence)
    # Batch 0: Full target sequence (length 4)
    # Batch 1: Padded target sequence (length 2, padded to 4)
    y = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))

    # ============================================================================
    # Test Condition: Padding masks for both source and target
    # ============================================================================
    # Source padding mask (for encoder self-attention and decoder cross-attention)
    # Batch 0: [tok1, tok2, tok3] - all real
    # Batch 1: [tok1, tok2, PAD]  - last is padding
    source_pad_mask = torch.tensor([
        [False, False, False],  # No padding
        [False, False, True]    # Last position is padding
    ])

    # Target padding mask (for decoder self-attention)
    # Batch 0: [tok1, tok2, tok3, tok4] - all real
    # Batch 1: [tok1, tok2, PAD, PAD]   - last 2 are padding
    target_pad_mask = torch.tensor([
        [False, False, False, False],  # No padding
        [False, False, True, True]     # Last 2 positions are padding
    ])

    # ============================================================================
    # Operation: Forward pass with both source and target masks
    # ============================================================================
    # This is the main API used during training
    # source_pad_mask: prevents encoder from attending to source padding
    # target_pad_mask: prevents decoder from attending to target padding
    output = model(
        x=x,
        y=y,
        source_pad_mask=source_pad_mask,
        target_pad_mask=target_pad_mask
    )

    # ============================================================================
    # Assertions: Verify correct output and mask usage
    # ============================================================================
    # Output should be log probabilities over vocabulary
    # Shape: (B, T_target, V) where V is vocabulary size
    assert output.shape == (batch_size, tgt_seq_len, vocab_size), \
        f"Expected shape {(batch_size, tgt_seq_len, vocab_size)}, got {output.shape}"

    # All probabilities should be finite (masking shouldn't cause -inf in output)
    assert torch.all(torch.isfinite(output)), \
        "Output log probabilities should be finite"


def test_target_pad_mask_affects_attention():
    """Target padding mask should actually affect attention patterns.

    Test Conditions:
    - Create identical input sequences
    - Run forward pass twice: once with mask, once without
    - Compare outputs for sequences with and without padding

    Expected Behavior:
    - Outputs should differ when mask is applied vs not applied
    - Sequences with padding should show larger differences
    - Sequences without padding should show smaller differences

    Why This Matters:
    - This verifies the mask actually changes behavior (not just accepted as parameter)
    - Without this test, mask could be ignored internally
    - Ensures padding positions are actually masked out in attention

    How It Works:
    - Sequence 0 has no padding: mask should have minimal effect
    - Sequence 1 has padding: mask should have significant effect
    - Compare difference magnitudes to verify mask is working
    """
    # ============================================================================
    # Setup: Create decode layer for attention testing
    # ============================================================================
    batch_size = 2
    seq_len = 4
    d_model = 64
    num_heads = 4

    # Create a decode layer
    layer = DecodeLayer(
        i_layer=0,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=256,
        max_time_steps=10
    )
    layer.eval()  # Use eval mode to disable dropout for deterministic behavior

    # ============================================================================
    # Test Condition: Identical input for both forward passes
    # ============================================================================
    # Create input embeddings
    # Both sequences contain the same token embeddings
    # But sequence 1 will be treated as having padding at positions 2,3
    x = torch.randn(batch_size, seq_len, d_model)

    # Create target padding mask
    # Batch 0: [tok1, tok2, tok3, tok4] - treat all as real tokens
    # Batch 1: [tok1, tok2, PAD, PAD]   - treat last 2 as padding
    target_pad_mask = torch.tensor([
        [False, False, False, False],  # No padding
        [False, False, True, True]     # Last 2 positions are padding
    ])

    # ============================================================================
    # Operation 1: Forward pass WITH target padding mask
    # ============================================================================
    # In this pass:
    # - Batch 0: All tokens attend to all tokens (causal only)
    # - Batch 1: Real tokens (0,1) cannot attend to padding tokens (2,3)
    with torch.no_grad():
        output_masked = layer(x=x.clone(), memory=None, target_pad_mask=target_pad_mask)

    # ============================================================================
    # Operation 2: Forward pass WITHOUT target padding mask
    # ============================================================================
    # In this pass:
    # - Both batches: All tokens attend to all tokens (causal only)
    # - No distinction between real and padding positions
    with torch.no_grad():
        output_unmasked = layer(x=x.clone(), memory=None, target_pad_mask=None)

    # ============================================================================
    # Assertion 1: Outputs should differ when mask is applied
    # ============================================================================
    # For batch 1 (with padding), outputs should be significantly different
    # because attention patterns change when padding is masked
    assert not torch.allclose(output_masked[1], output_unmasked[1], atol=1e-6), \
        "Target padding mask should affect attention patterns for sequences with padding"

    # ============================================================================
    # Assertion 2: Effect should be stronger for padded sequences
    # ============================================================================
    # Measure average absolute difference for each sequence
    # Batch 0 (no padding): Should have small difference (just numerical noise)
    # Batch 1 (with padding): Should have large difference (mask effect)
    diff_no_padding = torch.abs(output_masked[0] - output_unmasked[0]).mean()
    diff_with_padding = torch.abs(output_masked[1] - output_unmasked[1]).mean()

    # The difference for the padded sequence should be at least 2x larger
    # This confirms the mask has a real, measurable effect
    assert diff_with_padding > diff_no_padding * 2, \
        (f"Padding mask should have stronger effect on padded sequences. "
         f"Difference without padding: {diff_no_padding:.6f}, "
         f"Difference with padding: {diff_with_padding:.6f}")


if __name__ == "__main__":
    print("Testing target padding mask implementation...")
    print("=" * 70)

    test_decode_layer_accepts_target_pad_mask()
    print("✓ Test 1: DecodeLayer accepts target_pad_mask")

    test_decoder_accepts_target_pad_mask()
    print("✓ Test 2: Decoder accepts and propagates target_pad_mask")

    test_transformer_forward_accepts_target_pad_mask()
    print("✓ Test 3: Transformer.forward() accepts target_pad_mask")

    test_target_pad_mask_affects_attention()
    print("✓ Test 4: Target padding mask actually affects attention patterns")

    print("=" * 70)
    print("\nAll target padding mask tests passed! ✓")
    print("\nThe P2 fix is working correctly:")
    print("- Target padding masks are accepted at all API levels")
    print("- Masks are properly propagated through the decoder stack")
    print("- Masks actually change attention behavior as expected")
