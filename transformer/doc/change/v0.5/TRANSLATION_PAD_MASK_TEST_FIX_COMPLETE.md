✅ Translation Padding Mask Test Fix (COMPLETE)
   - Status: Test bug fixed - test now correctly validates length-based masking
   - Date: 2026-02-16
   - Type: TEST BUG FIX (implementation was correct, test was wrong)

   Test Location:
   - File: src/test/test_training_pitfalls.py
   - Function: test_translation_source_pad_mask_does_not_mask_real_eos_when_pad_equals_eos
   - Lines: 99-134

   Problem with Test (BEFORE fix):
   - Test incorrectly expected ALL tokens with value eos_id to not be masked
   - This is impossible when pad_token_id == eos_token_id
   - Test assertion:
     ```python
     # Get all positions where token value == eos_id
     eos_positions = source_ids == eos_id

     # Expect none of them to be masked (WRONG!)
     assert torch.all(source_pad_mask[eos_positions] == 0)
     ```
   - This fails because artificial padding tokens also have value eos_id

   Why Test Was Wrong:
   - When pad_token_id == eos_token_id, padding tokens have the same value as real EOS
   - Distinguishing them by token value alone is impossible
   - The correct approach is length-based masking (what the implementation does)
   - Test was checking for an impossible requirement

   Test Scenario:
   ```python
   # Input sequences with different lengths:
   batch[0]: source_ids = [1, 2, 0]  # Length 3, last token (0) is real EOS
   batch[1]: source_ids = [5, 0]     # Length 2, last token (0) is real EOS

   # After padding to max length (3):
   source_ids = [[1, 2, 0],    # All 3 tokens are real
                 [5, 0, 0]]    # First 2 tokens real, last 0 is PADDING

   # Implementation creates mask based on sequence lengths:
   source_lengths = [3, 2]

   mask = [[False, False, False],  # seq0: positions 0,1,2 < length 3 → not masked
           [False, False, True]]   # seq1: positions 0,1 < length 2 → not masked
                                   #       position 2 >= length 2 → MASKED
   ```

   The Issue:
   - Position [0, 2]: token=0 (real EOS), mask=False ✓ Correct
   - Position [1, 1]: token=0 (real EOS), mask=False ✓ Correct
   - Position [1, 2]: token=0 (PADDING), mask=True ✓ Correct

   - Old test expected position [1, 2] to have mask=False (WRONG!)
   - This position IS artificial padding, so it SHOULD be masked
   - Test was validating incorrect behavior

   Solution (Test Fix):
   - Changed test to validate the actual correct behavior: length-based masking
   - New test assertion:
     ```python
     # Expected mask based on sequence lengths (not token values)
     expected_mask = torch.tensor(
         [[False, False, False],  # seq0 length 3 → no padding
          [False, False, True]],  # seq1 length 2 → position 2 is padding
         dtype=torch.bool,
     )

     assert torch.equal(source_pad_mask, expected_mask)
     ```

   - Added clear documentation explaining why this is correct:
     ```python
     """Source padding mask should only hide artificial padding.

     When pad_token_id == eos_token_id, the only reliable way to identify padding
     is by sequence length (positions beyond the original length). This test
     asserts that real tokens (including EOS) are not masked, and only the
     padded positions are masked.
     """
     ```

   Why This Is The Correct Behavior:
   - Real EOS tokens (within sequence length) are preserved ✓
   - Artificial padding (beyond sequence length) is masked ✓
   - Works correctly regardless of pad_token_id == eos_token_id ✓
   - Implementation in translation_collate_fn uses this exact approach ✓

   Implementation Was Already Correct:
   - File: src/training/loader_translation.py
   - Function: translation_collate_fn() (lines 76-156)
   - Key implementation (lines 136-151):
     ```python
     # Get original lengths of each sequence
     source_lengths = torch.tensor([len(seq) for seq in source_seqs])
     target_lengths = torch.tensor([len(seq) for seq in target_seqs])

     # Create position indices
     source_positions = torch.arange(source_padded.size(1)).unsqueeze(0).expand(...)
     target_positions = torch.arange(target_padded.size(1)).unsqueeze(0).expand(...)

     # Create masks: True where position >= actual length (padding only)
     source_pad_mask = source_positions >= source_lengths.unsqueeze(1)
     target_pad_mask = target_positions >= target_lengths.unsqueeze(1)
     ```

   - This was documented in doc/change/v0.5/FIX_COMPLETE.md:
     * "Padding Mask Creation Bug (FIXED)"
     * "Create masks based on sequence lengths, not token comparisons"
     * Lines 61-97

   Test Results:
   - BEFORE fix: test_translation_source_pad_mask_does_not_mask_real_eos_when_pad_equals_eos FAILED
   - AFTER fix: test_translation_source_pad_mask_does_not_mask_real_eos_when_pad_equals_eos PASSED ✓

   - All 11 tests now pass:
     * test_warmup_scheduler_steps_per_batch ✓
     * test_warmup_lr_ramps_over_steps ✓
     * test_epoch_scheduler_steps_per_epoch_not_per_batch ✓
     * test_warmup_completes_in_expected_steps ✓
     * test_lm_scheduler_steps_per_batch_when_enabled ✓
     * test_translation_collate_does_not_poison_decoder_input ✓
     * test_transformer_generate_restores_training_mode ✓
     * test_transformer_evaluate_restores_training_mode ✓
     * test_language_model_generate_restores_training_mode ✓
     * test_translation_padding_does_not_mask_real_eos_when_pad_equals_eos ✓
     * test_translation_source_pad_mask_does_not_mask_real_eos_when_pad_equals_eos ✓ (FIXED)

   Files Modified:
   - src/test/test_training_pitfalls.py:
     * Updated test_translation_source_pad_mask_does_not_mask_real_eos_when_pad_equals_eos
     * Added detailed docstring explaining length-based masking
     * Changed assertion to validate correct behavior (lines 99-134)

   Commits:
   - 9acacfa1 "Update bugs. Added test"
   - f2b3e9d3 "Add notes"
   - Date: 2026-02-16

   Related Documentation:
   - Padding mask implementation: doc/change/v0.5/FIX_COMPLETE.md (Padding Mask Creation Bug)
   - Collate function: src/training/loader_translation.py translation_collate_fn()
