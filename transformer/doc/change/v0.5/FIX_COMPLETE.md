
✅ EOS Learning Fix (COMPLETE)
   - Changed ignore_index=pad_token_id → ignore_index=LABEL_IGNORE_VALUE (-100)
   - Implemented masking logic in translation_collate_fn() to replace padded
     positions with LABEL_IGNORE_VALUE
   - Model can now properly learn to generate EOS tokens
   - Files: train_lm.py, train_translation.py, loader_translation.py, constant.py

✅ Translation BOS Token (COMPLETE)
   - Added BOS token (EOS as START) to target sequences
   - Model now learns to predict first target token
   - File: loader_translation.py

✅ run_config.json Path (COMPLETE)
   - Fixed to save in result/{model_name}/snapshots/ matching trainer location
   - Resume no longer silently misses config
   - Files: train_lm.py, train_translation.py

✅ eval() Mode Missing (COMPLETE)
   - Added self.eval() to evaluate() and generate() methods
   - Disables dropout for deterministic evaluation/generation
   - Files: model.py, lm.py

✅ val_loss or train_loss bug (FIXED)
   - Bug: Used 'val_loss or train_loss' which fails when val_loss=0.0
   - Fix: Changed to 'val_loss if val_loss is not None else train_loss'
   - File: trainer.py line 605

✅ Attention Padding Mask (IMPLEMENTED)
   - Status: Fully implemented and tested

   Implementation:
   1. translation_collate_fn() creates and returns source_pad_mask
      File: loader_translation.py

   2. ScaledDotProductAttention.forward() accepts padding_mask parameter
      Signature: forward(q, k, v, padding_mask=None, return_similarities=False)
      File: common.py lines 433-540

   3. Mask applied before softmax in attention computation
      Code: similarities.masked_fill(padding_mask, float('-inf'))
      File: common.py line 529

   4. Mask propagated through all attention layers:
      - MultiHeadAttention.forward() accepts and passes padding_mask
      - EncodeLayer.forward() accepts source_pad_mask for self-attention
      - DecodeLayer.forward() accepts source_pad_mask for cross-attention
      - Encoder.forward() and Decoder.forward() accept and propagate mask
      - Transformer.forward() accepts source_pad_mask parameter
      - Trainer._train_one_step() and _validate() extract and pass mask

   Benefits:
   - Padded tokens no longer attended to in variable-length batches
   - Attention mass not wasted on padding positions
   - Real token representations no longer contaminated by padding
   - Same sentence produces consistent attention regardless of padding length

   Tests: All model tests pass (4/4)
   Files: loader_translation.py, common.py, encoder.py, decoder.py, model.py, trainer.py

✅ Padding Mask Creation Bug (FIXED)
   - Status: Critical bug fixed

   Problem:
   - OLD: Created masks by comparing token IDs: source_pad_mask = (source_padded == pad_token_id)
   - When pad_token_id == eos_token_id (true for tiktoken/GPT-2), this masked ALL EOS tokens
   - Bug #1: All EOS/BOS tokens in target were masked → EOS/BOS unlearnable
   - Bug #2: All EOS tokens in source were masked → Encoder didn't see sequence boundaries

   Solution:
   - NEW: Create masks based on actual sequence lengths, not token comparisons
   - Track original lengths and mark positions BEYOND length as padding
   - Only artificial padding is masked, real EOS/BOS tokens preserved

   Implementation (loader_translation.py lines 120-149):
   ```python
   # Get original sequence lengths
   source_lengths = torch.tensor([len(seq) for seq in source_seqs])
   target_lengths = torch.tensor([len(seq) for seq in target_seqs])

   # Create position indices
   source_positions = torch.arange(source_padded.size(1)).unsqueeze(0).expand(...)
   target_positions = torch.arange(target_padded.size(1)).unsqueeze(0).expand(...)

   # Mask positions >= actual length (artificial padding only)
   source_pad_mask = source_positions >= source_lengths.unsqueeze(1)
   target_pad_mask = target_positions >= target_lengths.unsqueeze(1)
   ```

   Benefits:
   - Real EOS tokens preserved in both source and target
   - BOS tokens (set to EOS) preserved in target
   - Works correctly regardless of pad_token_id == eos_token_id
   - Model can now learn sequence boundaries

   Tests: All model tests pass (4/4)
   File: loader_translation.py

✅ Decoder Input Corruption (FIXED)
   - Problem: Premature masking in collate caused decoder_input to contain LABEL_IGNORE_VALUE (-100)
   - Solution: Moved masking to trainer AFTER shifting sequences
   - decoder_input gets clean token IDs (can be embedded)
   - decoder_target gets masked PAD positions (loss ignores them)
   - Files: loader_translation.py, trainer.py

✅ Train Mode Preservation (FIXED)
   - Problem: generate() called self.eval() but never restored training mode
   - Solution: Save/restore training state with try-finally block
   - Files: model.py (Transformer.generate), lm.py (LanguageModel.generate)

✅ Option A Implementation (COMPLETE)
   - Implemented 45-50M parameter model configuration for WikiText-103
   - Model presets: tiny (~16M), small (~45-50M), medium (~117M)
   - Learning rate warmup scheduler (1000 steps default)
   - Fixed d_ff ratio: 2048 (4× d_model) instead of 512
   - Increased max_seq_len: 512 instead of 256
   - Updated training script with preset support
   - Expected perplexity: ~50-60 on WikiText-103
   - Files: train_lm.py, run_train_lm.sh, doc/note/option_a_training.md

✅ Source Padding Mask in generate() (FIXED)
   - Problem: generate() had no source_pad_mask parameter
   - Without it, encoder and cross-attention attended to padding tokens
   - Could corrupt outputs when generating from padded source batches
   - Solution: Added source_pad_mask parameter (Optional)
   - Propagated mask to encoder and decoder for proper masking
   - File: model.py (generate method)

✅ evaluate() Training Mode Restoration (FIXED)
   - Problem: evaluate() called self.eval() without restoring training mode
   - If called during training, dropout stayed disabled afterward
   - Solution: Save/restore training state with try-finally block
   - Same pattern as generate() method
   - File: model.py (evaluate method)

✅ Minor Fixes (COMPLETE)
   - Fixed missing space in error message (model.py:603-606)
   - Fixed invalid pylint directive: added missing colon (lm.py:367)
   - Fixed off-by-one in get_stats(): removed extra -1 (loader.py:255,259)
   - All model tests pass (4/4)

✅ Warmup Scheduler Test Coverage (FIXED)
   - Problem: Tests duplicated scheduler.step() logic instead of calling production code
   - Risk: Warmup scheduler could break without test detection (high regression risk)
   - Solution: Extracted _step_scheduler_if_configured() method for tests to call
   - Tests now exercise real production per-batch stepping logic
   - Improved assertions: calculated expected LR (77.5%) instead of magic number (50%)
   - Added detailed comments explaining two-phase schedule (linear warmup + cosine decay)
   - File: trainer.py, test_scheduler_stepping.py
   - See: doc/change/v0.5/WARMUP_SCHEDULER_TEST_COVERAGE_COMPLETE.md
