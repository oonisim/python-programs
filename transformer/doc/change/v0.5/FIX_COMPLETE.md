
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

