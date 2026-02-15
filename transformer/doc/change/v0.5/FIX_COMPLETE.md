
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
