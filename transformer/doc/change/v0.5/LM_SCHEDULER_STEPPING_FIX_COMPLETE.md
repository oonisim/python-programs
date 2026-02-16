✅ LanguageModelTrainer Scheduler Stepping Fix (COMPLETE)
   - Status: Critical implementation bug fixed - warmup scheduler now works for language model training
   - Date: 2026-02-16
   - Type: IMPLEMENTATION BUG FIX

   Bug Location:
   - File: src/training/trainer.py
   - Class: LanguageModelTrainer
   - Method: _train_one_step() (lines 1141-1194)
   - Issue: Missing scheduler stepping call after optimizer.step()

   Problem:
   - LanguageModelTrainer._train_one_step() did not step the scheduler per batch
   - The base Trainer class correctly calls _step_scheduler_if_configured() after optimizer.step()
   - LanguageModelTrainer overrode _train_one_step() but forgot to include scheduler stepping
   - Impact: Warmup schedules did not work at all for language model training
   - Severity: HIGH - warmup is critical for training stability with large learning rates
   - Affected: Language model training (train_lm.py) but NOT translation training

   Root Cause:
   - When LanguageModelTrainer was created, it overrode the base Trainer._train_one_step() method
   - The override implemented the LM-specific forward pass (single input, no encoder/decoder split)
   - But it missed copying the scheduler stepping logic from the base class
   - Result: config.step_scheduler_per_batch=True was ignored for LM training

   Why This Matters:
   - Warmup schedules are defined in STEPS, not epochs
   - Without per-batch stepping, LR stays near zero for entire first epoch
   - Example: 1000-step warmup with 14,394 steps/epoch
     * Expected: LR ramps from 0 → peak over first 1000 batches (6.9% of epoch 1)
     * Actual (broken): LR stays at ~0 for entire first epoch
   - Training diverges or fails to converge when LR is too low initially
   - Warmup is especially critical for large models and high learning rates

   How Bug Was Detected:
   - Test: test_lm_scheduler_steps_per_batch_when_enabled()
   - File: src/test/test_training_regressions.py (lines 31-71)
   - Test created a dummy scheduler that counts step() calls
   - After calling LanguageModelTrainer._train_one_step(), scheduler.step_calls == 0
   - Expected: scheduler.step_calls == 1
   - Test assertion failed, exposing the missing scheduler stepping

   Evidence:
   - Base Trainer._train_one_step() at line 605:
     ```python
     self.optimizer.step()

     # Step scheduler per batch if configured (e.g., warmup schedules)
     self._step_scheduler_if_configured()

     # Callback: on_step_end
     self.callbacks.on_step_end(self)
     ```

   - LanguageModelTrainer._train_one_step() BEFORE fix (lines 1188-1192):
     ```python
     self.optimizer.step()

     # Callback: on_step_end                    ← Missing scheduler stepping!
     self.callbacks.on_step_end(self)

     return loss.item()
     ```

   Solution (Implementation Fix):
   - Added _step_scheduler_if_configured() call after optimizer.step()
   - Location: src/training/trainer.py lines 1191-1192

   - LanguageModelTrainer._train_one_step() AFTER fix (lines 1188-1195):
     ```python
     self.optimizer.step()

     # Step scheduler per batch if configured (e.g., warmup schedules)
     self._step_scheduler_if_configured()          ← FIXED: Added this line

     # Callback: on_step_end
     self.callbacks.on_step_end(self)

     return loss.item()
     ```

   Why _step_scheduler_if_configured() Instead of Direct scheduler.step():
   - Encapsulates the conditional logic: only step if scheduler exists AND per-batch mode enabled
   - Matches base Trainer implementation for consistency
   - Testable: tests can verify production code even when _train_one_step is patched
   - Single source of truth for scheduler stepping logic
   - Method definition (trainer.py lines 612-622):
     ```python
     def _step_scheduler_if_configured(self) -> None:
         """Step LR scheduler if per-batch stepping is enabled."""
         if self.scheduler is not None and self.config.step_scheduler_per_batch:
             self.scheduler.step()
     ```

   Additional Test Cleanup (Not the main fix):
   - Removed obsolete enable_weight_monitor=False parameter from test configs
   - This parameter was removed from TrainerConfig API but tests still referenced it
   - Affected tests:
     * src/test/test_scheduler_stepping.py: 4 test functions
     * src/test/test_training_regressions.py: 1 test function
   - This was preventing tests from running, but was not the actual bug

   Impact:
   - Language model training with warmup schedules now works correctly
   - LR properly ramps up during warmup phase (e.g., first 1000 steps)
   - Training stability and convergence restored for LM training
   - Translation training was not affected (Trainer._train_one_step was already correct)

   Test Results After Fix:
   - test_lm_scheduler_steps_per_batch_when_enabled: NOW PASSING ✓
     * File: src/test/test_training_regressions.py
     * Verifies scheduler.step() is called once per training step
     * Uses LanguageModelTrainer with step_scheduler_per_batch=True
     * Confirms warmup scheduler integration works end-to-end

   - All 11 scheduler/regression/pitfalls tests pass:
     * test_warmup_scheduler_steps_per_batch ✓
     * test_warmup_lr_ramps_over_steps ✓
     * test_epoch_scheduler_steps_per_epoch_not_per_batch ✓
     * test_warmup_completes_in_expected_steps ✓
     * test_lm_scheduler_steps_per_batch_when_enabled ✓ (was failing, now fixed)
     * test_translation_collate_does_not_poison_decoder_input ✓
     * test_transformer_generate_restores_training_mode ✓
     * test_transformer_evaluate_restores_training_mode ✓
     * test_language_model_generate_restores_training_mode ✓
     * test_translation_padding_does_not_mask_real_eos_when_pad_equals_eos ✓
     * test_translation_source_pad_mask_does_not_mask_real_eos_when_pad_equals_eos ✓

   Files Modified:
   - src/training/trainer.py:
     * MAIN FIX: Added scheduler stepping in LanguageModelTrainer._train_one_step() (line 1191-1192)
   - src/test/test_scheduler_stepping.py:
     * CLEANUP: Removed obsolete enable_weight_monitor parameter (4 tests)
   - src/test/test_training_regressions.py:
     * CLEANUP: Removed obsolete enable_weight_monitor parameter (1 test)

   Commit:
   - ac4b3adc "Fix warmup scheduler stepping in LanguageModelTrainer"
   - Date: 2026-02-16

   Related Documentation:
   - Warmup scheduler test coverage: doc/change/v0.5/WARMUP_SCHEDULER_TEST_COVERAGE_COMPLETE.md
   - Warmup scheduler implementation: src/training/train_lm.py get_cosine_schedule_with_warmup()
   - Training configuration: doc/note/option_a_training.md
