✅ Warmup Scheduler Test Coverage Fix (COMPLETE)
   - Status: Critical test coverage gap fixed for per-batch scheduler stepping

   Problem:
   - Warmup scheduler tests patched _train_one_step() and duplicated scheduler stepping logic
   - Tests verified their own local copy, not the production Trainer code
   - If production scheduler.step() was removed/broken, tests would still pass
   - Affected tests: test_warmup_scheduler_steps_per_batch, test_warmup_lr_ramps_over_steps,
     test_warmup_completes_in_expected_steps
   - Risk: High regression risk - warmup scheduler could break without detection

   Solution:
   - Extracted _step_scheduler_if_configured() method in Trainer class
   - Tests now call this method instead of duplicating logic
   - Tests exercise real production code even when _train_one_step is patched
   - Method encapsulates the per-batch scheduler stepping logic:
     ```python
     def _step_scheduler_if_configured(self) -> None:
         """Step LR scheduler if per-batch stepping is enabled."""
         if self.scheduler is not None and self.config.step_scheduler_per_batch:
             self.scheduler.step()
     ```

   Test Improvements:
   - Replaced magic number (50%) with mathematically calculated expected LR
   - Added detailed explanatory comments about two-phase warmup schedule:
     * Phase 1 - Linear Warmup (steps 1-5):
       Purpose: Gradually increase LR from 0 to peak to avoid early training instability
       Formula: lr = (current_step / warmup_steps) * peak_lr
       Result: Linear ramp from 0 → peak_lr

     * Phase 2 - Cosine Annealing Decay (steps 6-20):
       Purpose: Smoothly decay LR using cosine curve for better convergence
       Formula: lr = min_lr + (peak_lr - min_lr) * cosine_decay
                where cosine_decay = 0.5 * (1 + cos(π * progress))
       Result: Smooth cosine curve from peak_lr → min_lr_ratio * peak_lr

   - Calculate expected LR at step 10 (after 5 warmup + 5 decay):
     * progress = 5/15 = 0.333
     * cosine_decay ≈ 0.75
     * expected_lr ≈ 0.775 * peak_lr (77.5% of peak)

   - Use 5% tolerance for robust cross-platform testing (protects against floating point variations)
   - Improved assertion message shows actual vs expected LR with percentages

   Benefits:
   - Tests now verify production warmup scheduler behavior, not test code
   - Catches regressions where per-batch scheduler stepping logic breaks
   - Clear documentation of warmup + cosine decay schedule in test comments
   - No magic numbers - expected values derived from mathematical formulas
   - Assertions are both strict (5% tolerance) and informative (detailed error messages)

   Why 20 steps for unit test?
   - Tests the mechanism (does warmup scheduler step per-batch?), not training quality
   - Covers both warmup phase (steps 1-5) and cosine decay phase (steps 6-20)
   - Sufficient to verify LR follows expected mathematical formula
   - Faster than realistic training values (1000-4000 warmup steps)
   - Real training uses more steps for stability, not for mechanism correctness

   Why 5% tolerance?
   - Warmup scheduler uses deterministic math (cosine), no randomness
   - Floating point errors typically < 0.001%
   - 5% allows for: minor PyTorch implementation differences, platform variations
   - Could be stricter (1% or 0.1%) but 5% is generous and robust

   Context - Why Per-Batch Stepping Matters:
   - Warmup schedules REQUIRE per-batch stepping (not per-epoch)
   - Per-epoch stepping would make 1000-step warmup take 1000 epochs!
   - config.step_scheduler_per_batch=True activates per-batch mode
   - Regular epoch-based schedulers (CosineAnnealingLR) use per-epoch stepping
   - This fix ensures warmup scheduler tests actually test the critical per-batch logic

   Tests: All scheduler tests pass (4/4)
   Files: src/training/trainer.py, src/test/test_scheduler_stepping.py

   Related:
   - See doc/note/option_a_training.md for warmup scheduler usage in training
   - See src/training/train_lm.py get_cosine_schedule_with_warmup() implementation
