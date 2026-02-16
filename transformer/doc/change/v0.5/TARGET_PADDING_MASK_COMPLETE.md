✅ Target Padding Mask in Decoder Self-Attention (COMPLETE)
   - Status: Critical P2 bug fixed

   Problem:
   - Decoder self-attention did not accept or apply target_pad_mask
   - Only source_pad_mask was used (for cross-attention only)
   - In variable-length target batches, padded target positions remained valid keys/values
   - Real tokens could attend to padding positions in decoder self-attention
   - This contaminated token representations and gradients for variable-length sequences
   - Same target sentence padded to different lengths produced different outputs

   Solution:
   - Added target_pad_mask parameter to DecodeLayer.forward()
   - Applied target_pad_mask to causal self-attention (line 199-212)
   - Added target_pad_mask parameter to Decoder.forward()
   - Propagated mask through all decoder layers (line 387)
   - Added target_pad_mask parameter to Transformer.forward()
   - Passed mask from model to decoder (line 519-530)
   - Updated Trainer._train_one_step() to shift and pass target mask
   - Updated Trainer._validate() to shift and pass target mask

   Implementation Details:

   1. **DecodeLayer** (src/model/decoder.py):
      ```python
      def forward(self, x, memory=None, source_pad_mask=None, target_pad_mask=None):
          # Causal self-attention with target padding mask
          x = x + self.dropout_causal(
              self.causal_self_attention(
                  q=_q, k=_k, v=_v,
                  padding_mask=target_pad_mask  # NEW: mask padding in self-attention
              )
          )
      ```

   2. **Decoder** (src/model/decoder.py):
      ```python
      def forward(self, y, memory=None, source_pad_mask=None, target_pad_mask=None):
          for _layer in self.layers:
              y = _layer(
                  x=y, memory=memory,
                  source_pad_mask=source_pad_mask,
                  target_pad_mask=target_pad_mask  # NEW: propagate to all layers
              )
      ```

   3. **Transformer** (src/model/model.py):
      ```python
      def forward(self, x, y, source_pad_mask=None, target_pad_mask=None):
          # Encode source
          memory = self.encoder(x=x, source_pad_mask=source_pad_mask)

          # Decode with both source and target masks
          y = self.decoder(
              y=y,
              memory=memory,
              source_pad_mask=source_pad_mask,  # For cross-attention
              target_pad_mask=target_pad_mask   # NEW: For self-attention
          )
      ```

   4. **Trainer** (src/training/trainer.py):
      ```python
      def _train_one_step(self, batch):
          decoder_input = target_ids[:, :-1]  # Exclude last token

          # Shift target_pad_mask to match decoder_input
          if target_pad_mask is not None:
              decoder_input_pad_mask = target_pad_mask[:, :-1]  # NEW: shift mask

          log_probabilities = self.model(
              x=source_ids,
              y=decoder_input,
              source_pad_mask=source_pad_mask,
              target_pad_mask=decoder_input_pad_mask  # NEW: pass shifted mask
          )
      ```

   Mask Shifting Logic:
   - Original target_ids:     [BOS, tok1, tok2, tok3, EOS]  (length T)
   - Original target_pad_mask: [F, F, F, T, T]              (length T)
   - decoder_input:            [BOS, tok1, tok2, tok3]      (length T-1, last dropped)
   - decoder_input_pad_mask:   [F, F, F, T]                 (length T-1, last dropped)
   - decoder_target:           [tok1, tok2, tok3, EOS]      (length T-1, first dropped)
   - target_pad_mask_shifted:  [F, F, T, T]                 (length T-1, first dropped)

   Why Shifting Matters:
   - decoder_input uses target_ids[:, :-1], so mask must also drop last position
   - decoder_target uses target_ids[:, 1:], so mask must also drop first position
   - This ensures mask positions align with actual token positions

   Benefits:
   - Padding tokens no longer attended to in decoder self-attention
   - Real target tokens have clean, uncontaminated representations
   - Same sentence produces consistent output regardless of padding length
   - Attention mass not wasted on meaningless padding positions
   - Gradients computed only from real tokens, not padding noise

   Mask Application:
   - Causal mask (upper triangular): Prevents future attention (always active)
   - Target padding mask: Prevents padding attention (active when provided)
   - Both masks work together: attention[i,j] masked if j > i OR j is padding
   - In ScaledDotProductAttention: masked positions set to -inf before softmax

   Test Coverage:
   - test_decode_layer_accepts_target_pad_mask: API acceptance at layer level
   - test_decoder_accepts_target_pad_mask: Propagation through decoder stack
   - test_transformer_forward_accepts_target_pad_mask: Top-level API integration
   - test_target_pad_mask_affects_attention: Verifies mask actually changes behavior

   Tests: All tests pass (src/test/test_target_padding_mask.py)
   Files:
   - src/model/decoder.py (DecodeLayer, Decoder)
   - src/model/model.py (Transformer)
   - src/training/trainer.py (Trainer._train_one_step, Trainer._validate)
   - src/test/test_target_padding_mask.py (new test suite)

   Related Context:
   - Source padding mask still used for encoder self-attention and cross-attention
   - Target padding mask is ONLY for decoder self-attention
   - Cross-attention uses source_pad_mask to avoid attending to source padding
   - This completes the padding mask implementation for all attention types:
     * Encoder self-attention: uses source_pad_mask ✓
     * Decoder self-attention: uses target_pad_mask ✓ (THIS FIX)
     * Cross-attention: uses source_pad_mask ✓
