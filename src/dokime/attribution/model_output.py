"""Custom TRAK ModelOutput for autoregressive language modeling (next-token prediction).

This enables TRAK-based data attribution for LLM fine-tuning data.
Given a fine-tuned model and a set of training examples, we can compute
per-example influence scores that indicate how much each training sample
contributes to the model's performance on a target benchmark.
"""

from __future__ import annotations

from typing import Any

import torch as ch
from torch import Tensor

try:
    from trak.modelout_functions import AbstractModelOutput
except ImportError:
    raise ImportError("Install attribution support: pip install dokime[attribution]") from None


class LanguageModelingModelOutput(AbstractModelOutput):
    """TRAK ModelOutput for autoregressive language modeling (next-token prediction).

    For a given input sequence [x1, x2, ..., xn], the model predicts [x2, x3, ..., xn+1].
    The output function is the sum of log-probabilities of the correct next tokens,
    which corresponds to the negative cross-entropy loss.

    This supports HuggingFace causal LM models (GPT-2, Llama, etc.) that return
    a CausalLMOutputWithPast object with a `.logits` attribute.

    IMPORTANT: All operations must be vmap-compatible — no data-dependent control
    flow (.item(), .any(), if/else on tensor values, boolean indexing).
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def get_output(
        model: Any,
        weights: Any,
        buffers: Any,
        input_id: Tensor,
        attention_mask: Tensor,
        label: Tensor,
    ) -> Tensor:
        """Compute the model output function for a single example.

        Returns the masked sum of log-probabilities of correct next tokens.
        Uses float masks instead of boolean indexing for vmap compatibility.
        """
        kw_inputs = {
            "input_ids": input_id.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }

        output = ch.func.functional_call(model, (weights, buffers), args=(), kwargs=kw_inputs)
        logits = output.logits

        # Shift for next-token prediction: logits[:-1] predict labels[1:]
        shift_logits = logits[0, :-1, :]  # (seq_len-1, vocab_size)
        shift_labels = label[1:]  # (seq_len-1,)

        # Float mask for non-padding tokens (no data-dependent branching)
        mask = (shift_labels != -100).float()

        # Clamp labels to valid range (replace -100 with 0 for gather, masked out later)
        safe_labels = shift_labels.clamp(min=0)

        # Log-probabilities of correct tokens
        log_probs = ch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs[ch.arange(shift_logits.shape[0]), safe_labels]

        # Masked sum: only count non-padding tokens
        return (token_log_probs * mask).sum()

    def get_out_to_loss_grad(self, model: Any, weights: Any, buffers: Any, batch: tuple[Tensor, ...]) -> Tensor:
        """Compute the gradient of the out-to-loss transformation.

        For cross-entropy loss: d(loss)/d(output) = -(1 - p(correct_token)).
        Uses float masks for vmap compatibility.

        Returns a tensor of shape (batch_size, 1).
        """
        input_ids, attention_mask, labels = batch

        kw_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        output = ch.func.functional_call(model, (weights, buffers), args=(), kwargs=kw_inputs)
        logits = output.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
        shift_labels = labels[:, 1:]  # (batch, seq_len-1)

        # Compute probabilities
        probs = ch.softmax(shift_logits / self.temperature, dim=-1)

        # Float mask for non-padding tokens
        mask = (shift_labels != -100).float()  # (batch, seq_len-1)
        safe_labels = shift_labels.clamp(min=0)

        # Gather correct token probabilities
        # probs: (batch, seq_len-1, vocab), safe_labels: (batch, seq_len-1)
        correct_probs = probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len-1)

        # Masked mean per example
        masked_probs = correct_probs * mask  # (batch, seq_len-1)
        token_counts = mask.sum(dim=1).clamp(min=1.0)  # (batch,) avoid div by zero
        avg_ps = masked_probs.sum(dim=1) / token_counts  # (batch,)

        return (1 - avg_ps).clone().detach().unsqueeze(-1)
