"""Minimal supervised bootstrap trainer for adapter-only warm starts."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from contextlib import nullcontext

from agentrl.core.config import GRPOConfig
from agentrl.memory.layout import SharedWeightLayout


class SFTBootstrapTrainer:
    """Run lightweight supervised fine-tuning on LoRA adapters only.

    This trainer is intentionally small. Its job is to teach the model a task
    format and give it a foothold before GRPO is asked to optimize a sparse
    real verifier.
    """

    def __init__(
        self,
        config: GRPOConfig,
        tokenizer: Any,
        layout: SharedWeightLayout,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.layout = layout
        self.device = layout.device
        self._set_seed()
        self.optimizer = torch.optim.AdamW(
            self.layout.trainable_parameters(),
            lr=self.config.lr,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )

    def train(
        self,
        samples: Sequence[tuple[str, str]],
        epochs: int = 1,
        shuffle: bool = True,
    ) -> list[dict[str, float]]:
        """Run adapter-only SFT over prompt/target pairs.

        Args:
            samples: Supervised `(prompt, target)` pairs.
            epochs: Number of full passes through the provided samples.
            shuffle: Whether to reshuffle sample order per epoch.

        Returns:
            Per-step scalar metrics.
        """

        if epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if not samples:
            raise ValueError("samples must contain at least one prompt/target pair.")

        history: list[dict[str, float]] = []
        steps_per_epoch = math.ceil(len(samples) / self.config.batch_size)
        step_index = 0
        self.layout.model.train()
        model_config = getattr(self.layout.model, "config", None)
        if model_config is not None:
            model_config.use_cache = False

        for epoch in range(epochs):
            epoch_samples = list(samples)
            if shuffle:
                random.shuffle(epoch_samples)
            for start in range(0, len(epoch_samples), self.config.batch_size):
                batch_samples = epoch_samples[start : start + self.config.batch_size]
                batch = self._encode_batch(batch_samples)
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    outputs = self.layout.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.layout.trainable_parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                history.append(
                    {
                        "step": float(step_index),
                        "epoch": float(epoch),
                        "loss": float(loss.detach().item()),
                        "batch_size": float(len(batch_samples)),
                        "steps_per_epoch": float(steps_per_epoch),
                    }
                )
                step_index += 1
        return history

    def save_adapter(self, output_dir: str | Path) -> Path:
        """Save the trained LoRA adapter for later GRPO initialization."""

        return self.layout.save_adapter(output_dir)

    def _encode_batch(self, samples: Sequence[tuple[str, str]]) -> dict[str, torch.Tensor]:
        encoded_inputs: list[list[int]] = []
        encoded_labels: list[list[int]] = []
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must expose pad_token_id for SFT batching.")

        for prompt, target in samples:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            input_ids = prompt_ids + target_ids
            labels = ([-100] * len(prompt_ids)) + target_ids
            if self.config.max_prompt_tokens is not None and len(input_ids) > self.config.max_prompt_tokens:
                input_ids = input_ids[-self.config.max_prompt_tokens :]
                labels = labels[-self.config.max_prompt_tokens :]
            encoded_inputs.append(input_ids)
            encoded_labels.append(labels)

        max_len = max(len(item) for item in encoded_inputs)
        if self.config.pad_to_multiple_of is not None:
            multiple = self.config.pad_to_multiple_of
            max_len = int(math.ceil(max_len / multiple) * multiple)

        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for input_ids, labels in zip(encoded_inputs, encoded_labels):
            pad_len = max_len - len(input_ids)
            input_rows.append(input_ids + ([pad_token_id] * pad_len))
            mask_rows.append(([1] * len(input_ids)) + ([0] * pad_len))
            label_rows.append(labels + ([-100] * pad_len))

        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long, device=self.device),
            "labels": torch.tensor(label_rows, dtype=torch.long, device=self.device),
        }

    def _set_seed(self) -> None:
        random.seed(self.config.seed)
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _autocast_context(self):
        if self.device.type not in {"cuda", "cpu"}:
            return nullcontext()
        if self.config.dtype not in {"float16", "bfloat16"}:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=getattr(torch, self.config.dtype))
