"""Shared-weight policy/reference model layout for AgentRL."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Iterator

import torch
from torch import nn

if TYPE_CHECKING:
    from peft import LoraConfig


class SharedWeightLayout:
    """Own one frozen base model plus one trainable LoRA adapter.

    The base model weights are loaded exactly once. Reference scoring disables
    adapter layers temporarily, while policy scoring keeps them enabled. This is
    the key VRAM-saving layout that keeps single-GPU GRPO practical.
    """

    def __init__(
        self,
        model_name: str,
        lora_config: LoraConfig,
        dtype: str = "float16",
        device: torch.device | str | None = None,
        trust_remote_code: bool = False,
        sdpa_backend: str = "auto",
    ) -> None:
        """Load the shared base model and attach LoRA adapters.

        Args:
            model_name: Hugging Face model identifier.
            lora_config: PEFT LoRA configuration used to wrap the base model.
            dtype: Torch dtype name for model loading.
            device: Device to move the model onto once.
            trust_remote_code: Whether to allow custom HF model code.
            sdpa_backend: Attention backend policy.
        """

        self.model_name = model_name
        self.lora_config = lora_config
        self.torch_dtype = getattr(torch, dtype)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.sdpa_backend = sdpa_backend
        self.active_attention_backend = "math"

        try:
            from peft import get_peft_model
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "SharedWeightLayout requires `transformers` and `peft` to be installed."
            ) from exc

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model = get_peft_model(base_model, lora_config)
        self._prepare_parameter_states()
        self.model.to(self.device)

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with LoRA adapter layers enabled.

        Args:
            input_ids: Token ids of shape `[batch, seq]`.
            attention_mask: Attention mask of shape `[batch, seq]`.

        Returns:
            Logits tensor of shape `[batch, seq, vocab]`.
        """

        self._enable_adapters()
        with self._sdpa_context():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with LoRA adapter layers disabled.

        Args:
            input_ids: Token ids of shape `[batch, seq]`.
            attention_mask: Attention mask of shape `[batch, seq]`.

        Returns:
            Logits tensor of shape `[batch, seq, vocab]`.
        """

        self._disable_adapters()
        try:
            with self._sdpa_context():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            self._enable_adapters()
        return outputs.logits

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only LoRA adapter parameters."""

        for parameter in self.model.parameters():
            if parameter.requires_grad:
                yield parameter

    def vram_report(self) -> dict[str, float]:
        """Estimate VRAM occupied by base and adapter parameters in megabytes."""

        adapter_bytes = 0
        base_bytes = 0
        for parameter in self.model.parameters():
            size_bytes = parameter.numel() * parameter.element_size()
            if parameter.requires_grad:
                adapter_bytes += size_bytes
            else:
                base_bytes += size_bytes

        total_bytes = base_bytes + adapter_bytes
        return {
            "base_mb": base_bytes / (1024 * 1024),
            "adapter_mb": adapter_bytes / (1024 * 1024),
            "total_mb": total_bytes / (1024 * 1024),
        }

    def _prepare_parameter_states(self) -> None:
        """Ensure base weights are frozen and only LoRA parameters remain trainable."""

        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = "lora_" in name

    def _enable_adapters(self) -> None:
        method = getattr(self.model, "enable_adapter_layers", None)
        if method is None:
            raise AttributeError("PEFT model does not expose enable_adapter_layers().")
        method()

    def _disable_adapters(self) -> None:
        method = getattr(self.model, "disable_adapter_layers", None)
        if method is None:
            raise AttributeError("PEFT model does not expose disable_adapter_layers().")
        method()

    @contextmanager
    def _sdpa_context(self) -> Iterator[None]:
        """Apply the configured SDPA policy.

        `auto` deliberately lets PyTorch/Transformers choose the backend. This
        avoids forcing a FlashAttention path that can be invalid for models or
        masks that require a different kernel.
        """

        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError:
            self.active_attention_backend = "math"
            with nullcontext():
                yield
            return

        if self.sdpa_backend == "auto":
            self.active_attention_backend = "auto"
            with nullcontext():
                yield
            return

        if not torch.cuda.is_available():
            self.active_attention_backend = "math"
            with sdpa_kernel(SDPBackend.MATH):
                yield
            return

        if self.sdpa_backend == "math":
            self.active_attention_backend = "math"
            with sdpa_kernel(SDPBackend.MATH):
                yield
            return

        self.active_attention_backend = "flash_attention"
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            yield
