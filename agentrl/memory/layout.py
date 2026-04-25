"""Shared-weight policy/reference model layout for AgentRL."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Iterator

import torch
from torch import nn

if TYPE_CHECKING:
    from peft import LoraConfig


class SharedWeightLayout:
    """Own one frozen base model plus policy/reference LoRA adapters.

    The base model weights are loaded exactly once. Policy and reference
    scoring switch between two named adapters on that shared base model. This
    is the key VRAM-saving layout that keeps single-GPU GRPO practical.
    """

    POLICY_ADAPTER_NAME = "policy"
    REFERENCE_ADAPTER_NAME = "reference"

    def __init__(
        self,
        model_name: str,
        lora_config: LoraConfig,
        dtype: str = "float16",
        device: torch.device | str | None = None,
        trust_remote_code: bool = False,
        sdpa_backend: str = "auto",
        adapter_path: str | None = None,
    ) -> None:
        """Load the shared base model and attach LoRA adapters.

        Args:
            model_name: Hugging Face model identifier.
            lora_config: PEFT LoRA configuration used to wrap the base model.
            dtype: Torch dtype name for model loading.
            device: Device to move the model onto once.
            trust_remote_code: Whether to allow custom HF model code.
            sdpa_backend: Attention backend policy.
            adapter_path: Optional path to a saved LoRA adapter to load.
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
        self.model = self._build_dual_adapter_model(
            base_model=base_model,
            get_peft_model=get_peft_model,
            adapter_path=adapter_path,
        )
        self._prepare_parameter_states()
        self.model.to(self.device)

    def policy_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with the policy LoRA adapter active.

        Args:
            input_ids: Token ids of shape `[batch, seq]`.
            attention_mask: Attention mask of shape `[batch, seq]`.

        Returns:
            Logits tensor of shape `[batch, seq, vocab]`.
        """

        self._set_active_adapter(self.POLICY_ADAPTER_NAME)
        with self._sdpa_context():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def reference_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with the frozen reference LoRA adapter active.

        Args:
            input_ids: Token ids of shape `[batch, seq]`.
            attention_mask: Attention mask of shape `[batch, seq]`.

        Returns:
            Logits tensor of shape `[batch, seq, vocab]`.
        """

        self._set_active_adapter(self.REFERENCE_ADAPTER_NAME)
        with self._sdpa_context():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
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
        for name, parameter in self.model.named_parameters():
            size_bytes = parameter.numel() * parameter.element_size()
            if "lora_" in name:
                adapter_bytes += size_bytes
            else:
                base_bytes += size_bytes

        total_bytes = base_bytes + adapter_bytes
        return {
            "base_mb": base_bytes / (1024 * 1024),
            "adapter_mb": adapter_bytes / (1024 * 1024),
            "total_mb": total_bytes / (1024 * 1024),
        }

    def save_adapter(self, path: str | Path) -> Path:
        """Persist just the LoRA adapter weights and config."""

        output_path = Path(path).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        self._save_named_adapter(output_path, adapter_name=self.POLICY_ADAPTER_NAME)
        return output_path

    def _build_dual_adapter_model(
        self,
        base_model: nn.Module,
        get_peft_model: Any,
        adapter_path: str | None,
    ) -> nn.Module:
        """Create or load the policy adapter, then snapshot it into a frozen reference adapter."""

        if adapter_path is not None:
            model = self._load_policy_adapter(base_model=base_model, adapter_path=adapter_path)
        else:
            model = self._create_policy_adapter(base_model=base_model, get_peft_model=get_peft_model)

        self.model = model
        self._set_active_adapter(self.POLICY_ADAPTER_NAME, model=model)
        self._create_reference_snapshot()
        self._set_active_adapter(self.POLICY_ADAPTER_NAME, model=model)
        return model

    def _create_policy_adapter(self, base_model: nn.Module, get_peft_model: Any) -> nn.Module:
        """Wrap the shared base model with a named trainable policy adapter."""

        try:
            return get_peft_model(
                base_model,
                self.lora_config,
                adapter_name=self.POLICY_ADAPTER_NAME,
            )
        except TypeError:
            model = get_peft_model(base_model, self.lora_config)
            self._ensure_named_policy_adapter(model)
            return model

    def _load_policy_adapter(self, base_model: nn.Module, adapter_path: str) -> nn.Module:
        """Load a saved trainable policy adapter onto the shared base model."""

        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "Loading a saved LoRA adapter requires `peft.PeftModel` to be available."
            ) from exc

        adapter_dir = Path(adapter_path).expanduser()
        if not adapter_dir.exists():
            raise FileNotFoundError(f"LoRA adapter path does not exist: {adapter_dir}")

        try:
            model = PeftModel.from_pretrained(
                base_model,
                adapter_dir,
                is_trainable=True,
                adapter_name=self.POLICY_ADAPTER_NAME,
            )
        except TypeError:
            model = PeftModel.from_pretrained(
                base_model,
                adapter_dir,
                is_trainable=True,
            )
            self._ensure_named_policy_adapter(model, adapter_dir=adapter_dir)
        return model

    def _ensure_named_policy_adapter(self, model: nn.Module, adapter_dir: Path | None = None) -> None:
        """Backfill an explicit `policy` adapter name when PEFT loads a default adapter name."""

        active_adapter = self._active_adapter_name(model)
        if active_adapter == self.POLICY_ADAPTER_NAME:
            return

        if adapter_dir is not None and hasattr(model, "load_adapter"):
            model.load_adapter(
                adapter_dir,
                adapter_name=self.POLICY_ADAPTER_NAME,
                is_trainable=True,
            )
            self._remove_adapter_or_raise(model, active_adapter)
            self._set_active_adapter(self.POLICY_ADAPTER_NAME, model=model)
            return

        if active_adapter is None:
            raise AttributeError("PEFT model must expose an active adapter to name the policy adapter.")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self._save_named_adapter(temp_path, adapter_name=active_adapter, model=model)
            load_adapter = getattr(model, "load_adapter", None)
            if load_adapter is None:
                raise AttributeError("PEFT model does not expose load_adapter().")
            load_adapter(
                temp_path,
                adapter_name=self.POLICY_ADAPTER_NAME,
                is_trainable=True,
            )
        self._remove_adapter_or_raise(model, active_adapter)
        self._set_active_adapter(self.POLICY_ADAPTER_NAME, model=model)

    def _create_reference_snapshot(self) -> None:
        """Create a frozen reference adapter snapshot from the current policy adapter state."""

        if self._has_adapter(self.REFERENCE_ADAPTER_NAME):
            return

        load_adapter = getattr(self.model, "load_adapter", None)
        if load_adapter is not None:
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                self._save_named_adapter(temp_path, adapter_name=self.POLICY_ADAPTER_NAME)
                load_adapter(
                    temp_path,
                    adapter_name=self.REFERENCE_ADAPTER_NAME,
                    is_trainable=False,
                )
            return

        self._clone_adapter_via_state_dict()

    def _clone_adapter_via_state_dict(self) -> None:
        """Clone the policy adapter into a named reference adapter without duplicating the base model."""

        add_adapter = getattr(self.model, "add_adapter", None)
        if add_adapter is None:
            raise AttributeError("PEFT model does not expose add_adapter().")

        try:
            from peft import get_peft_model_state_dict, set_peft_model_state_dict
        except ImportError as exc:
            raise ImportError(
                "Reference adapter cloning requires PEFT state-dict helpers when load_adapter() is unavailable."
            ) from exc

        add_adapter(self.REFERENCE_ADAPTER_NAME, self.lora_config)
        policy_state = get_peft_model_state_dict(self.model, adapter_name=self.POLICY_ADAPTER_NAME)
        set_peft_model_state_dict(
            self.model,
            policy_state,
            adapter_name=self.REFERENCE_ADAPTER_NAME,
        )

    def _prepare_parameter_states(self) -> None:
        """Ensure base weights are frozen, only policy LoRA weights are trainable."""

        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = "lora_" in name and self.POLICY_ADAPTER_NAME in name

    def _set_active_adapter(self, adapter_name: str, model: nn.Module | None = None) -> None:
        """Activate a named adapter explicitly."""

        target = self.model if model is None else model
        method = getattr(target, "set_adapter", None)
        if method is None:
            raise AttributeError("PEFT model does not expose set_adapter().")
        method(adapter_name)

    def _save_named_adapter(
        self,
        path: Path,
        adapter_name: str,
        model: nn.Module | None = None,
    ) -> None:
        """Persist one named adapter, falling back to active-adapter save semantics when needed."""

        target = self.model if model is None else model
        save_pretrained = getattr(target, "save_pretrained", None)
        if save_pretrained is None:
            raise AttributeError("PEFT model does not expose save_pretrained().")

        try:
            save_pretrained(path, selected_adapters=[adapter_name])
        except TypeError:
            self._set_active_adapter(adapter_name, model=target)
            save_pretrained(path)

    def _has_adapter(self, adapter_name: str) -> bool:
        peft_config = getattr(self.model, "peft_config", None)
        if isinstance(peft_config, dict):
            return adapter_name in peft_config
        adapters = getattr(self.model, "adapters", None)
        if isinstance(adapters, nn.ParameterDict):
            return adapter_name in adapters
        if isinstance(adapters, dict):
            return adapter_name in adapters
        return False

    def _active_adapter_name(self, model: nn.Module) -> str | None:
        active_adapter = getattr(model, "active_adapter", None)
        if isinstance(active_adapter, str):
            return active_adapter
        if isinstance(active_adapter, (list, tuple)) and active_adapter:
            return str(active_adapter[0])
        return None

    def _remove_adapter_or_raise(self, model: nn.Module, adapter_name: str) -> None:
        """Remove a superseded adapter or fail closed if cleanup is unsupported."""

        if adapter_name == self.POLICY_ADAPTER_NAME:
            return

        delete_adapter = getattr(model, "delete_adapter", None)
        if callable(delete_adapter):
            delete_adapter(adapter_name)
            return

        remove_adapter = getattr(model, "remove_adapter", None)
        if callable(remove_adapter):
            remove_adapter(adapter_name)
            return

        raise RuntimeError(
            "PEFT compatibility fallback created a legacy adapter that cannot be removed. "
            "Refusing to keep extra adapter state resident; expected exactly one base model plus "
            "'policy' and 'reference' adapters."
        )

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
