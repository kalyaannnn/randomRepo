from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from agentrl.memory.layout import SharedWeightLayout


class FakeBaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.tensor([10.0]))


class FakePeftModel(torch.nn.Module):
    saved_adapters: dict[str, dict[str, torch.Tensor]] = {}

    def __init__(self, base_model: FakeBaseModel) -> None:
        super().__init__()
        self.base_model = base_model
        self.base_weight = base_model.base_weight
        self.adapters = torch.nn.ParameterDict()
        self.active_adapter: str | None = None
        self.set_adapter_calls: list[str] = []
        self.loaded_adapters: list[tuple[str, str, bool]] = []
        self.saved_calls: list[tuple[str, tuple[str, ...] | None]] = []
        self.deleted_adapters: list[str] = []

    def add_adapter(self, adapter_name: str, config: object) -> None:
        del config
        self.adapters[adapter_name] = torch.nn.Parameter(torch.tensor([0.0]))
        self.active_adapter = adapter_name

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name
        self.set_adapter_calls.append(adapter_name)

    def load_adapter(
        self,
        adapter_dir: str | Path,
        adapter_name: str,
        is_trainable: bool = False,
    ) -> None:
        snapshot = self.saved_adapters[str(Path(adapter_dir))]
        source_name, tensor = next(iter(snapshot.items()))
        self.adapters[adapter_name] = torch.nn.Parameter(tensor.clone())
        self.adapters[adapter_name].requires_grad = is_trainable
        self.loaded_adapters.append((source_name, adapter_name, is_trainable))

    def save_pretrained(
        self,
        path: str | Path,
        selected_adapters: list[str] | None = None,
    ) -> None:
        adapter_names = tuple(selected_adapters) if selected_adapters is not None else None
        self.saved_calls.append((str(Path(path)), adapter_names))
        if selected_adapters is None:
            selected_adapters = [self.active_adapter]
        self.saved_adapters[str(Path(path))] = {
            name: self.adapters[name].detach().clone() for name in selected_adapters
        }

    def delete_adapter(self, adapter_name: str) -> None:
        self.deleted_adapters.append(adapter_name)
        if adapter_name in self.adapters:
            del self.adapters[adapter_name]

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        del prefix, recurse
        yield "base_model.weight", self.base_weight
        for adapter_name, parameter in self.adapters.items():
            yield f"lora_A.{adapter_name}.weight", parameter

    def parameters(self, recurse: bool = True):
        del recurse
        yield self.base_weight
        yield from self.adapters.values()

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        return self

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        batch, seq = input_ids.shape
        active = self.active_adapter or "none"
        logits = torch.zeros((batch, seq, 3), dtype=torch.float32)
        if active in self.adapters:
            logits[:, :, 1] = self.adapters[active].item()
        return SimpleNamespace(logits=logits)


def _install_fake_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_transformers = ModuleType("transformers")

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs) -> FakeBaseModel:
            del args, kwargs
            return FakeBaseModel()

    fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM

    fake_peft = ModuleType("peft")

    class _PeftModel:
        @staticmethod
        def from_pretrained(
            base_model: FakeBaseModel,
            adapter_dir: str | Path,
            is_trainable: bool = True,
            adapter_name: str = "policy",
        ) -> FakePeftModel:
            model = FakePeftModel(base_model)
            model.load_adapter(adapter_dir, adapter_name=adapter_name, is_trainable=is_trainable)
            model.set_adapter(adapter_name)
            return model

    def _get_peft_model(
        base_model: FakeBaseModel,
        lora_config: object,
        adapter_name: str = "policy",
    ) -> FakePeftModel:
        model = FakePeftModel(base_model)
        model.add_adapter(adapter_name, lora_config)
        model.set_adapter(adapter_name)
        return model

    fake_peft.get_peft_model = _get_peft_model
    fake_peft.PeftModel = _PeftModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)


def _install_legacy_fake_backends(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_delete: bool,
) -> None:
    fake_transformers = ModuleType("transformers")

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs) -> FakeBaseModel:
            del args, kwargs
            return FakeBaseModel()

    fake_transformers.AutoModelForCausalLM = _AutoModelForCausalLM

    fake_peft = ModuleType("peft")

    class LegacyFakePeftModel(FakePeftModel):
        if not allow_delete:
            delete_adapter = None

    class _PeftModel:
        @staticmethod
        def from_pretrained(
            base_model: FakeBaseModel,
            adapter_dir: str | Path,
            is_trainable: bool = True,
        ) -> LegacyFakePeftModel:
            model = LegacyFakePeftModel(base_model)
            model.load_adapter(adapter_dir, adapter_name="default", is_trainable=is_trainable)
            model.set_adapter("default")
            return model

    def _get_peft_model(
        base_model: FakeBaseModel,
        lora_config: object,
    ) -> LegacyFakePeftModel:
        model = LegacyFakePeftModel(base_model)
        model.add_adapter("default", lora_config)
        model.set_adapter("default")
        return model

    fake_peft.get_peft_model = _get_peft_model
    fake_peft.PeftModel = _PeftModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)


def test_named_adapters_switch_between_policy_and_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_backends(monkeypatch)

    layout = SharedWeightLayout(
        model_name="fake/model",
        lora_config=object(),
        device="cpu",
    )
    layout.model.adapters["policy"].data.fill_(1.5)
    layout.model.adapters["reference"].data.fill_(0.5)

    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    policy_logits = layout.policy_forward(input_ids=input_ids, attention_mask=attention_mask)
    reference_logits = layout.reference_forward(input_ids=input_ids, attention_mask=attention_mask)

    assert list(layout.model.adapters.keys()) == ["policy", "reference"]
    assert layout.model.set_adapter_calls[-2:] == ["policy", "reference"]
    assert torch.all(policy_logits[:, :, 1] == 1.5)
    assert torch.all(reference_logits[:, :, 1] == 0.5)


def test_layout_snapshots_initial_policy_adapter_and_freezes_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_backends(monkeypatch)

    layout = SharedWeightLayout(
        model_name="fake/model",
        lora_config=object(),
        device="cpu",
    )

    policy_parameter = layout.model.adapters["policy"]
    reference_parameter = layout.model.adapters["reference"]

    assert torch.equal(policy_parameter.detach(), reference_parameter.detach())
    assert policy_parameter.requires_grad is True
    assert reference_parameter.requires_grad is False
    assert layout.model.base_weight.requires_grad is False

    policy_parameter.data.add_(2.0)

    assert not torch.equal(policy_parameter.detach(), reference_parameter.detach())


def test_save_adapter_persists_policy_adapter_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_backends(monkeypatch)

    layout = SharedWeightLayout(
        model_name="fake/model",
        lora_config=object(),
        device="cpu",
    )

    output_path = layout.save_adapter(tmp_path / "adapter")

    assert output_path == tmp_path / "adapter"
    assert layout.model.saved_calls[-1] == (str(output_path), ("policy",))


def test_legacy_peft_fallback_removes_default_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_legacy_fake_backends(monkeypatch, allow_delete=True)

    layout = SharedWeightLayout(
        model_name="fake/model",
        lora_config=object(),
        device="cpu",
    )

    assert list(layout.model.adapters.keys()) == ["policy", "reference"]
    assert layout.model.deleted_adapters == ["default"]


def test_legacy_peft_fallback_fails_when_default_adapter_cannot_be_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_legacy_fake_backends(monkeypatch, allow_delete=False)

    with pytest.raises(RuntimeError, match="cannot be removed"):
        SharedWeightLayout(
            model_name="fake/model",
            lora_config=object(),
            device="cpu",
        )
