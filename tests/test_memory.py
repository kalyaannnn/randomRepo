from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from agentrl.memory.layout import SharedWeightLayout


class FakeBaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        batch, seq = input_ids.shape
        vocab = 3
        logits = torch.zeros((batch, seq, vocab), dtype=torch.float16)
        return SimpleNamespace(logits=logits)


class FakePeftModel(torch.nn.Module):
    saved_adapters: dict[str, dict[str, torch.Tensor]] = {}

    def __init__(self, base_model: FakeBaseModel, adapter_name: str) -> None:
        super().__init__()
        self.base_model = base_model
        self.base_weight = base_model.base_weight
        self.adapters = torch.nn.ParameterDict()
        self.active_adapter: str | None = None
        self.set_adapter_calls: list[str] = []
        self.add_adapter(adapter_name, config=None)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        batch, seq = input_ids.shape
        logits = torch.zeros((batch, seq, 3), dtype=torch.float16)
        if self.active_adapter is not None:
            logits[:, :, 1] = self.adapters[self.active_adapter].sum()
        return SimpleNamespace(logits=logits)

    def add_adapter(self, adapter_name: str, config: object) -> None:
        del config
        initial_value = 1.0 if adapter_name == "policy" else 0.0
        self.adapters[adapter_name] = torch.nn.Parameter(
            torch.full((2,), initial_value, dtype=torch.float16)
        )
        self.active_adapter = adapter_name

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name
        self.set_adapter_calls.append(adapter_name)

    def load_adapter(
        self,
        adapter_dir: str,
        adapter_name: str,
        is_trainable: bool = False,
    ) -> None:
        source_name, source_value = next(iter(self.saved_adapters[str(adapter_dir)].items()))
        del source_name
        self.adapters[adapter_name] = torch.nn.Parameter(source_value.clone())
        self.adapters[adapter_name].requires_grad = is_trainable

    def save_pretrained(self, path: str, selected_adapters: list[str] | None = None) -> None:
        adapter_names = selected_adapters or [self.active_adapter]
        self.saved_adapters[str(path)] = {
            name: self.adapters[name].detach().clone()
            for name in adapter_names
            if name is not None
        }

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
        del prefix, recurse
        yield "model.base_weight", self.base_weight
        for adapter_name, parameter in self.adapters.items():
            yield f"model.lora_A.{adapter_name}.weight", parameter

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        del recurse
        yield self.base_weight
        yield from self.adapters.values()

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        return self


def install_fake_hf_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = ModuleType("transformers")
    peft = ModuleType("peft")

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name: str, dtype: torch.dtype, **kwargs) -> FakeBaseModel:
            assert model_name == "fake/model"
            assert dtype == torch.float16
            assert "trust_remote_code" in kwargs
            assert "low_cpu_mem_usage" in kwargs
            return FakeBaseModel()

    def get_peft_model(
        model: FakeBaseModel,
        lora_config: object,
        adapter_name: str = "policy",
    ) -> FakePeftModel:
        assert lora_config == {"r": 16}
        return FakePeftModel(model, adapter_name=adapter_name)

    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    peft.get_peft_model = get_peft_model
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)


def test_shared_weight_layout_switches_named_adapters_for_reference_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_hf_modules(monkeypatch)
    layout = SharedWeightLayout(model_name="fake/model", lora_config={"r": 16})

    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    policy_logits = layout.policy_forward(input_ids, attention_mask)
    ref_logits = layout.reference_forward(input_ids, attention_mask)
    post_ref_logits = layout.policy_forward(input_ids, attention_mask)

    assert layout.model.set_adapter_calls[-3:] == ["policy", "reference", "policy"]
    assert torch.all(policy_logits[:, :, 1] == 2.0)
    assert torch.all(ref_logits[:, :, 1] == 2.0)
    assert torch.all(post_ref_logits[:, :, 1] == 2.0)


def test_shared_weight_layout_exposes_only_lora_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_hf_modules(monkeypatch)
    layout = SharedWeightLayout(model_name="fake/model", lora_config={"r": 16})

    parameters = list(layout.trainable_parameters())

    assert len(parameters) == 1
    assert parameters[0].requires_grad is True
    assert parameters[0] is layout.model.adapters["policy"]
    assert parameters[0].numel() == 2


def test_shared_weight_layout_reports_vram_breakdown(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_hf_modules(monkeypatch)
    layout = SharedWeightLayout(model_name="fake/model", lora_config={"r": 16})

    report = layout.vram_report()

    assert report["base_mb"] == pytest.approx(8 / (1024 * 1024), rel=1e-2)
    assert report["adapter_mb"] == pytest.approx(8 / (1024 * 1024), rel=1e-2)
    assert report["total_mb"] == pytest.approx(16 / (1024 * 1024), rel=1e-2)
