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
        self.adapter_enabled = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        batch, seq = input_ids.shape
        vocab = 3
        offset = 1.0 if self.adapter_enabled else 0.0
        logits = torch.full((batch, seq, vocab), offset, dtype=torch.float16)
        return SimpleNamespace(logits=logits)

    def enable_adapter_layers(self) -> None:
        self.adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self.adapter_enabled = False


class FakePeftModel(torch.nn.Module):
    def __init__(self, base_model: FakeBaseModel) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_parameter("base_weight", base_model.base_weight)
        self.register_parameter("lora_weight", torch.nn.Parameter(torch.ones(2, dtype=torch.float16)))
        self.enable_adapter_layers()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask)

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
        del prefix, recurse
        yield "model.base_weight", self.base_weight
        yield "model.lora_A.default.weight", self.lora_weight

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        del recurse
        yield self.base_weight
        yield self.lora_weight

    def enable_adapter_layers(self) -> None:
        self.base_model.enable_adapter_layers()

    def disable_adapter_layers(self) -> None:
        self.base_model.disable_adapter_layers()


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

    def get_peft_model(model: FakeBaseModel, lora_config: object) -> FakePeftModel:
        assert lora_config == {"r": 16}
        return FakePeftModel(model)

    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    peft.get_peft_model = get_peft_model
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)


def test_shared_weight_layout_toggles_adapters_for_reference_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_hf_modules(monkeypatch)
    layout = SharedWeightLayout(model_name="fake/model", lora_config={"r": 16})

    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    policy_logits = layout.policy_forward(input_ids, attention_mask)
    ref_logits = layout.reference_forward(input_ids, attention_mask)
    post_ref_logits = layout.policy_forward(input_ids, attention_mask)

    assert torch.all(policy_logits == 1.0)
    assert torch.all(ref_logits == 0.0)
    assert torch.all(post_ref_logits == 1.0)


def test_shared_weight_layout_exposes_only_lora_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_hf_modules(monkeypatch)
    layout = SharedWeightLayout(model_name="fake/model", lora_config={"r": 16})

    parameters = list(layout.trainable_parameters())

    assert len(parameters) == 1
    assert parameters[0].requires_grad is True
    assert parameters[0].numel() == 2


def test_shared_weight_layout_reports_vram_breakdown(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_hf_modules(monkeypatch)
    layout = SharedWeightLayout(model_name="fake/model", lora_config={"r": 16})

    report = layout.vram_report()

    assert report["base_mb"] == pytest.approx(8 / (1024 * 1024), rel=1e-2)
    assert report["adapter_mb"] == pytest.approx(4 / (1024 * 1024), rel=1e-2)
    assert report["total_mb"] == pytest.approx(12 / (1024 * 1024), rel=1e-2)
