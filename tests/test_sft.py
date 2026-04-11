from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from agentrl.core.config import GRPOConfig
from agentrl.core.sft import SFTBootstrapTrainer


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [max(1, ord(ch) % 17) for ch in text]


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.proj = nn.Linear(8, 32)

    def forward(self, input_ids, attention_mask, labels):
        del attention_mask
        logits = self.proj(self.embedding(input_ids))
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=-100,
        )
        return type("Output", (), {"loss": loss})


class DummyLayout:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = DummyModel()

    def trainable_parameters(self):
        return self.model.parameters()

    def save_adapter(self, path):
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter.bin").write_bytes(b"ok")
        return output


def test_sft_bootstrap_trainer_masks_prompt_tokens_and_saves_adapter(tmp_path) -> None:
    config = GRPOConfig(
        model_name="dummy/model",
        batch_size=2,
        steps=1,
        max_prompt_tokens=64,
        output_dir=str(tmp_path),
    )
    trainer = SFTBootstrapTrainer(
        config=config,
        tokenizer=DummyTokenizer(),
        layout=DummyLayout(),
    )

    encoded = trainer._encode_batch([("Prompt", "Final answer: 4")])
    prompt_len = len(DummyTokenizer().encode("Prompt"))

    assert encoded["labels"].shape[0] == 1
    assert torch.all(encoded["labels"][0, :prompt_len] == -100)

    history = trainer.train([("Prompt", "Final answer: 4"), ("Prompt", "Final answer: 5")], epochs=1)
    adapter_dir = trainer.save_adapter(tmp_path / "adapter")

    assert len(history) == 1
    assert history[0]["loss"] >= 0.0
    assert (adapter_dir / "adapter.bin").exists()
