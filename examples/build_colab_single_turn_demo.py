"""Generate the Colab notebook for the 1.5B AgentRL-vs-TRL demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "notebooks/agentrl_trl_15b_t4_demo.ipynb"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the AgentRL vs TRL 1.5B T4 Colab notebook.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_notebook(), indent=2), encoding="utf-8")
    print(output)
    return output


def _notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            "# AgentRL vs TRL: Fair GRPO + Systems Moat\n\n"
            "This Colab mirrors `codeDemo.ipynb`, but separates two claims:\n\n"
            "1. **Fairness track:** AgentRL standard rollout vs TRL after the same SFT bootstrap.\n"
            "2. **Systems track:** AgentRL-only runtime modes show continuous batching, speculative decoding, and paged-KV telemetry.\n\n"
            "The default model is `Qwen/Qwen2.5-1.5B-Instruct` because it is the practical T4 path."
        ),
        _code(
            "REPO_URL = \"https://github.com/kalyaannnn/agentRL.git\"\n"
            "MODEL = \"Qwen/Qwen2.5-1.5B-Instruct\"\n"
            "DRAFT_MODEL = \"Qwen/Qwen2.5-0.5B-Instruct\"\n"
            "SEED = 0\n"
            "LIMIT = 8\n"
            "SFT_EPOCHS = 1\n"
            "GRPO_STEPS = 3\n"
            "BATCH_SIZE = 1\n"
            "GROUP_SIZE = 4\n"
            "MAX_NEW_TOKENS = 64\n"
            "MAX_EPISODE_STEPS = 2\n"
            "OUTPUT_DIR = \"/content/agentrl_colab_demo\"\n"
        ),
        _code(
            "!cd /content && rm -rf agentRL\n"
            "!cd /content && git clone https://github.com/kalyaannnn/agentRL.git\n"
            "%cd /content/agentRL\n"
            "!pip install -U pip\n"
            "!pip install -e \".[benchmark]\"\n"
            "!pip install -U trl peft accelerate bitsandbytes datasets pandas\n"
        ),
        _code(
            "import torch\n"
            "print(\"CUDA:\", torch.cuda.is_available())\n"
            "if torch.cuda.is_available():\n"
            "    print(\"GPU:\", torch.cuda.get_device_name(0))\n"
            "    print(\"VRAM GB:\", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))\n"
        ),
        _markdown(
            "## Fair comparison track\n\n"
            "Both stacks use the same records, supervised targets, SFT bootstrap, seed, generation budget, "
            "and single-turn verifier. AgentRL uses standard rollout here so the comparison is about the "
            "post-training pipeline, not AgentRL-specific runtime systems."
        ),
        _code(
            "!python -m examples.compare_single_turn_baselines \\\n"
            "  --model Qwen/Qwen2.5-1.5B-Instruct \\\n"
            "  --limit 8 \\\n"
            "  --seed 0 \\\n"
            "  --sft-epochs 1 \\\n"
            "  --steps 3 \\\n"
            "  --batch-size 1 \\\n"
            "  --group-size 4 \\\n"
            "  --max-new-tokens 64 \\\n"
            "  --output-dir /content/agentrl_colab_demo/fair_compare\n"
        ),
        _code(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "comparison = json.loads(Path('/content/agentrl_colab_demo/fair_compare/comparison.json').read_text())\n"
            "pd.DataFrame([\n"
            "    {\n"
            "        'framework': result.get('framework', name),\n"
            "        'model': result.get('model_name'),\n"
            "        'sft_epochs': result.get('sft_epochs'),\n"
            "        'grpo_steps': result.get('steps'),\n"
            "        'mean_reward': result.get('mean_reward'),\n"
            "        'wall_time_s': result.get('wall_time_s'),\n"
            "        'peak_vram_mb': result.get('peak_vram_mb'),\n"
            "    }\n"
            "    for name, result in comparison.items()\n"
            "    if isinstance(result, dict) and 'framework' in result\n"
            "])\n"
        ),
        _markdown(
            "## AgentRL systems moat\n\n"
            "This section is AgentRL-only. It compares runtime modes on the same model scale so the telemetry "
            "shows where rollout time and memory go. Speculative decoding uses a smaller Qwen draft model. "
            "Paged KV is a memory/control-path demo and may not be the fastest mode on very small T4 runs."
        ),
        _code(
            "!python -m examples.benchmark_systems \\\n"
            "  --model Qwen/Qwen2.5-1.5B-Instruct \\\n"
            "  --draft-model Qwen/Qwen2.5-0.5B-Instruct \\\n"
            "  --task tool-use \\\n"
            "  --split easy \\\n"
            "  --steps 3 \\\n"
            "  --batch-size 1 \\\n"
            "  --group-size 4 \\\n"
            "  --max-new-tokens 64 \\\n"
            "  --max-episode-steps 2 \\\n"
            "  --output-dir /content/agentrl_colab_demo/systems_compare \\\n"
            "  --compare-runtime-modes \\\n"
            "  --include-speculative\n"
        ),
        _code(
            "systems = json.loads(Path('/content/agentrl_colab_demo/systems_compare/comparison.json').read_text())\n"
            "runs = systems.get('runs') or systems.get('summaries') or []\n"
            "pd.DataFrame([\n"
            "    {\n"
            "        'mode': run.get('mode_name'),\n"
            "        'mean_step_time_ms': run.get('mean_step_time_ms'),\n"
            "        'tokens_per_second': run.get('mean_tokens_per_second'),\n"
            "        'peak_vram_mb': run.get('peak_vram_mb'),\n"
            "        'rollout_peak_vram_mb': run.get('rollout_peak_vram_mb'),\n"
            "        'padding_ratio': run.get('mean_padding_ratio'),\n"
            "        'kv_pressure': run.get('mean_scheduler_kv_pressure'),\n"
            "        'bottleneck': run.get('dominant_runtime_bottleneck'),\n"
            "        'reward': run.get('mean_reward'),\n"
            "    }\n"
            "    for run in runs\n"
            "])\n"
        ),
        _markdown(
            "## Interpretation\n\n"
            "- TRL is the fair algorithmic baseline after the same SFT bootstrap.\n"
            "- AgentRL's moat is the runtime layer: continuous batching, speculative decoding, paged-KV telemetry, scheduler diagnostics, "
            "VRAM/headroom metrics, and rollout throughput.\n"
            "- For A100/L4, switch `MODEL` to `Qwen/Qwen2.5-7B-Instruct` and use "
            "`Qwen/Qwen2.5-1.5B-Instruct` as the draft model for speculative experiments."
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


if __name__ == "__main__":
    main()
