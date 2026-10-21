from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from novel_lab.settings import DATA_DIR

# LLaMA Factory 适配器（P9）：
# 本项目只负责把数据做好；训练交给 LLaMA-Factory 的 SFT/LoRA/QLoRA。
# 导出内容：
#   data/llama_factory/dataset/sy_{train,validation,test}.json  （alpaca 纯格式）
#   data/llama_factory/dataset_info.json                        （注册片段）
#   data/llama_factory/qlora_sy.yaml                            （训练配置模板）
#   data/llama_factory/README_训练步骤.md


ALPACA_KEYS = ("instruction", "input", "output")


def convert_split(
    src: Path, dst: Path
) -> int:

    count = 0
    with dst.open("w", encoding="utf-8") as f:
        for line in src.read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            clean = {
                k: sample.get(k, "")
                for k in ALPACA_KEYS
            }
            f.write(
                json.dumps(
                    clean, ensure_ascii=False
                )
                + "\n"
            )
            count += 1
    return count


DATASET_INFO = {
    "sy_train": {
        "file_name": "dataset/sy_train.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    },
    "sy_validation": {
        "file_name": "dataset/sy_validation.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    },
    "sy_test": {
        "file_name": "dataset/sy_test.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    },
}

QLORA_TEMPLATE = """### 《双月降临》文风 QLoRA 微调模板（LLaMA-Factory）
### 用法：拷进 LLaMA-Factory 目录，改 base_model 和 output_dir 后
###   llamafactory-cli train qlora_sy.yaml
### 显存参考：RTX 5080 16GB 可跑 7-8B QLoRA（cutoff_len 降到 1024 更稳）

### 模型
model_name_or_path: Qwen/Qwen3-8B      # ← 换成你下载的本地模型路径
trust_remote_code: true

### 方法
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
quantization_bit: 4                    # QLoRA；显存富余可删此行用 LoRA

### 数据
dataset: sy_train
eval_dataset: sy_validation
template: qwen                          # 与底座模型配套的对话模板
cutoff_len: 2048
max_samples: 100000
overwrite_cache: true

### 输出
output_dir: saves/sy-qlora-v1
logging_steps: 10
save_steps: 100
plot_loss: true

### 训练超参（小数据量 343 条：少 epoch 防_过拟合_）
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### 评估
val_size: 0.0                           # 已有独立 validation 集
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 100
"""

README_TRAIN = """# 训练步骤（LLaMA-Factory）

1. 安装 LLaMA-Factory（sick-cloud 上）：
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]"
   ```

2. 把本目录整体拷进 LLaMA-Factory/data/：
   - dataset/sy_*.json（训练数据，alpaca 格式）
   - dataset_info.json 的内容合并进 LLaMA-Factory/data/dataset_info.json
     （同名键不会覆盖，直接加在最外层）

3. 准备底座模型（Qwen3-8B 或其他中文底座，下载到本地路径）。

4. 训练：
   ```bash
   cp qlora_sy.yaml ~/LLaMA-Factory/ && cd ~/LLaMA-Factory
   # 改 yaml 里的 model_name_or_path / output_dir
   llamafactory-cli train qlora_sy.yaml
   ```

5. 推理验证（用 sy_test 里没见过的样本手测文风）：
   ```bash
   llamafactory-cli chat --adapter_name_or_path saves/sy-qlora-v1 \
     --model_name_or_path <底座路径> --template qwen
   ```

## 数据口径（重要）

- 样本只来自 rights ∈ authorized/user_provided/open_license 的文本
  （当前=作者本人《双月降临》前 20 章）。
- 343 条里 writing 类（自监督续写切片）才是文风主力，
  其余五类是知识问答，量小防灾难性遗忘即可。
- **343 条属于"流程验证版"规模**：文风微调通常需要 1k-10k 条切片。
  扩量路径：ch21-42 入库 ingest → understand/extract → dataset，
  或对每章滑动切多片（当前每章 2 片，可到 8-10 片，注意别重叠）。
"""


def export_llamafactory(
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:

    if dataset_dir is None:
        dataset_dir = DATA_DIR / "datasets" / "v2"
    if out_dir is None:
        out_dir = DATA_DIR / "llama_factory"

    dataset_out = out_dir / "dataset"
    dataset_out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ("train", "validation", "test"):
        src = dataset_dir / f"{split}.jsonl"
        if not src.is_file():
            raise FileNotFoundError(
                f"缺 {src}：先跑 quality-gate"
            )
        dst = dataset_out / f"sy_{split}.json"
        counts[split] = convert_split(src, dst)

    (out_dir / "dataset_info.json").write_text(
        json.dumps(
            DATASET_INFO, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    (out_dir / "qlora_sy.yaml").write_text(
        QLORA_TEMPLATE, encoding="utf-8"
    )
    (out_dir / "README_训练步骤.md").write_text(
        README_TRAIN, encoding="utf-8"
    )

    return {
        "exported_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "counts": counts,
        "out_dir": str(out_dir),
    }
