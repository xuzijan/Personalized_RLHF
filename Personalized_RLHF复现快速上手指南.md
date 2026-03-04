# Personalized_RLHF (P-RLHF) 复现快速上手指南

> 按 5 步递进：Fork → 环境隔离 → 跑通示例 → 整理 I/O 与配置 → 统一评估与 AmbiCoding

---

## 论文背景

**论文**：*Personalized Language Modeling from Personalized Human Feedback* (arXiv:2402.05133)  
**机构**：HumainLab  
**核心思想**：在 RLHF 框架下引入用户模型（user model），使用个性化偏好数据微调 LLM，支持 P-DPO 等个性化偏好优化算法。

**评估场景**：TLDR 摘要、PSOUPS 个性化问答、PRISM 对话偏好  

**当前项目**：`xuzijan/Personalized_RLHF` 是 `HumainLab/Personalized_RLHF` 的 fork。

---

## Step 1：Fork 到自己仓库

- [x] Fork `HumainLab/Personalized_RLHF` → `xuzijan/Personalized_RLHF`（或你的 GitHub 用户名）
- [ ] 在 README 或 commit 中注明 fork 来源与对应 commit
- [ ] `.gitignore` 排除大文件（模型权重、LoRA、wandb 等），只同步代码和配置

**建议 .gitignore 新增：**

```gitignore
*.safetensors
*.bin
*.pt
*.pth
wandb/
dpo/
outputs/
.cache/
```

---

## Step 2：环境隔离

每个 baseline 单独环境，避免依赖冲突。

**目录结构：**

```
/root/autodl-tmp/
├── Personalized_RLHF/         # 论文 5
│   ├── data/                 # 数据预处理、mock 数据
│   ├── prlhf/                # 核心实现
│   ├── scripts/              # 训练、生成、验证脚本
│   └── evaluate/             # Win-rate 评估
├── envs/
│   └── prlhf/                # 独立 conda 环境（数据盘）
└── experiments/
    └── configs/
        └── prlhf_baseline.yaml   # 统一 baseline 配置（已创建）
```

**P-RLHF 环境：**

```bash
# 创建环境（建议建在数据盘，避免系统盘空间不足）
conda create --prefix /root/autodl-tmp/envs/prlhf python=3.11 -y
conda activate /root/autodl-tmp/envs/prlhf

# pip 缓存指向数据盘（系统盘紧张时）
export PIP_CACHE_DIR=/root/autodl-tmp/.pip_cache TMPDIR=/root/autodl-tmp/.tmp

# PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# 其余依赖
pip install datasets==2.21.0 transformers==4.44.2 trl==0.10.1 peft==0.12.0 wandb==0.17.9 pydantic==2.9.0 pandas accelerate rich
```

**说明**：README 推荐 `--index-url https://download.pytorch.org/whl/cu118` 安装 CUDA 11.8 版；默认 PyPI 的 torch 含 CUDA 12，通常可直接使用。

**验证**：

```bash
conda activate /root/autodl-tmp/envs/prlhf
cd /root/autodl-tmp/Personalized_RLHF
python -c "import torch, datasets, transformers, trl, peft; print('OK')"
```

---

## Step 3：按原作者示例跑通一次

### 3.1 Mock 验证（无需网络、无需模型）

```bash
cd /root/autodl-tmp/Personalized_RLHF
conda activate /root/autodl-tmp/envs/prlhf
python scripts/validate_mock.py
# 或
bash scripts/validate_mock.sh
```

验证内容：环境导入、项目模块、Mock 数据加载与 DPO 格式转换。

### 3.2 真实训练（需网络下载模型）

```bash
cd /root/autodl-tmp/Personalized_RLHF
conda activate /root/autodl-tmp/envs/prlhf
WANDB_MODE=disabled accelerate launch prlhf/train_language_model_dpo.py \
  --dataset psoups \
  --downloads_data_path ./data/mock_psoups_mini.json \
  --model_class gptneo \
  --model_name EleutherAI/gpt-neo-125m \
  --tokenizer_name EleutherAI/gpt-neo-125m \
  --user_model individual \
  --sanity_check True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --output_dir ./output_mock \
  --report_to none
```

**使用 YAML 配置**：`experiments/configs/prlhf_baseline.yaml` 已创建，可被 `run_baseline` 或自定义脚本读取并映射为上述命令行参数。

### 3.3 数据准备

数据在 `data/` 下，支持 TLDR、PSOUPS、PRISM。Mock 数据：`data/mock_psoups_mini.json`。完整数据见 `data/` 内 `tldr_dataset.ipynb`、`psoups_dataset.ipynb`、`prism_dataset.ipynb`。

### 3.4 生成

```bash
accelerate launch prlhf/generate.py \
  --user_model individual \
  --lora_checkpoint $LORA_CKPT_PATH \
  --output_dir $GENERATION_OUTPUT_DIR \
  --model_name EleutherAI/gpt-neo-125m \
  --model_class gptneo \
  --dataset $DATASET_NAME
```

### 3.5 评估

Win-rate 评估见 `evaluate/`。

---

## Step 4：整理 I/O 与配置

- [x] 将数据路径、模型路径、输出路径抽到配置文件
- [x] 在 `experiments/configs/` 下新增 `prlhf_baseline.yaml`，与其它 baseline 对齐

**配置路径**：`experiments/configs/prlhf_baseline.yaml`

**主要字段**：`experiment`、`model`、`training`、`peft`、`data`、`output`。参数映射说明见配置文件头部注释。

---

## Step 5：统一评估与 AmbiCoding

- [ ] 与 MemGPT、PersonaRAG、DPO、LD-Agent 等 baseline 统一评估流程
- [ ] 对接 AmbiCoding 框架（若适用）

---

## 常见问题

| 问题 | 排查 |
|------|------|
| 系统盘空间不足 | 环境建在数据盘 `--prefix`，设置 `PIP_CACHE_DIR`、`TMPDIR` |
| iJIT_NotifyEvent 报错 | conda MKL 与 PyTorch 冲突，改用 pip 安装 torch |
| 缺少 pyyaml / requests | `pip install pyyaml requests` |
| 缺少 rich | `pip install rich` |
| 网络不可达 | 先跑 `validate_mock.py` 验证；有网络后再完整训练 |

---

## 参考链接

- [P-RLHF 论文](https://arxiv.org/abs/2402.05133)
- [HumainLab/Personalized_RLHF](https://github.com/HumainLab/Personalized_RLHF)
- [TLDR 数据集](https://huggingface.co/datasets/openai/summarize_from_feedback)
- [PRISM 数据集](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment)
