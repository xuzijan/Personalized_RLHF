#!/usr/bin/env python3
"""
P-RLHF Mock 验证：完全离线跑通流程
验证环境、数据加载、核心模块，不依赖 HuggingFace 模型下载
"""
import os
import sys

root = os.path.join(os.path.dirname(__file__), "..")
prlhf_dir = os.path.join(root, "prlhf")
sys.path.insert(0, root)
sys.path.insert(0, prlhf_dir)


def main():
    print("=== P-RLHF Mock 验证 ===\n")

    # 1. 环境与核心导入
    print("1. 验证核心依赖...")
    import torch
    import datasets
    import transformers
    import trl
    import peft
    import accelerate
    print(f"   torch {torch.__version__}, cuda={torch.cuda.is_available()}")

    # 2. 项目模块导入
    print("2. 验证项目模块...")
    from user_model import IndividualUserModel, ClusterUserModel
    from utils import load_psoups_comparisons, build_psoups_dataset_dpo
    from user_dpo_trainer import UserDPOTrainer
    from user_language_model import UserGPTNeoForCausalLM
    print("   导入成功")

    # 3. Mock 数据加载与 DPO 格式转换
    print("3. 验证 Mock 数据加载...")
    data_path = os.path.join(root, "data", "mock_psoups_mini.json")
    if not os.path.exists(data_path):
        print(f"   错误: Mock 数据不存在 {data_path}")
        return 1

    train_ds, eval_ds, n_users = load_psoups_comparisons(
        downloads_data_path=data_path,
        sanity_check=False,  # mock 数据少，不截断
        num_proc=1,
    )
    print(f"   train={len(train_ds)}, eval={len(eval_ds)}, n_users={n_users}")
    if len(train_ds) == 0:
        print("   错误: train 为空")
        return 1

    # 4. 检查数据格式
    ex = train_ds[0]
    assert "prompt" in ex and "chosen" in ex and "rejected" in ex
    print(f"   样本格式 OK: prompt_len={len(ex['prompt'])}, chosen_len={len(ex['chosen'])}")

    print("\n=== Mock 验证完成 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
