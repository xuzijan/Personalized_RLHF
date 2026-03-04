#!/usr/bin/env bash
# P-RLHF Mock 验证：完全离线跑通流程
# 使用: conda activate /root/autodl-tmp/envs/prlhf && bash scripts/validate_mock.sh
set -e
cd "$(dirname "$0")/.."

echo "=== P-RLHF Mock 验证 ==="
python scripts/validate_mock.py
echo "=== 完成 ==="
