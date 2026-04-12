#!/usr/bin/env bash
# 约束扰动实验：全量运行 + 分析
# 用法: bash run_all.sh
set -euo pipefail

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "=========================================="
echo " 约束扰动实验 (Null Model)"
echo "=========================================="
echo ""

# Step 1: 全量重建
echo "[Step 1/2] Running perturbation experiment (140 scenes × 9 fractions × R=20)..."
echo "  Output: results/perturbation_results.jsonl"
echo "  Supports Ctrl+C resume."
echo ""

python run_experiment.py \
    --workers 8 \
    --repeats 20 \
    --restarts 5

echo ""
echo "[Step 2/2] Analyzing results..."
echo ""

# Step 2: 分析 + 绘图
python analyze_results.py

echo ""
echo "=========================================="
echo " Done! Check results/ for figures and data."
echo "=========================================="
