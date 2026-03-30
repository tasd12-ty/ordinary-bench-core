"""
迭代冲突消解 (Iterative Conflict Resolution) Pipeline
=====================================================

目的：区分 VLM 空间推理中的**随机噪声**与**系统性错误**。

原理：
    当 VLM 回答大量空间关系问题时，即使正确率 80%，
    偶然错误也会在约束图中产生传递性违反（环 / cycle）。
    本模块通过迭代重问冲突题目来消除噪声：

    1. 从已有评测结果中提取 QRR 约束图
    2. 计算最小反馈弧集 (Minimum Feedback Arc Set, FAS)
       —— 即使图无环所需移除的最少约束
    3. 将 FAS 中的约束映射回原始问题
    4. 调用 VLM API 重新回答这些冲突问题
    5. 用新答案替换旧答案，重新检测冲突
    6. 重复 3-5 直到收敛（FAS 连续不减小）

    收敛后：
    - 被翻转的答案 = 随机噪声引起的偶然错误
    - 剩余冲突    = 模型的系统性误解（真实能力上限）
    - 收敛轮数    = 模型回答稳定性指标

模块结构：
    fas.py              — 反馈弧集算法（贪心近似）
    conflict_detector.py — 冲突检测 + 问题溯源
    vlm_requester.py    — 调 VLM API 重问冲突题
    resolver.py         — 迭代消解主循环
    report.py           — 收敛报告生成

用法：
    # 仅检测冲突（不调 API）
    python run_conflict_resolution.py --job API-test/jobs/conflict_resolution_gemini.toml --dry-run

    # 执行迭代消解
    python run_conflict_resolution.py --job API-test/jobs/conflict_resolution_gemini.toml
"""
