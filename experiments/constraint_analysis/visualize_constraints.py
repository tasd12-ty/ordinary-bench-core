"""
约束图可视化与环分析。

对一个场景的 VLM 评测结果:
1. 绘制 GT 距离矩阵热力图
2. 绘制 VLM 约束 DAG (绿=正确, 红=错误, 粗=环中的边)
3. 检测并列出所有环
4. 输出场景摘要 JSON

用法:
    cd experiments/constraint_analysis

    # 单场景单模型
    uv run python visualize_constraints.py \
        --scene n10_000082 --model claude_opus_single_v2

    # 所有 n10 场景所有模型
    uv run python visualize_constraints.py --split n10
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VLM_TEST_DIR = PROJECT_ROOT / "VLM-test"
for p in [str(VLM_TEST_DIR / "API-test"), str(VLM_TEST_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── 数据加载 ──

def load_scene(scenes_dir: Path, scene_id: str) -> dict:
    with open(scenes_dir / f"{scene_id}.json") as f:
        return json.load(f)


def compute_gt_distances(scene: dict) -> dict:
    """计算所有 C(N,2) pairwise 3D 距离。"""
    objs = scene["objects"]
    dists = {}
    for a, b in combinations(range(len(objs)), 2):
        oa, ob = objs[a], objs[b]
        ca, cb = oa["3d_coords"], ob["3d_coords"]
        d = math.sqrt(sum((x - y) ** 2 for x, y in zip(ca, cb)))
        pair = (oa["id"], ob["id"])
        dists[pair] = d
    return dists


def load_vlm_results(results_dir: Path, model: str, scene_id: str) -> dict:
    p = results_dir / model / "scenes" / f"{scene_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_questions(questions_dir: Path, scene_id: str) -> list:
    """加载 QRR 问题 (v2 格式)。"""
    p = questions_dir / "qrr" / f"{scene_id}.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    qs = []
    for batch in data.get("batches", []):
        qs.extend(batch["questions"])
    return qs


def extract_qrr_constraints(vlm_result: dict, questions: list) -> list:
    """从 VLM 结果中提取所有 QRR 约束 (disjoint + shared_anchor)。"""
    per_q = vlm_result.get("scores", vlm_result).get("per_question", [])
    q_lookup = {q["qid"]: q for q in questions}

    constraints = []
    for pq in per_q:
        qid = pq["qid"]
        if not qid.startswith("qrr"):
            continue

        pred = pq.get("predicted")
        if pred not in ("<", ">", "~="):
            continue

        q = q_lookup.get(qid, {})
        pair1 = pq.get("pair1", q.get("pair1", []))
        pair2 = pq.get("pair2", q.get("pair2", []))
        gt = pq.get("gt", pq.get("gt_comparator", q.get("gt_comparator", "")))
        variant = pq.get("variant", q.get("variant", "disjoint"))

        if not pair1 or not pair2:
            continue

        constraints.append({
            "qid": qid,
            "variant": variant,
            "anchor": pq.get("anchor", q.get("anchor", "")),
            "pair1": pair1,
            "pair2": pair2,
            "gt_comparator": gt,
            "predicted": pred,
            "correct": pq.get("correct", False),
        })
    return constraints


# ── 图构建 ──

def build_distance_dag(constraints: list) -> tuple:
    """
    从 QRR 约束构建距离偏序 DAG。

    节点 = 规范化的 object pair，如 ("obj_0","obj_1")，代表距离 d(obj_0, obj_1)
    边 A→B 表示 d(A) < d(B) (A 是更短的距离)

    支持 disjoint 和 shared_anchor 两种 QRR:
    - disjoint: pair1=(A,B), pair2=(C,D), 比较 d(A,B) vs d(C,D)
    - shared_anchor: anchor=A, pair1=(A,B), pair2=(A,C), 比较 d(A,B) vs d(A,C)

    Returns: (G, edge_metadata)
    """
    G = nx.DiGraph()
    edge_meta = {}

    for c in constraints:
        pred = c["predicted"]
        if pred == "~=":
            continue

        # 规范化为距离节点 (排序 pair 保持一致)
        dist_1 = tuple(sorted(c["pair1"]))
        dist_2 = tuple(sorted(c["pair2"]))

        if pred == "<":
            src, dst = dist_1, dist_2  # d(pair1) < d(pair2)
        else:
            src, dst = dist_2, dist_1  # d(pair2) < d(pair1)

        if G.has_edge(src, dst):
            continue  # 避免重复边

        G.add_edge(src, dst)
        edge_meta[(src, dst)] = {
            "qid": c["qid"],
            "variant": c.get("variant", ""),
            "correct": c["correct"],
            "predicted": pred,
            "gt": c["gt_comparator"],
        }

    return G, edge_meta


def find_all_cycles(G: nx.DiGraph, max_cycles: int = 100) -> list:
    """找到所有简单环 (限制数量避免爆炸)。"""
    cycles = []
    try:
        for cycle in nx.simple_cycles(G):
            cycles.append(cycle)
            if len(cycles) >= max_cycles:
                break
    except Exception:
        pass
    return cycles


def compute_fas_edges(G: nx.DiGraph) -> set:
    """贪心 FAS: 返回需移除的边集合。"""
    H = G.copy()
    removed = set()
    while True:
        try:
            cycle = nx.find_cycle(H, orientation="original")
            # 找环中度最高的边移除
            best_edge = None
            best_score = -1
            for u, v, _ in cycle:
                score = H.in_degree(v) + H.out_degree(u)
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            if best_edge:
                H.remove_edge(*best_edge)
                removed.add(best_edge)
        except nx.NetworkXNoCycle:
            break
    return removed


# ── 可视化 ──

def plot_gt_distance_matrix(scene: dict, gt_dists: dict, output_path: Path):
    """绘制 GT 距离矩阵热力图。"""
    objs = [o["id"] for o in scene["objects"]]
    n = len(objs)
    matrix = np.zeros((n, n))

    for (a, b), d in gt_dists.items():
        i, j = objs.index(a), objs.index(b)
        matrix[i][j] = d
        matrix[j][i] = d

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(objs, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(objs, fontsize=8)

    # 在格子中标注距离值
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{matrix[i][j]:.1f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if matrix[i][j] > np.max(matrix) * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="3D Distance")
    ax.set_title(f"GT Distance Matrix — {scene['scene_id']} ({n} objects)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cycles_only(
    G: nx.DiGraph,
    edge_meta: dict,
    cycles: list,
    gt_dists: dict,
    scene_id: str,
    model: str,
    output_path: Path,
):
    """将所有环合并到一张有向图。

    箭头语义: A → B 表示 VLM 认为 d(A) > d(B)  (A 是更大的距离, 指向更小的)
    颜色:
      - 绿色有向边: VLM 回答正确 (与 GT 一致)
      - 红色有向边: VLM 回答错误 (与 GT 矛盾)
      - 黑色无向边: VLM 回答 ~= (约等于)
    """
    if not cycles:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No cycles detected", ha="center", va="center",
                fontsize=16, color="#2ecc71", transform=ax.transAxes)
        ax.set_title(f"{scene_id} / {model} — FAS=0", fontsize=11)
        ax.axis("off")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    # 收集所有环中涉及的边 (从完整 DAG 中取)
    cycle_nodes = set()
    cycle_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            cycle_nodes.add(u)
            cycle_nodes.add(v)
            cycle_edges.add((u, v))

    # 同时收集这些节点之间的 ~= 边 (从原始约束中)
    # ~= 边没有方向，不在 DAG 中，但对理解环很重要
    approx_edges = []  # (u, v) pairs for ~= answers between cycle nodes

    # 分类边
    directed_correct = []   # 绿色有向: VLM 对了
    directed_wrong = []     # 红色有向: VLM 错了
    edge_labels = {}

    for (u, v) in cycle_edges:
        meta = edge_meta.get((u, v), {})
        correct = meta.get("correct", True)
        pred = meta.get("predicted", "?")
        gt = meta.get("gt", "?")
        if correct:
            directed_correct.append((u, v))
        else:
            directed_wrong.append((u, v))
        edge_labels[(u, v)] = f"VLM:{pred}  GT:{gt}"

    # 统计
    n_correct = len(directed_correct)
    n_wrong = len(directed_wrong)
    n_approx = len(approx_edges)
    n_nodes = len(cycle_nodes)

    # 构建绘图用子图
    H = nx.DiGraph()
    for n in cycle_nodes:
        H.add_node(n)
    for e in cycle_edges:
        H.add_edge(*e)

    # 布局 — 放大画布，节点间距更宽
    size = max(12, min(n_nodes * 2.5, 30))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    if n_nodes <= 5:
        pos = nx.circular_layout(H, scale=2.0)
    else:
        pos = nx.kamada_kawai_layout(H, scale=2.5)

    # 节点标签
    node_labels = {}
    for n in cycle_nodes:
        pair = tuple(sorted(n))
        d = gt_dists.get(pair, gt_dists.get((pair[1], pair[0]), 0))
        node_labels[n] = f"d({n[0].replace('obj_','')},{n[1].replace('obj_','')})\n={d:.2f}"

    # 绘制节点 — 更大
    nx.draw_networkx_nodes(H, pos, node_color="white", node_size=3500,
                           edgecolors="#2c3e50", linewidths=2.5, ax=ax)
    nx.draw_networkx_labels(H, pos, node_labels, font_size=12,
                            font_weight="bold", ax=ax)

    # 绘制绿色有向边 (正确) — 更粗，箭头更大
    if directed_correct:
        nx.draw_networkx_edges(
            H, pos, edgelist=directed_correct,
            edge_color="#27ae60", width=4,
            arrows=True, arrowsize=40, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.15",
            min_source_margin=30, min_target_margin=30, ax=ax,
        )

    # 绘制红色有向边 (错误)
    if directed_wrong:
        nx.draw_networkx_edges(
            H, pos, edgelist=directed_wrong,
            edge_color="#c0392b", width=4,
            arrows=True, arrowsize=40, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.15",
            min_source_margin=30, min_target_margin=30, ax=ax,
        )

    # 绘制黑色无向边 (~= 约等于)
    if approx_edges:
        nx.draw_networkx_edges(
            H, pos, edgelist=approx_edges,
            edge_color="#2c3e50", width=2.5, style="dashed",
            arrows=False, ax=ax,
        )

    # 边标签 — 更大字体
    nx.draw_networkx_edge_labels(H, pos, edge_labels, font_size=10,
                                 font_color="#333333", ax=ax,
                                 label_pos=0.35, bbox=dict(
                                     boxstyle="round,pad=0.15",
                                     facecolor="white", edgecolor="none",
                                     alpha=0.8))

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#27ae60", linewidth=2.5, label=f"Correct ({n_correct})",
               marker=">", markersize=8, markeredgecolor="#27ae60"),
        Line2D([0], [0], color="#c0392b", linewidth=2.5, label=f"Wrong ({n_wrong})",
               marker=">", markersize=8, markeredgecolor="#c0392b"),
        Line2D([0], [0], color="#2c3e50", linewidth=1.5, linestyle="dashed",
               label=f"Approx equal ({n_approx})"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", fontsize=11,
              frameon=True, edgecolor="#2c3e50", ncol=3)

    ax.set_title(
        f"Cycle Subgraph — {scene_id} / {model}\n"
        f"{len(cycles)} cycles, {n_nodes} nodes, "
        f"{len(cycle_edges)} directed + {n_approx} approx edges "
        f"(of {G.number_of_edges()} total)\n"
        f"Arrow: A → B means VLM thinks d(A) > d(B)",
        fontsize=12, fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── 主流程 ──

def analyze_scene(
    scene_id: str,
    model: str,
    scenes_dir: Path,
    results_dir: Path,
    questions_dir: Path,
    output_dir: Path,
) -> dict:
    """分析单个场景 + 模型，输出可视化和统计。"""
    scene = load_scene(scenes_dir, scene_id)
    vlm_result = load_vlm_results(results_dir, model, scene_id)
    if not vlm_result:
        return {"scene_id": scene_id, "model": model, "status": "no_results"}

    questions = load_questions(questions_dir, scene_id)
    gt_dists = compute_gt_distances(scene)
    constraints = extract_qrr_constraints(vlm_result, questions)

    if not constraints:
        return {"scene_id": scene_id, "model": model, "status": "no_constraints"}

    # 构建 DAG
    G, edge_meta = build_distance_dag(constraints)
    fas_edges = compute_fas_edges(G)
    cycles = find_all_cycles(G, max_cycles=50)

    # 统计
    n_correct = sum(1 for c in constraints if c["correct"])
    n_wrong = sum(1 for c in constraints if not c["correct"] and c["predicted"] != "~=")
    n_approx = sum(1 for c in constraints if c["predicted"] == "~=")

    # 环中涉及的 qid
    cycle_edge_set = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            if G.has_edge(u, v):
                cycle_edge_set.add((u, v))

    cycle_qids = []
    for e in cycle_edge_set:
        meta = edge_meta.get(e, {})
        if meta.get("qid"):
            cycle_qids.append(meta["qid"])

    # 输出目录
    out = output_dir / model / scene_id
    out.mkdir(parents=True, exist_ok=True)

    # 1. GT 距离矩阵
    plot_gt_distance_matrix(scene, gt_dists, out / "gt_distance_matrix.png")

    # 2. 环可视化 (只展示成环的部分)
    plot_cycles_only(G, edge_meta, cycles, gt_dists, scene_id, model, out / "cycles.png")

    # 3. 环列表
    cycles_data = []
    for i, cycle in enumerate(cycles):
        edges_in_cycle = []
        for j in range(len(cycle)):
            u, v = cycle[j], cycle[(j + 1) % len(cycle)]
            meta = edge_meta.get((u, v), {})
            edges_in_cycle.append({
                "from": f"d({u[0]},{u[1]})",
                "to": f"d({v[0]},{v[1]})",
                "qid": meta.get("qid", ""),
                "predicted": meta.get("predicted", ""),
                "gt": meta.get("gt", ""),
                "correct": meta.get("correct", None),
            })
        cycles_data.append({
            "cycle_id": i,
            "length": len(cycle),
            "edges": edges_in_cycle,
        })
    with open(out / "cycles.json", "w") as f:
        json.dump(cycles_data, f, indent=2, ensure_ascii=False)

    # 4. 摘要
    summary = {
        "scene_id": scene_id,
        "model": model,
        "n_objects": len(scene["objects"]),
        "n_pairs": len(gt_dists),
        "n_qrr_constraints": len(constraints),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_approx_equal": n_approx,
        "accuracy": round(n_correct / (n_correct + n_wrong), 4) if (n_correct + n_wrong) else 0,
        "n_dag_nodes": G.number_of_nodes(),
        "n_dag_edges": G.number_of_edges(),
        "n_cycles": len(cycles),
        "fas_size": len(fas_edges),
        "cycle_qids": sorted(set(cycle_qids)),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="约束图可视化与环分析")
    parser.add_argument("--scene", default=None, help="单场景 ID")
    parser.add_argument("--model", default=None, help="单模型名称")
    parser.add_argument("--split", default=None, help="处理该 split 的所有场景 (如 n10)")
    parser.add_argument("--scenes-dir", default="../../datasets/test-data/scenes")
    parser.add_argument("--results-dir", default="../../VLM-test/output/results")
    parser.add_argument("--questions-dir", default="../../datasets/test-data/questions")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir)
    results_dir = Path(args.results_dir)
    questions_dir = Path(args.questions_dir)
    output_dir = Path(args.output_dir)

    # 发现模型
    if args.model:
        models = [args.model]
    else:
        models = sorted([
            d.name for d in results_dir.iterdir()
            if d.is_dir() and (d / "scenes").is_dir()
        ])

    # 发现场景
    if args.scene:
        scene_ids = [args.scene]
    elif args.split:
        scene_ids = sorted([
            f.stem for f in scenes_dir.glob(f"{args.split}_*.json")
        ])
    else:
        scene_ids = sorted([f.stem for f in scenes_dir.glob("*.json")])

    print(f"Models: {len(models)}, Scenes: {len(scene_ids)}")

    all_summaries = []
    for model in models:
        for scene_id in scene_ids:
            print(f"  {model} / {scene_id}...", end=" ")
            summary = analyze_scene(
                scene_id, model, scenes_dir, results_dir, questions_dir, output_dir
            )
            status = summary.get("status", "ok")
            if status != "ok" and "fas_size" not in summary:
                print(f"[{status}]")
                continue
            print(f"FAS={summary.get('fas_size', '?')} "
                  f"cycles={summary.get('n_cycles', '?')} "
                  f"acc={summary.get('accuracy', '?')}")
            all_summaries.append(summary)

    # 跨模型汇总
    if all_summaries:
        cross = defaultdict(list)
        for s in all_summaries:
            cross[s["model"]].append(s)

        cross_summary = {}
        print(f"\n{'Model':<40} {'Scenes':>6} {'Avg FAS':>8} {'Avg Cycles':>10} {'Avg Acc':>8}")
        print("-" * 76)
        for model in sorted(cross):
            ss = cross[model]
            avg_fas = sum(s["fas_size"] for s in ss) / len(ss)
            avg_cyc = sum(s["n_cycles"] for s in ss) / len(ss)
            avg_acc = sum(s["accuracy"] for s in ss) / len(ss)
            print(f"{model:<40} {len(ss):>6} {avg_fas:>8.1f} {avg_cyc:>10.1f} {avg_acc:>8.4f}")
            cross_summary[model] = {
                "n_scenes": len(ss),
                "avg_fas_size": round(avg_fas, 2),
                "avg_n_cycles": round(avg_cyc, 2),
                "avg_accuracy": round(avg_acc, 4),
            }

        with open(output_dir / "cross_model_summary.json", "w") as f:
            json.dump(cross_summary, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
