"""
符号预处理：P_dist（QRR 偏序 DAG）+ P_ang（TRR 弧区间）+ 超图分析。

重建管线的阶段 0-1：
  - 从 QRR 约束构建距离偏序
  - 从 TRR 约束构建角度扇区系统
  - 可行性检查（环检测、弧交集、连通性）
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

from .utils import UnionFind, normalize_angle, pair_key


# ── QRR 约束表示 ──

@dataclass
class QRREntry:
    """单个 QRR 序数约束：pair1 <op> pair2。"""
    pair1: Tuple[str, str]
    pair2: Tuple[str, str]
    comparator: str  # "<", "~=", ">"
    weight: float = 1.0
    variant: str = "disjoint"
    anchor: Optional[str] = None


@dataclass
class TRREntry:
    """单个 TRR 角度约束。"""
    target: str
    ref1: str
    ref2: str
    hour: int
    weight: float = 1.0
    level: str = "hour"  # "hour" 或 "quadrant"


@dataclass
class FDREntry:
    """单个 FDR 排序约束：从锚点出发的全距离排序。"""
    anchor: str
    ranking: List[str]  # 从最近到最远
    weight: float = 1.0


# ── P_dist：距离偏序 DAG ──

@dataclass
class DistancePoset:
    """QRR 约束组织为对象对上的偏序 DAG。

    节点 = 对象对的等价类（通过 ~= 合并）。
    边 = 等价类之间的严格排序（< / >）。
    """
    equiv_classes: UnionFind = field(default_factory=UnionFind)
    # 有向边：class_rep -> 严格更大的 class_rep 集合
    edges_lt: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    # 涉及的所有对象对
    all_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    # 用于损失计算的原始约束
    constraints: List[QRREntry] = field(default_factory=list)
    # 检测到的问题
    has_cycle: bool = False
    cycle_info: Optional[str] = None

    def _pair_id(self, pair: Tuple[str, str]) -> str:
        return f"{pair[0]}_{pair[1]}"


def build_distance_poset(qrr_entries: List[QRREntry]) -> DistancePoset:
    """从 QRR 约束构建 P_dist。

    步骤：
      1. 并查集合并所有 ~= 对
      2. 为 < / > 构建 DAG 边
      3. 检测环
    """
    poset = DistancePoset()
    poset.constraints = list(qrr_entries)

    for entry in qrr_entries:
        p1 = pair_key(*entry.pair1)
        p2 = pair_key(*entry.pair2)
        poset.all_pairs.add(p1)
        poset.all_pairs.add(p2)
        id1 = poset._pair_id(p1)
        id2 = poset._pair_id(p2)

        if entry.comparator == "~=":
            poset.equiv_classes.union(id1, id2)
        elif entry.comparator == "<":
            # d(pair1) < d(pair2) => pair1 "小于" pair2
            r1 = poset.equiv_classes.find(id1)
            r2 = poset.equiv_classes.find(id2)
            poset.edges_lt[r1].add(r2)
        elif entry.comparator == ">":
            r1 = poset.equiv_classes.find(id1)
            r2 = poset.equiv_classes.find(id2)
            poset.edges_lt[r2].add(r1)

    # 使用最终等价类代表元重建边
    new_edges: Dict[str, Set[str]] = defaultdict(set)
    for src, dsts in poset.edges_lt.items():
        rs = poset.equiv_classes.find(src)
        for dst in dsts:
            rd = poset.equiv_classes.find(dst)
            if rs != rd:
                new_edges[rs].add(rd)
    poset.edges_lt = new_edges

    # 通过 DFS 检测环
    poset.has_cycle, poset.cycle_info = _detect_cycle(poset.edges_lt)

    return poset


def _detect_cycle(edges: Dict[str, Set[str]]) -> Tuple[bool, Optional[str]]:
    """使用 DFS 着色法检测有向图中的环。"""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = defaultdict(int)

    all_nodes: Set[str] = set()
    for src, dsts in edges.items():
        all_nodes.add(src)
        all_nodes.update(dsts)

    def dfs(node: str) -> Optional[str]:
        color[node] = GRAY
        for nb in edges.get(node, set()):
            if color[nb] == GRAY:
                return f"cycle involving {node} -> {nb}"
            if color[nb] == WHITE:
                result = dfs(nb)
                if result:
                    return result
        color[node] = BLACK
        return None

    for node in all_nodes:
        if color[node] == WHITE:
            result = dfs(node)
            if result:
                return True, result

    return False, None


# ── P_ang：角度扇区系统 ──

@dataclass
class ArcInterval:
    """角弧区间 [center - half_width, center + half_width]。"""
    center_deg: float
    half_width_deg: float
    weight: float = 1.0
    level: str = "hour"


@dataclass
class AngularSectorSystem:
    """TRR 约束组织为角度扇区。

    对于每个 (target, ref1, ref2) 三元组，存储一组弧区间。
    同一三元组的多个区间取交集可缩小可行弧范围。
    """
    # 键：(target, ref1, ref2) -> ArcInterval 列表
    sectors: Dict[Tuple[str, str, str], List[ArcInterval]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # 提及的所有对象
    all_objects: Set[str] = field(default_factory=set)
    # 原始约束
    constraints: List[TRREntry] = field(default_factory=list)
    # 检测到的冲突
    conflicts: List[Tuple[str, str, str]] = field(default_factory=list)


def build_angular_sectors(trr_entries: List[TRREntry]) -> AngularSectorSystem:
    """从 TRR 约束构建 P_ang。

    每个 TRR 约束映射为一个弧区间：
      - hour_correct: center = hour*30, half_width = 15 度
      - quadrant_correct: center = quadrant*90 - 45, half_width = 45 度
    """
    system = AngularSectorSystem()
    system.constraints = list(trr_entries)

    for entry in trr_entries:
        system.all_objects.update([entry.target, entry.ref1, entry.ref2])
        key = (entry.target, entry.ref1, entry.ref2)

        if entry.level == "hour":
            center = (entry.hour % 12) * 30.0
            half_width = 15.0
        else:  # quadrant
            quadrant = _hour_to_quadrant(entry.hour)
            center = (quadrant - 1) * 90.0 + 45.0
            half_width = 45.0

        system.sectors[key].append(ArcInterval(
            center_deg=center,
            half_width_deg=half_width,
            weight=entry.weight,
            level=entry.level,
        ))

    # 检查冲突（弧交集为空）
    for key, arcs in system.sectors.items():
        if len(arcs) > 1 and not _arcs_compatible(arcs):
            system.conflicts.append(key)

    return system


def _hour_to_quadrant(hour: int) -> int:
    if hour in (12, 1, 2):
        return 1
    elif hour in (3, 4, 5):
        return 2
    elif hour in (6, 7, 8):
        return 3
    else:
        return 4


def _arcs_compatible(arcs: List[ArcInterval]) -> bool:
    """检查弧区间的交集是否非空。

    逐对检查：对于每对弧，验证其角距是否小于两者半宽之和。
    这是充分的，因为所有弧的宽度均 < 180°（象限级别最大 90°），
    根据 S^1 上的 Helly 定理，逐对相交意味着宽度 < 180° 的弧
    的全局交集非空。
    """
    for i in range(len(arcs)):
        for j in range(i + 1, len(arcs)):
            dist = _angular_dist(arcs[i].center_deg, arcs[j].center_deg)
            max_gap = arcs[i].half_width_deg + arcs[j].half_width_deg
            if dist > max_gap:
                return False
    return True


def _angular_dist(a: float, b: float) -> float:
    diff = abs(normalize_angle(a) - normalize_angle(b))
    return min(diff, 360 - diff)


# ── 超图连通性 ──

@dataclass
class HypergraphInfo:
    """约束超图的连通性分析。"""
    n_objects: int = 0
    n_qrr_constraints: int = 0
    n_trr_constraints: int = 0
    is_connected: bool = False
    n_components: int = 0
    components: List[Set[str]] = field(default_factory=list)
    object_participation: Dict[str, int] = field(default_factory=dict)


def analyze_hypergraph(
    qrr_entries: List[QRREntry],
    trr_entries: List[TRREntry],
    object_ids: List[str],
) -> HypergraphInfo:
    """分析约束超图的连通性。

    共享约束的对象视为连通。
    """
    uf = UnionFind()
    participation: Dict[str, int] = defaultdict(int)

    # 初始化所有对象
    for oid in object_ids:
        uf.find(oid)

    # QRR：连接被比较对引用的所有唯一对象。
    for entry in qrr_entries:
        objs = sorted(set(entry.pair1) | set(entry.pair2))
        for obj in objs:
            participation[obj] += 1
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                uf.union(objs[i], objs[j])

    # TRR：每个约束连接 3 个对象
    for entry in trr_entries:
        objs = [entry.target, entry.ref1, entry.ref2]
        for obj in objs:
            participation[obj] += 1
        uf.union(objs[0], objs[1])
        uf.union(objs[1], objs[2])

    groups = uf.groups()
    components = [set(members) for members in groups.values()]

    info = HypergraphInfo(
        n_objects=len(object_ids),
        n_qrr_constraints=len(qrr_entries),
        n_trr_constraints=len(trr_entries),
        is_connected=len(components) <= 1,
        n_components=len(components),
        components=components,
        object_participation=dict(participation),
    )
    return info


# ── 从评分结果中提取约束 ──

def extract_qrr_from_scoring(
    per_question: List[dict],
    questions: List[dict],
    use_correct_only: bool = True,
) -> List[QRREntry]:
    """从评分结果中提取 QRR 约束。

    参数:
        per_question: score_batch_scene() 的 per_question 列表
        questions: 带有真值数据的原始问题列表
        use_correct_only: 若为 True，仅使用 VLM 的正确答案；
                         若为 False，使用 VLM 的预测答案（用于信念重建）
    """
    # 构建问题查找表
    q_lookup = {q["qid"]: q for q in questions}
    entries = []

    for pq in per_question:
        if pq["type"] != "qrr":
            continue
        qid = pq["qid"]
        q = q_lookup.get(qid)
        if q is None:
            continue

        predicted = pq.get("predicted")
        if predicted is None:
            continue

        if use_correct_only and not pq.get("correct", False):
            continue

        # 使��� VLM 的预测比较器进行信念重建
        comparator = str(predicted) if not use_correct_only else q["gt_comparator"]
        if comparator not in {"<", "~=", ">"}:
            continue

        entries.append(QRREntry(
            pair1=tuple(q["pair1"]),
            pair2=tuple(q["pair2"]),
            comparator=comparator,
            weight=1.0,
            variant=q.get("variant", "disjoint"),
            anchor=q.get("anchor"),
        ))

    return entries


def extract_trr_from_scoring(
    per_question: List[dict],
    questions: List[dict],
    use_correct_only: bool = True,
) -> List[TRREntry]:
    """从评分结果中提取 TRR ��束。

    使用层级选择：
      - hour_correct → level="hour"（最紧）
      - quadrant_correct → level="quadrant"（较松）
      - 否则 → 跳过（当 use_correct_only=True 时）
    """
    q_lookup = {q["qid"]: q for q in questions}
    entries = []

    for pq in per_question:
        if pq["type"] != "trr":
            continue
        qid = pq["qid"]
        q = q_lookup.get(qid)
        if q is None:
            continue

        predicted = pq.get("predicted")
        if predicted is None or predicted == -1:
            continue

        if use_correct_only:
            if pq.get("hour_correct", False):
                entries.append(TRREntry(
                    target=q["target"],
                    ref1=q["ref1"],
                    ref2=q["ref2"],
                    hour=q["gt_hour"],
                    weight=1.0,
                    level="hour",
                ))
            elif pq.get("quadrant_correct", False):
                entries.append(TRREntry(
                    target=q["target"],
                    ref1=q["ref1"],
                    ref2=q["ref2"],
                    hour=q["gt_hour"],
                    weight=0.5,
                    level="quadrant",
                ))
        else:
            # 使用 VLM 的预测小时进行信念重建
            try:
                pred_hour = int(predicted)
                if 1 <= pred_hour <= 12:
                    entries.append(TRREntry(
                        target=q["target"],
                        ref1=q["ref1"],
                        ref2=q["ref2"],
                        hour=pred_hour,
                        weight=1.0,
                        level="hour",
                    ))
            except (ValueError, TypeError):
                pass

    return entries


def extract_fdr_from_scoring(
    per_question: List[dict],
    questions: List[dict],
    use_correct_only: bool = True,
) -> List[FDREntry]:
    """从评分结果中提取 FDR 约束。"""
    q_lookup = {q["qid"]: q for q in questions}
    entries = []

    for pq in per_question:
        if pq["type"] != "fdr":
            continue
        qid = pq["qid"]
        q = q_lookup.get(qid)
        if q is None:
            continue

        if use_correct_only:
            if pq.get("pairwise_accuracy", 0.0) < 0.5:
                continue
            ranking = q["gt_ranking"]
        else:
            predicted = pq.get("predicted", [])
            if not isinstance(predicted, list):
                continue
            allowed = set(q.get("gt_ranking", []))
            ranking = []
            seen = set()
            for item in predicted:
                if not isinstance(item, str):
                    continue
                if item == q["anchor"] or item not in allowed or item in seen:
                    continue
                ranking.append(item)
                seen.add(item)
            if len(ranking) < 2:
                continue

        entries.append(FDREntry(
            anchor=q["anchor"],
            ranking=ranking,
            weight=1.0,
        ))

    return entries


def decompose_fdr_to_qrr(fdr_entries: List[FDREntry]) -> List[QRREntry]:
    """将 FDR 排序分解为等价的 QRR 成对约束。

    对于锚点 A 和排序 [B, C, D]：
      dist(A,B) < dist(A,C)  =>  QRR: (A,B) < (A,C)
      dist(A,B) < dist(A,D)  =>  QRR: (A,B) < (A,D)
      dist(A,C) < dist(A,D)  =>  QRR: (A,C) < (A,D)
    """
    qrr_entries = []
    for fdr in fdr_entries:
        anchor = fdr.anchor
        n = len(fdr.ranking)
        n_pairs = n * (n - 1) // 2 if n >= 2 else 1
        w = fdr.weight / n_pairs
        for i in range(n):
            for j in range(i + 1, n):
                nearer = fdr.ranking[i]
                farther = fdr.ranking[j]
                qrr_entries.append(QRREntry(
                    pair1=tuple(sorted((anchor, nearer))),
                    pair2=tuple(sorted((anchor, farther))),
                    comparator="<",
                    weight=w,
                    variant="shared_anchor",
                    anchor=anchor,
                ))
    return qrr_entries


# ── FDR 与 QRR 冲突检测 ──

_OPPOSITE = {"<": ">", ">": "<"}


def detect_fdr_qrr_conflicts(
    qrr_direct: List[QRREntry],
    qrr_from_fdr: List[QRREntry],
) -> dict:
    """检测直接 QRR 与 FDR 派生 QRR 约束之间的矛盾。

    同一 (pair1, pair2) 上的两个约束可能冲突：
      - 矛盾："\<" vs ">"（不可能同时满足）
      - 弱冲突："\<" vs "~=" 或 ">" vs "~="（有张力但非不可能）

    返回包含计数、一致性比率和冲突详情的字典。
    """
    # 按规范化的对象对键索引直接 QRR
    direct_index: Dict[Tuple, List[QRREntry]] = defaultdict(list)
    for entry in qrr_direct:
        key = (pair_key(*entry.pair1), pair_key(*entry.pair2))
        direct_index[key].append(entry)

    n_overlapping = 0
    n_consistent = 0
    n_contradictory = 0
    n_weak_conflict = 0
    conflicts = []

    for fdr_entry in qrr_from_fdr:
        key = (pair_key(*fdr_entry.pair1), pair_key(*fdr_entry.pair2))
        if key not in direct_index:
            continue

        for direct_entry in direct_index[key]:
            n_overlapping += 1
            fc = fdr_entry.comparator
            dc = direct_entry.comparator

            if fc == dc:
                n_consistent += 1
            elif fc == _OPPOSITE.get(dc):
                n_contradictory += 1
                conflicts.append({
                    "pair1": list(fdr_entry.pair1),
                    "pair2": list(fdr_entry.pair2),
                    "qrr_comparator": dc,
                    "fdr_comparator": fc,
                    "conflict_type": "contradictory",
                })
            else:
                n_weak_conflict += 1
                conflicts.append({
                    "pair1": list(fdr_entry.pair1),
                    "pair2": list(fdr_entry.pair2),
                    "qrr_comparator": dc,
                    "fdr_comparator": fc,
                    "conflict_type": "weak",
                })

    return {
        "n_overlapping": n_overlapping,
        "n_consistent": n_consistent,
        "n_contradictory": n_contradictory,
        "n_weak_conflict": n_weak_conflict,
        "consistency_rate": n_consistent / n_overlapping if n_overlapping else 1.0,
        "conflicts": conflicts,
    }


# ── 可行性汇总 ──

@dataclass
class FeasibilityReport:
    """符号可行性检查汇总。"""
    qrr_has_cycle: bool = False
    qrr_cycle_info: Optional[str] = None
    trr_n_conflicts: int = 0
    trr_conflict_keys: List[Tuple[str, str, str]] = field(default_factory=list)
    hypergraph_connected: bool = False
    n_components: int = 0
    n_qrr: int = 0
    n_trr: int = 0
    n_objects: int = 0


def check_feasibility(
    poset: DistancePoset,
    sectors: AngularSectorSystem,
    hyper: HypergraphInfo,
) -> FeasibilityReport:
    """运行所有符号可行性检查。"""
    return FeasibilityReport(
        qrr_has_cycle=poset.has_cycle,
        qrr_cycle_info=poset.cycle_info,
        trr_n_conflicts=len(sectors.conflicts),
        trr_conflict_keys=sectors.conflicts,
        hypergraph_connected=hyper.is_connected,
        n_components=hyper.n_components,
        n_qrr=len(poset.constraints),
        n_trr=len(sectors.constraints),
        n_objects=hyper.n_objects,
    )
