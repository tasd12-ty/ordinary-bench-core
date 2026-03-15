"""
Symbolic preprocessing: P_dist (QRR poset DAG) + P_ang (TRR arc intervals) + hypergraph analysis.

Stage 0-1 of the reconstruction pipeline:
  - Build distance poset from QRR constraints
  - Build angular sector system from TRR constraints
  - Check feasibility (cycle detection, arc intersection, connectivity)
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

from .utils import UnionFind, normalize_angle, pair_key


# ── QRR Constraint Representation ──

@dataclass
class QRREntry:
    """A single QRR ordinal constraint: pair1 <op> pair2."""
    pair1: Tuple[str, str]
    pair2: Tuple[str, str]
    comparator: str  # "<", "~=", ">"
    weight: float = 1.0


@dataclass
class TRREntry:
    """A single TRR angular constraint."""
    target: str
    ref1: str
    ref2: str
    hour: int
    weight: float = 1.0
    level: str = "hour"  # "hour" or "quadrant"


# ── P_dist: Distance Poset DAG ──

@dataclass
class DistancePoset:
    """QRR constraints organized as a poset DAG on object pairs.

    Nodes = equivalence classes of object pairs (merged via ~=).
    Edges = strict ordering (< / >) between classes.
    """
    equiv_classes: UnionFind = field(default_factory=UnionFind)
    # Directed edges: class_rep -> set of class_reps that are strictly greater
    edges_lt: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    # All object pairs involved
    all_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    # Original constraints for loss computation
    constraints: List[QRREntry] = field(default_factory=list)
    # Detected issues
    has_cycle: bool = False
    cycle_info: Optional[str] = None

    def _pair_id(self, pair: Tuple[str, str]) -> str:
        return f"{pair[0]}_{pair[1]}"


def build_distance_poset(qrr_entries: List[QRREntry]) -> DistancePoset:
    """Build P_dist from QRR constraints.

    Steps:
      1. Union-find merge all ~= pairs
      2. Build DAG edges for < / >
      3. Detect cycles
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
            # d(pair1) < d(pair2) => pair1 is "less" than pair2
            r1 = poset.equiv_classes.find(id1)
            r2 = poset.equiv_classes.find(id2)
            poset.edges_lt[r1].add(r2)
        elif entry.comparator == ">":
            r1 = poset.equiv_classes.find(id1)
            r2 = poset.equiv_classes.find(id2)
            poset.edges_lt[r2].add(r1)

    # Rebuild edges using final equivalence class representatives
    new_edges: Dict[str, Set[str]] = defaultdict(set)
    for src, dsts in poset.edges_lt.items():
        rs = poset.equiv_classes.find(src)
        for dst in dsts:
            rd = poset.equiv_classes.find(dst)
            if rs != rd:
                new_edges[rs].add(rd)
    poset.edges_lt = new_edges

    # Cycle detection via DFS
    poset.has_cycle, poset.cycle_info = _detect_cycle(poset.edges_lt)

    return poset


def _detect_cycle(edges: Dict[str, Set[str]]) -> Tuple[bool, Optional[str]]:
    """Detect cycle in directed graph using DFS coloring."""
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


# ── P_ang: Angular Sector System ──

@dataclass
class ArcInterval:
    """An angular arc interval [center - half_width, center + half_width]."""
    center_deg: float
    half_width_deg: float
    weight: float = 1.0
    level: str = "hour"


@dataclass
class AngularSectorSystem:
    """TRR constraints organized as angular sectors.

    For each (target, ref1, ref2) triple, we store a list of arc intervals.
    Intersecting multiple intervals for the same triple narrows the feasible arc.
    """
    # Key: (target, ref1, ref2) -> list of ArcInterval
    sectors: Dict[Tuple[str, str, str], List[ArcInterval]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # All objects mentioned
    all_objects: Set[str] = field(default_factory=set)
    # Original constraints
    constraints: List[TRREntry] = field(default_factory=list)
    # Detected conflicts
    conflicts: List[Tuple[str, str, str]] = field(default_factory=list)


def build_angular_sectors(trr_entries: List[TRREntry]) -> AngularSectorSystem:
    """Build P_ang from TRR constraints.

    Each TRR constraint maps to an arc interval:
      - hour_correct: center = hour*30, half_width = 15 deg
      - quadrant_correct: center = quadrant*90 - 45, half_width = 45 deg
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

    # Check for conflicts (empty arc intersections)
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
    """Check if the intersection of arc intervals is non-empty.

    Simple check: for each pair of arcs, verify their angular distance
    is less than the sum of their half-widths.
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


# ── Hypergraph Connectivity ──

@dataclass
class HypergraphInfo:
    """Connectivity analysis of the constraint hypergraph."""
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
    """Analyze connectivity of the constraint hypergraph.

    Objects are connected if they share constraints.
    """
    uf = UnionFind()
    participation: Dict[str, int] = defaultdict(int)

    # Initialize all objects
    for oid in object_ids:
        uf.find(oid)

    # QRR: each constraint connects 4 objects
    for entry in qrr_entries:
        objs = list(entry.pair1) + list(entry.pair2)
        for obj in objs:
            participation[obj] += 1
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                uf.union(objs[i], objs[j])

    # TRR: each constraint connects 3 objects
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


# ── Constraint Extraction from Scoring Results ──

def extract_qrr_from_scoring(
    per_question: List[dict],
    questions: List[dict],
    use_correct_only: bool = True,
) -> List[QRREntry]:
    """Extract QRR constraints from scoring results.

    Args:
        per_question: per_question list from score_batch_scene()
        questions: original question list with GT data
        use_correct_only: if True, only use VLM's correct answers;
                         if False, use VLM's predicted answers (for belief reconstruction)
    """
    # Build question lookup
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

        # Use VLM's predicted comparator for belief reconstruction
        comparator = str(predicted) if not use_correct_only else q["gt_comparator"]

        entries.append(QRREntry(
            pair1=tuple(q["pair1"]),
            pair2=tuple(q["pair2"]),
            comparator=comparator,
            weight=1.0,
        ))

    return entries


def extract_trr_from_scoring(
    per_question: List[dict],
    questions: List[dict],
    use_correct_only: bool = True,
) -> List[TRREntry]:
    """Extract TRR constraints from scoring results.

    Uses hierarchical selection:
      - hour_correct → level="hour" (tightest)
      - quadrant_correct → level="quadrant" (looser)
      - otherwise → skip (for use_correct_only=True)
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
            # Use VLM's predicted hour for belief reconstruction
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


# ── Feasibility Summary ──

@dataclass
class FeasibilityReport:
    """Summary of symbolic feasibility checks."""
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
    """Run all symbolic feasibility checks."""
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
