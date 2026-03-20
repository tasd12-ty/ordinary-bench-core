"""
Scene Belief Reconstruction module.

Reconstructs 2D spatial configurations from VLM ordinal spatial judgments.
"""

from .pipeline import (
    reconstruct,
    reconstruct_from_prepared,
    reconstruct_from_scoring,
    ReconstructResult,
)
from .preparation import (
    PreparedSceneInput,
    prepare_reconstruction_input_from_scoring,
    load_questions_auto,
    load_scene_gt_positions,
)
from .solver import SolverConfig, SolverSolution
from .evaluate import EvalMetrics
from .constraints import (
    QRREntry,
    TRREntry,
    FeasibilityReport,
)

__all__ = [
    "reconstruct",
    "reconstruct_from_prepared",
    "reconstruct_from_scoring",
    "ReconstructResult",
    "PreparedSceneInput",
    "prepare_reconstruction_input_from_scoring",
    "load_questions_auto",
    "load_scene_gt_positions",
    "SolverConfig",
    "SolverSolution",
    "EvalMetrics",
    "QRREntry",
    "TRREntry",
    "FeasibilityReport",
]
