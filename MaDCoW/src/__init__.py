"""Shared type aliases and dataclasses for the MaDCoW pipeline.

All inter-module data exchange goes through the structures defined here so
that the camera, mesh, projection, loss, optimization, and rendering modules
stay decoupled but interface-aligned.

Conventions:
    * Angles ``lambda`` (yaw) and ``phi`` (pitch) are always in radians.
    * Mesh-shaped arrays/tensors use layout ``(H, W)`` for scalar fields and
      ``(H, W, 2)`` for 2D vector fields whose last axis is ``(u, v)``.
    * Multi-ROI stacks use layout ``(K, H, W)`` or ``(K, H, W, 2)`` where
      ``K`` is the number of regions.
    * ``Array`` is for numpy data used in pre/post processing;
      ``Tensor`` is for torch data participating in autograd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
import torch

Array: TypeAlias = np.ndarray
Tensor: TypeAlias = torch.Tensor


@dataclass
class CameraConfig:
    """Intrinsic configuration of the input perspective camera.

    Attributes:
        fov_deg: Horizontal field of view of the input image, in degrees.
        width: Input image width in pixels.
        height: Input image height in pixels.
    """

    fov_deg: float
    width: int
    height: int


@dataclass
class MeshGrid:
    """Discretization of the view sphere used as the warp parameterization.

    Attributes:
        lambda_grid: Array of shape ``(H, W)`` holding the yaw of each vertex
            in radians.
        phi_grid: Array of shape ``(H, W)`` holding the pitch of each vertex
            in radians.
    """

    lambda_grid: Array
    phi_grid: Array


@dataclass
class LineAnnotation:
    """A single user-annotated straight-line constraint.

    Attributes:
        start_dir: View-sphere direction ``(lambda, phi)`` of the line start.
        end_dir: View-sphere direction ``(lambda, phi)`` of the line end.
    """

    start_dir: tuple[float, float]
    end_dir: tuple[float, float]


@dataclass
class RegionAnnotation:
    """A single user-annotated region of interest.

    Attributes:
        name: Human-readable identifier (used to name mask files).
        mask_path: Filesystem path to a binary PNG mask in input-image space.
    """

    name: str
    mask_path: str


@dataclass
class AnnotationData:
    """Full set of user annotations attached to one input image.

    Attributes:
        image_path: Path to the input perspective ``.jpg``.
        fov_deg: Horizontal field of view of the input image in degrees.
        lines: List of straight-line annotations.
        regions: List of region-of-interest annotations.
    """

    image_path: str
    fov_deg: float
    lines: list[LineAnnotation] = field(default_factory=list)
    regions: list[RegionAnnotation] = field(default_factory=list)


@dataclass
class LossWeights:
    """Scalar weights of the four loss terms used in the full-image warp.

    Attributes:
        w_l: Weight of the straight-line preservation loss ``E_l``.
        w_c: Weight of the conformal loss ``E_c``.
        w_s: Weight of the smoothness loss ``E_s``.
        w_dvc: Weight of the per-ROI DVC matching loss ``E_DVC``.
    """

    w_l: float
    w_c: float
    w_s: float
    w_dvc: float


@dataclass
class PipelineConfig:
    """Hyperparameters controlling the MaDCoW pipeline.

    Attributes:
        mesh_n_lambda: Number of mesh samples along the yaw direction.
        mesh_n_phi: Number of mesh samples along the pitch direction.
        loss_weights: Loss term weights for the Stage 2 optimization.
        blend_c: Scaling constant ``c`` in the exponential blend ``alpha_k``.
        stage1_max_iter: Maximum optimizer iterations per ROI in Stage 1.
        stage2_max_iter: Maximum optimizer iterations for the full warp.
        lbfgs_lr: Learning rate passed to the Stage 2 ``torch.optim.LBFGS``.
    """

    mesh_n_lambda: int
    mesh_n_phi: int
    loss_weights: LossWeights
    blend_c: float
    stage1_max_iter: int
    stage2_max_iter: int
    lbfgs_lr: float


__all__ = [
    "Array",
    "Tensor",
    "CameraConfig",
    "MeshGrid",
    "LineAnnotation",
    "RegionAnnotation",
    "AnnotationData",
    "LossWeights",
    "PipelineConfig",
]
