"""2D ribbon dynamic-programming snapping package."""

from .config import SnapConfig, SnapResult
from .snapping import snap_annotation

__all__ = ["SnapConfig", "SnapResult", "snap_annotation"]
