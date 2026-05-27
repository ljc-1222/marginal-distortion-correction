"""Local edge evidence for 2D annotation snapping."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SnapConfig


_EPS = 1e-9


@dataclass
class EdgeFeatures:
    """Dense edge strength and tangent maps."""

    strength: np.ndarray
    tangent_x: np.ndarray
    tangent_y: np.ndarray
    canny: np.ndarray | None


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        gray = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"image must have shape (H, W) or (H, W, C); got {arr.shape}.")

    if gray.dtype == np.uint8:
        return gray
    gray_f = gray.astype(np.float32)
    if float(gray_f.max(initial=0.0)) <= 1.0:
        gray_f = gray_f * 255.0
    return np.clip(gray_f, 0.0, 255.0).astype(np.uint8)


def compute_edge_features(image: np.ndarray, config: SnapConfig) -> EdgeFeatures:
    """Compute normalized edge strength and edge tangent direction."""
    gray_u8 = _to_gray_uint8(image)
    ksize = max(1, int(config.gaussian_blur_ksize))
    if ksize % 2 == 0:
        ksize += 1
    if ksize > 1:
        gray_blur = cv2.GaussianBlur(gray_u8, (ksize, ksize), 0)
    else:
        gray_blur = gray_u8

    gray_f = gray_blur.astype(np.float32) / 255.0
    gx = cv2.Scharr(gray_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray_f, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(gx * gx + gy * gy)
    scale = float(np.percentile(magnitude, 97.0))
    if scale <= _EPS:
        scale = float(magnitude.max(initial=0.0))
    if scale <= _EPS:
        strength = np.zeros_like(magnitude, dtype=np.float32)
    else:
        strength = np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)

    canny = None
    if int(config.canny_high) > int(config.canny_low) >= 0:
        canny_u8 = cv2.Canny(gray_blur, int(config.canny_low), int(config.canny_high))
        canny = (canny_u8 > 0).astype(np.float32)
        strength = np.maximum(strength, canny).astype(np.float32)

    tangent_x = -gy
    tangent_y = gx
    norm = np.sqrt(tangent_x * tangent_x + tangent_y * tangent_y)
    valid = norm > _EPS
    tx = np.zeros_like(tangent_x, dtype=np.float32)
    ty = np.zeros_like(tangent_y, dtype=np.float32)
    tx[valid] = tangent_x[valid] / norm[valid]
    ty[valid] = tangent_y[valid] / norm[valid]
    return EdgeFeatures(strength=strength, tangent_x=tx, tangent_y=ty, canny=canny)


def bilinear_sample(image_or_map: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Sample a 2D or channel-last array at floating-point ``(x, y)`` points."""
    arr = np.asarray(image_or_map)
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape (N, 2); got {points.shape}.")
    if arr.ndim not in (2, 3):
        raise ValueError(f"image_or_map must be 2D or 3D; got {arr.shape}.")

    height, width = arr.shape[:2]
    x = np.clip(points[:, 0], 0.0, max(0.0, width - 1.0))
    y = np.clip(points[:, 1], 0.0, max(0.0, height - 1.0))
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    if arr.ndim == 2:
        top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
        bottom = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
        return (top * (1.0 - wy) + bottom * wy).astype(np.float32)

    wx_c = wx[:, None]
    wy_c = wy[:, None]
    top = arr[y0, x0] * (1.0 - wx_c) + arr[y0, x1] * wx_c
    bottom = arr[y1, x0] * (1.0 - wx_c) + arr[y1, x1] * wx_c
    return (top * (1.0 - wy_c) + bottom * wy_c).astype(np.float32)
