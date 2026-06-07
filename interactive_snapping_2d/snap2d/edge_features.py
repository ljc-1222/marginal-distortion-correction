"""Local edge evidence for 2D annotation snapping."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SnapConfig


_EPS = 1e-9


@dataclass
class EdgeFeatures:
    """Dense edge strength, tangent, and color maps."""

    strength: np.ndarray
    tangent_x: np.ndarray
    tangent_y: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    lab: np.ndarray
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


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(_to_gray_uint8(arr), cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
    else:
        raise ValueError(f"image must have shape (H, W) or (H, W, C); got {arr.shape}.")
    if rgb.dtype == np.uint8:
        return rgb
    rgb_f = rgb.astype(np.float32)
    if float(rgb_f.max(initial=0.0)) <= 1.0:
        rgb_f = rgb_f * 255.0
    return np.clip(rgb_f, 0.0, 255.0).astype(np.uint8)


def _normalize_magnitude(magnitude: np.ndarray) -> np.ndarray:
    scale = float(np.percentile(magnitude, 97.0))
    if scale <= _EPS:
        scale = float(magnitude.max(initial=0.0))
    if scale <= _EPS:
        return np.zeros_like(magnitude, dtype=np.float32)
    return np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)


def compute_edge_features(image: np.ndarray, config: SnapConfig) -> EdgeFeatures:
    """Compute normalized edge strength and edge tangent direction."""
    gray_u8 = _to_gray_uint8(image)
    strength = np.zeros(gray_u8.shape, dtype=np.float32)
    best_gx = np.zeros(gray_u8.shape, dtype=np.float32)
    best_gy = np.zeros(gray_u8.shape, dtype=np.float32)
    scales = tuple(float(s) for s in config.edge_scales) or (0.0,)
    for sigma in scales:
        if sigma > 0.0:
            gray_blur_scale = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            gray_blur_scale = gray_u8
        gray_f = gray_blur_scale.astype(np.float32) / 255.0
        gx_scale = cv2.Scharr(gray_f, cv2.CV_32F, 1, 0)
        gy_scale = cv2.Scharr(gray_f, cv2.CV_32F, 0, 1)
        magnitude = np.sqrt(gx_scale * gx_scale + gy_scale * gy_scale)
        normalized = _normalize_magnitude(magnitude)
        better = normalized > strength
        strength = np.maximum(strength, normalized).astype(np.float32)
        best_gx[better] = gx_scale[better]
        best_gy[better] = gy_scale[better]

    ksize = max(1, int(config.gaussian_blur_ksize))
    if ksize % 2 == 0:
        ksize += 1
    gray_blur = cv2.GaussianBlur(gray_u8, (ksize, ksize), 0) if ksize > 1 else gray_u8

    canny = None
    if int(config.canny_high) > int(config.canny_low) >= 0:
        canny_u8 = cv2.Canny(gray_blur, int(config.canny_low), int(config.canny_high))
        canny = (canny_u8 > 0).astype(np.float32)
        strength = np.maximum(strength, canny).astype(np.float32)

    tangent_x = -best_gy
    tangent_y = best_gx
    norm = np.sqrt(tangent_x * tangent_x + tangent_y * tangent_y)
    valid = norm > _EPS
    tx = np.zeros_like(tangent_x, dtype=np.float32)
    ty = np.zeros_like(tangent_y, dtype=np.float32)
    tx[valid] = tangent_x[valid] / norm[valid]
    ty[valid] = tangent_y[valid] / norm[valid]
    gx = np.zeros_like(best_gx, dtype=np.float32)
    gy = np.zeros_like(best_gy, dtype=np.float32)
    gx[valid] = best_gx[valid] / norm[valid]
    gy[valid] = best_gy[valid] / norm[valid]
    lab = cv2.cvtColor(_to_rgb_uint8(image), cv2.COLOR_RGB2LAB).astype(np.float32)
    return EdgeFeatures(strength=strength, tangent_x=tx, tangent_y=ty, gradient_x=gx, gradient_y=gy, lab=lab, canny=canny)


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


def sample_color_contrast(
    edge_features: EdgeFeatures,
    points_xy: np.ndarray,
    normals: np.ndarray,
    radius_px: float,
    scale: float,
) -> np.ndarray:
    """Sample normalized Lab color contrast across candidate normals."""
    points = np.asarray(points_xy, dtype=np.float32)
    normals_arr = np.asarray(normals, dtype=np.float32)
    if normals_arr.shape != points.shape:
        raise ValueError(f"normals must have shape {points.shape}; got {normals_arr.shape}.")
    norm = np.linalg.norm(normals_arr, axis=1, keepdims=True)
    safe_normals = normals_arr / np.maximum(norm, _EPS)
    radius = max(float(radius_px), 0.0)
    left = bilinear_sample(edge_features.lab, points - safe_normals * radius)
    right = bilinear_sample(edge_features.lab, points + safe_normals * radius)
    delta = np.linalg.norm(left - right, axis=1)
    denom = max(float(scale), _EPS)
    return np.clip(delta / denom, 0.0, 1.0).astype(np.float32)
