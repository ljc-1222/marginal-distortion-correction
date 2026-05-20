"""Outer-package bootstrap for running local SAM2 tools from the project root."""

from __future__ import annotations

from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_INNER_PACKAGE_DIR = _PACKAGE_DIR / "sam2"

if _INNER_PACKAGE_DIR.is_dir():
    __path__ = [str(_INNER_PACKAGE_DIR), str(_PACKAGE_DIR)]

    try:
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except ImportError:
        pass
    else:
        if not GlobalHydra.instance().is_initialized():
            initialize_config_dir(config_dir=str(_INNER_PACKAGE_DIR), version_base="1.2")
