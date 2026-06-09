#!/usr/bin/env bash
# Run with either:
#   source ./setup.sh   # keeps .venv active in the current shell
#   bash setup.sh       # installs and verifies, but does not activate the caller shell

is_sourced() {
    [ "${BASH_SOURCE[0]}" != "$0" ]
}

finish_setup() {
    local status="$1"
    if is_sourced; then
        return "$status"
    fi
    exit "$status"
}

download_file() {
    local url="$1"
    local output_path="$2"
    local tmp_path="${output_path}.tmp"

    rm -f "$tmp_path"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --output "$tmp_path" "$url" || return 1
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp_path" "$url" || return 1
    else
        echo "Please install curl or wget to download SAM2 checkpoints."
        return 1
    fi

    mv "$tmp_path" "$output_path"
}

download_sam2_checkpoint() {
    local filename="$1"
    local url="$2"
    local output_path="sam2/checkpoints/${filename}"

    if [ -s "$output_path" ]; then
        echo "SAM2 checkpoint already exists: $output_path"
        return 0
    fi

    echo "Downloading SAM2 checkpoint: $filename"
    download_file "$url" "$output_path" || return 1
}

verify_setup() {
    python - <<'PY'
import importlib

modules = [
    "numpy",
    "PIL",
    "matplotlib",
    "cv2",
    "scipy",
    "torch",
    "torchvision",
    "hydra",
    "iopath",
    "piexif",
    "tqdm",
    "annotation_gui",
    "sam2.annotate_ROI_auto",
    "interactive_snapping_2d.annotate_line_aid",
    "MaDCoW.main",
    "pipeline.integrated_annotation",
    "pipeline.annotation_merge",
    "annotate",
    "run_mad",
]

for module in modules:
    importlib.import_module(module)

from sam2.annotate_ROI_auto import _bootstrap_sam2_package

_bootstrap_sam2_package()
importlib.import_module("sam2.build_sam")

print("Setup verification passed.")
PY
}

main() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required to create the virtual environment."
        return 1
    fi

    if [ ! -d ".venv" ]; then
        python3 -m venv .venv || return 1
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate || return 1
    python -m pip install --upgrade pip wheel || return 1
    python -m pip install -r requirements.txt || return 1

    mkdir -p sam2/checkpoints || return 1
    SAM2_CHECKPOINT_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
    download_sam2_checkpoint "sam2.1_hiera_tiny.pt" "${SAM2_CHECKPOINT_BASE_URL}/sam2.1_hiera_tiny.pt" || return 1
    download_sam2_checkpoint "sam2.1_hiera_small.pt" "${SAM2_CHECKPOINT_BASE_URL}/sam2.1_hiera_small.pt" || return 1
    download_sam2_checkpoint "sam2.1_hiera_base_plus.pt" "${SAM2_CHECKPOINT_BASE_URL}/sam2.1_hiera_base_plus.pt" || return 1
    download_sam2_checkpoint "sam2.1_hiera_large.pt" "${SAM2_CHECKPOINT_BASE_URL}/sam2.1_hiera_large.pt" || return 1

    verify_setup || return 1

    if is_sourced; then
        echo "Setup complete. The .venv environment is active."
    else
        echo "Setup complete. Run 'source .venv/bin/activate' before using the tools in a new shell."
    fi
}

main "$@"
finish_setup "$?"
