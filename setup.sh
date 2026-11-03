#!/usr/bin/env bash
# Run with either:
#   source ./setup.sh   # keeps .venv active in the current shell
#   bash setup.sh       # installs and verifies, but does not activate the caller shell
# Add --sam2-model tiny|small|base-plus|large to download only one checkpoint.

setup_usage() {
    cat <<'USAGE'
Usage: bash setup.sh [--sam2-model MODEL]
       source ./setup.sh [--sam2-model MODEL]

Create .venv, install requirements.txt, download SAM2.1 checkpoints, and check imports.

Options:
  --sam2-model MODEL  tiny, small, base-plus, large, or all (default: all)
  -h, --help         Show this help without installing or downloading anything.
USAGE
}

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
    local sam2_model="all"
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --sam2-model)
                if [ "$#" -lt 2 ]; then
                    echo "--sam2-model requires tiny, small, base-plus, large, or all." >&2
                    return 2
                fi
                sam2_model="$2"
                shift 2
                ;;
            -h|--help)
                setup_usage
                return 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                setup_usage >&2
                return 2
                ;;
        esac
    done

    case "$sam2_model" in
        tiny|small|base-plus|large|all) ;;
        *)
            echo "Unknown SAM2 model: $sam2_model. Choose tiny, small, base-plus, large, or all." >&2
            return 2
            ;;
    esac

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
    local checkpoint_base_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
    local selected_models=()
    case "$sam2_model" in
        all) selected_models=(tiny small base_plus large) ;;
        base-plus) selected_models=(base_plus) ;;
        *) selected_models=("$sam2_model") ;;
    esac
    local model_name
    for model_name in "${selected_models[@]}"; do
        download_sam2_checkpoint "sam2.1_hiera_${model_name}.pt" \
            "${checkpoint_base_url}/sam2.1_hiera_${model_name}.pt" || return 1
    done

    verify_setup || return 1

    if is_sourced; then
        echo "Setup complete. The .venv environment is active."
    else
        echo "Setup complete. Run 'source .venv/bin/activate' before using the tools in a new shell."
    fi
}

main "$@"
finish_setup "$?"
