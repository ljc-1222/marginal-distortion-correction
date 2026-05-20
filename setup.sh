#!/usr/bin/env bash
# Run this setup script with: source ./setup.sh

if [ ! -d ".venv" ]; then
    python3 -m venv .venv || return 1
fi

source .venv/bin/activate || return 1
python -m pip install -r requirements.txt || return 1

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

mkdir -p sam2/checkpoints || return 1
SAM2_CHECKPOINT_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
download_sam2_checkpoint "sam2.1_hiera_tiny.pt" "${SAM2_CHECKPOINT_BASE_URL}/sam2.1_hiera_tiny.pt" || return 1
download_sam2_checkpoint "sam2.1_hiera_small.pt" "${SAM2_CHECKPOINT_BASE_URL}/sam2.1_hiera_small.pt" || return 1
download_sam2_checkpoint "sam2.1_hiera_base_plus.pt" "${SAM2_CHECKPOINT_BASE_URL}/sam2.1_hiera_base_plus.pt" || return 1
download_sam2_checkpoint "sam2.1_hiera_large.pt" "${SAM2_CHECKPOINT_BASE_URL}/sam2.1_hiera_large.pt" || return 1
