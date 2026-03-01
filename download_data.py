#!/usr/bin/env python3
"""
download_and_extract.py

Downloads your VIGORL dataset tarballs from Hugging Face (tracking
dataset downloads in HF metrics), extracts them under DATA_ROOT, and
removes the tarballs.

Usage:
    export DATA_ROOT=/path/to/data
    python download_and_extract.py
"""

import os
import sys
import tarfile
from huggingface_hub import snapshot_download, hf_hub_download
import shutil

# -----------------------------------------------------------------------------
# 1️⃣ Check DATA_ROOT
# -----------------------------------------------------------------------------
DATA_ROOT = os.getenv("DATA_ROOT")
if not DATA_ROOT:
    sys.stderr.write("Error: DATA_ROOT environment variable is not set.\n")
    sys.stderr.write("Please set DATA_ROOT to the directory where you want to store the data.\n")
    sys.exit(1)

os.makedirs(DATA_ROOT, exist_ok=True)
print(f"✅ DATA_ROOT is set to: {DATA_ROOT}")

# -----------------------------------------------------------------------------
# 2️⃣ Specify which datasets to download
# -----------------------------------------------------------------------------
datasets = [
    #"spatial_reasoning",
    #"visual_search",
    "web_action",
    "web_grounding",
    # "MCTS_VSTAR_20250514_134727_images_1", # download if need visual search sft data (large ~50GB)
    # "MCTS_VSTAR_20250514_134727_images_2", # download if need visual search sft data (large ~50GB)
]

# Define the large files that we want to handle separately
LARGE_FILES = ["MCTS_VSTAR_20250514_134727_images_1", "MCTS_VSTAR_20250514_134727_images_2"]

# -----------------------------------------------------------------------------
# 3️⃣ Download the main dataset snapshot (excluding large files)
# -----------------------------------------------------------------------------
print("\n🔄 Downloading main dataset snapshot...")
snapshot_download(
    repo_id="gsarch/vigorl_datasets",
    repo_type="dataset",
    local_dir=DATA_ROOT,
    local_dir_use_symlinks=False,   # ensures real files, not symlinks
    ignore_patterns=["MCTS_VSTAR_20250514_134727_images_1.tar", "MCTS_VSTAR_20250514_134727_images_2.tar"]
)

# -----------------------------------------------------------------------------
# 4️⃣ Selectively download large files if they're in the datasets list
# -----------------------------------------------------------------------------
for large_file in LARGE_FILES:
    if large_file in datasets:
        tar_filename = f"{large_file}.tar"
        print(f"\n🔄 Downloading large file: {tar_filename}...")
        
        try:
            tar_path = hf_hub_download(
                repo_id="gsarch/vigorl_datasets",
                repo_type="dataset",
                filename=tar_filename,
                local_dir=DATA_ROOT,
                local_dir_use_symlinks=False
            )
            print(f"✅ Downloaded {tar_filename}")
        except Exception as e:
            print(f"⚠️  Warning: Could not download {tar_filename}: {e}")

# -----------------------------------------------------------------------------
# 5️⃣ Extract each tarball and clean up
# -----------------------------------------------------------------------------
for ds in datasets:
    tar_path = os.path.join(DATA_ROOT, f"{ds}.tar")
    if not os.path.isfile(tar_path):
        print(f"⚠️  Warning: {tar_path} not found, skipping.")
        continue

    print(f"\n📂 Extracting {ds}.tar …")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=DATA_ROOT)

    print(f"🧹 Removing {ds}.tar …")
    os.remove(tar_path)

    if ds in ["MCTS_VSTAR_20250514_134727_images_1", "MCTS_VSTAR_20250514_134727_images_2"]:
        print(f"🧹 Moving contents of {ds} to visual_search/MCTS_VSTAR_20250514_134727_images …")
        
        source_dir = os.path.join(DATA_ROOT, ds)
        target_dir = os.path.join(DATA_ROOT, "visual_search", "MCTS_VSTAR_20250514_134727_images")
        
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Move all contents from source to target directory
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            target_item = os.path.join(target_dir, item)
            shutil.move(source_item, target_item)
        
        print(f"🧹 Removing empty {ds} directory …")
        shutil.rmtree(source_dir)

print("\n🎉 All done! Your data folders are ready under:")
for ds in datasets:
    print(f" • {os.path.join(DATA_ROOT, ds)}")
