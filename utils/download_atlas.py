"""
Download script for the DoseRAD2026 dataset from HuggingFace.
Dataset: https://huggingface.co/datasets/LMUK-RADONC-PHYS-RES/DoseRAD2026

Usage:
    python download_dataset.py
    python download_dataset.py --output-dir /data --token your_hf_token
    HF_TOKEN=your_token python download_dataset.py
"""

import os
import sys
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Download DoseRAD2026 dataset from HuggingFace")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/users/ffbence/dose/data",
        help="Directory to save the dataset (default: /data)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token. Can also be set via HF_TOKEN env variable.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="LMUK-RADONC-PHYS-RES/DoseRAD2026",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Branch/revision to download (default: main)",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=None,
        help="File patterns to ignore (e.g. '*.json' '*.md')",
    )
    return parser.parse_args()


def download_dataset(repo_id, output_dir, token=None, revision="main", ignore_patterns=None):
    try:
        from huggingface_hub import snapshot_download, login
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)

    # Resolve token: argument > environment variable
    hf_token = "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" #token or os.environ.get("HF_TOKEN")

    if hf_token:
        print("HuggingFace token detected — logging in...")
        login(token=hf_token)
    else:
        print("No HF_TOKEN provided — attempting anonymous download.")
        print("Note: If the dataset is gated, set HF_TOKEN=<your_token>")

    output_path = Path(output_dir) / repo_id.replace("/", "__")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset  : {repo_id}")
    print(f"Revision : {revision}")
    print(f"Output   : {output_path}\n")

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(output_path),
            token=hf_token,
            ignore_patterns=ignore_patterns,
        )
        print(f"\n✅ Dataset downloaded successfully to: {local_dir}")
        return local_dir

    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print("\n❌ Authentication error — the dataset may be gated.")
            print("   1. Accept the dataset terms at:")
            print(f"      https://huggingface.co/datasets/{repo_id}")
            print("   2. Re-run with your token:")
            print("      HF_TOKEN=your_token python download_dataset.py")
        else:
            print(f"\n❌ HTTP error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


def print_summary(local_dir):
    """Print a summary of downloaded files."""
    path = Path(local_dir)
    files = list(path.rglob("*"))
    files = [f for f in files if f.is_file() and ".cache" not in str(f)]

    total_size = sum(f.stat().st_size for f in files)
    size_gb = total_size / (1024 ** 3)

    print(f"\n📁 Downloaded {len(files)} files ({size_gb:.2f} GB total)")

    # Group by extension
    from collections import Counter
    ext_counts = Counter(f.suffix.lower() for f in files)
    print("\nFile types:")
    for ext, count in ext_counts.most_common():
        print(f"  {ext or '(no ext)':15s}: {count}")


if __name__ == "__main__":
    args = parse_args()

    local_dir = download_dataset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        token=args.token,
        revision=args.revision,
        ignore_patterns=args.ignore_patterns,
    )

    print_summary(local_dir)
