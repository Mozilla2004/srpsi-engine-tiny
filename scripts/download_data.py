"""
Download Training Data Script
==============================

Automatically downloads burgers_1d.npy training data if not present.

This script handles the large data file (117MB) that cannot be stored
in the GitHub repository due to size limits.

Usage:
    python scripts/download_data.py

Author: SRΨ-Engine Tiny Experiment
"""

import os
import hashlib
from pathlib import Path
import urllib.request


def get_file_checksum(filepath: str, block_size=65536) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


def download_data():
    """Download training data if not present."""
    data_dir = Path("data")
    data_file = data_dir / "burgers_1d.npy"

    # Check if data already exists
    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"✓ Data file already exists: {data_file}")
        print(f"  Size: {size_mb:.1f} MB")

        # Verify checksum
        expected_checksum = "a1b2c3d4e5f6..."  # TODO: Add actual checksum
        actual_checksum = get_file_checksum(str(data_file))

        print(f"  Checksum: {actual_checksum[:16]}...")

        return

    # Create data directory
    data_dir.mkdir(exist_ok=True)

    print("Downloading training data...")
    print(f"  Target: {data_file}")
    print(f"  Size: ~117 MB")

    # TODO: Replace with actual download URL
    # Options:
    # 1. Upload to Dropbox/Google Drive and create share link
    # 2. Upload to GitHub Release
    # 3. Host on your own server
    # 4. Use rsync from another machine

    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("="*60)
    print("\nThe data file is too large for GitHub (117MB).")
    print("Please download it from one of these sources:")
    print("\nOption 1: From local Mac")
    print(f"  scp /Users/luxiangrong/.../data/burgers_1d.npy \\")
    print(f"      jules:~/srpsi-engine-tiny/data/")
    print("\nOption 2: From cloud storage")
    print("  [TODO: Add download link]")
    print("\nOption 3: Generate from scratch")
    print("  python src/data_gen.py")
    print("\nAfter downloading, verify:")
    print(f"  ls -lh {data_file}")
    print("  Should show ~117 MB")
    print("="*60)


def verify_data():
    """Verify data file integrity."""
    data_file = Path("data/burgers_1d.npy")

    if not data_file.exists():
        print("✗ Data file not found!")
        print("\nPlease run:")
        print("  python scripts/download_data.py")
        return False

    # Check file size
    size_mb = data_file.stat().st_size / (1024 * 1024)
    if size_mb < 100:
        print(f"✗ Data file too small: {size_mb:.1f} MB (expected ~117 MB)")
        return False

    print(f"✓ Data file verified: {size_mb:.1f} MB")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_data()
    else:
        download_data()
