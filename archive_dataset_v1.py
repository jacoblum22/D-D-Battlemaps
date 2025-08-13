#!/usr/bin/env python3
"""
Archive dataset_v1 for reproducibility
Creates a complete backup with version tracking
"""

import json
import shutil
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime

def get_git_info():
    """Get current git commit hash and branch"""
    try:
        # Get current commit hash
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd='.')
        commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        # Get current branch
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True, cwd='.')
        branch = result.stdout.strip() if result.returncode == 0 else "unknown"
        
        # Check if there are uncommitted changes
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, cwd='.')
        has_changes = len(result.stdout.strip()) > 0 if result.returncode == 0 else True
        
        return {
            'commit_hash': commit_hash,
            'branch': branch,
            'has_uncommitted_changes': has_changes
        }
    except:
        return {
            'commit_hash': "unknown",
            'branch': "unknown", 
            'has_uncommitted_changes': True
        }

def create_version_manifest():
    """Create a comprehensive version manifest"""
    
    git_info = get_git_info()
    
    # Load dataset summary
    summary_path = Path("dataset_v1/summary.json")
    dataset_summary = {}
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            dataset_summary = json.load(f)
    
    manifest = {
        'dataset_version': 'v1.0',
        'created_at': datetime.now().isoformat(),
        'git_info': git_info,
        'dataset_summary': dataset_summary,
        'source_files': {
            'processed_images': 'processed_images.json',
            'captions': 'phase4_captions.json',
            'vocabulary_analysis': 'phase4_vocabulary_analysis.txt',
            'preparation_script': 'prepare_dataset_v1.py'
        },
        'structure': {
            'images': 'dataset_v1/images/',
            'captions': 'dataset_v1/captions/', 
            'metadata': 'dataset_v1/meta.jsonl',
            'summary': 'dataset_v1/summary.json'
        },
        'reproducibility_notes': [
            "To reproduce this exact dataset:",
            "1. Use the same source files (processed_images.json, phase4_captions.json)",
            "2. Run the archived prepare_dataset_v1.py script",
            "3. Verify the git commit matches if using the same code version"
        ]
    }
    
    return manifest

def archive_dataset():
    """Create comprehensive archive of dataset v1.0"""
    
    print("Creating dataset v1.0 archive...")
    
    # Create archive directory
    archive_dir = Path("dataset_v1_archive")
    archive_dir.mkdir(exist_ok=True)
    
    # Create version manifest
    manifest = create_version_manifest()
    manifest_path = archive_dir / "version_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Version manifest created: {manifest_path}")
    
    # Copy key source files
    source_files = [
        'processed_images.json',
        'phase4_captions.json', 
        'phase4_vocabulary_analysis.txt',
        'prepare_dataset_v1.py'
    ]
    
    for filename in source_files:
        source_path = Path(filename)
        if source_path.exists():
            target_path = archive_dir / filename
            shutil.copy2(source_path, target_path)
            print(f"Copied: {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    # Create zip archive of the complete dataset
    zip_path = Path("dataset_v1_complete.zip")
    
    print("Creating zip archive...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add dataset_v1 directory
        dataset_dir = Path("dataset_v1")
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(".")
                zipf.write(file_path, arcname)
        
        # Add archive directory contents
        for file_path in archive_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(".")
                zipf.write(file_path, arcname)
    
    # Get file sizes for reporting
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    dataset_size_mb = sum(f.stat().st_size for f in Path("dataset_v1").rglob("*") if f.is_file()) / (1024 * 1024)
    
    print("\n" + "="*50)
    print("=== DATASET V1.0 ARCHIVED ===")
    print("="*50)
    print(f"Archive created: {zip_path}")
    print(f"Archive size: {zip_size_mb:.1f} MB")
    print(f"Dataset size: {dataset_size_mb:.1f} MB") 
    print(f"Git commit: {manifest['git_info']['commit_hash'][:8]}")
    print(f"Git branch: {manifest['git_info']['branch']}")
    if manifest['git_info']['has_uncommitted_changes']:
        print("⚠️  Warning: Uncommitted changes detected")
    print(f"Images: {manifest['dataset_summary'].get('successful', 'unknown')}")
    print(f"Created: {manifest['created_at']}")
    
    print(f"\nFiles in archive:")
    print(f"  dataset_v1/images/ ({manifest['dataset_summary'].get('successful', 0)} PNG files)")
    print(f"  dataset_v1/captions/ ({manifest['dataset_summary'].get('successful', 0)} TXT files)")
    print(f"  dataset_v1/meta.jsonl")
    print(f"  dataset_v1/summary.json")
    print(f"  dataset_v1_archive/version_manifest.json")
    print(f"  dataset_v1_archive/prepare_dataset_v1.py")
    print(f"  dataset_v1_archive/processed_images.json")
    print(f"  dataset_v1_archive/phase4_captions.json")
    print(f"  dataset_v1_archive/phase4_vocabulary_analysis.txt")
    
    print(f"\n✅ Dataset v1.0 is now locked and archived!")
    print(f"Archive path: {zip_path.absolute()}")

if __name__ == "__main__":
    archive_dataset()
