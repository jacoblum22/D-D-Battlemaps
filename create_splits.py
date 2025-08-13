#!/usr/bin/env python3
"""
Step 1: Create fixed train/val/test splits
Train: 1,250 | Val: 120 | Test: 77
Saves as text file lists and never reshuffles.
"""

import json
import random
from pathlib import Path

def create_splits():
    """Create fixed train/val/test splits"""
    
    # Load meta data to get total count
    dataset_dir = Path("dataset_v1")
    meta_path = dataset_dir / "meta.jsonl"
    
    if not meta_path.exists():
        print("Error: meta.jsonl not found. Run prepare_dataset_v1.py first.")
        return
    
    # Count total images
    image_ids = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            image_ids.append(entry['id'])
    
    total_images = len(image_ids)
    print(f"Total images: {total_images}")
    
    if total_images < 1447:
        print(f"Warning: Expected ~1447 images, got {total_images}")
    
    # Set split sizes
    train_size = 1250
    val_size = 120
    test_size = 77
    
    expected_total = train_size + val_size + test_size
    if total_images < expected_total:
        print(f"Error: Not enough images for desired splits. Need {expected_total}, have {total_images}")
        # Adjust proportionally
        ratio = total_images / expected_total
        train_size = int(train_size * ratio)
        val_size = int(val_size * ratio)
        test_size = total_images - train_size - val_size
        print(f"Adjusted splits: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Shuffle image IDs
    shuffled_ids = image_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Create splits
    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:train_size + val_size]
    test_ids = shuffled_ids[train_size + val_size:train_size + val_size + test_size]
    
    print(f"Split sizes: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Save splits
    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    # Save as text files
    with open(splits_dir / "train.txt", 'w') as f:
        for img_id in train_ids:
            f.write(f"{img_id}\n")
    
    with open(splits_dir / "val.txt", 'w') as f:
        for img_id in val_ids:
            f.write(f"{img_id}\n")
    
    with open(splits_dir / "test.txt", 'w') as f:
        for img_id in test_ids:
            f.write(f"{img_id}\n")
    
    # Save split info
    split_info = {
        'total_images': total_images,
        'train_size': len(train_ids),
        'val_size': len(val_ids),
        'test_size': len(test_ids),
        'random_seed': 42,
        'train_ids_sample': train_ids[:10],  # First 10 for reference
        'val_ids_sample': val_ids[:10],
        'test_ids_sample': test_ids[:10]
    }
    
    with open(splits_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Splits saved to {splits_dir}")
    print("Split files: train.txt, val.txt, test.txt")
    print("Never reshuffle these splits - they are now fixed for this dataset version!")

if __name__ == "__main__":
    create_splits()
