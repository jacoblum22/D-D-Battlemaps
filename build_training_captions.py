#!/usr/bin/env python3
"""
Step 2: Build training captions with tag dropout
Creates training-ready captions with 10-20% tag dropout on everything except description.
"""

import json
import random
import re
from pathlib import Path


def apply_tag_dropout(caption_data, dropout_rate=0.15):
    """
    Apply dropout to tag lines (not description) with specified rate
    """
    if random.random() < dropout_rate:
        # Randomly drop one of the tag categories
        tags_to_drop = ["terrain", "features", "scene_type", "attributes", "grid"]
        drop_tag = random.choice(tags_to_drop)

        # Create a copy and remove the selected tag
        modified_data = caption_data.copy()
        if drop_tag in modified_data:
            if drop_tag == "attributes":
                # For attributes, might drop individual sub-attributes
                if "attributes" in modified_data and modified_data["attributes"]:
                    attrs = modified_data["attributes"].copy()
                    attr_keys = list(attrs.keys())
                    if attr_keys:
                        drop_attr = random.choice(attr_keys)
                        del attrs[drop_attr]
                        modified_data["attributes"] = attrs
            else:
                # For other tags, drop entire category or partial items
                if (
                    isinstance(modified_data[drop_tag], list)
                    and len(modified_data[drop_tag]) > 1
                ):
                    # Drop some items from list
                    items = modified_data[drop_tag].copy()
                    num_to_keep = max(
                        1, len(items) - random.randint(1, min(2, len(items) - 1))
                    )
                    modified_data[drop_tag] = random.sample(items, num_to_keep)
                else:
                    # Drop entire tag
                    del modified_data[drop_tag]

        return modified_data

    return caption_data


def format_training_caption(caption_data, apply_dropout=True, dropout_rate=0.15):
    """
    Format caption for training with optional tag dropout
    """
    if apply_dropout:
        caption_data = apply_tag_dropout(caption_data, dropout_rate)

    parts = []

    # Description (never dropped)
    if "description" in caption_data:
        parts.append(caption_data["description"])

    # Terrain
    if "terrain" in caption_data and caption_data["terrain"]:
        terrain_str = ", ".join([t.lower() for t in caption_data["terrain"]])
        parts.append(f"terrain: {terrain_str}")

    # Features
    if "features" in caption_data and caption_data["features"]:
        features_str = ", ".join([f.lower() for f in caption_data["features"]])
        parts.append(f"features: {features_str}")

    # Scene type
    if "scene_type" in caption_data and caption_data["scene_type"]:
        scene_type = caption_data["scene_type"].lower()
        parts.append(f"scene_type: {scene_type}")

    # Attributes
    attr_parts = []
    if "attributes" in caption_data and caption_data["attributes"]:
        attrs = caption_data["attributes"]
        if "lighting" in attrs and attrs["lighting"]:
            lighting_str = ", ".join([l.lower() for l in attrs["lighting"]])
            attr_parts.append(f"lighting({lighting_str})")
        if "condition" in attrs and attrs["condition"]:
            condition_str = ", ".join([c.lower() for c in attrs["condition"]])
            attr_parts.append(f"condition({condition_str})")
        if "coverage" in attrs and attrs["coverage"]:
            coverage_str = ", ".join([c.lower() for c in attrs["coverage"]])
            attr_parts.append(f"coverage({coverage_str})")
        if "vegetation" in attrs and attrs["vegetation"]:
            vegetation_str = ", ".join([v.lower() for v in attrs["vegetation"]])
            attr_parts.append(f"vegetation({vegetation_str})")
        if "layout" in attrs and attrs["layout"]:
            layout_str = ", ".join([l.lower() for l in attrs["layout"]])
            attr_parts.append(f"layout({layout_str})")

    if attr_parts:
        parts.append(f"attributes: {', '.join(attr_parts)}")

    # Grid
    if "grid" in caption_data and caption_data["grid"]:
        grid_val = caption_data["grid"].lower()
        parts.append(f"grid: {grid_val}")

    return ". ".join(parts) + "."


def create_training_captions():
    """Create training captions with dropout for each split"""

    dataset_dir = Path("dataset_v1")
    splits_dir = dataset_dir / "splits"
    captions_training_dir = dataset_dir / "captions_training"
    captions_training_dir.mkdir(exist_ok=True)

    # Load meta data
    meta_path = dataset_dir / "meta.jsonl"
    meta_data = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            meta_data[entry["id"]] = entry

    print(f"Loaded {len(meta_data)} image entries")

    # Process each split
    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}.txt"
        if not split_file.exists():
            print(f"Warning: {split_file} not found, skipping")
            continue

        # Load split IDs
        with open(split_file, "r") as f:
            split_ids = [line.strip() for line in f if line.strip()]

        print(f"Processing {split_name} split: {len(split_ids)} images")

        # Create training captions
        for img_id in split_ids:
            if img_id not in meta_data:
                print(f"Warning: {img_id} not found in meta data")
                continue

            caption_json = meta_data[img_id]["caption_json"]

            # Apply dropout for train split, no dropout for val/test
            apply_dropout = split_name == "train"
            dropout_rate = 0.15  # 15% dropout rate

            training_caption = format_training_caption(
                caption_json, apply_dropout=apply_dropout, dropout_rate=dropout_rate
            )

            # Save training caption
            caption_filename = f"{img_id}.txt"
            caption_path = captions_training_dir / caption_filename
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(training_caption)

    # Create summary
    summary = {
        "description": "Training captions with tag dropout",
        "dropout_rate": 0.15,
        "dropout_applied_to": ["train"],
        "no_dropout": ["val", "test"],
        "caption_format": "<description>. terrain: t1, t2. features: f1, f2. scene_type: <scene>. attributes: lighting(l1), condition(c1). grid: <yes|no>.",
    }

    with open(captions_training_dir / "training_captions_info.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Training captions saved to {captions_training_dir}")
    print(f"Applied {dropout_rate*100}% tag dropout to training set")


if __name__ == "__main__":
    create_training_captions()
