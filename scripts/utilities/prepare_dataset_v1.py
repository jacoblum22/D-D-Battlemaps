#!/usr/bin/env python3
"""
Dataset v1.0 Preparation Script
Creates dataset_v1/ structure with:
- images/ (512x512 processed images)
- captions/ (individual .txt files from structured JSON)
- meta.jsonl (full JSON per image for auditing)

Updated to work with transformed phase4_captions.json that includes structured data.
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image


def format_caption(caption_data):
    """
    Format caption according to user's specification:
    <description>. terrain: t1, t2, t3. features: f1, f2, f3, f4. scene_type: <scene>.
    attributes: lighting(l1, l2), condition(c1), coverage(k1). grid: <yes|no>.
    """
    if not caption_data:
        return ""

    parts = []

    # Description
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
        if "elevation" in attrs and attrs["elevation"]:
            elevation_str = ", ".join([e.lower() for e in attrs["elevation"]])
            attr_parts.append(f"elevation({elevation_str})")

    if attr_parts:
        parts.append(f"attributes: {', '.join(attr_parts)}")

    # Grid
    if "grid" in caption_data and caption_data["grid"]:
        grid_val = caption_data["grid"].lower()
        parts.append(f"grid: {grid_val}")

    return ". ".join(parts) + "."


def resize_image_to_512(image_path, output_path):
    """Resize image to 512x512 with padding/cropping"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Calculate aspect ratio
            width, height = img.size
            aspect_ratio = width / height
            operation = "resize"

            if aspect_ratio > 1:  # Wider than square
                # Crop to square first, then resize
                new_width = height
                left = (width - new_width) // 2
                img = img.crop((left, 0, left + new_width, height))
                operation = "crop"
            elif aspect_ratio < 1:  # Taller than square
                # Crop to square first, then resize
                new_height = width
                top = (height - new_height) // 2
                img = img.crop((0, top, width, top + new_height))
                operation = "crop"

            # Resize to 512x512
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            img.save(output_path, "PNG", quality=95)
            return True, operation
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False, "error"


def main():
    import time

    start_time = time.time()

    # Paths
    base_dir = Path(".")
    dataset_dir = base_dir / "dataset_v1"
    images_dir = dataset_dir / "images"
    captions_dir = dataset_dir / "captions"

    # Create directories
    dataset_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    captions_dir.mkdir(exist_ok=True)

    # Tracking variables for logging
    pad_count = 0
    crop_count = 0
    warnings = []
    sample_checks = []

    # Load data
    print("Loading captions with image paths...")
    with open("phase4_captions.json", "r") as f:
        captions_data = json.load(f)

    print(f"Found {len(captions_data)} image-caption pairs")

    # Process each image-caption pair
    meta_data = []
    successful_count = 0
    failed_count = 0

    for i, caption_entry in enumerate(captions_data):
        # Get image path from the caption entry
        image_path = caption_entry.get("image_path", "")
        if not image_path:
            warnings.append(f"Missing image_path in entry {i}")
            failed_count += 1
            continue

        print(f"Processing {i+1}/{len(captions_data)}: {os.path.basename(image_path)}")

        # Early termination check: if we have too many missing fields in the first batch
        if i == 99 and len(warnings) > 50:  # More than 50% failure rate in first 100
            print(
                f"\nERROR: High validation failure rate detected ({len(warnings)} warnings in first 100 files)."
            )
            print("Stopping processing to prevent generating incomplete dataset.")
            print("Please check the caption data structure and validation logic.")
            return

        # Use the already-parsed structured data
        caption_json = caption_entry

        # Create unique filename first for error reporting
        image_filename = f"image_{i:04d}.png"
        caption_filename = f"image_{i:04d}.txt"

        # Check for missing fields
        missing_fields = []
        for field in ["description", "terrain", "features", "scene_type"]:
            if field not in caption_json:
                missing_fields.append(f"{field} (not present)")
            elif field == "description" and not caption_json[field]:
                missing_fields.append(f"{field} (empty)")
            elif field in ["terrain", "features"] and not isinstance(
                caption_json[field], list
            ):
                missing_fields.append(f"{field} (not a list)")
            elif field == "scene_type" and not caption_json[field]:
                missing_fields.append(f"{field} (empty)")

        if missing_fields:
            warnings.append(
                f"Missing/invalid fields in {image_filename}: {missing_fields}"
            )
            # Don't skip for empty arrays - they're valid

        # Generate formatted caption
        formatted_caption = format_caption(caption_json)
        if not formatted_caption.strip():
            warnings.append(f"Empty caption generated for {image_filename}")
            failed_count += 1
            continue

        # Copy and resize image
        source_path = base_dir / image_path
        target_image_path = images_dir / image_filename

        if not source_path.exists():
            warnings.append(f"Image not found: {source_path}")
            failed_count += 1
            continue

        success, operation = resize_image_to_512(source_path, target_image_path)
        if success:
            # Track aspect ratio handling
            if operation == "crop":
                crop_count += 1
            elif operation == "resize":
                pad_count += 1

            # Save caption
            caption_path = captions_dir / caption_filename
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(formatted_caption)

            # Add to meta data
            meta_entry = {
                "id": f"image_{i:04d}",
                "original_path": str(image_path),
                "caption_formatted": formatted_caption,
                "caption_json": caption_json,
            }
            meta_data.append(meta_entry)

            # Collect sample for verification (first 5 successful)
            if len(sample_checks) < 5:
                sample_checks.append(
                    {
                        "image_path": str(target_image_path),
                        "caption_path": str(caption_path),
                        "caption_text": formatted_caption[:200]
                        + ("..." if len(formatted_caption) > 200 else ""),
                    }
                )

            successful_count += 1
        else:
            warnings.append(f"Failed to process image: {image_filename}")
            failed_count += 1

    # Save meta.jsonl
    meta_path = dataset_dir / "meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for entry in meta_data:
            f.write(json.dumps(entry) + "\n")

    # Save summary
    summary = {
        "total_processed": len(captions_data),
        "successful": successful_count,
        "failed": failed_count,
        "dataset_size": len(meta_data),
        "pad_count": pad_count,
        "crop_count": crop_count,
        "warnings_count": len(warnings),
    }

    summary_path = dataset_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # Print comprehensive summary
    print("\n" + "=" * 50)
    print("=== DATASET PREP SUMMARY ===")
    print("=" * 50)
    print(f"Images processed: {successful_count}")
    print(f"Final size: 512x512 (pad: {pad_count}, crop: {crop_count})")
    print(f"Caption files: {successful_count}")
    print(f"Elapsed time: {minutes}m {seconds}s")
    print(f"Dataset saved to: {dataset_dir}")

    # Sample verification
    print("\n" + "-" * 50)
    print(f"--- SAMPLE CHECK ({len(sample_checks)}) ---")
    print("-" * 50)
    for i, sample in enumerate(sample_checks, 1):
        print(f"Sample {i}:")
        print(f"  Image: {sample['image_path']}")
        print(f"  Caption file: {sample['caption_path']}")
        print(f"  Caption text: {sample['caption_text']}")
        print()

    # Warnings and anomalies
    print("-" * 50)
    print("--- WARNINGS & ANOMALIES ---")
    print("-" * 50)
    if warnings:
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    else:
        print("  ✅ No warnings or anomalies detected")

    # File structure check
    print("\n" + "-" * 50)
    print("--- FILE STRUCTURE ---")
    print("-" * 50)
    print("dataset_v1/")
    print("  images/")
    print(f"    {len(list(images_dir.glob('*.png')))} PNG files")
    print("  captions/")
    print(f"    {len(list(captions_dir.glob('*.txt')))} TXT files")
    print("  meta.jsonl")
    print("  summary.json")

    if failed_count > 0:
        print(f"\n⚠️  {failed_count} images failed to process")
    else:
        print("\n✅ All images processed successfully!")


if __name__ == "__main__":
    main()
