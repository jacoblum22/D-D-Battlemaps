#!/usr/bin/env python3
"""
Dataset v1.0 Preparation Script
Creates dataset_v1/ structure with:
- images/ (512x512 processed images)
- captions/ (individual .txt files from JSON)
- meta.jsonl (full JSON per image for auditing)
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import re


def clean_json_response(raw_response):
    """Extract JSON from markdown code block"""
    # Remove ```json and ``` markers
    cleaned = re.sub(r"^```json\s*", "", raw_response, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def parse_caption_json(raw_response):
    """Parse the JSON caption data"""
    try:
        cleaned = clean_json_response(raw_response)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {raw_response[:200]}...")
        return None


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
    print("Loading processed images list...")
    with open("processed_images.json", "r") as f:
        image_paths = json.load(f)

    print("Loading captions...")
    with open("phase4_captions.json", "r") as f:
        captions_data = json.load(f)

    print(f"Found {len(image_paths)} images and {len(captions_data)} captions")

    if len(image_paths) != len(captions_data):
        print("WARNING: Mismatch between number of images and captions!")
        min_len = min(len(image_paths), len(captions_data))
        print(f"Using first {min_len} entries")
        image_paths = image_paths[:min_len]
        captions_data = captions_data[:min_len]

    # Process each image-caption pair
    meta_data = []
    successful_count = 0
    failed_count = 0

    for i, (image_path, caption_raw) in enumerate(zip(image_paths, captions_data)):
        print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

        # Parse caption
        caption_json = parse_caption_json(caption_raw["raw_response"])
        if not caption_json:
            warnings.append(
                f"Failed to parse caption for image {i}: {os.path.basename(image_path)}"
            )
            failed_count += 1
            continue

        # Check for missing fields
        missing_fields = []
        for field in ["description", "terrain", "features", "scene_type", "grid"]:
            if field not in caption_json or not caption_json[field]:
                missing_fields.append(field)
        if missing_fields:
            warnings.append(f"Missing fields in {image_filename}: {missing_fields}")

        # Generate formatted caption
        formatted_caption = format_caption(caption_json)
        if not formatted_caption.strip():
            warnings.append(f"Empty caption generated for {image_filename}")
            failed_count += 1
            continue

        # Create unique filename
        image_filename = f"image_{i:04d}.png"
        caption_filename = f"image_{i:04d}.txt"

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
        "total_processed": len(image_paths),
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
