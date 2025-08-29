#!/usr/bin/env python3
"""
Create grassland captions file from phase4_captions_restored_merged.json
Matches actual filenames in wilderness_images/ with captions and updates paths
"""

import json
import os
from pathlib import Path


def find_image_files(directory):
    """Find all image files in wilderness_images directory"""
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    image_files = []

    wilderness_path = Path(directory)
    if not wilderness_path.exists():
        print(f"Error: {directory} not found!")
        return []

    # Recursively find all image files
    for file_path in wilderness_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # Store relative path from wilderness_images/
            rel_path = file_path.relative_to(wilderness_path)
            image_files.append(str(rel_path))

    return image_files


def create_grassland_captions():
    """Create captions file by matching wilderness_images filenames with existing captions"""

    # Input and output files
    captions_file = "phase4_captions_restored_merged.json"
    wilderness_dir = "wilderness_images"
    output_file = "captions_grassland.json"

    if not os.path.exists(captions_file):
        print(f"Error: {captions_file} not found!")
        return

    print(f"Scanning {wilderness_dir} for image files...")
    wilderness_files = find_image_files(wilderness_dir)
    print(f"Found {len(wilderness_files)} image files in wilderness_images")

    if not wilderness_files:
        print("No image files found in wilderness_images!")
        return

    print(f"Loading captions from {captions_file}...")

    # Load the captions
    with open(captions_file, "r", encoding="utf-8") as f:
        all_captions = json.load(f)

    print(f"Total captions loaded: {len(all_captions)}")

    # Create lookup by filename for faster matching
    print("Creating filename lookup...")
    caption_lookup = {}
    for caption in all_captions:
        image_path = caption.get("image_path", "")
        if image_path:
            # Extract just the filename from the full path
            filename = Path(image_path).name
            caption_lookup[filename] = caption

    print(f"Created lookup for {len(caption_lookup)} caption filenames")

    # Match wilderness files with captions
    matched_captions = []
    unmatched_files = []

    for wilderness_file in wilderness_files:
        # Extract just the filename (not the directory structure)
        filename = Path(wilderness_file).name

        if filename in caption_lookup:
            # Found a matching caption
            original_caption = caption_lookup[filename]

            # Create a copy and update the image path
            matched_caption = original_caption.copy()

            # Update path to wilderness_images with the actual file structure
            new_path = f"wilderness_images/{wilderness_file}"
            matched_caption["image_path"] = new_path

            matched_captions.append(matched_caption)
        else:
            unmatched_files.append(wilderness_file)

    print(f"Matched {len(matched_captions)} files with captions")
    print(f"Unmatched files: {len(unmatched_files)}")

    if unmatched_files:
        print("First few unmatched files:")
        for unmatched in unmatched_files[:5]:
            print(f"  - {unmatched}")

    # Save matched captions
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(matched_captions, f, indent=2, ensure_ascii=False)

    print(f"Matched captions saved to {output_file}")

    # Show some examples
    if matched_captions:
        print("\nFirst few matched examples:")
        for i, caption in enumerate(matched_captions[:3]):
            print(f"{i+1}. {caption['image_path']}")
            print(f"   Terrain: {caption.get('terrain', [])}")
            print(f"   Description: {caption['description'][:80]}...")
            print()


if __name__ == "__main__":
    create_grassland_captions()
