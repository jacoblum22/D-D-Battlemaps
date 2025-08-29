#!/usr/bin/env python3
"""
Move all images with terrain="grassland" to wilderness_images/ folder.
Also cleans up existing wilderness_images/ by moving non-grassland images back
and removing empty folders.
Preserves the original folder structure within the new directory.
"""

import json
import shutil
from pathlib import Path


def get_grassland_images(captions_data):
    """Find all images that have 'grassland' in their terrain list."""
    grassland_images = []
    for entry in captions_data:
        if "terrain" in entry and "image_path" in entry:
            terrain_list = entry.get("terrain", [])
            if "grassland" in terrain_list:
                # Convert Windows backslashes to forward slashes for consistency
                image_path = entry["image_path"].replace("\\", "/")
                grassland_images.append(image_path)
    return grassland_images


def get_image_by_path(captions_data, image_path):
    """Find the caption entry for a given image path."""
    # Normalize the path for comparison
    normalized_path = image_path.replace("\\", "/")

    for entry in captions_data:
        if "image_path" in entry:
            entry_path = entry["image_path"].replace("\\", "/")
            if entry_path == normalized_path:
                return entry
    return None


def cleanup_wilderness_folder(captions_data, wilderness_dir):
    """Remove non-grassland images from wilderness_images and clean up empty folders."""
    if not wilderness_dir.exists():
        return 0, 0

    print(f"\n=== CLEANING UP {wilderness_dir}/ ===")

    # Find all images currently in wilderness_images
    current_images = []
    for png_file in wilderness_dir.rglob("*.png"):
        current_images.append(png_file)

    if not current_images:
        print("No images found in wilderness_images/")
        return 0, 0

    moved_back = 0
    kept_images = 0

    for img_file in current_images:
        # Reconstruct the original path
        rel_path = img_file.relative_to(wilderness_dir)
        original_path = f"generated_images/{rel_path.as_posix()}"

        # Find the caption entry for this image
        entry = get_image_by_path(captions_data, original_path)

        if entry is None:
            print(f"  ⚠️  No caption found for: {img_file}")
            kept_images += 1
            continue

        # Check if this image should be in wilderness (has grassland terrain)
        terrain_list = entry.get("terrain", [])
        if "grassland" not in terrain_list:
            # This shouldn't be in wilderness_images, move it back
            original_file = Path(original_path)

            # Create parent directories if needed
            original_file.parent.mkdir(parents=True, exist_ok=True)

            if original_file.exists():
                print(f"  ⚠️  Destination exists: {original_file}")
                kept_images += 1
            else:
                try:
                    shutil.move(str(img_file), str(original_file))
                    print(f"  🔄 Moved back: {img_file} → {original_file}")
                    moved_back += 1
                except Exception as e:
                    print(f"  ❌ Error moving back {img_file}: {e}")
                    kept_images += 1
        else:
            kept_images += 1

    # Clean up empty directories
    removed_dirs = 0
    for root, dirs, files in wilderness_dir.walk(top_down=False):
        if root != wilderness_dir:  # Don't remove the root wilderness_images folder
            try:
                if not any(root.rglob("*")):  # Directory is empty
                    root.rmdir()
                    print(f"  🗑️  Removed empty directory: {root}")
                    removed_dirs += 1
            except OSError:
                pass  # Directory not empty or other error

    return moved_back, removed_dirs


def main():
    print("=== MANAGING GRASSLAND IMAGES ===")

    # Load captions data
    try:
        with open("phase4_captions.json", "r", encoding="utf-8") as f:
            captions_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: phase4_captions.json not found")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to parse phase4_captions.json: {e}")
        return

    # Clean up existing wilderness_images folder first
    wilderness_dir = Path("wilderness_images")
    moved_back, removed_dirs = cleanup_wilderness_folder(captions_data, wilderness_dir)

    if moved_back > 0:
        print(f"✅ Moved {moved_back} non-grassland images back to generated_images/")
    if removed_dirs > 0:
        print(f"✅ Removed {removed_dirs} empty directories")

    # Find all grassland images
    grassland_images = get_grassland_images(captions_data)

    print(f"\nFound {len(grassland_images)} images with terrain='grassland'")

    if not grassland_images:
        print("No grassland images found to move.")
        return

    # Show what would be moved
    print("\nImages to be moved:")
    for i, img_path in enumerate(sorted(grassland_images)):
        # Show only first 10 and last 10 if there are many
        if len(grassland_images) > 20:
            if i < 10:
                print(f"  {img_path}")
            elif i == 10:
                print(f"  ... ({len(grassland_images) - 20} more) ...")
            elif i >= len(grassland_images) - 10:
                print(f"  {img_path}")
        else:
            print(f"  {img_path}")

    # Ask for confirmation
    response = (
        input(
            f"\nMove {len(grassland_images)} grassland images to wilderness_images/? (Y/n): "
        )
        .strip()
        .lower()
    )

    if response and response not in ["y", "yes"]:
        print("Operation cancelled.")
        return

    # Create wilderness directory
    wilderness_dir.mkdir(exist_ok=True)

    # Move the files
    moved_count = 0
    skipped_count = 0
    error_count = 0

    print(f"\nMoving images to {wilderness_dir}/")

    for img_path in grassland_images:
        try:
            src_path = Path(img_path)

            # Check if source file exists
            if not src_path.exists():
                print(f"  ⚠️  Source not found: {img_path}")
                skipped_count += 1
                continue

            # Create destination path maintaining structure relative to generated_images
            if src_path.parts[0] == "generated_images":
                # Remove "generated_images" from the path
                rel_path = Path(*src_path.parts[1:])
            else:
                # If it doesn't start with generated_images, use as-is
                rel_path = src_path

            dst_path = wilderness_dir / rel_path

            # Create parent directories if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if destination already exists
            if dst_path.exists():
                print(f"  ⚠️  Destination exists: {dst_path}")
                skipped_count += 1
                continue

            # Move the file
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✅ Moved: {src_path} → {dst_path}")
            moved_count += 1

        except Exception as e:
            print(f"  ❌ Error moving {img_path}: {e}")
            error_count += 1

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"✅ Successfully moved: {moved_count} images")
    if skipped_count > 0:
        print(f"⚠️  Skipped: {skipped_count} images")
    if error_count > 0:
        print(f"❌ Errors: {error_count} images")

    print(f"\nGrassland images are now in: {wilderness_dir.absolute()}")

    # Verify some moved files exist
    if moved_count > 0:
        print("\nVerifying moved files...")
        verification_files = list(wilderness_dir.rglob("*.png"))[:5]  # Check first 5
        for file_path in verification_files:
            if file_path.exists():
                print(f"  ✅ Verified: {file_path}")
            else:
                print(f"  ❌ Missing: {file_path}")


if __name__ == "__main__":
    main()
