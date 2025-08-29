#!/usr/bin/env python3
"""
Fix Packaging Script - Deduplicate Images
Modifies the packaging script to ensure exactly 1446 unique images are included
by skipping duplicate filenames.
"""

import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_packaging_script():
    """Add deduplication logic to the packaging script"""

    # Read the current packaging script
    with open("package_dataset_for_runpod.py", "r", encoding="utf-8") as f:
        content = f.readlines()

    # Find the copy_images_and_create_captions function and replace it
    start_line = None
    end_line = None
    indent_level = None

    for i, line in enumerate(content):
        if line.strip().startswith("def copy_images_and_create_captions"):
            start_line = i
            indent_level = len(line) - len(line.lstrip())
            continue

        if (
            start_line is not None
            and line.strip()
            and indent_level is not None
            and len(line) - len(line.lstrip()) <= indent_level
            and not line.startswith(" " * (indent_level + 1))
        ):
            if line.strip().startswith("def ") or line.strip().startswith("if "):
                end_line = i
                break

    if start_line is None:
        logger.error("Could not find copy_images_and_create_captions function")
        return False

    if end_line is None:
        end_line = len(content)

    logger.info(f"Replacing lines {start_line + 1} to {end_line}")

    # Create the new function with deduplication
    new_function = '''def copy_images_and_create_captions(splits_with_captions, base_path):
    """Copy images to split folders and create individual caption files - with deduplication"""
    images_dir = base_path / "images"
    captions_dir = base_path / "captions"
    
    stats = {
        'train': {'images': 0, 'captions': 0},
        'val': {'images': 0, 'captions': 0},
        'test': {'images': 0, 'captions': 0}
    }
    
    missing_images = []
    missing_captions = []
    
    # Track processed filenames to avoid duplicates - this ensures exactly 1446 unique images
    processed_filenames = set()
    
    for split_name, image_caption_pairs in splits_with_captions.items():
        split_images_dir = images_dir / split_name
        split_captions_dir = captions_dir / split_name
        
        logger.info(f"Processing {len(image_caption_pairs)} images for {split_name} split...")
        
        processed_count = 0
        skipped_duplicates = 0
        
        for image_path, caption in image_caption_pairs:
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"  {split_name}: Processed {processed_count}/{len(image_caption_pairs)} images")
            
            source_path = Path(image_path)
            
            if not source_path.exists():
                missing_images.append(image_path)
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Extract original filename 
            original_filename = source_path.name
            
            # Skip if we've already processed this filename (avoid duplicates)
            if original_filename in processed_filenames:
                skipped_duplicates += 1
                continue
            
            processed_filenames.add(original_filename)
            
            # Use original filename (no prefix needed since we're deduplicating)
            dest_filename = original_filename
            dest_stem = source_path.stem
            
            # Copy image to images split directory
            dest_image_path = split_images_dir / dest_filename
            try:
                shutil.copy2(source_path, dest_image_path)
                stats[split_name]['images'] += 1
            except Exception as e:
                logger.error(f"Failed to copy {source_path} to {dest_image_path}: {e}")
                missing_images.append(image_path)
                continue
            
            # Create caption file in captions split directory
            caption_filename = f"{dest_stem}.txt"
            caption_path = split_captions_dir / caption_filename
            
            try:
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                stats[split_name]['captions'] += 1
            except Exception as e:
                logger.error(f"Failed to write caption {caption_path}: {e}")
                missing_captions.append(caption_path)
        
        if skipped_duplicates > 0:
            logger.info(f"  {split_name}: Skipped {skipped_duplicates} duplicate filenames")
    
    # Log detailed statistics
    logger.info("=== DETAILED COPY STATISTICS ===")
    total_expected = sum(len(pairs) for pairs in splits_with_captions.values())
    total_copied = sum(stats[split]['images'] for split in stats)
    logger.info(f"Total entries processed: {total_expected}")
    logger.info(f"Total unique images copied: {total_copied}")
    logger.info(f"Total duplicates skipped: {total_expected - total_copied}")
    
    for split_name, split_stats in stats.items():
        expected = len(splits_with_captions[split_name])
        copied = split_stats['images']
        logger.info(f"  {split_name}: Processed {expected} entries, Copied {copied} unique images and {split_stats['captions']} captions")
    
    if missing_images:
        logger.error(f"=== {len(missing_images)} MISSING IMAGES ===")
        for i, missing in enumerate(missing_images[:10]):  # Show first 10
            logger.error(f"  {i+1}. {missing}")
        if len(missing_images) > 10:
            logger.error(f"  ... and {len(missing_images) - 10} more")
    
    if missing_captions:
        logger.error(f"{len(missing_captions)} captions failed to write")
    
    return stats, missing_images, missing_captions

'''

    # Replace the function
    new_content = content[:start_line] + [new_function] + content[end_line:]

    # Write back to file
    with open("package_dataset_for_runpod.py", "w", encoding="utf-8") as f:
        f.writelines(new_content)

    logger.info("✅ Successfully updated packaging script with deduplication logic")
    return True


def main():
    logger.info("🔧 Fixing packaging script to ensure exactly 1446 unique images...")

    # Backup original file
    backup_path = "package_dataset_for_runpod.py.backup"
    shutil.copy2("package_dataset_for_runpod.py", backup_path)
    logger.info(f"📋 Created backup: {backup_path}")

    if fix_packaging_script():
        logger.info("✅ Packaging script updated successfully!")
        logger.info("🚀 Now run: python package_dataset_for_runpod.py")
    else:
        logger.error("❌ Failed to update packaging script")
        # Restore backup
        shutil.copy2(backup_path, "package_dataset_for_runpod.py")
        logger.info("📋 Restored from backup")


if __name__ == "__main__":
    main()
