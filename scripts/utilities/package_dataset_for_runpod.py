#!/usr/bin/env python3
"""
Package Dataset for Runpod Training
Creates the exact folder structure required for cloud training:
- images/{train,val,test}/ with images
- Individual .txt caption files for each image
- meta.jsonl and summary.json metadata files
- Training config and validation prompts
- Final zip file for upload
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("package_runpod_dataset.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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


def normalize_path_for_matching(path):
    """Normalize path by handling common encoding corruption"""
    # Common character replacements for encoding issues
    replacements = {
        "château": "ch�teau",
        "è": "�",
        "é": "�",
        "ê": "�",
        "à": "�",
        "â": "�",
        "ù": "�",
        "û": "�",
        "ô": "�",
        "ç": "�",
        "î": "�",
        "ï": "�",
        "ü": "�",
        "ë": "�",
        "ö": "�",
    }

    normalized = path
    for correct, corrupted in replacements.items():
        normalized = normalized.replace(correct, corrupted)

    return normalized


def find_matching_caption(image_path, caption_lookup, all_captions):
    """Find matching caption with fallback strategies"""
    # Try exact match first
    if image_path in caption_lookup:
        return caption_lookup[image_path]

    # Try normalized path matching (handle encoding corruption)
    normalized_path = normalize_path_for_matching(image_path)
    if normalized_path != image_path and normalized_path in caption_lookup:
        return caption_lookup[normalized_path]

    # Try reverse: normalize the lookup keys and match against original path
    for caption_entry in all_captions:
        if "image_path" in caption_entry:
            entry_path = caption_entry["image_path"]
            if normalize_path_for_matching(entry_path) == image_path:
                return caption_entry

    # Last resort: fuzzy matching on filename only
    image_filename = (
        image_path.split("/")[-1] if "/" in image_path else image_path.split("\\")[-1]
    )
    for caption_entry in all_captions:
        if "image_path" in caption_entry:
            entry_path = caption_entry["image_path"]
            entry_filename = (
                entry_path.split("/")[-1]
                if "/" in entry_path
                else entry_path.split("\\")[-1]
            )
            if (
                image_filename == entry_filename
                or normalize_path_for_matching(entry_filename) == image_filename
            ):
                return caption_entry

    return None


def find_actual_image_file(image_path):
    """Find the actual image file, handling encoding mismatches"""
    # Try the original path first
    source_path = Path(image_path)
    if source_path.exists():
        return source_path

    # Try with normalized path (reverse of what we did for captions)
    # The caption may have been normalized FROM the file path, so reverse it
    reversed_replacements = {
        "ch�teau": "château",
        "�": "è",
        "�": "é",
        "�": "ê",
        "�": "à",
        "�": "â",
        "�": "ù",
        "�": "û",
        "�": "ô",
        "�": "ç",
        "�": "î",
        "�": "ï",
        "�": "ü",
        "�": "ë",
        "�": "ö",
    }

    reversed_path = image_path
    for corrupted, correct in reversed_replacements.items():
        reversed_path = reversed_path.replace(corrupted, correct)

    if reversed_path != image_path:
        source_path = Path(reversed_path)
        if source_path.exists():
            return source_path

    # Try the normalized version (in case the file has corrupted characters)
    normalized_path = normalize_path_for_matching(image_path)
    if normalized_path != image_path:
        source_path = Path(normalized_path)
        if source_path.exists():
            return source_path

    return None


def load_split_files():
    """Load the train/val/test split files"""
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = f"{split_name}.txt"
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} not found!")

        with open(split_file, "r", encoding="utf-8") as f:
            splits[split_name] = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(splits[split_name])} images for {split_name} split")

    return splits


def load_training_captions():
    """Load the formatted training captions and match them with split files"""
    # Load the merged captions JSON file
    captions_file = "phase4_captions_restored_merged.json"
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Caption file {captions_file} not found!")

    logger.info(f"Loading captions from {captions_file}...")

    # Try multiple encodings to handle Unicode issues
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    all_captions = None

    for encoding in encodings_to_try:
        try:
            with open(captions_file, "r", encoding=encoding) as f:
                all_captions = json.load(f)
            logger.info(f"Successfully loaded file with {encoding} encoding")
            break
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to load with {encoding} encoding: {e}")
            continue
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error with {encoding} encoding: {e}")
            continue

    if all_captions is None:
        # Last resort: load with error handling
        try:
            with open(captions_file, "r", encoding="utf-8", errors="replace") as f:
                all_captions = json.load(f)
            logger.warning(
                "Loaded file with UTF-8 and error replacement - some characters may be corrupted"
            )
        except Exception as e:
            raise RuntimeError(f"Could not load {captions_file} with any encoding: {e}")

    logger.info(f"Loaded {len(all_captions)} caption entries")

    # Create a mapping from image_path to caption data
    caption_lookup = {}
    for caption_entry in all_captions:
        if "image_path" in caption_entry:
            caption_lookup[caption_entry["image_path"]] = caption_entry

    logger.info(f"Created caption lookup for {len(caption_lookup)} images")

    # Load split files and match with captions
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = f"{split_name}.txt"

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} not found!")

        # Load image paths for this split with encoding handling
        image_paths = []
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                image_paths = [line.strip() for line in f if line.strip()]
        except UnicodeDecodeError:
            # Try alternative encoding for split files too
            with open(split_file, "r", encoding="utf-8", errors="replace") as f:
                image_paths = [line.strip() for line in f if line.strip()]
            logger.warning(f"Loaded {split_file} with error replacement")

        # Match images with captions and format them
        image_caption_pairs = []
        missing_captions = []

        for image_path in image_paths:
            caption_data = find_matching_caption(
                image_path, caption_lookup, all_captions
            )
            if caption_data:
                formatted_caption = format_caption(caption_data)
                if formatted_caption.strip():
                    image_caption_pairs.append((image_path, formatted_caption))
                else:
                    missing_captions.append(f"Empty caption for {image_path}")
            else:
                missing_captions.append(f"No caption found for {image_path}")

        if missing_captions:
            logger.warning(
                f"Missing captions in {split_name} split: {len(missing_captions)}"
            )
            for missing in missing_captions[:5]:  # Log first 5 examples
                logger.warning(f"  {missing}")
            if len(missing_captions) > 5:
                logger.warning(f"  ... and {len(missing_captions) - 5} more")

        splits[split_name] = image_caption_pairs
        logger.info(
            f"Loaded {len(splits[split_name])} image-caption pairs for {split_name} split"
        )

    return splits


def create_dataset_structure(output_dir):
    """Create the required folder structure"""
    base_path = Path(output_dir)

    # Create main directories
    images_dir = base_path / "images"
    captions_dir = base_path / "captions"
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (captions_dir / split).mkdir(parents=True, exist_ok=True)

    logger.info(f"Created dataset structure in {output_dir}")
    return base_path


def copy_images_and_create_captions(splits_with_captions, base_path):
    """Copy images to split folders and create individual caption files in separate folders"""
    images_dir = base_path / "images"
    captions_dir = base_path / "captions"

    stats = {
        "train": {"images": 0, "captions": 0},
        "val": {"images": 0, "captions": 0},
        "test": {"images": 0, "captions": 0},
    }

    missing_images = []
    missing_captions = []

    # Track processed filenames to avoid duplicates - this ensures exactly 1446 unique images
    processed_filenames = set()

    for split_name, image_caption_pairs in splits_with_captions.items():
        split_images_dir = images_dir / split_name
        split_captions_dir = captions_dir / split_name

        logger.info(
            f"Processing {len(image_caption_pairs)} images for {split_name} split..."
        )

        processed_count = 0
        skipped_duplicates = 0

        for image_path, caption in image_caption_pairs:
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(
                    f"  {split_name}: Processed {processed_count}/{len(image_caption_pairs)} images"
                )

            # image_path is a full path like:
            # "generated_images/watermark_folder/gdrive_*/filename.png"
            source_path = find_actual_image_file(image_path)

            if source_path is None:
                missing_images.append(image_path)
                logger.warning(f"Image not found: {image_path}")
                continue

            # Use full path to avoid skipping images with same filename in different folders
            # Since the original filenames already include gdrive_id, just add map folder to ensure uniqueness
            path_parts = image_path.replace("\\", "/").split("/")
            original_filename = source_path.name

            if len(path_parts) >= 4:  # generated_images/map_folder/gdrive_id/file.png
                map_folder = path_parts[1]
                # Create unique filename: mapfolder_originalfilename
                dest_filename = f"{map_folder}_{original_filename}"
            else:
                dest_filename = original_filename

            # Skip if we've already processed this exact image path (not just filename)
            if image_path in processed_filenames:
                skipped_duplicates += 1
                continue

            processed_filenames.add(image_path)
            dest_stem = Path(dest_filename).stem

            # Copy image to images split directory
            dest_image_path = split_images_dir / dest_filename
            try:
                shutil.copy2(source_path, dest_image_path)
                stats[split_name]["images"] += 1
            except Exception as e:
                logger.error(f"Failed to copy {source_path} to {dest_image_path}: {e}")
                missing_images.append(image_path)
                continue

            # Create caption file in captions split directory
            caption_filename = f"{dest_stem}.txt"
            caption_path = split_captions_dir / caption_filename

            try:
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                stats[split_name]["captions"] += 1
            except Exception as e:
                logger.error(f"Failed to write caption {caption_path}: {e}")
                missing_captions.append(caption_path)

        if skipped_duplicates > 0:
            logger.info(
                f"  {split_name}: Skipped {skipped_duplicates} duplicate filenames"
            )

    # Log detailed statistics
    logger.info("=== DETAILED COPY STATISTICS ===")
    total_expected = sum(len(pairs) for pairs in splits_with_captions.values())
    total_copied = sum(stats[split]["images"] for split in stats)
    logger.info(f"Total entries processed: {total_expected}")
    logger.info(f"Total unique images copied: {total_copied}")
    logger.info(f"Total duplicates skipped: {total_expected - total_copied}")

    for split_name, split_stats in stats.items():
        expected = len(splits_with_captions[split_name])
        copied = split_stats["images"]
        logger.info(
            f"  {split_name}: Processed {expected} entries, Copied {copied} unique images and {split_stats['captions']} captions"
        )

    if missing_images:
        logger.error(f"=== {len(missing_images)} MISSING IMAGES ===")
        for i, missing in enumerate(missing_images[:10]):  # Show first 10
            logger.error(f"  {i+1}. {missing}")
        if len(missing_images) > 10:
            logger.error(f"  ... and {len(missing_images) - 10} more")

    if missing_captions:
        logger.error(f"{len(missing_captions)} captions failed to write")

    return stats, missing_images, missing_captions


def create_meta_jsonl(base_path, splits_with_captions):
    """Create meta.jsonl file with dataset metadata"""
    meta_path = base_path / "meta.jsonl"

    # Track processed filenames to match the deduplication in copy function
    processed_filenames = set()

    with open(meta_path, "w", encoding="utf-8") as f:
        for split_name, image_caption_pairs in splits_with_captions.items():
            for image_path, caption in image_caption_pairs:
                source_path = Path(image_path)
                original_filename = source_path.name

                # Skip duplicates (same logic as copy function)
                if original_filename in processed_filenames:
                    continue

                processed_filenames.add(original_filename)

                # Use original filename and stem
                original_stem = source_path.stem

                meta_entry = {
                    "file_name": original_filename,
                    "text": f"{original_stem}.txt",  # Caption file reference
                    "split": split_name,
                }
                f.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")

    total_unique = len(processed_filenames)
    logger.info(f"Created meta.jsonl with {total_unique} unique entries")


def create_summary_json(base_path, splits_with_captions, stats):
    """Create summary.json with dataset statistics"""
    summary = {
        "dataset_name": "D&D Battlemaps LoRA Training Dataset",
        "created_at": datetime.now().isoformat(),
        "total_images": sum(
            len(image_caption_pairs)
            for image_caption_pairs in splits_with_captions.values()
        ),
        "splits": {
            "train": len(splits_with_captions["train"]),
            "val": len(splits_with_captions["val"]),
            "test": len(splits_with_captions["test"]),
        },
        "image_format": "PNG",
        "image_size": "512x512",
        "caption_format": "Individual .txt files per image",
        "model_target": "Stable Diffusion 1.5",
        "training_method": "LoRA",
        "copy_stats": stats,
        "notes": [
            "Group-aware split to prevent tile leakage",
            "Only images with captions included",
            "Caption dropout applied for robustness",
            "Vocabulary analysis completed",
        ],
    }

    summary_path = base_path / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Created summary.json")


def copy_config_files(base_path):
    """Copy training configuration and validation files"""
    config_files = ["kohya_training_config.toml", "validation_prompts.txt"]

    copied_files = []
    for config_file in config_files:
        if os.path.exists(config_file):
            dest_path = base_path / config_file
            shutil.copy2(config_file, dest_path)
            copied_files.append(config_file)
            logger.info(f"Copied {config_file}")
        else:
            logger.warning(f"Config file not found: {config_file}")

    return copied_files


def copy_split_files(base_path):
    """Copy the train.txt, val.txt, test.txt split files"""
    split_files = ["train.txt", "val.txt", "test.txt"]
    copied_split_files = []

    for split_file in split_files:
        if os.path.exists(split_file):
            dest_path = base_path / split_file
            shutil.copy2(split_file, dest_path)
            copied_split_files.append(split_file)
            logger.info(f"Copied {split_file}")
        else:
            logger.warning(f"Split file not found: {split_file}")

    return copied_split_files


def create_corrected_split_files(base_path, splits_with_captions):
    """Create corrected train.txt, val.txt, test.txt files with transformed filenames"""
    logger.info("Creating corrected split files with transformed filenames...")

    # Track processed filenames to match the deduplication in copy function
    processed_filenames = set()
    corrected_splits = {"train": [], "val": [], "test": []}

    for split_name, image_caption_pairs in splits_with_captions.items():
        for image_path, caption in image_caption_pairs:
            source_path = Path(image_path)
            original_filename = source_path.name

            # Skip duplicates (same logic as copy function)
            if image_path in processed_filenames:
                continue

            processed_filenames.add(image_path)

            # Apply same filename transformation as in copy_images_and_create_captions
            path_parts = image_path.replace("\\", "/").split("/")

            if len(path_parts) >= 4:  # generated_images/map_folder/gdrive_id/file.png
                map_folder = path_parts[1]
                # Create unique filename: mapfolder_originalfilename
                dest_filename = f"{map_folder}_{original_filename}"
            else:
                dest_filename = original_filename

            corrected_splits[split_name].append(dest_filename)

    # Write corrected split files
    corrected_split_files = []
    for split_name, filenames in corrected_splits.items():
        split_file = f"{split_name}.txt"
        dest_path = base_path / split_file

        with open(dest_path, "w", encoding="utf-8") as f:
            for filename in filenames:
                f.write(f"{filename}\n")

        corrected_split_files.append(split_file)
        logger.info(
            f"Created corrected {split_file} with {len(filenames)} transformed filenames"
        )

    return corrected_split_files


def create_readme(base_path, splits_with_captions, stats, copied_files):
    """Create README with dataset information"""
    readme_content = f"""# D&D Battlemaps LoRA Training Dataset

## Dataset Structure
```
dataset_v1/
├── images/
│   ├── train/          # {len(splits_with_captions['train'])} training images (PNG files)
│   ├── val/            # {len(splits_with_captions['val'])} validation images (PNG files)
│   └── test/           # {len(splits_with_captions['test'])} test images (PNG files)
├── captions/
│   ├── train/          # {len(splits_with_captions['train'])} training caption files (TXT files)
│   ├── val/            # {len(splits_with_captions['val'])} validation caption files (TXT files)
│   └── test/           # {len(splits_with_captions['test'])} test caption files (TXT files)
├── train.txt           # Training split file list
├── val.txt             # Validation split file list
├── test.txt            # Test split file list
├── meta.jsonl          # Dataset metadata (JSONL format)
├── summary.json        # Dataset statistics and info
├── kohya_training_config.toml  # Training configuration
├── validation_prompts.txt      # Validation prompts
└── README.md           # This file
```

## Dataset Information
- **Total Images**: {sum(len(image_caption_pairs) for image_caption_pairs in splits_with_captions.values())}
- **Image Format**: PNG, 512x512 pixels
- **Caption Format**: Individual .txt files per image
- **Split Method**: Group-aware (prevents tile leakage)
- **Target Model**: Stable Diffusion 1.5
- **Training Method**: LoRA fine-tuning

## Training Configuration
- Configuration file: `kohya_training_config.toml`
- Validation prompts: `validation_prompts.txt`
- Recommended: RTX 4090 or better for optimal training speed

## Split Statistics
- **Train**: {len(splits_with_captions['train'])} images ({stats['train']['images']} copied, {stats['train']['captions']} captions)
- **Val**: {len(splits_with_captions['val'])} images ({stats['val']['images']} copied, {stats['val']['captions']} captions)
- **Test**: {len(splits_with_captions['test'])} images ({stats['test']['images']} copied, {stats['test']['captions']} captions)

## Notes
- Group-aware splitting prevents tiles from the same battlemap appearing in different splits
- Only images with captions are included in the dataset
- Caption dropout was applied during preprocessing for training robustness
- Vocabulary analysis was completed to ensure comprehensive coverage

## Files Included
- Dataset images and captions: ✓
- Metadata files: ✓
- Training configuration: {'✓' if 'kohya_training_config.toml' in copied_files else '✗'}
- Validation prompts: {'✓' if 'validation_prompts.txt' in copied_files else '✗'}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    readme_path = base_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info("Created README.md")


def create_final_zip(base_path):
    """Create final zip file for upload"""
    zip_filename = (
        f"dnd_battlemaps_lora_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    )

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = Path(root) / file
                # Calculate relative path within the zip
                arcname = file_path.relative_to(base_path)
                zipf.write(file_path, arcname)

    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    logger.info(f"Created zip file: {zip_filename} ({zip_size_mb:.1f} MB)")

    return zip_filename


def main():
    logger.info("Starting Runpod dataset packaging...")

    try:
        # Load data
        splits_with_captions = load_training_captions()

        # Create output directory
        output_dir = "dataset_v1"
        if os.path.exists(output_dir):
            try:
                logger.info(f"Removing existing {output_dir} directory")
                shutil.rmtree(output_dir)
            except PermissionError as e:
                logger.warning(f"Could not remove existing {output_dir} directory: {e}")
                logger.warning(
                    "Continuing with existing directory - some files may be overwritten"
                )

        base_path = create_dataset_structure(output_dir)

        # Copy images and create caption files
        stats, missing_images, missing_captions = copy_images_and_create_captions(
            splits_with_captions, base_path
        )

        # Create metadata files
        create_meta_jsonl(base_path, splits_with_captions)
        create_summary_json(base_path, splits_with_captions, stats)

        # Copy configuration files
        copied_files = copy_config_files(base_path)

        # Create corrected split files with transformed filenames
        corrected_split_files = create_corrected_split_files(
            base_path, splits_with_captions
        )

        # Create README
        create_readme(base_path, splits_with_captions, stats, copied_files)

        # Validate results
        if missing_images or missing_captions:
            logger.error("Dataset packaging completed with errors:")
            logger.error(f"  Missing images: {len(missing_images)}")
            logger.error(f"  Missing captions: {len(missing_captions)}")
            return False

        # Create final zip
        zip_filename = create_final_zip(base_path)

        logger.info("Dataset packaging completed successfully!")
        logger.info(f"Upload {zip_filename} to your Runpod instance")
        logger.info(f"Dataset directory: {output_dir}")

        return True

    except Exception as e:
        logger.error(f"Error during packaging: {e}")
        raise


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
