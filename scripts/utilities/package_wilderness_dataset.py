#!/usr/bin/env python3
"""
Package Wilderness Dataset for Training
Creates a dataset package from wilderness_images/ with all images in train split
"""

import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_caption(caption_data):
    """Format caption data into training text"""
    if not caption_data:
        return ""

    # Build formatted caption
    parts = []

    # Add description
    description = caption_data.get("description", "").strip()
    if description:
        parts.append(description)

    # Add terrain information
    terrain = caption_data.get("terrain", [])
    if terrain:
        terrain_str = ", ".join(terrain)
        parts.append(f"Terrain: {terrain_str}")

    # Add features
    features = caption_data.get("features", [])
    if features:
        features_str = ", ".join(features[:5])  # Limit to avoid too long captions
        parts.append(f"Features: {features_str}")

    # Add scene type
    scene_type = caption_data.get("scene_type", "")
    if scene_type:
        parts.append(f"Scene: {scene_type}")

    # Join with periods and ensure proper formatting
    formatted = ". ".join(parts)
    if formatted and not formatted.endswith("."):
        formatted += "."

    return formatted


def load_wilderness_captions():
    """Load the wilderness captions file"""
    captions_file = "captions_grassland.json"

    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file {captions_file} not found!")

    logger.info(f"Loading wilderness captions from {captions_file}...")

    # Load captions with encoding handling
    try:
        with open(captions_file, "r", encoding="utf-8") as f:
            all_captions = json.load(f)
    except UnicodeDecodeError:
        logger.warning("UTF-8 failed, trying with error replacement...")
        with open(captions_file, "r", encoding="utf-8", errors="replace") as f:
            all_captions = json.load(f)

    logger.info(f"Loaded {len(all_captions)} wilderness caption entries")

    # Create image-caption pairs (all go to train)
    train_pairs = []
    missing_captions = []

    for caption_data in all_captions:
        image_path = caption_data.get("image_path", "")
        if not image_path:
            missing_captions.append("Missing image_path")
            continue

        formatted_caption = format_caption(caption_data)
        if formatted_caption.strip():
            train_pairs.append((image_path, formatted_caption))
        else:
            missing_captions.append(f"Empty caption for {image_path}")

    if missing_captions:
        logger.warning(f"Missing captions: {len(missing_captions)}")
        for missing in missing_captions[:5]:  # Log first 5 examples
            logger.warning(f"  {missing}")

    logger.info(f"Created {len(train_pairs)} train image-caption pairs")

    # Return as splits format (only train, no val/test)
    return {"train": train_pairs, "val": [], "test": []}


def create_dataset_structure(output_dir):
    """Create the required folder structure"""
    output_path = Path(output_dir)

    # Remove existing directory if it exists
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create main structure
    output_path.mkdir(parents=True)

    # Create split directories (only train needed)
    (output_path / "images" / "train").mkdir(parents=True)
    (output_path / "captions" / "train").mkdir(parents=True)

    # Val and test are empty but create for compatibility
    (output_path / "images" / "val").mkdir(parents=True)
    (output_path / "images" / "test").mkdir(parents=True)
    (output_path / "captions" / "val").mkdir(parents=True)
    (output_path / "captions" / "test").mkdir(parents=True)

    logger.info(f"Created dataset structure in {output_path}")
    return output_path


def copy_images_and_create_captions(splits_with_captions, base_path):
    """Copy images to split folders and create individual caption files"""
    images_dir = base_path / "images"
    captions_dir = base_path / "captions"

    stats = {
        "train": {"images": 0, "captions": 0},
        "val": {"images": 0, "captions": 0},
        "test": {"images": 0, "captions": 0},
    }

    missing_images = []
    missing_captions = []

    # Only process train split (val and test should be empty)
    for split_name, image_caption_pairs in splits_with_captions.items():
        if not image_caption_pairs:  # Skip empty splits
            continue

        split_images_dir = images_dir / split_name
        split_captions_dir = captions_dir / split_name

        logger.info(
            f"Processing {len(image_caption_pairs)} images for {split_name} split..."
        )

        for image_path, caption in image_caption_pairs:
            source_path = Path(image_path)
            original_filename = source_path.name

            # Check if source image exists
            if not source_path.exists():
                logger.warning(f"Image not found: {source_path}")
                missing_images.append(image_path)
                continue

            # Create unique filename for destination
            dest_filename = original_filename
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

    # Log statistics
    logger.info("=== COPY STATISTICS ===")
    for split_name, split_stats in stats.items():
        if split_stats["images"] > 0 or split_stats["captions"] > 0:
            logger.info(
                f"  {split_name}: {split_stats['images']} images, {split_stats['captions']} captions"
            )

    if missing_images:
        logger.error(f"{len(missing_images)} images failed to copy")
        for missing in missing_images[:5]:
            logger.error(f"  {missing}")

    if missing_captions:
        logger.error(f"{len(missing_captions)} captions failed to write")

    return stats, missing_images, missing_captions


def create_meta_jsonl(base_path, splits_with_captions):
    """Create meta.jsonl file with dataset metadata"""
    meta_path = base_path / "meta.jsonl"

    with open(meta_path, "w", encoding="utf-8") as f:
        for split_name, image_caption_pairs in splits_with_captions.items():
            for image_path, caption in image_caption_pairs:
                filename = Path(image_path).name
                stem = Path(image_path).stem

                meta_entry = {
                    "id": stem,
                    "filename": filename,
                    "split": split_name,
                    "caption": caption,
                    "source_path": image_path,
                }

                f.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")

    logger.info(f"Created {meta_path}")


def create_summary_json(base_path, splits_with_captions, stats):
    """Create summary.json with dataset statistics"""
    summary_path = base_path / "summary.json"

    summary = {
        "dataset_name": "D&D Battlemaps Wilderness Dataset",
        "creation_date": datetime.now().isoformat(),
        "total_images": sum(len(pairs) for pairs in splits_with_captions.values()),
        "splits": {
            split_name: {
                "count": len(pairs),
                "images_copied": stats[split_name]["images"],
                "captions_created": stats[split_name]["captions"],
            }
            for split_name, pairs in splits_with_captions.items()
        },
        "source_directory": "wilderness_images/",
        "caption_source": "captions_grassland.json",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Created {summary_path}")


def copy_config_files(base_path):
    """Copy training configuration and validation files"""
    copied_files = []

    # Copy kohya config if it exists
    if os.path.exists("kohya_training_config.toml"):
        dest_path = base_path / "kohya_training_config.toml"
        shutil.copy2("kohya_training_config.toml", dest_path)
        copied_files.append("kohya_training_config.toml")
        logger.info("Copied kohya_training_config.toml")

    # Copy validation prompts if they exist
    if os.path.exists("validation_prompts.txt"):
        dest_path = base_path / "validation_prompts.txt"
        shutil.copy2("validation_prompts.txt", dest_path)
        copied_files.append("validation_prompts.txt")
        logger.info("Copied validation_prompts.txt")

    return copied_files


def create_split_files(base_path, splits_with_captions):
    """Create train.txt, val.txt, test.txt files"""
    split_files = []

    for split_name, image_caption_pairs in splits_with_captions.items():
        split_file = f"{split_name}.txt"
        dest_path = base_path / split_file

        with open(dest_path, "w", encoding="utf-8") as f:
            for image_path, caption in image_caption_pairs:
                filename = Path(image_path).name
                f.write(f"{filename}\n")

        split_files.append(split_file)
        logger.info(f"Created {split_file} with {len(image_caption_pairs)} entries")

    return split_files


def create_readme(base_path, splits_with_captions, stats):
    """Create README with dataset information"""
    readme_path = base_path / "README.md"

    total_images = sum(len(pairs) for pairs in splits_with_captions.values())

    readme_content = f"""# D&D Battlemaps Wilderness Dataset

## Overview
This dataset contains D&D battlemap images from the wilderness collection with associated captions for LoRA training.

## Dataset Structure
```
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images (empty)
│   └── test/           # Test images (empty)
├── captions/
│   ├── train/          # Training captions (.txt files)
│   ├── val/            # Validation captions (empty)
│   └── test/           # Test captions (empty)
├── train.txt           # Training split file list
├── val.txt             # Validation split file list (empty)
├── test.txt            # Test split file list (empty)
├── meta.jsonl          # Dataset metadata (JSONL format)
├── summary.json        # Dataset statistics and info
├── kohya_training_config.toml  # Training configuration (if available)
├── validation_prompts.txt      # Validation prompts (if available)
└── README.md           # This file
```

## Dataset Information
- **Total Images**: {total_images}
- **Image Format**: PNG/WEBP, various sizes
- **Caption Format**: Individual .txt files per image
- **Split Method**: All images in train split
- **Target Model**: Stable Diffusion 1.5
- **Training Method**: LoRA fine-tuning

## Split Statistics
- **Train**: {len(splits_with_captions['train'])} images ({stats['train']['images']} copied, {stats['train']['captions']} captions)
- **Val**: {len(splits_with_captions['val'])} images (empty)
- **Test**: {len(splits_with_captions['test'])} images (empty)

## Source
- Source directory: `wilderness_images/`
- Caption source: `captions_grassland.json`
- Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage
This dataset is designed for LoRA fine-tuning of Stable Diffusion models for D&D battlemap generation.
All images are placed in the training split for maximum training data utilization.
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info(f"Created {readme_path}")


def create_final_zip(base_path):
    """Create final zip file for upload"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"wilderness_battlemaps_dataset_{timestamp}.zip"

    logger.info(f"Creating final zip: {zip_name}")

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_path):
            for file in files:
                file_path = Path(root) / file
                arc_name = file_path.relative_to(base_path)
                zipf.write(file_path, arc_name)

    zip_size = Path(zip_name).stat().st_size / (1024 * 1024)  # MB
    logger.info(f"Created {zip_name} ({zip_size:.1f} MB)")

    return zip_name


def main():
    """Main packaging function"""
    try:
        # Configuration
        output_dir = "wilderness_dataset_package"

        logger.info("=== WILDERNESS DATASET PACKAGING ===")

        # Load wilderness captions and create splits
        splits_with_captions = load_wilderness_captions()

        # Create dataset structure
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

        # Create split files
        split_files = create_split_files(base_path, splits_with_captions)

        # Create README
        create_readme(base_path, splits_with_captions, stats)

        # Create final zip
        zip_name = create_final_zip(base_path)

        # Final summary
        total_copied = sum(stats[split]["images"] for split in stats)
        logger.info("=== PACKAGING COMPLETE ===")
        logger.info(f"Total images processed: {total_copied}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Final package: {zip_name}")

        if missing_images:
            logger.warning(f"Missing images: {len(missing_images)}")
        if missing_captions:
            logger.warning(f"Missing captions: {len(missing_captions)}")

        return True

    except Exception as e:
        logger.error(f"Packaging failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
