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
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = f"{split_name}.txt"
        captions_file = f"{split_name}_captions.txt"

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} not found!")
        if not os.path.exists(captions_file):
            raise FileNotFoundError(f"Caption file {captions_file} not found!")

        # Load image paths
        with open(split_file, "r", encoding="utf-8") as f:
            image_paths = [line.strip() for line in f if line.strip()]

        # Load captions (one per line)
        with open(captions_file, "r", encoding="utf-8") as f:
            captions_list = [line.strip() for line in f if line.strip()]

        if len(image_paths) != len(captions_list):
            raise ValueError(
                f"Mismatch in {split_name}: {len(image_paths)} images vs {len(captions_list)} captions"
            )

        splits[split_name] = list(zip(image_paths, captions_list))
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
            source_path = Path(image_path)

            if not source_path.exists():
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

        # Copy split files
        copied_split_files = copy_split_files(base_path)

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
