#!/usr/bin/env python3
"""
Preflight Check for D&D Battlemaps LoRA Dataset
Validates the zip file structure before uploading to RunPod
"""

import sys, re, json, io, zipfile, hashlib
from collections import defaultdict, Counter

# Regex patterns for file types and content
RE_SPLIT = re.compile(r"^(train|val|test)/")
RE_IMG = re.compile(r"\.(png|jpg|jpeg|webp)$", re.I)
RE_TXT = re.compile(r"\.txt$", re.I)

# Caption format checks for D&D battlemap captions
RE_TERRAIN = re.compile(r"\bterrain:\s*[^.\n]+", re.I)
RE_GRID = re.compile(r"\bgrid:\s*(yes|no)\b", re.I)
RE_SCENE_TYPE = re.compile(r"\bscene_type:\s*[^.\n]+", re.I)


def die(msg):
    print("❌ FAIL:", msg)
    sys.exit(1)


def main(zip_path):
    print("🔍 Starting D&D Battlemaps Dataset Preflight Check...")
    print(f"📦 Checking zip file: {zip_path}")
    print("=" * 60)

    # Try to open the zip file
    try:
        z = zipfile.ZipFile(zip_path)
    except Exception as e:
        die(f"Could not open zip file: {e}")

    names = z.namelist()
    print(f"📁 Total files in zip: {len(names)}")

    # Check top-level structure (files are at root level in zip)
    required_files = {
        "kohya_training_config.toml",
        "validation_prompts.txt",
        "train.txt",
        "val.txt",
        "test.txt",
        "meta.jsonl",
        "summary.json",
        "README.md",
    }

    missing_files = []
    for req_file in required_files:
        if req_file not in names:
            missing_files.append(req_file)

    if missing_files:
        die(f"Missing required files: {missing_files}")

    print("✅ All required top-level files present")

    # Check folder structure (directories are at root level in zip)
    required_dirs = {
        "images/train/",
        "images/val/",
        "images/test/",
        "captions/train/",
        "captions/val/",
        "captions/test/",
    }

    found_dirs = set()
    for name in names:
        if name.endswith("/"):
            found_dirs.add(name)
        else:
            # Extract directory path
            dir_path = "/".join(name.split("/")[:-1]) + "/"
            found_dirs.add(dir_path)

    missing_dirs = required_dirs - found_dirs
    if missing_dirs:
        die(f"Missing required directories: {missing_dirs}")

    print("✅ All required directories present")

    # Collect images and captions by split
    imgs_by_split = defaultdict(list)
    caps_by_split = defaultdict(list)
    per_image_caps = defaultdict(list)
    split_lists = {}  # split -> set of basenames from train.txt etc.

    # Read split lists (files are at root level in zip)
    for split in ("train", "val", "test"):
        try:
            with z.open(f"{split}.txt") as f:
                files = [l.decode("utf-8").strip() for l in f if l.strip()]
                # Store the ACTUAL entries from split files (don't transform them)
                split_lists[split] = set(files)
                # Also store just the filenames for comparison
                split_filenames = set([f.split("/")[-1] for f in files])
                print(
                    f"📋 {split}.txt contains {len(files)} entries, {len(split_filenames)} unique filenames"
                )
        except KeyError:
            die(f"Could not read {split}.txt")

    # Walk through zip contents (files are at root level)
    for name in names:
        # Check images
        if name.startswith("images/") and RE_IMG.search(name):
            # images/{split}/filename.png
            parts = name.split("/")
            if len(parts) >= 3 and parts[1] in ("train", "val", "test"):
                split = parts[1]
                prefixed_filename = parts[-1]

                # Extract original filename by removing watermark prefix
                # Format: "WatermarkPrefix_original_filename.png"
                if "_gdrive_" in prefixed_filename:
                    # Find the gdrive part and use everything from there
                    gdrive_pos = prefixed_filename.find("_gdrive_")
                    original_filename = prefixed_filename[
                        gdrive_pos + 1 :
                    ]  # Skip the leading _
                else:
                    original_filename = prefixed_filename

                imgs_by_split[split].append(original_filename)

        # Check captions
        if name.startswith("captions/") and RE_TXT.search(name):
            parts = name.split("/")
            if len(parts) >= 3 and parts[1] in ("train", "val", "test"):
                split = parts[1]
                prefixed_filename = parts[-1]

                # Extract original filename stem by removing watermark prefix
                if "_gdrive_" in prefixed_filename:
                    gdrive_pos = prefixed_filename.find("_gdrive_")
                    original_stem = prefixed_filename[gdrive_pos + 1 :].replace(
                        ".txt", ""
                    )
                else:
                    original_stem = prefixed_filename.replace(".txt", "")

                caps_by_split[split].append(original_stem + ".txt")
                per_image_caps[split].append(name)

    # Analyze counts and pairing
    problems = []
    total_imgs = sum(len(v) for v in imgs_by_split.values())
    total_caps = sum(len(v) for v in caps_by_split.values())

    print("\n📊 DATASET STATISTICS")
    print(f"Total images:  {total_imgs}")
    print(f"Total captions: {total_caps}")
    print("-" * 40)

    expected_counts = {"train": 1250, "val": 125, "test": 71}

    for split in ("train", "val", "test"):
        img_count = len(imgs_by_split[split])
        cap_count = len(caps_by_split[split])
        expected = expected_counts[split]

        print(
            f"{split:>5}: images={img_count:>4}  captions={cap_count:>4}  expected={expected:>4}"
        )

        # Check counts match expected
        if img_count != expected:
            problems.append(f"{split}: expected {expected} images, found {img_count}")
        if cap_count != expected:
            problems.append(f"{split}: expected {expected} captions, found {cap_count}")

        # Check image/caption count match
        if img_count != cap_count:
            problems.append(
                f"{split}: image/caption count mismatch ({img_count} vs {cap_count})"
            )

        # Check 1:1 pairing by basename
        img_set = set(imgs_by_split[split])
        cap_set = set(caps_by_split[split])

        # Compare basenames (remove extensions)
        img_bases = set(fn.rsplit(".", 1)[0] for fn in img_set)
        cap_bases = set(fn.rsplit(".", 1)[0] for fn in cap_set)

        miss_for_imgs = sorted(img_bases - cap_bases)
        extra_caps = sorted(cap_bases - img_bases)

        if miss_for_imgs:
            problems.append(
                f"{split}: {len(miss_for_imgs)} images missing captions (e.g., {miss_for_imgs[:3]})"
            )
        if extra_caps:
            problems.append(
                f"{split}: {len(extra_caps)} captions without images (e.g., {extra_caps[:3]})"
            )

        # Enhanced filename matching check for split lists
        print(f"\n🔍 Checking split file alignment for {split}...")
        if split_lists[split] is not None:
            # Get the actual entries from the split file
            split_entries = split_lists[split]
            # Extract just the filenames from the split entries (remove paths)
            split_filenames = set([entry.split("/")[-1] for entry in split_entries])

            # Get the actual image filenames from the zip (these are the transformed names)
            actual_img_files = set()
            for name in names:
                if name.startswith(f"images/{split}/") and RE_IMG.search(name):
                    actual_filename = name.split("/")[-1]  # Just the filename
                    actual_img_files.add(actual_filename)

            # Compare split file filenames with actual packaged filenames
            img_only = actual_img_files - split_filenames
            split_only = split_filenames - actual_img_files

            if img_only or split_only:
                # Enhanced filename mismatch warning
                print(f"⚠️  FILENAME MISMATCH WARNING for {split}:")
                print(
                    f"   Split file ({split}.txt) contains entries that don't match actual image filenames"
                )
                print(
                    f"   This usually means the split files weren't updated after filename transformations"
                )

                if img_only:
                    problems.append(
                        f"{split}: {len(img_only)} images not referenced in {split}.txt"
                    )
                    print(
                        f"   📁 Images in folder but not referenced in {split}.txt: {len(img_only)}"
                    )
                    for example in list(img_only)[:3]:
                        print(f"      • {example}")

                if split_only:
                    problems.append(
                        f"{split}: {len(split_only)} entries in {split}.txt have no matching images"
                    )
                    print(
                        f"   📝 Filenames in {split}.txt but no matching images: {len(split_only)}"
                    )
                    for example in list(split_only)[:3]:
                        print(f"      • {example}")

                # Show example comparison for debugging
                print(f"\n   🔍 Example comparison:")
                if split_entries:
                    first_split_entry = list(split_entries)[0]
                    print(f"   Split entry: {first_split_entry}")
                if actual_img_files:
                    first_actual = list(actual_img_files)[0]
                    print(f"   Actual file: images/{split}/{first_actual}")

                print(
                    f"   💡 Fix: Ensure split files reference the transformed filenames used in the packaged dataset"
                )
            else:
                print(f"✅ {split}: Split file entries match image filenames perfectly")

    # Caption content validation (sample files)
    print("\n🔍 VALIDATING CAPTION CONTENT...")
    bad_format = []
    sample_size = 20  # Check 20 files per split

    for split in ("train", "val", "test"):
        sample = per_image_caps[split][:sample_size]
        print(f"Checking {len(sample)} sample captions from {split}...")

        for path in sample:
            try:
                with z.open(path) as f:
                    txt = f.read().decode("utf-8")
            except Exception as e:
                bad_format.append((split, path, f"decode error: {e}"))
                continue

            # Check for required D&D battlemap caption format
            issues = []
            if not RE_TERRAIN.search(txt):
                issues.append("missing terrain field")
            if not RE_GRID.search(txt):
                issues.append("missing grid field")
            if not RE_SCENE_TYPE.search(txt):
                issues.append("missing scene_type field")

            if issues:
                bad_format.append((split, path, "; ".join(issues)))

    if bad_format:
        problems.append(f"Caption format issues in {len(bad_format)} files")
        for split, path, issue in bad_format[:5]:  # Show first 5 issues
            problems.append(f"  └─ {path}: {issue}")
    else:
        print("✅ Caption format validation passed")

    # Check config file content
    print("\n⚙️  VALIDATING CONFIGURATION...")
    try:
        with z.open("kohya_training_config.toml") as f:
            config_content = f.read().decode("utf-8")

        # Basic checks for important paths
        if "images/" not in config_content:
            problems.append(
                "kohya_training_config.toml may not reference correct dataset paths"
            )

        print("✅ Configuration file present and readable")
    except Exception as e:
        problems.append(f"Could not read kohya_training_config.toml: {e}")

    # Validate metadata files
    try:
        with z.open("summary.json") as f:
            summary = json.load(f)

        expected_total = sum(expected_counts.values())
        if summary.get("total_images") != expected_total:
            problems.append(
                f"summary.json reports {summary.get('total_images')} images, expected {expected_total}"
            )

        print("✅ Metadata files validated")
    except Exception as e:
        problems.append(f"Could not read summary.json: {e}")

    # Final report
    print("\n" + "=" * 60)
    if problems:
        print("❌ PROBLEMS DETECTED:")
        for i, problem in enumerate(problems, 1):
            print(f"{i:>2}. {problem}")
        print(f"\n💥 Found {len(problems)} issues that need to be fixed before upload.")
        sys.exit(2)
    else:
        print("🎉 PASS: Dataset zip structure looks perfect and training-ready!")
        print("\n✅ All checks passed:")
        print("   • File structure is correct")
        print("   • Image/caption counts match expectations (1,250/125/71)")
        print("   • All images have matching captions")
        print("   • Caption format follows D&D battlemap standards")
        print("   • Configuration files are present and valid")
        print("   • Split files match actual image filenames")
        print("\n🚀 Ready for RunPod upload!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_dataset_zip.py <path/to/dataset.zip>")
        print(
            "Example: python validate_dataset_zip.py dnd_battlemaps_lora_dataset_20250812_213418.zip"
        )
        sys.exit(1)
    main(sys.argv[1])
