#!/usr/bin/env python3
"""
Check for duplicate filenames in the generated_images directory.
This could explain why the zip has fewer files than expected.
"""

import os
from pathlib import Path
from collections import Counter, defaultdict


def main():
    print("🔍 Checking for duplicate filenames in generated_images...")

    # Check for duplicate filenames across all watermark folders
    all_files = []
    file_locations = defaultdict(list)

    for root, dirs, files in os.walk("generated_images"):
        for file in files:
            if file.endswith(".png"):
                all_files.append(file)
                file_locations[file].append(os.path.join(root, file))

    counts = Counter(all_files)
    duplicates = {name: count for name, count in counts.items() if count > 1}

    print(f"📊 Total PNG files found: {len(all_files)}")
    print(f"📊 Unique filenames: {len(counts)}")
    print(f"📊 Duplicate filenames: {len(duplicates)}")

    if duplicates:
        print(f"\n❌ Found {len(duplicates)} duplicate filenames!")
        print("First 10 duplicates:")
        for i, (name, count) in enumerate(list(duplicates.items())[:10]):
            print(f"  {i+1}. {name}: {count} copies")
            # Show where each copy is located
            for location in file_locations[name]:
                print(f"      {location}")

        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more duplicates")

        # Check if duplicates affect our dataset
        print(f"\n🔍 Checking if duplicates affect our splits...")

        # Load split files to see if they contain duplicates
        for split_name in ["train", "val", "test"]:
            split_file = f"{split_name}.txt"
            if os.path.exists(split_file):
                with open(split_file, "r", encoding="utf-8") as f:
                    split_paths = [line.strip() for line in f if line.strip()]

                # Extract just filenames from full paths
                split_filenames = [Path(path).name for path in split_paths]
                split_duplicates = [
                    name for name in split_filenames if duplicates.get(name, 0) > 1
                ]

                if split_duplicates:
                    print(
                        f"  ❌ {split_name}: {len(split_duplicates)} duplicate filenames in split"
                    )
                else:
                    print(f"  ✅ {split_name}: No duplicate filenames in split")

    else:
        print("✅ No duplicate filenames found!")
        print("The issue must be something else...")


if __name__ == "__main__":
    main()
