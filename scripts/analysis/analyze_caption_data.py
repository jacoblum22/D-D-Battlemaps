#!/usr/bin/env python3
"""
Check the original caption data to understand the 1446 vs unique filename issue
"""

import json


def main():
    print("🔍 Analyzing original caption data...")

    with open("phase4_captions.json", "r") as f:
        data = json.load(f)

    print(f"📊 Total captions in phase4_captions.json: {len(data)}")
    print(f"📊 Data type: {type(data)}")

    if isinstance(data, list):
        print("📊 Data is a list - checking structure...")
        if data:
            print(
                f"📊 First entry keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}"
            )

        # Extract image paths from list structure
        img_paths = []
        for entry in data:
            if isinstance(entry, dict) and "image_path" in entry:
                img_paths.append(entry["image_path"])
            elif isinstance(entry, dict) and "file_path" in entry:
                img_paths.append(entry["file_path"])
    else:
        img_paths = list(data.keys())

    print(f"📊 Found {len(img_paths)} image paths")

    # Check unique filenames
    unique_filenames = set()
    filename_counts = {}

    for img_path in img_paths:
        filename = img_path.split("/")[-1]
        unique_filenames.add(filename)
        filename_counts[filename] = filename_counts.get(filename, 0) + 1

    print(f"📊 Unique filenames in captions: {len(unique_filenames)}")

    # Check for duplicates
    duplicates = {name: count for name, count in filename_counts.items() if count > 1}
    print(f"📊 Duplicate filenames: {len(duplicates)}")

    if duplicates:
        print("\n🔍 First 5 duplicate filenames:")
        for i, (name, count) in enumerate(list(duplicates.items())[:5]):
            print(f"  {i+1}. {name}: {count} copies")
            # Show the full paths for this filename
            paths = [path for path in img_paths if path.endswith(name)]
            for path in paths[:3]:  # Show first 3 paths
                print(f"      {path}")

    print(f"\n✅ Expected unique images: {len(unique_filenames)}")
    print(f"✅ Total caption entries: {len(data)}")


if __name__ == "__main__":
    main()
