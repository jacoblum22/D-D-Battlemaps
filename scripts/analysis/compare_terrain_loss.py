#!/usr/bin/env python3
"""
Compare phase4_captions.json and phase4_captions original.json to identify terrain loss.
"""

import json
import sys
from typing import Dict, List, Any


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Load JSON file and return data."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def extract_terrain_from_raw_response(raw_response: str) -> List[str]:
    """Extract terrain from raw_response JSON string."""
    try:
        # Find the JSON part in the raw response
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = raw_response[start:end]
            data = json.loads(json_str)
            return data.get("terrain", [])
    except:
        pass
    return []


def compare_terrain_data():
    """Compare terrain data between original and current files."""

    # Load both files
    print("Loading files...")
    original_data = load_json_file("phase4_captions original.json")
    current_data = load_json_file("phase4_captions.json")

    print(f"Original file has {len(original_data)} entries")
    print(f"Current file has {len(current_data)} entries")

    if len(original_data) != len(current_data):
        print("WARNING: Files have different number of entries!")
        return

    # Track issues
    empty_terrain_current = []
    terrain_lost = []
    terrain_changed = []
    total_checked = 0

    for i, (orig, curr) in enumerate(zip(original_data, current_data)):
        total_checked += 1

        # Get terrain from original raw_response
        orig_terrain = extract_terrain_from_raw_response(orig.get("raw_response", ""))

        # Get current terrain
        curr_terrain = curr.get("terrain", [])

        # Get terrain from current raw_response for comparison
        curr_raw_terrain = extract_terrain_from_raw_response(
            curr.get("raw_response", "")
        )

        # Check if current has empty terrain
        if not curr_terrain:
            empty_terrain_current.append(
                {
                    "index": i,
                    "image_path": curr.get("image_path", "unknown"),
                    "original_terrain": orig_terrain,
                    "raw_response_terrain": curr_raw_terrain,
                    "description": curr.get("description", "")[:100] + "...",
                }
            )

        # Check if terrain was lost from original to current
        if orig_terrain and not curr_terrain:
            terrain_lost.append(
                {
                    "index": i,
                    "image_path": curr.get("image_path", "unknown"),
                    "original_terrain": orig_terrain,
                    "current_terrain": curr_terrain,
                    "description": curr.get("description", "")[:100] + "...",
                }
            )

        # Check if terrain changed unexpectedly
        if orig_terrain and curr_terrain and set(orig_terrain) != set(curr_terrain):
            terrain_changed.append(
                {
                    "index": i,
                    "image_path": curr.get("image_path", "unknown"),
                    "original_terrain": orig_terrain,
                    "current_terrain": curr_terrain,
                    "description": curr.get("description", "")[:100] + "...",
                }
            )

    # Report findings
    print(f"\n=== TERRAIN ANALYSIS RESULTS ===")
    print(f"Total entries checked: {total_checked}")
    print(f"Entries with empty terrain in current file: {len(empty_terrain_current)}")
    print(f"Entries that lost terrain from original to current: {len(terrain_lost)}")
    print(f"Entries with changed terrain: {len(terrain_changed)}")

    if empty_terrain_current:
        print(f"\n=== ENTRIES WITH EMPTY TERRAIN (showing first 10) ===")
        for i, entry in enumerate(empty_terrain_current[:10]):
            print(f"\nEntry {entry['index']}:")
            print(f"  Image: {entry['image_path']}")
            print(f"  Original terrain: {entry['original_terrain']}")
            print(f"  Raw response terrain: {entry['raw_response_terrain']}")
            print(f"  Description: {entry['description']}")

    if terrain_lost:
        print(f"\n=== TERRAIN COMPLETELY LOST (showing first 10) ===")
        for i, entry in enumerate(terrain_lost[:10]):
            print(f"\nEntry {entry['index']}:")
            print(f"  Image: {entry['image_path']}")
            print(f"  Original terrain: {entry['original_terrain']}")
            print(f"  Current terrain: {entry['current_terrain']}")
            print(f"  Description: {entry['description']}")

    if terrain_changed:
        print(f"\n=== TERRAIN CHANGED (showing first 5) ===")
        for i, entry in enumerate(terrain_changed[:5]):
            print(f"\nEntry {entry['index']}:")
            print(f"  Image: {entry['image_path']}")
            print(f"  Original terrain: {entry['original_terrain']}")
            print(f"  Current terrain: {entry['current_terrain']}")
            print(f"  Description: {entry['description']}")

    # Also check if raw_response terrain differs from processed terrain
    raw_vs_processed_diff = []
    for i, curr in enumerate(current_data):
        curr_terrain = curr.get("terrain", [])
        curr_raw_terrain = extract_terrain_from_raw_response(
            curr.get("raw_response", "")
        )

        if curr_raw_terrain and set(curr_raw_terrain) != set(curr_terrain):
            raw_vs_processed_diff.append(
                {
                    "index": i,
                    "image_path": curr.get("image_path", "unknown"),
                    "processed_terrain": curr_terrain,
                    "raw_terrain": curr_raw_terrain,
                    "description": curr.get("description", "")[:100] + "...",
                }
            )

    if raw_vs_processed_diff:
        print(
            f"\n=== RAW RESPONSE vs PROCESSED TERRAIN DIFFERENCES (showing first 10) ==="
        )
        print(f"Total entries with differences: {len(raw_vs_processed_diff)}")
        for i, entry in enumerate(raw_vs_processed_diff[:10]):
            print(f"\nEntry {entry['index']}:")
            print(f"  Image: {entry['image_path']}")
            print(f"  Raw terrain: {entry['raw_terrain']}")
            print(f"  Processed terrain: {entry['processed_terrain']}")
            print(f"  Description: {entry['description']}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"🔍 Total issues found:")
    print(f"  - Empty terrain in current: {len(empty_terrain_current)}")
    print(f"  - Terrain completely lost: {len(terrain_lost)}")
    print(f"  - Terrain changed: {len(terrain_changed)}")
    print(f"  - Raw vs processed differences: {len(raw_vs_processed_diff)}")

    if raw_vs_processed_diff:
        print(
            f"\n⚠️  MAJOR FINDING: Raw responses contain terrain data but processed terrain is different!"
        )
        print(
            f"This suggests the issue is in the post-processing pipeline that extracts data from raw_response."
        )


if __name__ == "__main__":
    compare_terrain_data()
