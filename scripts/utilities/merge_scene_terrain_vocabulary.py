#!/usr/bin/env python3
"""
Script to merge and consolidate vocabulary according to specified groupings.

Merges:
Scene Types:
- warehouse, kitchen, study, library, chapel, barracks, prison → "interiors"
- sewer, crypt, mine, dungeon → "underground/dark"
- throne room, tower, courtyard → "castle/noble"

Features:
- pillar, chasm, statue, archway, gate, secret door → "stone/architecture"
- brazier, candlestick, chandelier → "fire/light"
- anvil, forge, tools → "smithing/workshop"
- sarcophagus, skull, bones → "grave/death"
- rug, tapestry → "fabric/decoration"
- trap, lever, rope, chain, barricade → "mechanisms/traps"
- buoy → boat (merge into existing)
- rail tracks, pipe, grate → "rails/industrial"

Also drops extras with less than 0.5% frequency.
"""

import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set


def load_captions(filepath: str) -> List[Dict[str, Any]]:
    """Load the captions JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_captions(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save the captions JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_extras_frequency(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate frequency percentages for extras."""
    extras_counter = Counter()
    total_entries = len(data)

    for entry in data:
        extras = entry.get("extras", [])
        if isinstance(extras, list):
            for extra in extras:
                if extra and isinstance(extra, str):
                    extras_counter[extra.lower().strip()] += 1

    # Calculate percentages
    frequency_percentages = {}
    for extra, count in extras_counter.items():
        frequency_percentages[extra] = (count / total_entries) * 100

    return frequency_percentages


def transform_entry(entry: Dict[str, Any], extras_to_drop: Set[str]) -> Dict[str, Any]:
    """Transform a single caption entry according to the merging rules."""
    # Create a copy to avoid modifying the original
    new_entry = entry.copy()

    # Scene type merging mappings
    scene_type_merges = {
        "interiors": {
            "warehouse",
            "kitchen",
            "study",
            "library",
            "chapel",
            "barracks",
            "prison",
        },
        "underground/dark": {"sewer", "crypt", "mine", "dungeon"},
        "castle/noble": {"throne room", "tower", "courtyard"},
    }

    # Feature merging mappings
    feature_merges = {
        "stone/architecture": {
            "pillar",
            "chasm",
            "statue",
            "archway",
            "gate",
            "secret door",
        },
        "fire/light": {"brazier", "candlestick", "chandelier"},
        "smithing/workshop": {"anvil", "forge", "tools"},
        "grave/death": {"sarcophagus", "skull", "bones"},
        "fabric/decoration": {"rug", "tapestry"},
        "mechanisms/traps": {"trap", "lever", "rope", "chain", "barricade"},
        "boat": {"buoy"},  # Merge buoy into existing boat
        "rails/industrial": {"rail tracks", "pipe", "grate"},
    }

    # Transform scene_type
    current_scene_type = new_entry.get("scene_type", "").lower().strip()
    new_scene_type = current_scene_type

    for target_scene, source_scenes in scene_type_merges.items():
        if current_scene_type in source_scenes:
            new_scene_type = target_scene
            break

    new_entry["scene_type"] = new_scene_type

    # Transform features
    current_features = new_entry.get("features", [])
    if isinstance(current_features, list):
        new_features = []
        merged_features = set()

        for feature in current_features:
            if feature and isinstance(feature, str):
                feature = feature.lower().strip()

                # Check if this feature should be merged
                was_merged = False
                for target_feature, source_features in feature_merges.items():
                    if feature in source_features:
                        merged_features.add(target_feature)
                        was_merged = True
                        break

                # If not merged, keep original
                if not was_merged:
                    new_features.append(feature)

        # Add merged features
        new_features.extend(list(merged_features))
        new_entry["features"] = new_features

    # Filter extras by frequency (drop those with < 0.5% frequency)
    current_extras = new_entry.get("extras", [])
    if isinstance(current_extras, list):
        filtered_extras = []
        for extra in current_extras:
            if extra and isinstance(extra, str):
                extra_clean = extra.lower().strip()
                if extra_clean not in extras_to_drop:
                    filtered_extras.append(extra)
        new_entry["extras"] = filtered_extras

    return new_entry


def analyze_vocabulary(data: List[Dict[str, Any]]) -> Dict[str, Counter]:
    """Analyze vocabulary usage in the transformed data."""
    terrain_counter = Counter()
    features_counter = Counter()
    scene_type_counter = Counter()
    extras_counter = Counter()

    for entry in data:
        # Count scene_types
        scene_type = entry.get("scene_type", "")
        if scene_type:
            scene_type_counter[scene_type] += 1

        # Count terrain
        for terrain in entry.get("terrain", []):
            if terrain:
                terrain_counter[terrain] += 1

        # Count features
        for feature in entry.get("features", []):
            if feature:
                features_counter[feature] += 1

        # Count extras
        for extra in entry.get("extras", []):
            if extra:
                extras_counter[extra] += 1

    return {
        "terrain": terrain_counter,
        "features": features_counter,
        "scene_type": scene_type_counter,
        "extras": extras_counter,
    }


def generate_vocabulary_report(
    vocab_stats: Dict[str, Counter], total_images: int, dropped_extras: Set[str]
) -> str:
    """Generate a vocabulary analysis report."""
    report = "=== MERGED VOCABULARY ANALYSIS REPORT ===\n\n"
    report += f"Total images processed: {total_images}\n\n"

    for category, counter in vocab_stats.items():
        report += f"{category.upper()}:\n"
        for term, count in counter.most_common():
            pct = (count / total_images) * 100
            report += f"  {term}: {count} ({pct:.1f}%)\n"
        report += f"\nTotal unique {category}: {len(counter)}\n\n"

    if dropped_extras:
        report += f"DROPPED EXTRAS (< 0.5% frequency): {len(dropped_extras)} terms\n"
        for extra in sorted(dropped_extras):
            report += f"  {extra}\n"
        report += "\n"

    return report


def main():
    """Main execution function."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Merge and consolidate vocabulary in caption files"
    )
    parser.add_argument("input_file", help="Input caption JSON file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output merged caption file (default: based on input filename)",
    )
    parser.add_argument(
        "--analysis", help="Output analysis file (default: based on input filename)"
    )

    args = parser.parse_args()

    input_file = args.input_file

    # Generate output filenames based on input if not specified
    if args.output:
        output_file = args.output
    else:
        input_path = Path(input_file)
        output_file = str(
            input_path.parent / f"{input_path.stem}_merged{input_path.suffix}"
        )

    if args.analysis:
        vocab_file = args.analysis
    else:
        input_path = Path(input_file)
        vocab_file = str(
            input_path.parent / f"{input_path.stem}_merged_vocabulary_analysis.txt"
        )

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Analysis file: {vocab_file}")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    # Load data
    data = load_captions(input_file)
    print(f"Loaded {len(data)} entries")

    # Calculate extras frequencies to determine which to drop
    print("Calculating extras frequencies...")
    extras_frequencies = calculate_extras_frequency(data)

    # Find extras with < 0.5% frequency to drop
    extras_to_drop = set()
    for extra, frequency in extras_frequencies.items():
        if frequency < 0.5:
            extras_to_drop.add(extra)

    print(f"Found {len(extras_to_drop)} extras with < 0.5% frequency to drop")

    print("Transforming entries...")
    transformed_data = []
    for i, entry in enumerate(data):
        try:
            transformed_entry = transform_entry(entry, extras_to_drop)
            transformed_data.append(transformed_entry)

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(data)} entries...")

        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            print(f"Entry: {entry}")
            # Keep original entry on error
            transformed_data.append(entry)

    print("Analyzing new vocabulary...")
    vocab_stats = analyze_vocabulary(transformed_data)

    print("Saving transformed data...")
    save_captions(transformed_data, output_file)
    print(f"Saved transformed data to {output_file}")

    print("Generating vocabulary report...")
    vocab_report = generate_vocabulary_report(
        vocab_stats, len(transformed_data), extras_to_drop
    )

    with open(vocab_file, "w", encoding="utf-8") as f:
        f.write(vocab_report)
    print(f"Saved vocabulary analysis to {vocab_file}")

    # Print summary statistics
    print("\n=== TRANSFORMATION SUMMARY ===")
    print(f"Total entries processed: {len(transformed_data)}")

    print("\nScene Type Distribution:")
    for scene_type, count in vocab_stats["scene_type"].most_common():
        pct = (count / len(transformed_data)) * 100
        print(f"  {scene_type}: {count} ({pct:.1f}%)")

    print("\nTop Features:")
    for feature, count in vocab_stats["features"].most_common(15):
        pct = (count / len(transformed_data)) * 100
        print(f"  {feature}: {count} ({pct:.1f}%)")

    print(f"\nDropped {len(extras_to_drop)} low-frequency extras (< 0.5%)")
    print(f"Transformation complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
