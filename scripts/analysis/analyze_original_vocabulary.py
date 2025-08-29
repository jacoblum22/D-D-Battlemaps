#!/usr/bin/env python3
"""
Vocabulary Analysis Script for phase4_captions.json

Analyzes the vocabulary usage in the caption file and generates statistics.
"""

import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional


def load_captions(filepath: str) -> List[Dict[str, Any]]:
    """Load the captions JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def extract_raw_response_data(raw_response: str) -> Dict[str, Any]:
    """Extract data from raw_response JSON string."""
    try:
        # Find the JSON part in the raw response
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = raw_response[start:end]
            return json.loads(json_str)
    except:
        pass
    return {}


def analyze_vocabulary(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze vocabulary usage in the caption data."""

    # Counters for different categories
    terrain_counter = Counter()
    features_counter = Counter()
    scene_type_counter = Counter()
    style_counter = Counter()
    extras_counter = Counter()

    # Attribute counters
    attribute_counters = defaultdict(Counter)

    # Raw response vs processed comparison
    raw_terrain_counter = Counter()
    raw_features_counter = Counter()
    raw_scene_type_counter = Counter()

    # Validation counters
    missing_fields = defaultdict(int)
    empty_fields = defaultdict(int)

    total_entries = len(data)
    processed_entries = 0

    for entry in data:
        processed_entries += 1

        # Analyze processed data
        terrain = entry.get("terrain", [])
        features = entry.get("features", [])
        scene_type = entry.get("scene_type", "")
        style = entry.get("style", "")
        extras = entry.get("extras", [])
        attributes = entry.get("attributes", {})

        # Count processed data
        if terrain:
            for t in terrain:
                terrain_counter[t.lower().strip()] += 1
        else:
            empty_fields["terrain"] += 1

        if features:
            for f in features:
                features_counter[f.lower().strip()] += 1
        else:
            empty_fields["features"] += 1

        if scene_type:
            scene_type_counter[scene_type.lower().strip()] += 1
        else:
            empty_fields["scene_type"] += 1

        if style:
            style_counter[style.lower().strip()] += 1
        else:
            empty_fields["style"] += 1

        if extras:
            for e in extras:
                extras_counter[e.lower().strip()] += 1
        else:
            empty_fields["extras"] += 1

        # Count attributes
        if attributes:
            for attr_type, values in attributes.items():
                if isinstance(values, list):
                    for value in values:
                        attribute_counters[attr_type][value.lower().strip()] += 1
                else:
                    attribute_counters[attr_type][str(values).lower().strip()] += 1
        else:
            empty_fields["attributes"] += 1

        # Analyze raw response data if available
        raw_response = entry.get("raw_response", "")
        if raw_response:
            raw_data = extract_raw_response_data(raw_response)

            # Count raw terrain
            raw_terrain = raw_data.get("terrain", [])
            if raw_terrain:
                for t in raw_terrain:
                    raw_terrain_counter[t.lower().strip()] += 1

            # Count raw features
            raw_features = raw_data.get("features", [])
            if raw_features:
                for f in raw_features:
                    raw_features_counter[f.lower().strip()] += 1

            # Count raw scene_type
            raw_scene_type = raw_data.get("scene_type", "")
            if raw_scene_type:
                raw_scene_type_counter[raw_scene_type.lower().strip()] += 1

        # Check for missing required fields
        required_fields = ["description", "terrain", "features", "scene_type", "style"]
        for field in required_fields:
            if field not in entry:
                missing_fields[field] += 1

    return {
        "total_entries": total_entries,
        "processed_entries": processed_entries,
        "terrain": terrain_counter,
        "features": features_counter,
        "scene_type": scene_type_counter,
        "style": style_counter,
        "extras": extras_counter,
        "attributes": dict(attribute_counters),
        "raw_terrain": raw_terrain_counter,
        "raw_features": raw_features_counter,
        "raw_scene_type": raw_scene_type_counter,
        "missing_fields": dict(missing_fields),
        "empty_fields": dict(empty_fields),
    }


def generate_report(stats: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Generate a comprehensive vocabulary analysis report."""

    report = "=== VOCABULARY ANALYSIS REPORT ===\n\n"
    report += f"Total entries analyzed: {stats['total_entries']}\n"
    report += f"Successfully processed: {stats['processed_entries']}\n\n"

    # Missing and empty fields analysis
    if stats["missing_fields"]:
        report += "MISSING FIELDS:\n"
        for field, count in stats["missing_fields"].items():
            report += f"  {field}: {count} entries missing\n"
        report += "\n"

    if stats["empty_fields"]:
        report += "EMPTY FIELDS:\n"
        for field, count in stats["empty_fields"].items():
            pct = (count / stats["total_entries"]) * 100
            report += f"  {field}: {count} entries empty ({pct:.1f}%)\n"
        report += "\n"

    # Terrain analysis
    report += "TERRAIN:\n"
    for terrain, count in stats["terrain"].most_common():
        pct = (count / stats["total_entries"]) * 100
        report += f"  {terrain}: {count} ({pct:.1f}%)\n"
    report += f"\nTotal unique terrain types: {len(stats['terrain'])}\n\n"

    # Features analysis
    report += "FEATURES:\n"
    for feature, count in stats["features"].most_common():
        pct = (count / stats["total_entries"]) * 100
        report += f"  {feature}: {count} ({pct:.1f}%)\n"
    report += f"\nTotal unique features: {len(stats['features'])}\n\n"

    # Scene type analysis
    report += "SCENE TYPES:\n"
    for scene_type, count in stats["scene_type"].most_common():
        pct = (count / stats["total_entries"]) * 100
        report += f"  {scene_type}: {count} ({pct:.1f}%)\n"
    report += f"\nTotal unique scene types: {len(stats['scene_type'])}\n\n"

    # Style analysis
    report += "STYLES:\n"
    for style, count in stats["style"].most_common():
        pct = (count / stats["total_entries"]) * 100
        report += f"  {style}: {count} ({pct:.1f}%)\n"
    report += "\n"

    # Extras analysis (if any)
    if stats["extras"]:
        report += "EXTRAS:\n"
        for extra, count in stats["extras"].most_common():
            pct = (count / stats["total_entries"]) * 100
            report += f"  {extra}: {count} ({pct:.1f}%)\n"
        report += f"\nTotal unique extras: {len(stats['extras'])}\n\n"

    # Attributes analysis
    if stats["attributes"]:
        report += "ATTRIBUTES:\n"
        for attr_type, counter in stats["attributes"].items():
            report += f"  {attr_type.upper()}:\n"
            for value, count in counter.most_common(10):
                pct = (count / stats["total_entries"]) * 100
                report += f"    {value}: {count} ({pct:.1f}%)\n"
            report += f"    Total unique {attr_type}: {len(counter)}\n\n"

    # Raw vs processed comparison
    if stats["raw_terrain"]:
        report += "RAW RESPONSE vs PROCESSED COMPARISON:\n\n"

        report += "Raw Response Terrain (top 15):\n"
        for terrain, count in stats["raw_terrain"].most_common(15):
            pct = (count / stats["total_entries"]) * 100
            processed_count = stats["terrain"].get(terrain, 0)
            diff = count - processed_count
            report += f"  {terrain}: {count} raw → {processed_count} processed"
            if diff > 0:
                report += f" (LOST {diff})"
            report += f"\n"
        report += "\n"

        report += "Terms only in raw responses (lost during processing):\n"
        raw_only = set(stats["raw_terrain"].keys()) - set(stats["terrain"].keys())
        for term in sorted(raw_only):
            count = stats["raw_terrain"][term]
            report += f"  {term}: {count} (completely lost)\n"
        report += "\n"

    # Save to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {output_file}")

    return report


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze vocabulary in caption files")
    parser.add_argument(
        "input_file",
        nargs="?",
        default="phase4_captions.json",
        help="Input caption JSON file (default: phase4_captions.json)",
    )
    parser.add_argument(
        "-o", "--output", help="Output analysis file (default: based on input filename)"
    )

    args = parser.parse_args()

    input_file = args.input_file

    # Generate output filename based on input if not specified
    if args.output:
        output_file = args.output
    else:
        from pathlib import Path

        input_path = Path(input_file)
        output_file = str(
            input_path.parent / f"{input_path.stem}_vocabulary_analysis.txt"
        )

    print(f"Analyzing vocabulary in {input_file}...")

    # Load data
    data = load_captions(input_file)
    print(f"Loaded {len(data)} caption entries")

    # Analyze vocabulary
    print("Analyzing vocabulary usage...")
    stats = analyze_vocabulary(data)

    # Generate report
    print("Generating vocabulary report...")
    report = generate_report(stats, output_file)

    # Print summary to console
    print("\n=== VOCABULARY ANALYSIS SUMMARY ===")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Empty terrain entries: {stats['empty_fields'].get('terrain', 0)}")
    print(f"Empty features entries: {stats['empty_fields'].get('features', 0)}")
    print(f"Empty scene_type entries: {stats['empty_fields'].get('scene_type', 0)}")
    print(f"Unique terrain types: {len(stats['terrain'])}")
    print(f"Unique features: {len(stats['features'])}")
    print(f"Unique scene types: {len(stats['scene_type'])}")

    if stats["raw_terrain"]:
        raw_only = set(stats["raw_terrain"].keys()) - set(stats["terrain"].keys())
        print(f"Terrain types lost during processing: {len(raw_only)}")

    print(f"\nFull report saved to: {output_file}")


if __name__ == "__main__":
    main()
