"""
Script to analyze empty terrain entries and show what terrain terms were lost.

This script finds captions with empty terrain fields and examines their raw responses
to identify what terrain terms were originally present but got deleted during processing.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter


class EmptyTerrainAnalyzer:
    """Analyze captions with empty terrain fields to find lost terrain terms."""

    # Define the controlled vocabulary (same as in captioning.py)
    TERRAIN = {
        "forest",
        "desert",
        "swamp",
        "snow",
        "ice",
        "coastline",
        "river",
        "lake",
        "lava",
        "cave",
        "cliffs",
        "grassland",
        "ruins",
        "dungeon",
        "sewer",
        "street",
        "plaza",
        "rooftop",
        "interior",
        "corridor",
        "garden",
        "path",
    }

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def extract_raw_terrain(self, raw_response: str) -> List[str]:
        """Extract terrain terms from raw response JSON."""
        terrain_terms = []

        if not raw_response:
            return terrain_terms

        # Extract the JSON from the raw response
        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = raw_response[start:end]
                raw_data = json.loads(json_str)

                # Get terrain from the original JSON
                raw_terrain = raw_data.get("terrain", [])
                if isinstance(raw_terrain, list):
                    for term in raw_terrain:
                        if term and isinstance(term, str):
                            clean_term = term.lower().strip()
                            terrain_terms.append(clean_term)
        except (json.JSONDecodeError, ValueError):
            pass

        return terrain_terms

    def analyze_empty_terrain_captions(self, input_file: str) -> Dict:
        """Analyze captions with empty terrain fields."""
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            captions = json.load(f)

        print(f"Analyzing {len(captions)} captions for empty terrain fields...")

        empty_terrain_captions = []
        lost_terrain_counter = Counter()
        total_empty = 0

        for i, caption in enumerate(captions):
            # Check if terrain is empty
            terrain = caption.get("terrain", [])
            if not terrain or len(terrain) == 0:
                total_empty += 1

                # Get raw terrain terms
                raw_response = caption.get("raw_response", "")
                raw_terrain = self.extract_raw_terrain(raw_response)

                if raw_terrain:
                    # This caption had terrain in raw but lost it during processing
                    empty_terrain_captions.append(
                        {
                            "index": i,
                            "image_path": caption.get("image_path", "unknown"),
                            "raw_terrain": raw_terrain,
                            "current_terrain": terrain,
                            "description": (
                                caption.get("description", "")[:100] + "..."
                                if len(caption.get("description", "")) > 100
                                else caption.get("description", "")
                            ),
                        }
                    )

                    # Count the lost terrain terms
                    for term in raw_terrain:
                        lost_terrain_counter[term] += 1

        return {
            "total_captions": len(captions),
            "total_empty_terrain": total_empty,
            "captions_with_lost_terrain": len(empty_terrain_captions),
            "empty_terrain_captions": empty_terrain_captions,
            "lost_terrain_counts": dict(lost_terrain_counter),
        }

    def print_analysis_report(self, analysis: Dict):
        """Print a detailed analysis report."""
        print("\n" + "=" * 70)
        print("EMPTY TERRAIN ANALYSIS REPORT")
        print("=" * 70)

        print(f"\nTotal captions: {analysis['total_captions']}")
        print(f"Captions with empty terrain: {analysis['total_empty_terrain']}")
        print(
            f"Empty terrain captions that had raw terrain: {analysis['captions_with_lost_terrain']}"
        )

        if analysis["lost_terrain_counts"]:
            print(f"\nLOST TERRAIN TERMS (from raw responses):")
            print("-" * 40)
            for term, count in sorted(
                analysis["lost_terrain_counts"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {term}: {count} times")

        if analysis["empty_terrain_captions"]:
            print(f"\nDETAILED BREAKDOWN:")
            print("-" * 40)

            for i, caption_info in enumerate(
                analysis["empty_terrain_captions"][:20], 1
            ):  # Show first 20
                print(f"\n{i}. Image: {Path(caption_info['image_path']).name}")
                print(f"   Raw terrain: {caption_info['raw_terrain']}")
                print(f"   Description: {caption_info['description']}")

            if len(analysis["empty_terrain_captions"]) > 20:
                print(
                    f"\n... and {len(analysis['empty_terrain_captions']) - 20} more captions with lost terrain"
                )

        print("\n" + "=" * 70)

    def generate_lost_terrain_list(self, analysis: Dict) -> List[str]:
        """Generate a list of all unique terrain terms that were lost."""
        return list(analysis["lost_terrain_counts"].keys())


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze empty terrain entries for lost terrain terms"
    )
    parser.add_argument("input_file", help="Input caption JSON file")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed breakdown of each empty terrain caption",
    )

    args = parser.parse_args()

    analyzer = EmptyTerrainAnalyzer()

    try:
        analysis = analyzer.analyze_empty_terrain_captions(args.input_file)
        analyzer.print_analysis_report(analysis)

        if args.detailed and analysis["empty_terrain_captions"]:
            print(
                f"\nFULL DETAILED LIST OF ALL {len(analysis['empty_terrain_captions'])} CAPTIONS:"
            )
            print("=" * 70)
            for i, caption_info in enumerate(analysis["empty_terrain_captions"], 1):
                print(f"\n{i}. Image: {caption_info['image_path']}")
                print(f"   Raw terrain: {caption_info['raw_terrain']}")
                print(f"   Current terrain: {caption_info['current_terrain']}")
                print(f"   Description: {caption_info['description']}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
