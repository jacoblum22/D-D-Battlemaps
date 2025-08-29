"""
Script to restore and fix deleted terms from raw text responses.

This script processes caption files and attempts to recover terms that were
lost during the normalization process by re-examining the raw_response field.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict


class TermRestorer:
    """Restore lost terms from raw responses in caption data."""

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

    FEATURES = {
        "altar",
        "anvil",
        "archway",
        "barricade",
        "barrel",
        "bed",
        "bench",
        "bloodstain",
        "boat",
        "bones",
        "bookshelf",
        "boulder",
        "brazier",
        "bridge",
        "broken wall",
        "buoy",
        "bush",
        "campfire",
        "candlestick",
        "cart",
        "chain",
        "chair",
        "chandelier",
        "chasm",
        "chest",
        "cobweb",
        "crates",
        "crystals",
        "debris",
        "desk",
        "dock",
        "door",
        "fence",
        "forge",
        "fountain",
        "gate",
        "gear",
        "grate",
        "ladder",
        "lantern",
        "lever",
        "mushrooms",
        "pipe",
        "pillar",
        "pit",
        "pond",
        "pool",
        "pulley",
        "rail tracks",
        "rope",
        "roots",
        "rubble",
        "rug",
        "sarcophagus",
        "secret door",
        "skull",
        "stairs",
        "statue",
        "stump",
        "table",
        "tapestry",
        "tent",
        "tile mosaic",
        "tools",
        "torch",
        "trap",
        "tree",
        "vines",
        "wall",
        "waterfall",
        "well",
        "window",
        "workbench",
    }

    SCENE_TYPES = {
        "tavern",
        "inn",
        "kitchen",
        "library",
        "study",
        "laboratory",
        "workshop",
        "armory",
        "barracks",
        "chapel",
        "crypt",
        "throne room",
        "market",
        "warehouse",
        "prison",
        "courtyard",
        "garden",
        "street",
        "sewer",
        "cave",
        "mine",
        "tower",
        "ruins",
        "wilderness",
        "road",
        "clearing",
        "camp",
        "bridge",
        "dock",
        "plaza",
        "dungeon",
    }

    # Terms that were identified as lost during processing
    LOST_TERMS = {"lava"}

    def __init__(self):
        """Initialize the term restorer."""
        # Create mapping from lost terms to their appropriate categories
        self.term_categories = {
            # Terrain terms
            "lava": "terrain"
        }

    def extract_terms_from_raw(self, raw_response: str) -> Dict[str, List[str]]:
        """Extract relevant terms from raw response JSON structure."""
        found_terms = {
            "terrain": [],
            "features": [],
            "scene_type": [],
            "attributes": [],
        }

        if not raw_response:
            return found_terms

        # Extract the JSON from the raw response
        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = raw_response[start:end]
                raw_data = json.loads(json_str)

                # Only look for lava in the terrain field of the original JSON
                raw_terrain = raw_data.get("terrain", [])
                if isinstance(raw_terrain, list):
                    for term in raw_terrain:
                        if term and term.lower().strip() == "lava":
                            found_terms["terrain"].append("lava")
        except (json.JSONDecodeError, ValueError):
            pass

        return found_terms

    def restore_caption_terms(self, caption: Dict) -> Dict:
        """Restore lost terms to a single caption."""
        if "raw_response" not in caption:
            return caption

        # Extract terms from raw response
        raw_terms = self.extract_terms_from_raw(caption["raw_response"])

        # Create a working copy of the caption
        restored_caption = caption.copy()

        # Restore terrain terms
        if raw_terms["terrain"]:
            current_terrain = set(restored_caption.get("terrain", []))
            for term in raw_terms["terrain"]:
                if term in self.TERRAIN and term not in current_terrain:
                    if "terrain" not in restored_caption:
                        restored_caption["terrain"] = []
                    restored_caption["terrain"].append(term)

        # Restore feature terms
        if raw_terms["features"]:
            current_features = set(restored_caption.get("features", []))
            for term in raw_terms["features"]:
                if term in self.FEATURES and term not in current_features:
                    if "features" not in restored_caption:
                        restored_caption["features"] = []
                    restored_caption["features"].append(term)

        # Restore scene type (only if current scene_type is missing/unknown)
        if raw_terms["scene_type"]:
            current_scene = restored_caption.get("scene_type", "").lower()
            if not current_scene or current_scene == "unknown":
                for term in raw_terms["scene_type"]:
                    if term in self.SCENE_TYPES:
                        restored_caption["scene_type"] = term
                        break

        # Note: Attributes restoration removed since we're only handling lava

        # Sort lists to maintain consistency
        if "terrain" in restored_caption:
            restored_caption["terrain"] = sorted(list(set(restored_caption["terrain"])))
        if "features" in restored_caption:
            restored_caption["features"] = sorted(
                list(set(restored_caption["features"]))
            )

        return restored_caption

    def process_caption_file(
        self, input_file: str, output_file: Optional[str] = None
    ) -> Dict[str, int]:
        """Process a caption file and restore lost terms."""
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Load captions
        with open(input_file, "r", encoding="utf-8") as f:
            captions = json.load(f)

        if not isinstance(captions, list):
            raise ValueError("Caption file must contain a list of caption objects")

        print(f"Processing {len(captions)} captions from {input_file}")

        # Track restoration statistics
        stats = {
            "total_captions": len(captions),
            "captions_modified": 0,
            "terms_restored": 0,
            "terrain_restored": 0,
            "features_restored": 0,
            "scene_type_restored": 0,
            "attributes_restored": 0,
        }

        # Process each caption
        restored_captions = []
        for i, caption in enumerate(captions):
            original_caption = json.dumps(caption, sort_keys=True)
            restored_caption = self.restore_caption_terms(caption)
            restored_json = json.dumps(restored_caption, sort_keys=True)

            if original_caption != restored_json:
                stats["captions_modified"] += 1

                # Count specific types of changes
                orig_terrain = set(caption.get("terrain", []))
                new_terrain = set(restored_caption.get("terrain", []))
                terrain_added = len(new_terrain - orig_terrain)
                stats["terrain_restored"] += terrain_added

                # Only counting terrain changes since we're only restoring lava
                stats["terms_restored"] += terrain_added

            restored_captions.append(restored_caption)

            # Progress update
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(captions)} captions...")

        # Save restored captions
        if output_file is None:
            # Create output filename based on input
            input_path = Path(input_file)
            output_file = str(
                input_path.parent / f"{input_path.stem}_restored{input_path.suffix}"
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(restored_captions, f, indent=2, ensure_ascii=False)

        print(f"\nRestoration complete! Saved to {output_file}")
        print(
            f"Modified {stats['captions_modified']}/{stats['total_captions']} captions"
        )
        print(f"Restored {stats['terms_restored']} lava terrain terms")

        return stats

    def analyze_restoration_potential(self, input_file: str) -> Dict[str, int]:
        """Analyze how many terms could be restored from a caption file."""
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            captions = json.load(f)

        print(f"Analyzing restoration potential for {len(captions)} captions...")

        found_terms = defaultdict(int)
        captions_with_restorable = 0

        for caption in captions:
            raw_response = caption.get("raw_response", "")
            if not raw_response:
                continue

            caption_has_restorable = False
            raw_terms = self.extract_terms_from_raw(raw_response)

            for category, terms in raw_terms.items():
                for term in terms:
                    found_terms[term] += 1
                    caption_has_restorable = True

            if caption_has_restorable:
                captions_with_restorable += 1

        print(f"\nRestorable terms found:")
        print(
            f"Captions with restorable terms: {captions_with_restorable}/{len(captions)}"
        )

        if found_terms:
            print("Term frequencies:")
            for term, count in sorted(
                found_terms.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {term}: {count}")
        else:
            print("No restorable terms found")

        return dict(found_terms)


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Restore lost terms from caption files"
    )
    parser.add_argument("input_file", help="Input caption JSON file")
    parser.add_argument(
        "-o", "--output", help="Output file (default: input_restored.json)"
    )
    parser.add_argument(
        "-a",
        "--analyze",
        action="store_true",
        help="Only analyze restoration potential, don't restore",
    )

    args = parser.parse_args()

    restorer = TermRestorer()

    try:
        if args.analyze:
            restorer.analyze_restoration_potential(args.input_file)
        else:
            restorer.process_caption_file(args.input_file, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
