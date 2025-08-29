"""
Vocabulary Management Tool for Iterative Improvement

This script helps manage and improve the controlled vocabulary by:
1. Analyzing previous captioning results
2. Suggesting vocabulary additions
3. Updating vocabulary lists
4. Testing vocabulary changes
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from battlemap_processor.captioning import ControlledVocabularyCaptioner


class VocabularyManager:
    """Manages vocabulary improvements and testing."""

    def __init__(self):
        self.captioner = ControlledVocabularyCaptioner()

    def load_previous_results(
        self, results_file: str = "captions_batch.json"
    ) -> List[Dict]:
        """Load previous captioning results."""
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"No previous results found at {results_file}")
            return []

    def analyze_extras(
        self, captions: List[Dict], min_frequency: int = 3
    ) -> Dict[str, List[str]]:
        """Analyze extras field to find common terms."""
        all_extras = []
        for caption in captions:
            all_extras.extend(caption.get("extras", []))

        # Count frequencies
        extras_counter = Counter(all_extras)

        # Categorize potential additions
        suggestions = {"terrain": [], "features": [], "scene_type": [], "other": []}

        # Keywords to help categorize
        terrain_keywords = [
            "mountain",
            "hill",
            "valley",
            "beach",
            "marsh",
            "bog",
            "tundra",
            "volcano",
            "canyon",
            "plains",
            "field",
        ]
        feature_keywords = [
            "pillar",
            "column",
            "beam",
            "pipe",
            "machinery",
            "crystal",
            "gem",
            "stone",
            "rock",
            "wall",
            "floor",
        ]
        scene_keywords = [
            "hall",
            "chamber",
            "room",
            "building",
            "structure",
            "area",
            "zone",
            "quarters",
            "sanctum",
        ]

        for term, count in extras_counter.items():
            if count >= min_frequency:
                if any(keyword in term.lower() for keyword in terrain_keywords):
                    suggestions["terrain"].append(f"{term} ({count})")
                elif any(keyword in term.lower() for keyword in feature_keywords):
                    suggestions["features"].append(f"{term} ({count})")
                elif any(keyword in term.lower() for keyword in scene_keywords):
                    suggestions["scene_type"].append(f"{term} ({count})")
                else:
                    suggestions["other"].append(f"{term} ({count})")

        return suggestions

    def suggest_vocabulary_updates(self, results_file: str = "captions_batch.json"):
        """Suggest vocabulary updates based on previous results."""
        captions = self.load_previous_results(results_file)

        if not captions:
            print("No captions to analyze. Run captioning first.")
            return

        print(f"Analyzing {len(captions)} captions...")

        suggestions = self.analyze_extras(captions, min_frequency=2)

        print("\n" + "=" * 60)
        print("VOCABULARY IMPROVEMENT SUGGESTIONS")
        print("=" * 60)

        for category, terms in suggestions.items():
            if terms:
                print(f"\n{category.upper()} CANDIDATES:")
                for term in terms:
                    print(f"  {term}")

        # Show current vocabulary sizes
        print(f"\nCURRENT VOCABULARY SIZES:")
        print(f"  Terrain: {len(self.captioner.TERRAIN)} terms")
        print(f"  Features: {len(self.captioner.FEATURES)} terms")
        print(f"  Scene Types: {len(self.captioner.SCENE_TYPES)} terms")

        # Interactive updates
        self.interactive_vocabulary_update(suggestions)

    def interactive_vocabulary_update(self, suggestions: Dict[str, List[str]]):
        """Interactively update vocabulary lists."""
        print(f"\n" + "=" * 60)
        print("INTERACTIVE VOCABULARY UPDATE")
        print("=" * 60)
        print("Review suggestions and update vocabulary lists.")
        print("(This is for planning - you'll need to manually update captioning.py)")

        updates = {"terrain": [], "features": [], "scene_type": []}

        for category in ["terrain", "features", "scene_type"]:
            if suggestions.get(category):
                print(f"\n{category.upper()} candidates:")
                for i, term_with_count in enumerate(suggestions[category], 1):
                    term = term_with_count.split(" (")[0]  # Remove count
                    print(f"  {i}. {term_with_count}")

                response = input(
                    f"\nEnter numbers to add to {category} (comma-separated, or 'skip'): "
                )

                if response.lower() != "skip":
                    try:
                        indices = [
                            int(x.strip()) for x in response.split(",") if x.strip()
                        ]
                        for idx in indices:
                            if 1 <= idx <= len(suggestions[category]):
                                term = suggestions[category][idx - 1].split(" (")[0]
                                updates[category].append(term)
                    except ValueError:
                        print("Invalid input, skipping category.")

        # Show update summary
        if any(updates.values()):
            print(f"\n" + "=" * 40)
            print("UPDATE SUMMARY")
            print("=" * 40)
            for category, terms in updates.items():
                if terms:
                    print(f"\n{category.upper()} additions:")
                    for term in terms:
                        print(f"  + {term}")

            print(f"\nTo apply these changes:")
            print(f"1. Edit battlemap_processor/captioning.py")
            print(f"2. Add the above terms to the respective vocabulary lists")
            print(f"3. Run captioning again to test improvements")
        else:
            print("\nNo vocabulary updates selected.")

    def compare_results(self, old_file: str, new_file: str):
        """Compare results between two captioning runs."""
        try:
            with open(old_file, "r", encoding="utf-8") as f:
                old_results = json.load(f)
            with open(new_file, "r", encoding="utf-8") as f:
                new_results = json.load(f)
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return

        print(f"Comparing {old_file} vs {new_file}")
        print(f"Old results: {len(old_results)} captions")
        print(f"New results: {len(new_results)} captions")

        # Compare extras usage
        old_extras = []
        new_extras = []

        for caption in old_results:
            old_extras.extend(caption.get("extras", []))

        for caption in new_results:
            new_extras.extend(caption.get("extras", []))

        old_unique = len(set(old_extras))
        new_unique = len(set(new_extras))

        print(f"\nExtras comparison:")
        print(f"  Old: {len(old_extras)} total, {old_unique} unique")
        print(f"  New: {len(new_extras)} total, {new_unique} unique")
        print(f"  Improvement: {old_unique - new_unique} fewer unique OOV terms")

    def test_single_image(self, image_path: str):
        """Test captioning on a single image for debugging."""
        print(f"Testing captioning on: {image_path}")

        caption, tokens_used, cost_usd = self.captioner.caption_single_image(image_path)

        print(f"\nCost for this image: ${cost_usd:.4f} ({tokens_used} tokens)")

        if caption:
            print(f"\nRaw caption:")
            print(f"  Description: {caption.description}")
            print(f"  Terrain: {caption.terrain}")
            print(f"  Features: {caption.features}")
            print(f"  Scene type: {caption.scene_type}")
            print(f"  Style: {caption.style}")
            print(f"  Grid: {caption.grid}")
            print(f"  Extras: {caption.extras}")

            # Normalize and show differences
            normalized, oov = self.captioner.normalize_caption(caption)

            print(f"\nNormalized caption:")
            print(f"  Terrain: {normalized.terrain}")
            print(f"  Features: {normalized.features}")
            print(f"  Scene type: {normalized.scene_type}")
            print(f"  Extras: {normalized.extras}")

            if oov:
                print(f"\nOOV terms found: {oov}")
        else:
            print("Failed to caption image")


def main():
    """Main function for vocabulary management."""
    manager = VocabularyManager()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "analyze":
            # Analyze previous results
            results_file = sys.argv[2] if len(sys.argv) > 2 else "captions_batch.json"
            manager.suggest_vocabulary_updates(results_file)

        elif command == "compare":
            # Compare two result files
            if len(sys.argv) != 4:
                print(
                    "Usage: python manage_vocabulary.py compare <old_file> <new_file>"
                )
                return
            manager.compare_results(sys.argv[2], sys.argv[3])

        elif command == "test":
            # Test single image
            if len(sys.argv) != 3:
                print("Usage: python manage_vocabulary.py test <image_path>")
                return
            manager.test_single_image(sys.argv[2])

        else:
            print(f"Unknown command: {command}")
            print("Available commands: analyze, compare, test")

    else:
        # Default: analyze current results
        print("=== Vocabulary Management Tool ===")
        print("Analyzing current captioning results...")
        manager.suggest_vocabulary_updates()


if __name__ == "__main__":
    main()
