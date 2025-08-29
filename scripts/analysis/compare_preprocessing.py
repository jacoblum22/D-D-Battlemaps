"""
Compare vocabulary analysis before and after post-processing.

This script re-analyzes existing caption data to compare OOV rates
before and after applying our smart reclassification logic.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from dataclasses import asdict

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from battlemap_processor.captioning import (
    ControlledVocabularyCaptioner,
    Caption,
    VocabularyStats,
)


def load_phase4_captions() -> List[Dict]:
    """Load the existing phase4 captions."""
    captions_file = "phase4_captions.json"
    if not Path(captions_file).exists():
        print(f"Captions file not found: {captions_file}")
        return []

    with open(captions_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_caption_from_dict(data: Dict) -> Caption:
    """Create a Caption object from dictionary data."""
    return Caption(
        description=data.get("description", ""),
        terrain=data.get("terrain", []),
        features=data.get("features", []),
        scene_type=data.get("scene_type", ""),
        style=data.get("style", ""),
        extras=data.get("extras", []),
        attributes=data.get("attributes", {}),
        image_path=data.get("image_path", ""),
        raw_response=data.get("raw_response", ""),
    )


def analyze_captions_without_postprocessing(
    captions_data: List[Dict], captioner: ControlledVocabularyCaptioner
) -> VocabularyStats:
    """
    Analyze captions as if they came directly from ChatGPT without our post-processing.
    This simulates the original state by moving extras back to their original locations.
    """
    # First, we need to reconstruct what the original ChatGPT responses might have looked like
    # Since we don't have the original raw responses, we'll use the current data but treat
    # extras as additional OOV terms that weren't reclassified

    all_oov = []
    category_usage = {
        "terrain": defaultdict(int),
        "features": defaultdict(int),
        "scene_type": defaultdict(int),
    }

    successful = len(captions_data)

    for data in captions_data:
        caption = create_caption_from_dict(data)

        # Count vocabulary usage from current categories
        for term in caption.terrain:
            if term in captioner.terrain_set:
                category_usage["terrain"][term] += 1

        for term in caption.features:
            if term in captioner.features_set:
                category_usage["features"][term] += 1

        if caption.scene_type in captioner.scene_types_set:
            category_usage["scene_type"][caption.scene_type] += 1

        # Add all extras as OOV (this represents terms that weren't properly classified)
        all_oov.extend(caption.extras)

    # Count OOV terms
    oov_counter = Counter(all_oov)

    return VocabularyStats(
        total_images=len(captions_data),
        oov_terms=dict(oov_counter),
        category_usage=dict(category_usage),
        successful_captions=successful,
        failed_captions=0,  # These are already successful captions
        total_tokens_used=0,  # Not recalculating costs
        total_cost_usd=0.0,
    )


def analyze_captions_with_postprocessing(
    captions_data: List[Dict], captioner: ControlledVocabularyCaptioner
) -> VocabularyStats:
    """Analyze captions after applying our smart reclassification post-processing."""

    all_oov = []
    category_usage = {
        "terrain": defaultdict(int),
        "features": defaultdict(int),
        "scene_type": defaultdict(int),
    }

    successful = 0

    for data in captions_data:
        original_caption = create_caption_from_dict(data)

        # Apply our smart reclassification
        reclassified_caption = captioner.smart_reclassify_terms(original_caption)

        # Now apply full normalization to get OOV terms
        normalized_caption, oov_terms = captioner.normalize_caption(
            reclassified_caption
        )

        successful += 1
        all_oov.extend(oov_terms)

        # Count vocabulary usage from normalized caption
        for term in normalized_caption.terrain:
            if term in captioner.terrain_set:
                category_usage["terrain"][term] += 1

        for term in normalized_caption.features:
            if term in captioner.features_set:
                category_usage["features"][term] += 1

        if normalized_caption.scene_type in captioner.scene_types_set:
            category_usage["scene_type"][normalized_caption.scene_type] += 1

    # Count OOV terms
    oov_counter = Counter(all_oov)

    return VocabularyStats(
        total_images=len(captions_data),
        oov_terms=dict(oov_counter),
        category_usage=dict(category_usage),
        successful_captions=successful,
        failed_captions=0,
        total_tokens_used=0,
        total_cost_usd=0.0,
    )


def compare_vocabulary_stats(
    before_stats: VocabularyStats, after_stats: VocabularyStats
) -> Dict:
    """Compare vocabulary statistics before and after post-processing."""

    before_oov_total = sum(before_stats.oov_terms.values())
    after_oov_total = sum(after_stats.oov_terms.values())

    before_unique_oov = len(before_stats.oov_terms)
    after_unique_oov = len(after_stats.oov_terms)

    # Find terms that were fixed (appeared in before but not after, or reduced frequency)
    fixed_terms = {}
    for term, before_count in before_stats.oov_terms.items():
        after_count = after_stats.oov_terms.get(term, 0)
        if after_count < before_count:
            fixed_terms[term] = {
                "before": before_count,
                "after": after_count,
                "reduction": before_count - after_count,
            }

    # Find terms that are still OOV
    still_oov_terms = {
        term: count for term, count in after_stats.oov_terms.items() if count > 0
    }

    return {
        "before_oov_total": before_oov_total,
        "after_oov_total": after_oov_total,
        "reduction_total": before_oov_total - after_oov_total,
        "reduction_percentage": (
            (before_oov_total - after_oov_total) / before_oov_total * 100
            if before_oov_total > 0
            else 0
        ),
        "before_unique_oov": before_unique_oov,
        "after_unique_oov": after_unique_oov,
        "fixed_terms": fixed_terms,
        "still_oov_terms": still_oov_terms,
    }


def main():
    """Main function to compare pre- and post-processing analysis."""
    print("=== Pre/Post Processing Vocabulary Comparison ===")

    # Load existing captions
    captions_data = load_phase4_captions()
    if not captions_data:
        return

    print(f"Loaded {len(captions_data)} captions from phase4_captions.json")

    # Initialize captioner with our updated vocabulary and post-processing
    captioner = ControlledVocabularyCaptioner()

    print("\nAnalyzing vocabulary usage WITHOUT post-processing...")
    before_stats = analyze_captions_without_postprocessing(captions_data, captioner)

    print("Analyzing vocabulary usage WITH post-processing...")
    after_stats = analyze_captions_with_postprocessing(captions_data, captioner)

    # Compare results
    comparison = compare_vocabulary_stats(before_stats, after_stats)

    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Total OOV occurrences:")
    print(f"  Before: {comparison['before_oov_total']}")
    print(f"  After:  {comparison['after_oov_total']}")
    print(
        f"  Reduction: {comparison['reduction_total']} ({comparison['reduction_percentage']:.1f}%)"
    )

    print(f"\nUnique OOV terms:")
    print(f"  Before: {comparison['before_unique_oov']}")
    print(f"  After:  {comparison['after_unique_oov']}")

    print(f"\n=== TERMS FIXED BY POST-PROCESSING ===")
    if comparison["fixed_terms"]:
        sorted_fixed = sorted(
            comparison["fixed_terms"].items(),
            key=lambda x: x[1]["reduction"],
            reverse=True,
        )
        for term, data in sorted_fixed:
            print(
                f"  {term}: {data['before']} → {data['after']} (reduced by {data['reduction']})"
            )
    else:
        print("  No terms were fixed")

    print(f"\n=== REMAINING OOV TERMS ===")
    if comparison["still_oov_terms"]:
        sorted_remaining = sorted(
            comparison["still_oov_terms"].items(), key=lambda x: x[1], reverse=True
        )
        print(f"Top 20 remaining OOV terms:")
        for term, count in sorted_remaining[:20]:
            print(f"  {term}: {count}")
    else:
        print("  No OOV terms remaining!")

    # Generate detailed reports
    print(f"\nGenerating detailed analysis reports...")

    # Generate suggestions for both
    before_suggestions = captioner.analyze_oov_terms(before_stats, min_frequency=2)
    after_suggestions = captioner.analyze_oov_terms(after_stats, min_frequency=2)

    # Generate reports
    captioner.generate_report(
        before_stats, before_suggestions, "phase4_analysis_before_postprocessing.txt"
    )
    captioner.generate_report(
        after_stats, after_suggestions, "phase4_analysis_after_postprocessing.txt"
    )

    print(f"\nDetailed reports saved:")
    print(f"  - phase4_analysis_before_postprocessing.txt")
    print(f"  - phase4_analysis_after_postprocessing.txt")

    print(
        f"\nComparison complete! Post-processing reduced OOV by {comparison['reduction_total']} occurrences ({comparison['reduction_percentage']:.1f}%)"
    )


if __name__ == "__main__":
    main()
