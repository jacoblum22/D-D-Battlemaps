"""
Run the battlemap captioning process with controlled vocabulary.

This script processes a batch of battlemap images and generates captions
using OpenAI's GPT-4o with a controlled vocabulary system.
"""

import sys
import os
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from battlemap_processor.captioning import ControlledVocabularyCaptioner


def main():
    """Run the captioning process."""
    print("=== D&D Battlemap Captioning System ===")
    print("Initializing captioner with controlled vocabulary...")

    try:
        # Initialize the captioner
        captioner = ControlledVocabularyCaptioner()

        # Configuration
        root_dir = "generated_images"
        max_images = 10000  # Set high to process all available images

        print(f"\nFinding images in {root_dir}...")

        # Find images to process
        if not os.path.exists(root_dir):
            print(f"Error: {root_dir} directory not found!")
            return

        image_paths = captioner.find_images(root_dir, max_images=max_images)

        if not image_paths:
            print("No images found to process!")
            return

        print(f"Found {len(image_paths)} images to process")

        # Ask user for confirmation
        response = input(
            f"\nProceed with captioning {len(image_paths)} images? (y/n): "
        )
        if response.lower() != "y":
            print("Cancelled.")
            return

        # Process the batch
        print(f"\nStarting captioning process...")
        print(f"This will process {len(image_paths)} images...")

        # Dynamic file naming based on image count
        if max_images <= 10:
            output_file = "test_captions.json"
            analysis_file = "test_vocabulary_analysis.txt"
            phase_name = "Test"
        elif max_images <= 100:
            output_file = "phase3_captions.json"
            analysis_file = "phase3_vocabulary_analysis.txt"
            phase_name = "Phase 3"
        elif max_images <= 500:
            output_file = "phase2_captions.json"
            analysis_file = "phase2_vocabulary_analysis.txt"
            phase_name = "Phase 2"
        else:
            output_file = "phase4_captions.json"
            analysis_file = "phase4_vocabulary_analysis.txt"
            phase_name = "Phase 4"

        stats = captioner.process_batch(image_paths, output_file=output_file)

        # Analyze OOV terms
        print("\nAnalyzing out-of-vocabulary terms...")
        suggestions = captioner.analyze_oov_terms(stats, min_frequency=2)

        # Generate comprehensive report
        print(f"Generating {phase_name} analysis report...")
        captioner.generate_report(stats, suggestions, analysis_file)

        # Display summary
        print("\n" + "=" * 50)
        print("CAPTIONING COMPLETE - SUMMARY")
        print("=" * 50)
        print(f"Total images: {stats.total_images}")
        print(f"Successful captions: {stats.successful_captions}")
        print(f"Failed captions: {stats.failed_captions}")
        print(f"Success rate: {stats.successful_captions/stats.total_images*100:.1f}%")

        # Cost information
        print(f"\n💰 COST ANALYSIS:")
        print(f"Total cost: ${stats.total_cost_usd:.4f}")
        print(f"Total tokens: {stats.total_tokens_used:,}")
        if stats.successful_captions > 0:
            avg_cost = stats.total_cost_usd / stats.successful_captions
            avg_tokens = stats.total_tokens_used / stats.successful_captions
            print(f"Average per image: ${avg_cost:.4f} ({avg_tokens:.0f} tokens)")

            # Cost projections
            print(f"\n📊 COST PROJECTIONS:")
            projections = [100, 500, 1000, 5000]
            for count in projections:
                projected_cost = avg_cost * count
                print(f"  {count:,} images: ${projected_cost:.2f}")

        print(f"\nUnique OOV terms: {len(stats.oov_terms)}")
        print(f"High frequency OOV terms: {len(suggestions['high_frequency'])}")

        # Show top OOV terms
        if stats.oov_terms:
            print(f"\nTop 10 OOV terms to consider adding:")
            sorted_oov = sorted(
                stats.oov_terms.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for term, count in sorted_oov:
                print(f"  {term}: {count} occurrences")

        print(f"\nDetailed results saved to:")
        print(f"  - {output_file} (all captions)")
        print(f"  - {analysis_file} (analysis report)")

        if max_images < 1000:
            print(f"\nNext steps:")
            print(f"  1. Review {analysis_file}")
            print(f"  2. Update vocabulary lists in captioning.py if needed")
            print(f"  3. Scale up to larger batch if results look good")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
