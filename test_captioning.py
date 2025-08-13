"""
Test the captioning system on a sma        # Take first 5 images for validation test
        test_images = all_images[:5]

        print(f"Testing with {len(test_images)} images:")
        for img in test_images:
            print(f"  - {Path(img).name}")ch of images.

This script tests the captioning system on just 3-5 images to verify
everything works correctly before running the full batch.
"""

import sys
import os
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from battlemap_processor.captioning import ControlledVocabularyCaptioner


def main():
    """Test captioning on a small batch."""
    print("=== Captioning System Test ===")
    print("Testing on a small batch of images...")

    try:
        # Initialize the captioner
        captioner = ControlledVocabularyCaptioner()

        # Find a few images to test
        root_dir = "generated_images"

        if not os.path.exists(root_dir):
            print(f"Error: {root_dir} directory not found!")
            return

        # Get just 3-5 images for testing
        all_images = captioner.find_images(
            root_dir, max_images=500
        )  # Get more to choose from

        if not all_images:
            print("No images found to test!")
            return

        # Take first 5 images for validation test
        test_images = all_images[:5]

        print(f"Testing with {len(test_images)} images:")
        for img in test_images:
            print(f"  - {Path(img).name}")

        # Process the test batch
        print(f"\nStarting test captioning...")
        stats = captioner.process_batch(test_images, output_file="test_captions.json")

        # Quick analysis
        suggestions = captioner.analyze_oov_terms(
            stats, min_frequency=1
        )  # Lower threshold for test

        # Generate test report
        captioner.generate_report(stats, suggestions, "test_vocabulary_analysis.txt")

        # Display results
        print(f"\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        print(f"Images processed: {stats.total_images}")
        print(f"Successful: {stats.successful_captions}")
        print(f"Failed: {stats.failed_captions}")

        if stats.successful_captions > 0:
            avg_cost = stats.total_cost_usd / stats.successful_captions
            avg_tokens = stats.total_tokens_used / stats.successful_captions
            print(f"\n💰 COST PER IMAGE:")
            print(f"  Average cost: ${avg_cost:.4f}")
            print(f"  Average tokens: {avg_tokens:.0f}")

            # Estimate costs for different batch sizes
            print(f"\n📊 ESTIMATED COSTS:")
            for count in [10, 50, 100, 500]:
                estimated = avg_cost * count
                print(f"  {count:3d} images: ${estimated:7.2f}")

        print(f"\nTotal cost for this test: ${stats.total_cost_usd:.4f}")
        print(f"Total tokens used: {stats.total_tokens_used}")

        if stats.oov_terms:
            print(f"\nOOV terms found ({len(stats.oov_terms)} unique):")
            sorted_oov = sorted(
                stats.oov_terms.items(), key=lambda x: x[1], reverse=True
            )
            for term, count in sorted_oov:
                print(f"  {term}: {count}")

        print(f"\nTest files created:")
        print(f"  - test_captions.json")
        print(f"  - test_vocabulary_analysis.txt")

        print(f"\n✅ Test completed successfully!")
        print(f"Ready to run full batch processing.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
