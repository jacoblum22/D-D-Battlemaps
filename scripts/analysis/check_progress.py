"""
Check processing status and statistics for the captioning system.
Shows how many images have been processed and how many remain.
"""

import sys
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from battlemap_processor.captioning import ControlledVocabularyCaptioner


def main():
    """Show processing statistics."""
    print("=== Processing Status Check ===")

    try:
        captioner = ControlledVocabularyCaptioner()

        # Get statistics
        stats = captioner.get_processing_stats()

        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"Total available images: {stats['total_available']:,}")
        print(f"Already processed: {stats['already_processed']:,}")
        print(f"Remaining unprocessed: {stats['remaining_unprocessed']:,}")
        print(f"Progress: {stats['processed_percentage']:.1f}% complete")

        # Cost estimates for remaining
        remaining = stats["remaining_unprocessed"]
        if remaining > 0:
            est_cost_per_image = 0.0057  # Based on recent runs

            print(f"\n💰 COST ESTIMATES FOR REMAINING IMAGES:")
            batch_sizes = [100, 300, 500, remaining]
            for size in batch_sizes:
                if size <= remaining:
                    cost = size * est_cost_per_image
                    print(f"  {size:,} images: ${cost:.2f}")

            if remaining > 1000:
                print(f"\n🎯 RECOMMENDED BATCHES:")
                print(f"  Phase 2: 500 images (${500 * est_cost_per_image:.2f})")
                print(
                    f"  Phase 3: {min(1000, remaining - 500):,} images (${min(1000, remaining - 500) * est_cost_per_image:.2f})"
                )
        else:
            print(f"\n✅ All images have been processed!")

        # Show recently processed files
        processed_images = captioner.load_processed_images()
        if processed_images:
            print(f"\n📁 RECENT CAPTION FILES FOUND:")
            caption_files = [
                "captions_batch.json",
                "test_captions.json",
                "validation_captions.json",
            ]
            for filename in caption_files:
                if Path(filename).exists():
                    print(f"  ✓ {filename}")
                else:
                    print(f"  - {filename} (not found)")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
