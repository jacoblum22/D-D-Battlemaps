"""
D&D Battlemap Processor - Usage Examples

This file demonstrates various ways to use the battlemap processing pipeline.
"""

import os
from battlemap_processor import BattlemapProcessor


def example_basic_usage():
    """Basic usage example - process a single image"""

    # Initialize processor
    processor = BattlemapProcessor(output_dir="output", tile_size=512)

    # Process a single image
    image_path = "path/to/your/battlemap.jpg"
    tiles_extracted = processor.process_source(image_path, squares_per_tile=12)

    print(f"Extracted {tiles_extracted} tiles from {image_path}")


def example_batch_processing():
    """Example of batch processing multiple images"""

    processor = BattlemapProcessor(output_dir="output_batch", tile_size=512)

    # Process a zip file containing multiple battlemaps
    zip_path = "data/datasets/battlemap_collection.zip"
    total_tiles = processor.process_source(zip_path, squares_per_tile=14)

    print(f"Batch processing complete. Total tiles: {total_tiles}")


def example_custom_configuration():
    """Example with custom configuration"""

    # Custom settings for different use cases
    processor = BattlemapProcessor(
        output_dir="custom_output", tile_size=256  # Smaller tiles for faster processing
    )

    # Extract larger tiles (16x16 grid squares)
    directory_path = "path/to/battlemap/directory"
    tiles = processor.process_source(directory_path, squares_per_tile=16)

    print(f"Custom processing complete: {tiles} tiles of 256x256 pixels")


def example_quality_control():
    """Example demonstrating quality control features"""

    processor = BattlemapProcessor(output_dir="quality_filtered", tile_size=512)

    # The processor automatically filters out:
    # - Very dark tiles (low brightness)
    # - Tiles that don't align properly with the detected grid
    # - Tiles smaller than the minimum required size

    source = "high_quality_battlemaps.zip"
    tiles = processor.process_source(source, squares_per_tile=12)

    print(f"Quality-filtered extraction: {tiles} tiles")


if __name__ == "__main__":
    # Run examples (commented out to prevent accidental execution)

    print("D&D Battlemap Processor Examples")
    print("================================")
    print()
    print("Uncomment the example functions below to try them:")
    print()
    print("# example_basic_usage()")
    print("# example_batch_processing()")
    print("# example_custom_configuration()")
    print("# example_quality_control()")
    print()
    print("For more information, see the README.md or run:")
    print("python main.py --help")
