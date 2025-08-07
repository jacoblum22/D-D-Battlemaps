"""
Main script for processing D&D battlemap images

Usage:
    python main.py <source> [--squares N] [--output DIR]

Where source can be:
    - Path to a zip file
    - Path to a directory containing images
    - Path to a single image file
    - Google Drive folder URL (not yet implemented)
"""

import argparse
import os
import sys
from battlemap_processor import BattlemapProcessor


def main():
    parser = argparse.ArgumentParser(description="Process D&D battlemap images")
    parser.add_argument(
        "source", help="Source: zip file, directory, image file, or Google Drive URL"
    )
    parser.add_argument(
        "--squares",
        type=int,
        default=12,
        help="Number of grid squares per tile (default: 12)",
    )
    parser.add_argument(
        "--output", default="output", help="Output directory (default: 'output')"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Output tile size in pixels (default: 512)",
    )

    args = parser.parse_args()

    # Validate source
    if not (os.path.exists(args.source) or "drive.google.com" in args.source):
        print(f"Error: Source '{args.source}' does not exist")
        return 1

    # Validate parameters
    if args.squares < 8 or args.squares > 20:
        print("Warning: squares per tile should typically be between 8-20")

    # Create processor
    print(f"Initializing battlemap processor...")
    print(f"  Output directory: {args.output}")
    print(f"  Tile size: {args.tile_size}x{args.tile_size}")
    print(f"  Squares per tile: {args.squares}x{args.squares}")

    processor = BattlemapProcessor(output_dir=args.output, tile_size=args.tile_size)

    try:
        # Process the source
        total_tiles = processor.process_source(args.source, args.squares)

        if total_tiles > 0:
            print(f"\nSuccess! Extracted {total_tiles} tiles.")
            print(f"Check the output directory: {args.output}")
        else:
            print(
                "\nNo tiles were extracted. Check if your images have detectable grids."
            )

        return 0

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during processing: {e}")
        return 1
    finally:
        # Cleanup any temporary files
        processor.input_handler.cleanup()


if __name__ == "__main__":
    sys.exit(main())
