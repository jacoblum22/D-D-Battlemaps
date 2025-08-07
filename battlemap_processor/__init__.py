"""
D&D Battlemap Processor

A tool for extracting grid-aligned tiles from D&D battlemap images.
Supports Google Drive links and local zip files as input sources.
"""

from .core.input_handler import InputHandler
from .core.grid_detector import GridDetector
from .core.tile_extractor import TileExtractor
from .core.image_processor import ImageProcessor

__version__ = "0.1.0"


class BattlemapProcessor:
    """Main class for processing D&D battlemaps"""

    def __init__(self, output_dir: str = "output", tile_size: int = 512):
        self.output_dir = output_dir
        self.tile_size = tile_size

        self.input_handler = InputHandler()
        self.grid_detector = GridDetector()
        self.tile_extractor = TileExtractor(tile_size=tile_size)
        self.image_processor = ImageProcessor()

    def process_source(self, source: str, squares_per_tile: int = 12):
        """
        Process a Google Drive link or zip file path

        Args:
            source: Google Drive URL or local zip file path
            squares_per_tile: Number of grid squares per tile (width/height)
        """
        print(f"Processing source: {source}")

        # Step 1: Get images from source
        images = self.input_handler.get_images_from_source(source)
        print(f"Found {len(images)} images to process")

        # Step 2: Process each image
        total_tiles = 0
        for img_name, img_data in images.items():
            print(f"\nProcessing {img_name}...")

            try:
                # Detect grid
                grid_info = self.grid_detector.detect_grid(img_data)
                if grid_info is None:
                    print(f"  No grid detected in {img_name}, skipping")
                    continue

                print(f"  Grid detected: {grid_info['nx']}x{grid_info['ny']} cells")

                # Extract tiles
                tiles = self.tile_extractor.extract_tiles(
                    img_data, grid_info, squares_per_tile
                )

                # Save tiles
                saved = self.image_processor.save_tiles(
                    tiles, img_name, self.output_dir
                )

                print(f"  Saved {saved} tiles from {img_name}")
                total_tiles += saved

            except Exception as e:
                print(f"  Error processing {img_name}: {e}")
                continue

        print(
            f"\nProcessing complete! Saved {total_tiles} total tiles to {self.output_dir}"
        )
        return total_tiles
