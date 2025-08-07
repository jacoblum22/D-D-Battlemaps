"""
D&D Battlemap Processor

A tool for extracting grid-aligned tiles from D&D battlemap images.
Supports Google Drive links and local zip files as input sources.
"""

import logging
from .core.input_handler import InputHandler
from .core.grid_detector import GridDetector
from .core.tile_extractor import TileExtractor
from .core.image_processor import ImageProcessor

__version__ = "0.1.0"

# Set up logging
logger = logging.getLogger(__name__)

# Add console handler by default if no handlers exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")  # Simple format for console
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


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
        logger.info(f"Processing source: {source}")

        # Process images one at a time (streaming for memory efficiency)
        total_tiles = 0
        processed_count = 0

        for img_name, img_data in self.input_handler.stream_images_from_source(source):
            processed_count += 1
            logger.info(f"Processing image {processed_count}: {img_name}...")

            try:
                # Detect grid
                grid_info = self.grid_detector.detect_grid(img_data)
                if grid_info is None:
                    logger.warning(f"  No grid detected in {img_name}, skipping")
                    continue

                logger.info(
                    f"  Grid detected: {grid_info['nx']}x{grid_info['ny']} cells"
                )

                # Extract tiles
                tiles = self.tile_extractor.extract_tiles(
                    img_data, grid_info, squares_per_tile
                )

                # Save tiles
                saved = self.image_processor.save_tiles(
                    tiles, img_name, self.output_dir
                )

                logger.info(f"  Saved {saved} tiles from {img_name}")
                total_tiles += saved

            except Exception as e:
                logger.exception(f"Error processing {img_name}: {e}")
                continue

        logger.info(
            f"Processing complete! Processed {processed_count} images and saved {total_tiles} total tiles to {self.output_dir}"
        )
        return total_tiles
