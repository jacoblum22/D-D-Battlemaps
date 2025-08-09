"""
Battlemap Processing Pipeline

This module provides a complete end-to-end pipeline for processing battlemap images:
1. Find images from various sources (Google Drive, zip files, local directories)
2. Apply smart image selection to prefer optimal variants
3. Detect grid dimensions using filename hints and visual analysis
4. Detect boring areas in images
5. Place optimal 12x12 tiles to maximize non-boring coverage
6. Save tiles as 512x512 images in organized subfolders

Features:
- Memory efficient: processes one image at a time
- Progress saving: can resume interrupted processing
- Proof-of-concept limits: configurable max images and tiles
- Organized output: tiles saved in subfolders by source image
"""

import logging
import os
import json
import tempfile
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from PIL import Image

from .image_source_handler import ImageSourceHandler, ImageInfo
from .grid_detector import GridDetector
from .advanced_boring_detector import AdvancedBoringDetector
from .optimal_tile_placer import OptimalTilePlacer
from .tile_saver import TileSaver

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the battlemap pipeline"""

    # Input settings
    sources: List[str]  # List of source paths/URLs to process
    use_smart_selection: bool = True  # Apply smart image selection

    # Processing limits (for proof-of-concept)
    max_images: Optional[int] = 20  # Max images to process (None = unlimited)
    max_tiles_per_image: Optional[int] = 50  # Max tiles per image (None = unlimited)

    # Grid and boring detection settings
    tile_size: int = 12  # Size of each tile in grid squares
    boring_threshold: float = 0.5  # Max fraction of boring squares per tile

    # Output settings
    output_dir: str = "generated_images"  # Base output directory
    tile_output_size: int = 512  # Size of output tile images (pixels)

    # Processing settings
    temp_dir: Optional[str] = None  # Temporary directory for processing
    save_progress: bool = True  # Save progress for resuming
    debug: bool = False  # Enable debug output


@dataclass
class ProcessingStats:
    """Statistics for pipeline processing"""

    images_found: int = 0
    images_processed: int = 0
    images_skipped: int = 0
    images_failed: int = 0
    total_tiles_generated: int = 0
    total_boring_tiles_rejected: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        if self.start_time:
            result["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingStats":
        """Create from dictionary (JSON deserialization)"""
        if "start_time" in data and data["start_time"]:
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if "end_time" in data and data["end_time"]:
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        return cls(**data)


@dataclass
class ProcessingProgress:
    """Progress tracking for resumable processing"""

    config: PipelineConfig
    stats: ProcessingStats
    processed_images: Optional[List[str]] = (
        None  # List of image paths that have been processed
    )
    current_source_index: int = 0  # Current source being processed

    def __post_init__(self):
        if self.processed_images is None:
            self.processed_images = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "config": asdict(self.config),
            "stats": self.stats.to_dict(),
            "processed_images": self.processed_images,
            "current_source_index": self.current_source_index,
            "progress_version": "1.0",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingProgress":
        """Create from dictionary (JSON deserialization)"""
        config = PipelineConfig(**data["config"])
        stats = ProcessingStats.from_dict(data["stats"])
        return cls(
            config=config,
            stats=stats,
            processed_images=data.get("processed_images", []),
            current_source_index=data.get("current_source_index", 0),
        )


class BattlemapPipeline:
    """
    Complete pipeline for processing battlemap images into training tiles
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.progress = ProcessingProgress(config=config, stats=ProcessingStats())

        # Initialize components
        self.image_handler = ImageSourceHandler(temp_dir=config.temp_dir)
        self.grid_detector = GridDetector()
        self.boring_detector = AdvancedBoringDetector()
        self.tile_placer = OptimalTilePlacer()
        self.tile_saver = TileSaver()

        # Set up output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Progress file path
        self.progress_file = self.output_dir / "pipeline_progress.json"

        if config.debug:
            print(f"🚀 Battlemap Pipeline initialized")
            print(f"   Output directory: {self.output_dir}")
            print(f"   Max images: {config.max_images or 'unlimited'}")
            print(
                f"   Max tiles per image: {config.max_tiles_per_image or 'unlimited'}"
            )

    def run(self) -> ProcessingStats:
        """
        Run the complete pipeline

        Returns:
            Final processing statistics
        """
        if self.config.debug:
            print(f"\n🎯 Starting battlemap pipeline processing...")

        self.progress.stats.start_time = datetime.now()

        try:
            # Load existing progress if available
            if self.config.save_progress and self.progress_file.exists():
                self._load_progress()
                if self.config.debug:
                    print(f"📁 Resumed from saved progress")
                    print(
                        f"   Previously processed: {len(self.progress.processed_images or [])} images"
                    )
                    print(
                        f"   Starting from source index: {self.progress.current_source_index}"
                    )

            # Process each source
            for source_idx in range(
                self.progress.current_source_index, len(self.config.sources)
            ):
                source = self.config.sources[source_idx]
                self.progress.current_source_index = source_idx

                if self.config.debug:
                    print(
                        f"\n📂 Processing source {source_idx + 1}/{len(self.config.sources)}: {source}"
                    )

                # Check if we've hit the image limit
                if (
                    self.config.max_images
                    and self.progress.stats.images_processed >= self.config.max_images
                ):
                    if self.config.debug:
                        print(
                            f"🛑 Reached maximum image limit ({self.config.max_images})"
                        )
                    break

                self._process_source(source)

                # Save progress after each source
                if self.config.save_progress:
                    self._save_progress()

            self.progress.stats.end_time = datetime.now()

            if self.config.debug:
                self._print_final_stats()

            return self.progress.stats

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            if self.config.debug:
                print(f"❌ Pipeline error: {e}")
            raise

        finally:
            # Clean up temporary files
            self.image_handler.cleanup_temp_files(debug=self.config.debug)

    def _process_source(self, source: str):
        """Process a single source (Google Drive, zip, local directory)"""
        try:
            if self.config.debug:
                print(f"🔍 Finding images in source...")

            # Find images in source
            images = self.image_handler.find_images_from_source(
                source=source,
                debug=self.config.debug,
                list_only=True,  # Memory efficient - list first
                use_smart_selection=self.config.use_smart_selection,
            )

            if not images:
                if self.config.debug:
                    print(f"⚠️  No images found in source")
                return

            self.progress.stats.images_found += len(images)

            if self.config.debug:
                print(f"📊 Found {len(images)} images to process")

            # Process each image individually
            for img_info in images:
                # Check if already processed
                processed_list = self.progress.processed_images or []
                if img_info.path in processed_list:
                    if self.config.debug:
                        print(f"⏭️  Skipping already processed: {img_info.filename}")
                    continue

                # Check if we've hit the image limit
                if (
                    self.config.max_images
                    and self.progress.stats.images_processed >= self.config.max_images
                ):
                    if self.config.debug:
                        print(
                            f"🛑 Reached maximum image limit ({self.config.max_images})"
                        )
                    break

                # Process single image
                self._process_single_image(img_info)

                # Mark as processed
                if self.progress.processed_images is None:
                    self.progress.processed_images = []
                self.progress.processed_images.append(img_info.path)

        except Exception as e:
            logger.error(f"Error processing source '{source}': {e}")
            if self.config.debug:
                print(f"❌ Source processing error: {e}")

    def _process_single_image(self, img_info: ImageInfo):
        """Process a single image to generate tiles"""
        if self.config.debug:
            print(f"\n🖼️  Processing: {img_info.filename}")

        try:
            # Download/access the image
            image_path = self.image_handler.download_single_image(
                img_info, debug=self.config.debug
            )
            if not image_path:
                if self.config.debug:
                    print(f"❌ Failed to access image")
                self.progress.stats.images_failed += 1
                return

            # Load the image
            try:
                pil_image = Image.open(image_path)

                # Skip images with transparency (tokens, UI elements, etc.)
                if pil_image.mode in ("RGBA", "LA") or "transparency" in pil_image.info:
                    # Check if image actually has transparent pixels
                    if pil_image.mode == "RGBA":
                        img_array = np.array(pil_image)
                        alpha_channel = img_array[:, :, 3]
                        has_transparency = np.any(alpha_channel < 255)
                    elif pil_image.mode == "LA":
                        img_array = np.array(pil_image)
                        alpha_channel = img_array[:, :, 1]
                        has_transparency = np.any(alpha_channel < 255)
                    else:
                        has_transparency = True  # Has transparency info

                    if has_transparency:
                        if self.config.debug:
                            print(
                                f"⏭️ Image has transparent parts - skipping analysis (likely token or UI element)"
                            )
                        self.progress.stats.images_skipped += 1
                        return

                # Convert to RGB for processing
                pil_image = pil_image.convert("RGB")
                image_array = np.array(pil_image)
            except Exception as e:
                if self.config.debug:
                    print(f"❌ Failed to load image: {e}")
                self.progress.stats.images_failed += 1
                return

            if self.config.debug:
                print(
                    f"📐 Image size: {image_array.shape[1]}x{image_array.shape[0]} pixels"
                )

            # Detect grid dimensions
            grid_info = self.grid_detector.detect_grid_with_filename_fallback(
                pil_image, img_info.filename
            )

            if not grid_info:
                if self.config.debug:
                    print(f"⚠️  Could not detect grid - skipping image")
                self.progress.stats.images_skipped += 1
                return

            if self.config.debug:
                print(f"🔲 Grid detected: {grid_info['nx']}x{grid_info['ny']} squares")

            # Analyze boring areas
            square_analysis, boring_reasons = (
                self.boring_detector.analyze_image_regions(
                    pil_image, grid_info, debug=self.config.debug
                )
            )

            # Place optimal tiles
            placed_tiles = self.tile_placer.find_optimal_placements(
                grid_info=grid_info,
                square_analysis=square_analysis,
                debug=self.config.debug,
            )

            # Apply limits if configured
            if (
                self.config.max_tiles_per_image
                and len(placed_tiles) > self.config.max_tiles_per_image
            ):
                placed_tiles = placed_tiles[: self.config.max_tiles_per_image]
                if self.config.debug:
                    print(f"🔢 Limited to {self.config.max_tiles_per_image} tiles")

            if not placed_tiles:
                if self.config.debug:
                    print(f"⚠️  No suitable tile positions found - skipping image")
                self.progress.stats.images_skipped += 1
                return

            # Create output subfolder for this image
            image_name = Path(img_info.filename).stem
            image_output_dir = self.output_dir / image_name
            image_output_dir.mkdir(parents=True, exist_ok=True)

            if self.config.debug:
                print(f"💾 Saving {len(placed_tiles)} tiles to: {image_output_dir}")

            # Save tiles
            result = self.tile_saver.save_tiles(
                source_image=pil_image,
                grid_info=grid_info,
                placed_tiles=placed_tiles,
                source_path_or_url=img_info.source_url or img_info.path,
                output_folder=str(image_output_dir),
                debug=self.config.debug,
            )

            # Update statistics
            self.progress.stats.images_processed += 1
            self.progress.stats.total_tiles_generated += len(result.saved_tiles)

            if self.config.debug:
                print(
                    f"✅ Successfully processed: {len(result.saved_tiles)} tiles saved"
                )

            # Clean up temporary file if it was downloaded
            if image_path != img_info.path and image_path.startswith(
                self.image_handler.temp_dir
            ):
                try:
                    os.remove(image_path)
                except:
                    pass  # Ignore cleanup errors

        except Exception as e:
            logger.error(f"Error processing image {img_info.filename}: {e}")
            if self.config.debug:
                print(f"❌ Image processing error: {e}")
            self.progress.stats.images_failed += 1

    def _save_progress(self):
        """Save current progress to file"""
        try:
            with open(self.progress_file, "w") as f:
                json.dump(self.progress.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")

    def _load_progress(self):
        """Load progress from file"""
        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
            self.progress = ProcessingProgress.from_dict(data)
        except Exception as e:
            logger.warning(f"Could not load progress: {e}")

    def _print_final_stats(self):
        """Print final processing statistics"""
        stats = self.progress.stats
        duration = None
        if stats.start_time and stats.end_time:
            duration = stats.end_time - stats.start_time

        print(f"\n📊 Pipeline Processing Complete!")
        print(f"════════════════════════════════════")
        print(f"Images found:      {stats.images_found}")
        print(f"Images processed:  {stats.images_processed}")
        print(f"Images skipped:    {stats.images_skipped}")
        print(f"Images failed:     {stats.images_failed}")
        print(f"Total tiles:       {stats.total_tiles_generated}")
        if duration:
            print(f"Duration:          {duration}")
        print(f"Output directory:  {self.output_dir}")
        print(f"════════════════════════════════════")

    def get_stats(self) -> ProcessingStats:
        """Get current processing statistics"""
        return self.progress.stats

    def resume_from_progress(self) -> bool:
        """
        Try to resume from saved progress

        Returns:
            True if progress was loaded successfully, False otherwise
        """
        if self.progress_file.exists():
            try:
                self._load_progress()
                return True
            except Exception as e:
                logger.warning(f"Could not resume from progress: {e}")
                return False
        return False

    def clear_progress(self):
        """Clear saved progress (start fresh)"""
        if self.progress_file.exists():
            self.progress_file.unlink()
        self.progress = ProcessingProgress(config=self.config, stats=ProcessingStats())
