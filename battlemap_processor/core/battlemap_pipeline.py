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
import contextlib
import tempfile
import shutil
import numpy as np
import multiprocessing
import concurrent.futures
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict


def detect_grid_brightness_method(image_path):
    """
    Brightness-based grid detection method using analyze_brightness.py
    """
    try:
        # Run the analyze_brightness script and capture output
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, "analyze_brightness.py", str(image_path), "--no-display"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            return {
                "method": "Brightness Analysis",
                "success": False,
                "error": f"Script failed: {result.stderr}",
            }

        output = result.stdout

        # Parse the output to extract grid information
        grid_size = None
        confidence = 0.0
        cols = 0
        rows = 0

        # Look for grid size
        grid_size_match = re.search(r"Square grid size: (\d+(?:\.\d+)?) pixels", output)
        if grid_size_match:
            grid_size = float(grid_size_match.group(1))

        # Look for confidence
        confidence_match = re.search(r"Confidence: (\d+(?:\.\d+)?)%", output)
        if confidence_match:
            confidence = float(confidence_match.group(1)) / 100.0

        # Look for grid dimensions
        cols_match = re.search(r"Grid columns: (\d+)", output)
        if cols_match:
            cols = int(cols_match.group(1))

        rows_match = re.search(r"Grid rows: (\d+)", output)
        if rows_match:
            rows = int(rows_match.group(1))

        # If grid size was found, we have a successful detection
        success = grid_size is not None and grid_size > 0 and cols > 0 and rows > 0

        if success and grid_size is not None:
            return {
                "method": "brightness",
                "success": True,
                "nx": cols,
                "ny": rows,
                "cell_width": grid_size,
                "cell_height": grid_size,
                "x_edges": [i * grid_size for i in range(cols + 1)],
                "y_edges": [i * grid_size for i in range(rows + 1)],
                "score": confidence,  # Use confidence as score
                "size_px": grid_size,
                "confidence": confidence,
                "detection_method": "brightness",
            }
        else:
            return {"method": "brightness", "success": False}

    except subprocess.TimeoutExpired:
        return {
            "method": "brightness",
            "success": False,
            "error": "Analysis timeout",
        }
    except Exception as e:
        return {"method": "brightness", "success": False, "error": str(e)}


def convert_score_to_confidence(detection_score):
    """
    Convert morphological detection score to confidence percentage.
    """
    if detection_score is None:
        return 0.0

    if detection_score <= 0:
        return 0.0

    # Base confidence for any successful detection
    base_confidence = 55.0

    # Logarithmic boost - scores have diminishing returns
    log_boost = np.log(1 + detection_score) * 15.0

    # Additional linear boost for very high scores
    if detection_score > 5:
        linear_boost = (detection_score - 5) * 1.0
    else:
        linear_boost = 0

    confidence = base_confidence + log_boost + linear_boost

    # Cap at 95% (never 100% confident)
    return min(95.0, max(0.0, confidence)) / 100.0


def enhanced_grid_detection(pil_image, image_path, img_info, debug=False):
    """
    Enhanced grid detection using both brightness and morphological methods.
    Returns the detection with higher confidence.
    """
    grid_detector = None

    # Import here to avoid circular imports
    try:
        from .grid_detector import GridDetector

        grid_detector = GridDetector()
    except ImportError:
        # Fallback for when not in package context
        import sys

        sys.path.append(str(Path(__file__).parent))
        from grid_detector import GridDetector

        grid_detector = GridDetector()

    results = []

    # Method 1: Try brightness detection
    if debug:
        print(f"    🔍 Trying brightness-based grid detection...")

    brightness_result = detect_grid_brightness_method(image_path)
    if brightness_result.get("success"):
        brightness_confidence = brightness_result.get("confidence", 0)
        if debug:
            print(
                f"    ✅ Brightness detection: {brightness_result['nx']}x{brightness_result['ny']} grid, confidence: {brightness_confidence:.1%}"
            )
        results.append(brightness_result)
    else:
        if debug:
            print(
                f"    ❌ Brightness detection failed: {brightness_result.get('error', 'Unknown error')}"
            )

    # Method 2: Try morphological detection
    if debug:
        print(f"    🔍 Trying morphological grid detection...")

    # Try filename-based detection first if available
    morphological_result = None
    if img_info and img_info.has_dimensions:
        try:
            filename_dims = grid_detector.extract_dimensions_from_filename(
                img_info.filename
            )
            if filename_dims:
                nx, ny = filename_dims
                img_width, img_height = pil_image.size

                cell_width = img_width / float(nx)
                cell_height = img_height / float(ny)

                morphological_result = {
                    "method": "morphological",
                    "success": True,
                    "nx": nx,
                    "ny": ny,
                    "cell_width": cell_width,
                    "cell_height": cell_height,
                    "x_edges": [i * cell_width for i in range(nx + 1)],
                    "y_edges": [i * cell_height for i in range(ny + 1)],
                    "score": 0.95,  # High confidence for filename dimensions
                    "size_px": min(cell_width, cell_height),
                    "confidence": 0.95,
                    "detection_method": "filename_dimensions",
                    "filename_dimensions": filename_dims,
                }
                if debug:
                    print(
                        f"    ✅ Filename detection: {nx}x{ny} grid, confidence: 95.0%"
                    )
        except Exception:
            pass

    # If filename detection didn't work, try visual morphological detection
    if not morphological_result:
        try:
            morphological_result = grid_detector.detect_grid(
                pil_image, str(image_path) if img_info else None
            )
            if morphological_result:
                # Convert score to confidence
                confidence = convert_score_to_confidence(
                    morphological_result.get("score", 0)
                )
                morphological_result["confidence"] = confidence
                morphological_result["method"] = "morphological"
                morphological_result["success"] = True
                if debug:
                    print(
                        f"    ✅ Morphological detection: {morphological_result['nx']}x{morphological_result['ny']} grid, confidence: {confidence:.1%}"
                    )
            else:
                if debug:
                    print(f"    ❌ Morphological detection failed")
        except Exception as e:
            if debug:
                print(f"    ❌ Morphological detection error: {e}")

    if morphological_result and morphological_result.get("success"):
        results.append(morphological_result)

    # Choose the best result
    if not results:
        if debug:
            print(f"    ❌ All grid detection methods failed")
        return None

    if len(results) == 1:
        best_result = results[0]
        if debug:
            print(
                f"    🎯 Using {best_result['method']} detection (only successful method)"
            )
    else:
        # Compare confidence levels
        brightness_conf = next(
            (r["confidence"] for r in results if r["method"] == "brightness"), 0
        )
        morphological_conf = next(
            (r["confidence"] for r in results if r["method"] == "morphological"), 0
        )

        if brightness_conf > morphological_conf:
            best_result = next(r for r in results if r["method"] == "brightness")
            if debug:
                print(
                    f"    🎯 Using brightness detection (confidence: {brightness_conf:.1%} vs {morphological_conf:.1%})"
                )
        else:
            best_result = next(r for r in results if r["method"] == "morphological")
            if debug:
                print(
                    f"    🎯 Using morphological detection (confidence: {morphological_conf:.1%} vs {brightness_conf:.1%})"
                )

    return best_result


from PIL import Image

from .image_source_handler import ImageSourceHandler, ImageInfo
from .smart_image_selector import SmartImageSelector
from .grid_detector import GridDetector
from .advanced_boring_detector import AdvancedBoringDetector
from .optimal_tile_placer import OptimalTilePlacer
from .tile_saver import TileSaver


def get_tile_size_for_image(
    image_index: int, current_stats: Optional[dict] = None
) -> int:
    """
    Get the optimal tile size for an image to maintain target tile count ratios.
    Target: 70% 12x12, 20% 20x20, 10% 30x30 tiles

    Args:
        image_index: Index of current image (for fallback)
        current_stats: Dictionary with current tile counts {'tiles_12x12': X, 'tiles_20x20': Y, 'tiles_30x30': Z}

    Returns:
        int: tile size (12, 20, or 30)
    """
    # Target ratios for tile counts (not image counts)
    target_12x12_ratio = 0.70  # 70%
    target_20x20_ratio = 0.20  # 20%
    target_30x30_ratio = 0.10  # 10%

    # If no stats provided, fall back to simple rotation for first few images
    if current_stats is None:
        # Simple rotation for the first images before we have enough data
        cycle_position = image_index % 10
        if cycle_position < 7:  # positions 0,1,2,3,4,5,6
            return 12
        elif cycle_position < 9:  # positions 7,8
            return 20
        else:  # position 9
            return 30

    # Get current counts
    tiles_12x12 = current_stats.get("tiles_12x12", 0)
    tiles_20x20 = current_stats.get("tiles_20x20", 0)
    tiles_30x30 = current_stats.get("tiles_30x30", 0)
    total_tiles = tiles_12x12 + tiles_20x20 + tiles_30x30

    # If no tiles yet, start with 12x12
    if total_tiles == 0:
        return 12

    # Calculate current ratios
    current_12x12_ratio = tiles_12x12 / total_tiles
    current_20x20_ratio = tiles_20x20 / total_tiles
    current_30x30_ratio = tiles_30x30 / total_tiles

    # Calculate how far each ratio is from target
    deficit_12x12 = target_12x12_ratio - current_12x12_ratio
    deficit_20x20 = target_20x20_ratio - current_20x20_ratio
    deficit_30x30 = target_30x30_ratio - current_30x30_ratio

    # Choose the tile size with the largest deficit (most under-represented)
    max_deficit = max(deficit_12x12, deficit_20x20, deficit_30x30)

    if max_deficit == deficit_12x12:
        return 12
    elif max_deficit == deficit_20x20:
        return 20
    else:
        return 30


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
    # Note: tile_size is now determined per-image using rotation pattern
    boring_threshold: float = 0.5  # Max fraction of boring squares per tile

    # Output settings
    output_dir: str = "generated_images"  # Base output directory
    tile_output_size: int = 512  # Size of output tile images (pixels)

    # Processing settings
    temp_dir: Optional[str] = None  # Temporary directory for processing
    save_progress: bool = True  # Save progress for resuming
    debug: bool = False  # Enable debug output
    test_mode: bool = False  # Test mode: process but don't save images
    use_multiprocessing: bool = True  # Enable parallel processing of images
    clear_hash_cache: bool = False  # Clear hash cache at startup


@dataclass
class ProcessingStats:
    """Statistics for pipeline processing"""

    images_found: int = 0
    images_processed: int = 0
    images_skipped: int = 0
    images_failed: int = 0
    total_tiles_generated: int = 0
    total_boring_tiles_rejected: int = 0

    # Tile size distribution tracking
    tiles_12x12: int = 0
    tiles_20x20: int = 0
    tiles_30x30: int = 0

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


def _process_downloaded_image_multiprocessing(args) -> Dict:
    """
    Process a single pre-downloaded image (for multiprocessing).
    Image has already been downloaded, so this only handles processing.

    Args:
        args: Tuple of (img_info_dict, config_dict, image_path, output_dir, image_index, tile_size)

    Returns:
        Dict with processing results
    """
    img_info_dict, config_dict, image_path, output_dir, image_index, tile_size = args

    try:
        # Import here to avoid issues with multiprocessing
        from .image_source_handler import ImageInfo
        from .grid_detector import GridDetector
        from .advanced_boring_detector import AdvancedBoringDetector
        from .optimal_tile_placer import OptimalTilePlacer
        from .tile_saver import TileSaver
        from PIL import Image
        import os
        from pathlib import Path

        # Recreate image info object from dict
        img_info = ImageInfo(
            path=img_info_dict["path"],
            filename=img_info_dict["filename"],
            source_type=img_info_dict["source_type"],
            relative_path=img_info_dict["relative_path"],
        )

        # Copy additional attributes
        if "size_bytes" in img_info_dict:
            img_info.size_bytes = img_info_dict["size_bytes"]
        if "source_url" in img_info_dict:
            img_info.source_url = img_info_dict["source_url"]
        if "has_dimensions" in img_info_dict:
            img_info.has_dimensions = img_info_dict["has_dimensions"]
        if "is_gridless" in img_info_dict:
            img_info.is_gridless = img_info_dict["is_gridless"]
        if "is_gridded" in img_info_dict:
            img_info.is_gridded = img_info_dict["is_gridded"]
        if "gridded_variant_path" in img_info_dict:
            img_info.gridded_variant_path = img_info_dict["gridded_variant_path"]
        if "gridded_variant_filename" in img_info_dict:
            img_info.gridded_variant_filename = img_info_dict[
                "gridded_variant_filename"
            ]
        if "has_both_variants" in img_info_dict:
            img_info.has_both_variants = img_info_dict["has_both_variants"]

        # Tile size is passed as parameter (calculated with dynamic balancing)
        # Debug output is handled in the main batch preparation phase

        # Load the pre-downloaded image
        pil_image = Image.open(image_path)

        # Debug: Always log image mode and transparency info
        if config_dict.get("debug", False):
            print(
                f"🔍 Image mode: {pil_image.mode}, transparency in info: {'transparency' in pil_image.info}"
            )

        # Skip images with significant transparency (>75% transparent pixels)
        if pil_image.mode in ("RGBA", "LA") or "transparency" in pil_image.info:
            has_significant_transparency = False
            transparency_percentage = 0.0

            if pil_image.mode == "RGBA":
                alpha_channel = pil_image.getchannel("A")
                import numpy as np

                alpha_array = np.array(alpha_channel)
                total_pixels = alpha_array.size
                transparent_pixels = np.sum(alpha_array < 255)
                transparency_percentage = (transparent_pixels / total_pixels) * 100
                has_significant_transparency = transparency_percentage > 75.0

            elif pil_image.mode == "LA":
                alpha_channel = pil_image.getchannel("A")
                import numpy as np

                alpha_array = np.array(alpha_channel)
                total_pixels = alpha_array.size
                transparent_pixels = np.sum(alpha_array < 255)
                transparency_percentage = (transparent_pixels / total_pixels) * 100
                has_significant_transparency = transparency_percentage > 75.0
            else:
                has_significant_transparency = True
                transparency_percentage = 100.0  # Unknown transparency format

            # Always log transparency percentage if image has any transparency
            print(f"🔍 Transparency: {transparency_percentage:.1f}% transparent pixels")

            if has_significant_transparency:
                return {
                    "success": False,
                    "skipped": True,  # Mark as skipped, not failed
                    "filename": img_info.filename,
                    "error": f"Image has {transparency_percentage:.1f}% transparent pixels (>75% threshold) - likely token or UI element",
                }

        # Convert to RGB for processing
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Create detector instances (matching main pipeline)
        grid_detector = GridDetector()
        boring_detector = AdvancedBoringDetector()
        # Use the calculated tile size based on image index
        tile_placer = OptimalTilePlacer(
            tile_size=tile_size,  # Use the calculated tile size!
            max_boring_percentage=config_dict.get("boring_threshold", 50.0)
            * 100,  # Convert to percentage
        )
        tile_saver = TileSaver()

        # Step 1: Enhanced grid detection using both brightness and morphological methods
        grid_info = enhanced_grid_detection(
            pil_image=pil_image,
            image_path=image_path,
            img_info=img_info,
            debug=config_dict.get("debug", False),
        )

        if not grid_info:
            return {
                "success": False,
                "filename": img_info.filename,
                "error": "Grid detection failed with all methods",
            }

        # Step 2: Boring detection
        square_analysis, boring_reasons = boring_detector.analyze_image_regions(
            pil_image, grid_info, debug=False
        )

        # Step 3: Optimal tile placement with automatic size fallback
        placed_tiles = []
        placement_error = None
        final_tile_size = tile_size

        # Try tile sizes in order: original -> 20x20 -> 12x12 (if original fails)
        tile_sizes_to_try = [tile_size]
        if tile_size == 30:
            tile_sizes_to_try.extend([20, 12])
        elif tile_size == 20:
            tile_sizes_to_try.append(12)

        for try_tile_size in tile_sizes_to_try:
            # Create tile placer for this size
            fallback_tile_placer = OptimalTilePlacer(
                tile_size=try_tile_size,
                max_boring_percentage=config_dict.get("boring_threshold", 50.0) * 100,
            )

            placed_tiles, placement_error = (
                fallback_tile_placer.find_optimal_placements(
                    grid_info=grid_info,
                    square_analysis=square_analysis,
                    debug=False,
                )
            )

            if placed_tiles and len(placed_tiles) > 0:
                # Success! Use this tile size
                final_tile_size = try_tile_size
                if try_tile_size != tile_size:
                    print(
                        f"    ↩️ Fallback: {tile_size}x{tile_size} failed, using {try_tile_size}x{try_tile_size} (got {len(placed_tiles)} tiles)"
                    )
                break
            elif placement_error and (
                "TOO_SMALL" in placement_error
                or "BORING_THRESHOLD_EXCEEDED" in placement_error
            ):
                # Size too small OR boring threshold exceeded, try next smaller size
                if "BORING_THRESHOLD_EXCEEDED" in placement_error:
                    error_details = (
                        placement_error.split(": ", 1)[1]
                        if ": " in placement_error
                        else placement_error
                    )
                    print(
                        f"    📐 {try_tile_size}x{try_tile_size} failed: {error_details}"
                    )
                else:
                    print(
                        f"    📐 {try_tile_size}x{try_tile_size} failed: {placement_error.split(': ', 1)[1] if ': ' in placement_error else placement_error}"
                    )
                continue
            else:
                # Other error (not fallback-compatible), don't try smaller sizes
                break

        # Apply limits if configured
        if (
            config_dict.get("max_tiles_per_image")
            and len(placed_tiles) > config_dict["max_tiles_per_image"]
        ):
            placed_tiles = placed_tiles[: config_dict["max_tiles_per_image"]]

        if not placed_tiles:
            return {
                "success": False,
                "error": placement_error
                or "No suitable tile positions found after trying all available tile sizes",
            }

        # Handle output based on test mode
        tile_count = len(placed_tiles)
        tiles_placed = tile_count

        if config_dict["test_mode"]:
            # Test mode: just count tiles, don't save
            tiles_saved = 0
        else:
            # Full mode: extract and save tiles
            try:
                # Create output subfolder for this image
                image_name = Path(img_info.filename).stem
                image_output_dir = Path(output_dir) / image_name
                image_output_dir.mkdir(parents=True, exist_ok=True)

                # Save tiles
                result = tile_saver.save_tiles(
                    source_image=pil_image,
                    grid_info=grid_info,
                    placed_tiles=placed_tiles,
                    source_path_or_url=img_info.source_url or img_info.path,
                    output_folder=str(image_output_dir),
                    debug=False,
                )
                tiles_saved = len(result.saved_tiles)

            except Exception as e:
                return {"success": False, "error": f"Failed to save tiles: {e}"}

        return {
            "success": True,
            "tiles_saved": tiles_saved,
            "tiles_placed": tiles_placed,
            "filename": img_info.filename,
            "tile_size": final_tile_size,  # Use the actual tile size used (may be different due to fallback)
            "grid_info": {"nx": grid_info["nx"], "ny": grid_info["ny"]},
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _process_single_image_multiprocessing(args) -> Optional[Dict[str, Any]]:
    """Process a single image for multiprocessing - includes full pipeline processing

    Args:
        args: Tuple of (img_info_data, config_data, image_path, output_dir, image_index)

    Returns:
        Dictionary with processing results or None if failed
    """
    img_info_data, config_data, image_path, output_dir, image_index = args

    # Import here to avoid issues with multiprocessing
    from .image_source_handler import ImageSourceHandler, ImageInfo
    from .smart_image_selector import SmartImageSelector
    from .grid_detector import GridDetector
    from .advanced_boring_detector import AdvancedBoringDetector
    from .optimal_tile_placer import OptimalTilePlacer
    from .tile_saver import TileSaver
    from PIL import Image
    import tempfile
    import shutil
    import os

    try:
        # Recreate ImageInfo object from data
        img_info = ImageInfo(
            path=img_info_data["path"],
            filename=img_info_data["filename"],
            source_type=img_info_data["source_type"],
            relative_path=img_info_data["relative_path"],
            size_bytes=img_info_data.get("size_bytes"),
            source_url=img_info_data.get("source_url"),
            has_dimensions=img_info_data.get("has_dimensions", False),
            is_gridless=img_info_data.get("is_gridless", False),
            is_gridded=img_info_data.get("is_gridded", False),
            gridded_variant_path=img_info_data.get("gridded_variant_path"),
            gridded_variant_filename=img_info_data.get("gridded_variant_filename"),
            has_both_variants=img_info_data.get("has_both_variants", False),
        )

        # Get tile size for this image based on rotation pattern
        tile_size = get_tile_size_for_image(image_index)

        # Create handlers for this process
        grid_detector = GridDetector()
        boring_detector = AdvancedBoringDetector()
        tile_placer = OptimalTilePlacer(
            tile_size=tile_size,
            max_boring_percentage=config_data.get("boring_threshold", 50.0)
            * 100,  # Convert to percentage
        )

        # Image is already downloaded at image_path
        if not image_path or not os.path.exists(image_path):
            return {
                "success": False,
                "filename": img_info.filename,
                "error": "Image file not found",
            }

        # Load the image
        try:
            pil_image = Image.open(image_path)

            # Skip images with significant transparency (>75% transparent pixels)
            if pil_image.mode in ("RGBA", "LA") or "transparency" in pil_image.info:
                has_significant_transparency = False
                transparency_percentage = 0.0

                if pil_image.mode == "RGBA":
                    alpha_channel = pil_image.getchannel("A")
                    import numpy as np

                    alpha_array = np.array(alpha_channel)
                    total_pixels = alpha_array.size
                    transparent_pixels = np.sum(alpha_array < 255)
                    transparency_percentage = (transparent_pixels / total_pixels) * 100
                    has_significant_transparency = transparency_percentage > 75.0

                elif pil_image.mode == "LA":
                    alpha_channel = pil_image.getchannel("A")
                    import numpy as np

                    alpha_array = np.array(alpha_channel)
                    total_pixels = alpha_array.size
                    transparent_pixels = np.sum(alpha_array < 255)
                    transparency_percentage = (transparent_pixels / total_pixels) * 100
                    has_significant_transparency = transparency_percentage > 75.0
                else:
                    has_significant_transparency = True
                    transparency_percentage = 100.0  # Unknown transparency format

                # Always log transparency percentage if image has any transparency
                print(
                    f"🔍 Transparency: {transparency_percentage:.1f}% transparent pixels"
                )

                if has_significant_transparency:
                    return {
                        "success": False,
                        "skipped": True,  # Mark as skipped, not failed
                        "filename": img_info.filename,
                        "error": f"Image has {transparency_percentage:.1f}% transparent pixels (>75% threshold) - likely token or UI element",
                    }

            # Convert to RGB for processing
            pil_image = pil_image.convert("RGB")

        except Exception as e:
            return {
                "success": False,
                "filename": img_info.filename,
                "error": f"Failed to load image: {e}",
            }

        # Enhanced grid detection using both brightness and morphological methods
        grid_info = enhanced_grid_detection(
            pil_image=pil_image,
            image_path=image_path,
            img_info=img_info,
            debug=config_data.get("debug", False),
        )

        if not grid_info:
            return {
                "success": False,
                "filename": img_info.filename,
                "error": "Grid detection failed with all methods",
            }

        # Analyze boring areas
        square_analysis, boring_reasons = boring_detector.analyze_image_regions(
            pil_image, grid_info, debug=False
        )

        # Place optimal tiles with automatic size fallback
        placed_tiles = []
        placement_error = None
        final_tile_size = tile_size

        # Try tile sizes in order: original -> 20x20 -> 12x12 (if original fails)
        tile_sizes_to_try = [tile_size]
        if tile_size == 30:
            tile_sizes_to_try.extend([20, 12])
        elif tile_size == 20:
            tile_sizes_to_try.append(12)

        for try_tile_size in tile_sizes_to_try:
            # Create tile placer for this size
            fallback_tile_placer = OptimalTilePlacer(
                tile_size=try_tile_size,
                max_boring_percentage=config_data.get("boring_threshold", 50.0) * 100,
            )

            placed_tiles, placement_error = (
                fallback_tile_placer.find_optimal_placements(
                    grid_info=grid_info,
                    square_analysis=square_analysis,
                    debug=False,
                )
            )

            if placed_tiles and len(placed_tiles) > 0:
                # Success! Use this tile size
                final_tile_size = try_tile_size
                if try_tile_size != tile_size:
                    print(
                        f"    ↩️ Fallback: {tile_size}x{tile_size} failed, using {try_tile_size}x{try_tile_size} (got {len(placed_tiles)} tiles)"
                    )
                break
            elif placement_error and (
                "TOO_SMALL" in placement_error
                or "BORING_THRESHOLD_EXCEEDED" in placement_error
            ):
                # Size too small OR boring threshold exceeded, try next smaller size
                if "BORING_THRESHOLD_EXCEEDED" in placement_error:
                    error_details = (
                        placement_error.split(": ", 1)[1]
                        if ": " in placement_error
                        else placement_error
                    )
                    print(
                        f"    📐 {try_tile_size}x{try_tile_size} failed: {error_details}"
                    )
                else:
                    print(
                        f"    📐 {try_tile_size}x{try_tile_size} failed: {placement_error.split(': ', 1)[1] if ': ' in placement_error else placement_error}"
                    )
                continue
            else:
                # Other error (not fallback-compatible), don't try smaller sizes
                break

        # Apply limits if configured
        if (
            config_data.get("max_tiles_per_image")
            and len(placed_tiles) > config_data["max_tiles_per_image"]
        ):
            placed_tiles = placed_tiles[: config_data["max_tiles_per_image"]]

        if not placed_tiles:
            return {
                "success": False,
                "filename": img_info.filename,
                "error": placement_error
                or "No suitable tile positions found after trying all available tile sizes",
            }

        # Handle output based on test mode
        tile_count = len(placed_tiles)
        saved_tiles = []

        if not config_data.get("test_mode", False):
            # Normal mode: save tiles
            try:
                # Create output subfolder for this image
                image_name = Path(img_info.filename).stem
                image_output_dir = Path(output_dir) / image_name
                image_output_dir.mkdir(parents=True, exist_ok=True)

                # Save tiles
                tile_saver = TileSaver()
                result = tile_saver.save_tiles(
                    source_image=pil_image,
                    grid_info=grid_info,
                    placed_tiles=placed_tiles,
                    source_path_or_url=img_info.source_url or img_info.path,
                    output_folder=str(image_output_dir),
                    debug=False,
                )
                saved_tiles = result.saved_tiles

            except Exception as e:
                return {
                    "success": False,
                    "filename": img_info.filename,
                    "error": f"Failed to save tiles: {e}",
                }

        # Clean up temporary file if it was downloaded
        # Cleanup is handled by the main batch processing loop

        return {
            "success": True,
            "filename": img_info.filename,
            "tiles_placed": tile_count,
            "tiles_saved": len(saved_tiles),
            "grid_info": grid_info,
            "tile_size": final_tile_size,  # Use the actual tile size used (may be different due to fallback)
        }

    except Exception as e:
        return {
            "success": False,
            "filename": img_info_data.get("filename", "unknown"),
            "error": f"Processing error: {e}",
        }


class BattlemapPipeline:
    """
    Complete pipeline for processing battlemap images into training tiles
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.progress = ProcessingProgress(config=config, stats=ProcessingStats())

        # Initialize components
        self.smart_selector = SmartImageSelector()
        self.image_handler = ImageSourceHandler(temp_dir=config.temp_dir)
        self.grid_detector = GridDetector()
        self.boring_detector = AdvancedBoringDetector()
        # Note: tile_placer will be created per-image with rotating tile sizes
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
        print("🏁 Pipeline.run() method started")
        if self.config.debug:
            print(f"\n🎯 Starting battlemap pipeline processing...")

        # Clear hash cache at startup if requested
        if self.config.clear_hash_cache:
            self.clear_hash_cache_on_startup()

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

            # First pass: collect all images from all sources with FOLDER-AWARE smart selection
            if self.config.debug:
                print(f"\n📂 Collecting images from all sources...")

            all_images = []
            for source_idx, source in enumerate(self.config.sources):
                if self.config.debug:
                    print(
                        f"\n📁 Finding images in source {source_idx + 1}/{len(self.config.sources)}: {source}"
                    )

                try:
                    if self.config.use_smart_selection:
                        # First, get ALL images without smart selection
                        all_images_in_source = (
                            self.image_handler.find_images_from_source(
                                source=source,
                                debug=False,  # Disable debug for initial collection
                                list_only=True,
                                use_smart_selection=False,  # Get everything first
                            )
                        )

                        if not all_images_in_source:
                            if self.config.debug:
                                print(f"  ⚠️  No images found")
                            continue

                        if self.config.debug:
                            print(f"  � Found {len(all_images_in_source)} total images")

                        # Group images by their parent folder path
                        from collections import defaultdict

                        folder_groups = defaultdict(list)

                        for img in all_images_in_source:
                            # Extract folder path from relative_path
                            folder_path = str(Path(img.relative_path).parent)
                            folder_groups[folder_path].append(img)

                        if self.config.debug:
                            print(f"  📁 Organized into {len(folder_groups)} folders")

                        # Apply smart selection within each folder
                        selected_images = []
                        for folder_path, folder_images in folder_groups.items():
                            # Convert ImageInfo objects to the format expected by SmartImageSelector
                            folder_images_dict = []
                            for img in folder_images:
                                folder_images_dict.append(
                                    {
                                        "path": getattr(img, "path", ""),
                                        "filename": img.filename,
                                    }
                                )

                            # Apply smart selection to this folder's images only
                            folder_selected = self.smart_selector.select_optimal_images(
                                folder_images_dict
                            )

                            # Convert back to ImageInfo objects by matching with original images
                            for selected_dict in folder_selected:
                                # Find the original ImageInfo object
                                original_img = next(
                                    img
                                    for img in folder_images
                                    if img.filename == selected_dict["filename"]
                                )

                                # Update the ImageInfo object with smart selection metadata
                                original_img.has_dimensions = selected_dict.get(
                                    "has_dimensions", False
                                )
                                original_img.has_both_variants = selected_dict.get(
                                    "has_both_variants", False
                                )
                                original_img.gridded_variant_path = selected_dict.get(
                                    "gridded_variant_path"
                                )
                                original_img.gridded_variant_filename = (
                                    selected_dict.get("gridded_variant_filename")
                                )

                                selected_images.append(original_img)

                            # Compact folder summary
                            if self.config.debug and len(folder_selected) > 0:
                                first_result = folder_selected[0]
                                strategy = first_result.get(
                                    "selection_reason", "unknown"
                                )
                                strategy_short = {
                                    "gridless_for_output_detect_on_gridded": "gridless",
                                    "only_variant": "only",
                                    "best_variant": "best",
                                    "prefer_gridless": "gridless",
                                }.get(strategy, strategy[:8])

                                variant_info = ""
                                if first_result.get("has_both_variants", False):
                                    variant_info = f"+gridded"

                                print(
                                    f"    📂 {folder_path} ({len(folder_images)}→{len(folder_selected)}, {strategy_short}{variant_info})"
                                )

                        all_images.extend(selected_images)
                        if self.config.debug:
                            print(
                                f"  ✅ Selected {len(selected_images)} optimal images from {len(folder_groups)} folders"
                            )

                    else:
                        # No smart selection - get all images
                        images = self.image_handler.find_images_from_source(
                            source=source,
                            debug=self.config.debug,
                            list_only=True,
                            use_smart_selection=False,
                        )

                        if images:
                            all_images.extend(images)
                            if self.config.debug:
                                print(f"  📊 Found {len(images)} images")
                        else:
                            if self.config.debug:
                                print(f"  ⚠️  No images found")

                except Exception as e:
                    logger.error(f"Error finding images in source '{source}': {e}")
                    if self.config.debug:
                        print(f"  ❌ Error: {e}")
                    # Fallback to original approach
                    try:
                        images = self.image_handler.find_images_from_source(
                            source=source,
                            debug=True,
                            list_only=True,
                            use_smart_selection=self.config.use_smart_selection,
                        )
                        if images:
                            all_images.extend(images)
                            if self.config.debug:
                                print(f"  📊 Fallback: Found {len(images)} images")
                    except Exception as fallback_error:
                        if self.config.debug:
                            print(f"  ❌ Fallback also failed: {fallback_error}")

            if not all_images:
                if self.config.debug:
                    print("⚠️  No images found in any source")
                return self.progress.stats

            self.progress.stats.images_found = len(all_images)

            if self.config.debug:
                print(f"\n📊 Total images found: {len(all_images)}")

            # Skip separate duplicate detection phase - will be integrated into processing
            # This avoids downloading all images twice (once for hashing, once for processing)
            # Note: all_images already contains all selected images from all sources

            if self.config.debug:
                print(
                    f"✅ Will process {len(all_images)} images with integrated duplicate detection"
                )
                print(
                    f"   Duplicate detection will happen during processing to avoid double-downloading"
                )

            # Apply image limit if configured
            if self.config.max_images and len(all_images) > self.config.max_images:
                if self.config.debug:
                    print(
                        f"🔢 Limiting to {self.config.max_images} images (out of {len(all_images)})"
                    )
                all_images = all_images[: self.config.max_images]

            # Process images
            if self.config.debug:
                print(f"\n🚀 Processing {len(all_images)} images...")
                print(f"DEBUG: use_multiprocessing = {self.config.use_multiprocessing}")
                print(f"DEBUG: len(all_images) = {len(all_images)}")
                print(
                    f"DEBUG: multiprocessing condition = {self.config.use_multiprocessing and len(all_images) > 1}"
                )

            if self.config.use_multiprocessing and len(all_images) > 1:
                # Use batched multiprocessing for better memory management
                if self.config.debug:
                    print(f"� Using BATCHED MULTIPROCESSING for image processing...")

                self._process_images_in_batches(all_images)

                if self.config.debug:
                    print(f"*** BATCHED MULTIPROCESSING COMPLETE ***")

            else:
                # Use single-threaded processing (fallback or when disabled)
                if self.config.debug:
                    if self.config.use_multiprocessing:
                        print(
                            f"🔄 Using SINGLE-THREADED processing (only 1 image or multiprocessing disabled)..."
                        )
                    else:
                        print(
                            f"🔄 Using SINGLE-THREADED processing (multiprocessing disabled)..."
                        )

                # Prepare normalized processed paths for better comparison
                processed_list = self.progress.processed_images or []
                processed_set = set()
                for path in processed_list:
                    # Add both original path and normalized versions
                    processed_set.add(path)
                    processed_set.add(str(Path(path).as_posix()))  # Unix-style path
                    processed_set.add(str(Path(path)))  # OS-native path

                if self.config.debug:
                    print(
                        f"🔄 Single-threaded processing: {len(all_images)} images, {len(processed_list)} previously processed"
                    )

                for image_info in all_images:
                    # Check if already processed with multiple path format checks
                    img_path_variants = [
                        image_info.path,
                        str(Path(image_info.path).as_posix()),  # Unix-style
                        str(Path(image_info.path)),  # OS-native
                    ]

                    is_processed = any(
                        variant in processed_set for variant in img_path_variants
                    )

                    if is_processed:
                        if self.config.debug:
                            print(
                                f"⏭️  Skipping already processed: {image_info.filename}"
                            )
                        continue

                    # Check if we've hit the image limit
                    if (
                        self.config.max_images
                        and self.progress.stats.images_processed
                        >= self.config.max_images
                    ):
                        if self.config.debug:
                            print(
                                f"🛑 Reached maximum image limit ({self.config.max_images})"
                            )
                        break

                    # Process single image
                    success = self._process_single_image(image_info)

                    # Only mark as processed if successful
                    if success:
                        if self.progress.processed_images is None:
                            self.progress.processed_images = []
                        self.progress.processed_images.append(image_info.path)

                    # Save progress periodically
                    if (
                        self.config.save_progress
                        and self.progress.stats.images_processed % 10 == 0
                    ):
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

            # Clean up hash cache to avoid false duplicates in future runs
            print("🧹 Cleaning up hash cache...")
            self._cleanup_hash_cache()

    def _process_source(self, source: str):
        """
        Legacy method - kept for compatibility but no longer used in the main pipeline.
        The new pipeline approach processes all images after deduplication.
        """
        # This method is kept for compatibility but is no longer used
        # The new pipeline collects all images first, deduplicates them, then processes
        pass

    def _process_images_in_batches(self, all_images: List) -> None:
        """Process images in batches with controlled downloading to avoid overwhelming system resources"""

        # Initialize integrated duplicate detection
        import hashlib
        import json
        import os
        from pathlib import Path

        # Load existing duplicate detection cache
        duplicate_cache_file = self.output_dir / "duplicate_detection_progress.json"
        known_hashes = {}  # hash -> original_image_path
        duplicate_count = 0

        try:
            if duplicate_cache_file.exists():
                with open(duplicate_cache_file, "r") as f:
                    data = json.load(f)
                    cache_data = data.get("hash_cache", {})
                    for path, hash_val in cache_data.items():
                        if hash_val in known_hashes:
                            duplicate_count += 1
                        else:
                            known_hashes[hash_val] = path
                if self.config.debug and cache_data:
                    print(
                        f"📁 Loaded {len(cache_data)} cached hashes ({len(known_hashes)} unique, {duplicate_count} duplicates)"
                    )
        except Exception:
            pass

        # Helper function to calculate file hash
        def calculate_file_hash(file_path: str) -> str:
            """Calculate MD5 hash of a file"""
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        # Determine optimal batch sizes
        num_workers = min(len(all_images), max(1, multiprocessing.cpu_count() // 2))
        # Download batch size should be smaller than processing batch to control memory
        download_batch_size = max(5, num_workers)  # Download 5-8 images at a time
        process_batch_size = max(num_workers * 2, 10)  # Process more in parallel

        if self.config.debug:
            print(f"  🖥️ Using {num_workers} CPU cores")
            print(f"  📥 Download batch size: {download_batch_size}")
            print(f"  ⚙️ Processing batch size: {process_batch_size}")

        # Prepare configuration data for multiprocessing (pickle-safe)
        config_data = {
            "boring_threshold": self.config.boring_threshold,
            "max_tiles_per_image": self.config.max_tiles_per_image,
            "test_mode": self.config.test_mode,
        }

        # Filter out already processed images with improved path comparison
        processed_list = self.progress.processed_images or []
        images_to_process = []

        # Normalize processed paths for better comparison
        processed_set = set()
        for path in processed_list:
            # Add both original path and normalized versions
            processed_set.add(path)
            processed_set.add(str(Path(path).as_posix()))  # Unix-style path
            processed_set.add(str(Path(path)))  # OS-native path

        if self.config.debug:
            print(
                f"  📊 Resume filtering: {len(all_images)} total images, {len(processed_list)} previously processed"
            )

        skipped_count = 0
        for img_info in all_images:
            # Skip already processed images with multiple path format checks
            img_path_variants = [
                img_info.path,
                str(Path(img_info.path).as_posix()),  # Unix-style
                str(Path(img_info.path)),  # OS-native
            ]

            is_processed = any(
                variant in processed_set for variant in img_path_variants
            )

            if is_processed:
                if (
                    self.config.debug and skipped_count < 5
                ):  # Show first few skipped images
                    print(f"⏭️  Skipping already processed: {img_info.filename}")
                elif self.config.debug and skipped_count == 5:
                    print("⏭️  (and more already processed images...)")
                skipped_count += 1
                continue

            # Check image limit
            if (
                self.config.max_images
                and self.progress.stats.images_processed >= self.config.max_images
            ):
                if self.config.debug:
                    print(f"🛑 Reached maximum image limit ({self.config.max_images})")
                break

            images_to_process.append(img_info)

        if self.config.debug:
            print(
                f"  🎯 Found {len(images_to_process)} new images to process (skipped {skipped_count} already processed)"
            )

        if not images_to_process:
            if self.config.debug:
                print(
                    "  ℹ️ No new images to process - all remaining images have been completed"
                )
                # Additional debugging to help identify the issue
                if len(all_images) > 0 and len(processed_list) > 0:
                    print(f"  🔍 Debug info:")
                    print(f"     Total images found: {len(all_images)}")
                    print(f"     Previously processed: {len(processed_list)}")
                    print(
                        f"     Expected remaining: {len(all_images) - len(processed_list)}"
                    )
                    print(f"     Actual remaining: {len(images_to_process)}")
                    if len(all_images) > 0:
                        print(f"     Sample found image: {all_images[0].path}")
                    if len(processed_list) > 0:
                        print(f"     Sample processed image: {processed_list[0]}")
            return

        # Process images in download + process batches
        total_batches = (
            len(images_to_process) + download_batch_size - 1
        ) // download_batch_size

        # Track overall progress for tile size rotation
        overall_image_index = 0

        for batch_num in range(total_batches):
            start_idx = batch_num * download_batch_size
            end_idx = min((batch_num + 1) * download_batch_size, len(images_to_process))
            download_batch = images_to_process[start_idx:end_idx]

            if self.config.debug:
                print(
                    f"  � Download batch {batch_num + 1}/{total_batches} ({len(download_batch)} images)"
                )

            # Step 1: Download this batch sequentially with integrated duplicate detection
            downloaded_images = []
            for img_info in download_batch:
                try:
                    # Download the image
                    image_path = self.image_handler.download_single_image(
                        img_info, debug=False
                    )
                    if image_path:
                        # Calculate hash of downloaded image for duplicate detection
                        try:
                            image_hash = calculate_file_hash(image_path)

                            # Check if this hash already exists (duplicate)
                            if image_hash in known_hashes:
                                original_path = known_hashes[image_hash]
                                if self.config.debug:
                                    print(
                                        f"    🔄 Duplicate: {img_info.filename} (same as {Path(original_path).name})"
                                    )

                                # Clean up downloaded duplicate
                                try:
                                    import os

                                    os.remove(image_path)
                                except:
                                    pass

                                duplicate_count += 1
                                continue  # Skip processing this duplicate
                            else:
                                # New unique image - remember its hash
                                known_hashes[image_hash] = img_info.path
                                downloaded_images.append((img_info, image_path))
                                # Compact download confirmation

                        except Exception as hash_error:
                            # If hashing fails, proceed anyway (better to process than lose images)
                            if self.config.debug:
                                print(
                                    f"    ⚠️ Hash failed for {img_info.filename}, processing anyway: {hash_error}"
                                )
                            downloaded_images.append((img_info, image_path))
                    else:
                        if self.config.debug:
                            print(f"    ❌ Failed to download: {img_info.filename}")
                        self.progress.stats.images_failed += 1
                except Exception as e:
                    if self.config.debug:
                        print(f"    ❌ Download error for {img_info.filename}: {e}")
                    self.progress.stats.images_failed += 1

            if not downloaded_images:
                if self.config.debug:
                    print(f"    ⚠️ No images downloaded in this batch")
                continue

            # Step 2: Process downloaded images with multiprocessing (full CPU utilization)
            if self.config.debug:
                print(
                    f"\n  ⚙️ Processing {len(downloaded_images)} downloaded images with multiprocessing..."
                )

            # Prepare args for the processing-only function with enhanced tile balancing
            # Start with actual stats from previous batches
            working_stats = {
                "tiles_12x12": self.progress.stats.tiles_12x12,
                "tiles_20x20": self.progress.stats.tiles_20x20,
                "tiles_30x30": self.progress.stats.tiles_30x30,
            }

            # Calculate tile sizes sequentially using working stats + imaginary tiles
            tile_sizes = []
            processing_args = []

            if self.config.debug:
                actual_total = sum(working_stats.values())
                print(
                    f"  📊 Batch {batch_num + 1}/{total_batches}: {len(downloaded_images)} images, starting stats: {working_stats['tiles_12x12']}/{working_stats['tiles_20x20']}/{working_stats['tiles_30x30']} (total: {actual_total})"
                )

                # Show tile size assignments in one compact line
                tile_assignments = []
                for idx, tile_size in enumerate(tile_sizes):
                    tile_assignments.append(f"{tile_size}")
                print(f"  🎯 Tile sizes: {' '.join(tile_assignments)}")

            for idx, (img_info, image_path) in enumerate(downloaded_images):
                # Calculate overall image index for fallback
                current_image_index = overall_image_index + idx

                # Get optimal tile size based on working stats (includes imaginary tiles)
                optimal_tile_size = get_tile_size_for_image(
                    current_image_index, working_stats
                )
                tile_sizes.append(optimal_tile_size)

                # Add imaginary tiles to working stats for next image in batch
                # Estimates: 12x12→4 tiles, 20x20→2 tiles, 30x30→1 tile
                if optimal_tile_size == 12:
                    working_stats["tiles_12x12"] += 4
                elif optimal_tile_size == 20:
                    working_stats["tiles_20x20"] += 2
                elif optimal_tile_size == 30:
                    working_stats["tiles_30x30"] += 1

                img_info_data = {
                    "path": getattr(img_info, "path", ""),
                    "filename": img_info.filename,
                    "source_type": img_info.source_type,
                    "relative_path": img_info.relative_path,
                    "size_bytes": getattr(img_info, "size_bytes", None),
                    "source_url": getattr(img_info, "source_url", None),
                    "has_dimensions": getattr(img_info, "has_dimensions", False),
                    "is_gridless": getattr(img_info, "is_gridless", False),
                    "is_gridded": getattr(img_info, "is_gridded", False),
                    "gridded_variant_path": getattr(
                        img_info, "gridded_variant_path", None
                    ),
                    "gridded_variant_filename": getattr(
                        img_info, "gridded_variant_filename", None
                    ),
                    "has_both_variants": getattr(img_info, "has_both_variants", False),
                }
                processing_args.append(
                    (
                        img_info_data,
                        config_data,
                        image_path,  # Pre-downloaded path
                        str(self.output_dir),
                        current_image_index,  # Add image index for tile size rotation
                        optimal_tile_size,  # Pass the pre-calculated optimal tile size
                    )
                )

            # Process this batch with multiprocessing
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                # Submit all tasks in this batch
                future_to_img_info = {}
                for args in processing_args:
                    future = executor.submit(
                        _process_downloaded_image_multiprocessing, args
                    )
                    img_filename = args[0]["filename"]
                    matching_img_info = next(
                        (
                            img
                            for img, _ in downloaded_images
                            if img.filename == img_filename
                        ),
                        None,
                    )
                    future_to_img_info[future] = (img_filename, matching_img_info)

                # Collect results from this batch
                for future in concurrent.futures.as_completed(future_to_img_info):
                    img_name, img_info = future_to_img_info[future]
                    try:
                        result = future.result()
                        if result and result.get("success", False):
                            # DEBUG: Print what tile_size we got from the result
                            result_tile_size = result.get("tile_size", "NOT_FOUND")
                            print(
                                f"📊 Result tile_size: {result_tile_size} for {result.get('filename', 'unknown')}"
                            )

                            # Update statistics
                            self.progress.stats.images_processed += 1

                            # Use appropriate tile count based on mode
                            tiles_count = (
                                result["tiles_placed"]
                                if self.config.test_mode
                                else result["tiles_saved"]
                            )
                            tile_size = result.get(
                                "tile_size", 12
                            )  # Default to 12 if not specified

                            # Update total count
                            self.progress.stats.total_tiles_generated += tiles_count

                            # Update tile size specific counts
                            if tile_size == 12:
                                self.progress.stats.tiles_12x12 += tiles_count
                            elif tile_size == 20:
                                self.progress.stats.tiles_20x20 += tiles_count
                            elif tile_size == 30:
                                self.progress.stats.tiles_30x30 += tiles_count

                            # Mark as processed
                            if self.progress.processed_images is None:
                                self.progress.processed_images = []
                            if img_info:
                                self.progress.processed_images.append(img_info.path)

                            if self.config.debug:
                                grid_info = result.get("grid_info", {})
                                nx, ny = grid_info.get("nx", 0), grid_info.get("ny", 0)
                                tiles_info = (
                                    f"{result['tiles_saved']} tiles"
                                    if not self.config.test_mode
                                    else f"{result['tiles_placed']} tiles (test mode)"
                                )
                                print(
                                    f"    ✅ {img_name}: Grid {nx}x{ny}, {tiles_info}"
                                )
                        elif result and result.get("skipped", False):
                            # Image was skipped (e.g., due to transparency)
                            self.progress.stats.images_skipped += 1
                            error_msg = result.get(
                                "error", "Skipped for unknown reason"
                            )
                            if self.config.debug:
                                print(f"    ⏭️ {img_name}: {error_msg}")
                        else:
                            self.progress.stats.images_failed += 1
                            error_msg = (
                                result.get("error", "Unknown error")
                                if result
                                else "No result"
                            )
                            if self.config.debug:
                                print(f"    ❌ {img_name}: {error_msg}")
                    except Exception as e:
                        self.progress.stats.images_failed += 1
                        if self.config.debug:
                            print(f"    ❌ {img_name}: Processing exception: {e}")

                    # Check image limit after each processed image
                    if (
                        self.config.max_images
                        and self.progress.stats.images_processed
                        >= self.config.max_images
                    ):
                        if self.config.debug:
                            print(
                                f"  🛑 Reached maximum image limit ({self.config.max_images})"
                            )
                        return

            # Step 3: Clean up downloaded files from this batch
            for img_info, image_path in downloaded_images:
                if image_path and image_path != img_info.path:
                    try:
                        os.remove(image_path)
                    except (FileNotFoundError, PermissionError):
                        pass

            if self.config.debug:
                print(f"  🧹 Cleaned up {len(downloaded_images)} temporary files")
                print(f"  ✅ BATCH {batch_num + 1}/{total_batches} COMPLETE")

            # Update overall image index for next batch
            overall_image_index += len(downloaded_images)

            # Save progress after each batch
            if self.config.save_progress:
                self._save_progress()

            # Brief pause between batches to allow system cleanup
            if batch_num < total_batches - 1:  # Don't pause after the last batch
                import time

                time.sleep(0.5)

        # Save duplicate detection progress at the end
        try:
            cache_data = {path: hash_val for hash_val, path in known_hashes.items()}
            duplicate_cache_data = {"hash_cache": cache_data, "version": "1.0"}
            with open(duplicate_cache_file, "w") as f:
                json.dump(duplicate_cache_data, f, indent=2)

            if self.config.debug:
                total_unique = len(known_hashes)
                print(
                    f"💾 Saved duplicate detection cache: {total_unique} unique images, {duplicate_count} duplicates skipped"
                )
        except Exception as e:
            if self.config.debug:
                print(f"⚠️ Failed to save duplicate cache: {e}")

    def _process_single_image(self, img_info: ImageInfo) -> bool:
        """Process a single image to generate tiles

        Returns:
            True if processing was successful, False otherwise
        """
        # Get tile size based on processing order with dynamic balancing
        current_index = (
            self.progress.stats.images_processed + self.progress.stats.images_skipped
        )
        current_stats = {
            "tiles_12x12": self.progress.stats.tiles_12x12,
            "tiles_20x20": self.progress.stats.tiles_20x20,
            "tiles_30x30": self.progress.stats.tiles_30x30,
        }
        tile_size = get_tile_size_for_image(current_index, current_stats)

        if self.config.debug:
            print(f"\n🖼️  Processing: {img_info.filename}")
            print(
                f"📐 Using {tile_size}x{tile_size} tiles (image #{current_index + 1})"
            )

            # Show current tile ratios
            total_tiles = sum(current_stats.values())
            if total_tiles > 0:
                ratio_12 = current_stats["tiles_12x12"] / total_tiles * 100
                ratio_20 = current_stats["tiles_20x20"] / total_tiles * 100
                ratio_30 = current_stats["tiles_30x30"] / total_tiles * 100
                print(
                    f"    📊 Current ratios: 12x12={ratio_12:.1f}%, 20x20={ratio_20:.1f}%, 30x30={ratio_30:.1f}%"
                )

        try:
            # Download/access the image
            image_path = self.image_handler.download_single_image(
                img_info, debug=self.config.debug
            )
            if not image_path:
                if self.config.debug:
                    print(f"❌ Failed to access image")
                self.progress.stats.images_failed += 1
                return False

            # Load the image
            try:
                pil_image = Image.open(image_path)

                # Debug: Always log image mode and transparency info
                if self.config.debug:
                    print(
                        f"🔍 Image mode: {pil_image.mode}, transparency in info: {'transparency' in pil_image.info}"
                    )

                # Skip images with significant transparency (>75% transparent pixels)
                if pil_image.mode in ("RGBA", "LA") or "transparency" in pil_image.info:
                    # Check transparency percentage
                    has_significant_transparency = False

                    if pil_image.mode == "RGBA":
                        # Use PIL's getchannel instead of full numpy array
                        alpha_channel = pil_image.getchannel("A")
                        # Convert to array to count transparent pixels
                        import numpy as np

                        alpha_array = np.array(alpha_channel)
                        total_pixels = alpha_array.size
                        transparent_pixels = np.sum(alpha_array < 255)
                        transparency_percentage = (
                            transparent_pixels / total_pixels
                        ) * 100
                        has_significant_transparency = transparency_percentage > 75.0

                        if self.config.debug:
                            print(
                                f"🔍 Transparency analysis: {transparency_percentage:.1f}% transparent pixels"
                            )

                    elif pil_image.mode == "LA":
                        # Use PIL's getchannel for L+Alpha mode
                        alpha_channel = pil_image.getchannel("A")
                        import numpy as np

                        alpha_array = np.array(alpha_channel)
                        total_pixels = alpha_array.size
                        transparent_pixels = np.sum(alpha_array < 255)
                        transparency_percentage = (
                            transparent_pixels / total_pixels
                        ) * 100
                        has_significant_transparency = transparency_percentage > 75.0

                        if self.config.debug:
                            print(
                                f"🔍 Transparency: {transparency_percentage:.1f}% transparent pixels"
                            )
                    else:
                        # If transparency info exists but no alpha channel, assume significant
                        has_significant_transparency = True
                        transparency_percentage = 100.0

                    if has_significant_transparency:
                        if self.config.debug:
                            print(
                                f"⏭️ Image has {transparency_percentage:.1f}% transparent pixels (>75% threshold) - skipping analysis"
                            )
                        self.progress.stats.images_skipped += 1
                        return False

                # Convert to RGB for processing
                pil_image = pil_image.convert("RGB")
            except Exception as e:
                if self.config.debug:
                    print(f"❌ Failed to load image: {e}")
                self.progress.stats.images_failed += 1
                return False

            if self.config.debug:
                # Get image dimensions efficiently without numpy conversion
                width, height = pil_image.size
                print(f"📐 Image size: {width}x{height} pixels")

            # Detect grid dimensions using new smart logic
            if self.config.debug:
                print(f"🧠 Smart grid detection:")
                print(f"  Has dimensions in filename: {img_info.has_dimensions}")
                print(f"  Is gridless: {img_info.is_gridless}")
                print(f"  Is gridded: {img_info.is_gridded}")
                print(f"  Has both variants: {img_info.has_both_variants}")
                if img_info.gridded_variant_filename:
                    print(f"  Gridded variant: {img_info.gridded_variant_filename}")

            # Use enhanced grid detection with brightness + morphological comparison
            grid_info = enhanced_grid_detection(
                pil_image=pil_image,
                image_path=image_path,
                img_info=img_info,
                debug=self.config.debug,
            )

            # Grid detection should always succeed now
            if not grid_info:
                if self.config.debug:
                    print(f"⚠️  Grid detection failed for all methods - skipping image")
                self.progress.stats.images_skipped += 1
                return False

            if self.config.debug:
                print(f"🔲 Grid detected: {grid_info['nx']}x{grid_info['ny']} squares")

            # Analyze boring areas
            square_analysis, boring_reasons = (
                self.boring_detector.analyze_image_regions(
                    pil_image, grid_info, debug=self.config.debug
                )
            )

            # Place optimal tiles with appropriate tile size and automatic fallback
            placed_tiles = []
            placement_error = None
            final_tile_size = tile_size

            # Try tile sizes in order: original -> 20x20 -> 12x12 (if original fails)
            tile_sizes_to_try = [tile_size]
            if tile_size == 30:
                tile_sizes_to_try.extend([20, 12])
            elif tile_size == 20:
                tile_sizes_to_try.append(12)

            for try_tile_size in tile_sizes_to_try:
                # Create tile placer for this size
                tile_placer = OptimalTilePlacer(
                    tile_size=try_tile_size,
                    max_boring_percentage=self.config.boring_threshold
                    * 100.0,  # Convert to percentage
                    batch_size=3,  # Optimized balanced mode
                )
                placed_tiles, placement_error = tile_placer.find_optimal_placements(
                    grid_info=grid_info,
                    square_analysis=square_analysis,
                    debug=self.config.debug,
                )

                if placed_tiles and len(placed_tiles) > 0:
                    # Success! Use this tile size
                    final_tile_size = try_tile_size
                    if try_tile_size != tile_size and self.config.debug:
                        print(
                            f"    ↩️ Fallback: {tile_size}x{tile_size} failed, using {try_tile_size}x{try_tile_size} (got {len(placed_tiles)} tiles)"
                        )
                    break
                elif placement_error and (
                    "TOO_SMALL" in placement_error
                    or "BORING_THRESHOLD_EXCEEDED" in placement_error
                ):
                    # Size too small OR boring threshold exceeded, try next smaller size
                    if self.config.debug:
                        if "BORING_THRESHOLD_EXCEEDED" in placement_error:
                            error_details = (
                                placement_error.split(": ", 1)[1]
                                if ": " in placement_error
                                else placement_error
                            )
                            print(
                                f"    📐 {try_tile_size}x{try_tile_size} failed: {error_details}"
                            )
                        else:
                            print(
                                f"    📐 {try_tile_size}x{try_tile_size} failed: {placement_error.split(': ', 1)[1] if ': ' in placement_error else placement_error}"
                            )
                    continue
                else:
                    # Other error (not fallback-compatible), don't try smaller sizes
                    break

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
                    print(
                        f"⚠️  {placement_error or 'No suitable tile positions found'} - skipping image"
                    )
                self.progress.stats.images_skipped += 1
                return False

            # Create output subfolder for this image
            image_name = Path(img_info.filename).stem
            image_output_dir = self.output_dir / image_name
            image_output_dir.mkdir(parents=True, exist_ok=True)

            if self.config.test_mode:
                # Test mode: count tiles but don't save them
                if self.config.debug:
                    print(
                        f"🧪 TEST MODE: Would save {len(placed_tiles)} tiles (not actually saving)"
                    )

                # Create simplified mock result for statistics
                class MockResult:
                    def __init__(self, tile_count):
                        self.saved_tiles = [
                            None
                        ] * tile_count  # Just for len() operations
                        self.total_saved = tile_count

                result = MockResult(len(placed_tiles))

                if self.config.debug:
                    print(f"🧪 TEST: Would generate {len(result.saved_tiles)} tiles")

            else:
                # Normal mode: actually save the tiles
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

                if self.config.debug:
                    print(
                        f"✅ Successfully processed: {len(result.saved_tiles)} tiles saved"
                    )

            # Update statistics (for both test and normal mode)
            self.progress.stats.images_processed += 1
            tiles_count = (
                len(placed_tiles) if self.config.test_mode else len(result.saved_tiles)
            )
            self.progress.stats.total_tiles_generated += tiles_count

            # Update tile size specific counts using the final tile size used
            if final_tile_size == 12:
                self.progress.stats.tiles_12x12 += tiles_count
            elif final_tile_size == 20:
                self.progress.stats.tiles_20x20 += tiles_count
            elif final_tile_size == 30:
                self.progress.stats.tiles_30x30 += tiles_count

            # Clean up temporary file if it was downloaded
            if (
                image_path != img_info.path
                and self.image_handler.temp_dir  # Check temp_dir is not None
                and image_path.startswith(self.image_handler.temp_dir)
            ):
                with contextlib.suppress(FileNotFoundError, PermissionError):
                    os.remove(image_path)

            return True  # Success

        except Exception as e:
            logger.error(f"Error processing image {img_info.filename}: {e}")
            if self.config.debug:
                print(f"❌ Image processing error: {e}")
            self.progress.stats.images_failed += 1
            return False  # Failure

    def _convert_score_to_confidence(self, detection_score):
        """
        Convert the original morphological detection score to a confidence percentage.
        Uses a logarithmic-inspired approach since the original method:
        1. Already filtered candidates and picked the best one
        2. Morphological scores have diminishing returns (1->2 more significant than 10->11)
        3. Any successful detection should have reasonable baseline confidence
        """
        if detection_score is None:
            return 0.0

        if detection_score <= 0:
            return 0.0

        # Base confidence for any successful detection
        base_confidence = 55.0

        # Logarithmic boost - scores have diminishing returns
        # log(1 + score) gives: score 1→69%, score 2→75%, score 5→85%, score 10→92%
        import math

        log_boost = math.log(1 + detection_score) * 15.0

        # Additional linear boost for very high scores
        if detection_score > 5:
            linear_boost = (detection_score - 5) * 1.0
        else:
            linear_boost = 0

        confidence = base_confidence + log_boost + linear_boost

        # Cap at 95% (never 100% confident)
        return min(95.0, max(0.0, confidence)) / 100.0  # Return as decimal (0-1)

    def _brightness_to_grid_info(
        self, image_path: str, no_display: bool = False
    ) -> Optional[Dict]:
        """
        Run brightness-based grid detection using analyze_brightness.py subprocess

        Args:
            image_path: Path to the image
            no_display: If True, skip matplotlib display

        Returns:
            Dict in the same format as GridDetector.detect_grid() with confidence score
        """
        try:
            import subprocess
            import re
            import sys

            # Prepare command with optional --no-display flag
            cmd = [sys.executable, "analyze_brightness.py"]
            if no_display:
                cmd.append("--no-display")
            cmd.append(image_path)

            # Run the analyze_brightness script and capture output
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                encoding="utf-8",
                errors="replace",
            )

            if result.returncode != 0:
                return None

            output = result.stdout

            # Parse the output to extract grid information
            grid_size = None
            confidence = 0.0
            cols = 0
            rows = 0

            # Look for grid size
            grid_size_match = re.search(
                r"Square grid size: (\d+(?:\.\d+)?) pixels", output
            )
            if grid_size_match:
                grid_size = float(grid_size_match.group(1))

            # Look for confidence
            confidence_match = re.search(r"Confidence: (\d+(?:\.\d+)?)%", output)
            if confidence_match:
                confidence = float(confidence_match.group(1)) / 100.0

            # Look for grid dimensions
            cols_match = re.search(r"Grid columns: (\d+)", output)
            if cols_match:
                cols = int(cols_match.group(1))

            rows_match = re.search(r"Grid rows: (\d+)", output)
            if rows_match:
                rows = int(rows_match.group(1))

            # If grid size was found, we have a successful detection
            if grid_size and grid_size > 0:
                # Load image to get dimensions
                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                    # Calculate cell dimensions - use defaults if cols/rows not found
                    if cols > 0 and rows > 0:
                        cell_width = img_width / cols
                        cell_height = img_height / rows
                    else:
                        # Estimate grid dimensions from image size and grid_size
                        cols = int(img_width / grid_size)
                        rows = int(img_height / grid_size)
                        cell_width = img_width / cols if cols > 0 else grid_size
                        cell_height = img_height / rows if rows > 0 else grid_size

                    # Generate grid edges
                    x_edges = [i * cell_width for i in range(cols + 1)]
                    y_edges = [i * cell_height for i in range(rows + 1)]

                    return {
                        "nx": cols,
                        "ny": rows,
                        "cell_width": cell_width,
                        "cell_height": cell_height,
                        "x_edges": x_edges,
                        "y_edges": y_edges,
                        "score": confidence,
                        "size_px": grid_size,
                        "detection_method": "brightness_analysis",
                        "filename_match": False,
                    }

            return None

        except Exception as e:
            if self.config.debug:
                print(f"    ❌ Error in brightness analysis: {e}")
            return None

    def _compare_detection_methods(
        self, image_path: str, filename: str, pil_image: Image.Image
    ) -> Optional[Dict]:
        """
        Compare morphological and brightness-based detection, return the better result

        Args:
            image_path: Path to the image file
            filename: Filename for the image
            pil_image: PIL Image object

        Returns:
            Best grid_info dict with comparison details
        """
        # Create a minimal ImageInfo for the enhanced detection
        from .image_source_handler import ImageInfo

        img_info = ImageInfo(
            path=image_path,
            filename=filename,
            source_type="local",
            relative_path=filename,
        )

        return enhanced_grid_detection(
            pil_image=pil_image,
            image_path=image_path,
            img_info=img_info,
            debug=self.config.debug,
        )

    def _enhanced_grid_detection(
        self, img_info: ImageInfo, pil_image: Image.Image, image_path: str
    ) -> Optional[Dict]:
        """
        Enhanced grid detection using the 4-step smart logic:
        1. Grid in filename → use gridless + filename dimensions
        2. Only one variant, no filename grid → morphological + brightness detection
        3. All same type variants, no filename grid → prefer non-transparent, morphological + brightness detection
        4. Mixed variants, no filename grid → detect on gridded, apply to gridless

        Args:
            img_info: ImageInfo with variant analysis
            pil_image: Current image
            image_path: Path to current image

        Returns:
            Grid info dict or None
        """

        # Step 1: Grid in filename → use filename dimensions directly (skip visual detection)
        if img_info.has_dimensions:
            if self.config.debug:
                print(f"🔄 STEP 1: Grid dimensions found in filename")

            # Extract dimensions directly from filename and create grid info
            filename_dims = self.grid_detector.extract_dimensions_from_filename(
                img_info.filename
            )
            if filename_dims:
                nx, ny = filename_dims
                img_width, img_height = pil_image.size

                cell_width = img_width / float(nx)
                cell_height = img_height / float(ny)

                # Generate grid edges
                x_edges = [i * cell_width for i in range(nx + 1)]
                y_edges = [i * cell_height for i in range(ny + 1)]

                grid_info = {
                    "nx": nx,
                    "ny": ny,
                    "cell_width": cell_width,
                    "cell_height": cell_height,
                    "x_edges": x_edges,
                    "y_edges": y_edges,
                    "score": 0.95,  # High confidence for filename dimensions
                    "size_px": None,
                    "filename_dimensions": filename_dims,
                    "filename_match": True,
                    "detection_method": "filename_direct",
                }

                if self.config.debug:
                    print(f"    📐 Using filename dimensions: {nx}x{ny} squares")
                return grid_info
            else:
                if self.config.debug:
                    print(
                        f"    ⚠️  has_dimensions=True but could not extract dimensions from filename"
                    )
                # Fall back to visual detection
                grid_info = self._compare_detection_methods(
                    image_path, img_info.filename, pil_image
                )
                return grid_info

        # Step 2: Only one variant OR no both_variants, no filename grid → morphological + brightness detection
        elif not img_info.has_both_variants:
            if self.config.debug:
                print(f"🔄 STEP 2: Single variant, comparing detection methods")

            grid_info = self._compare_detection_methods(
                image_path, img_info.filename, pil_image
            )
            return grid_info

        # Step 4: Mixed variants, no filename grid → detect on gridded, apply to gridless
        elif (
            img_info.has_both_variants
            and img_info.gridded_variant_path
            and not img_info.is_gridded
        ):
            if self.config.debug:
                print(
                    f"🔄 STEP 4: Mixed variants, detect on gridded, apply to gridless"
                )
                print(f"  🔍 Detecting grid on: {img_info.gridded_variant_filename}")
                print(f"  📁 Will apply to: {img_info.filename}")

            # Run detection on gridded variant
            try:
                with Image.open(img_info.gridded_variant_path) as gridded_pil:
                    gridded_pil = gridded_pil.convert("RGB")
                    # Make sure we have a filename
                    gridded_filename = (
                        img_info.gridded_variant_filename
                        or os.path.basename(img_info.gridded_variant_path)
                    )
                    grid_info = self._compare_detection_methods(
                        img_info.gridded_variant_path, gridded_filename, gridded_pil
                    )

                if grid_info:
                    if self.config.debug:
                        print(
                            f"    📐 Detected grid: {grid_info['nx']}x{grid_info['ny']} squares"
                        )

                    # Update grid dimensions for current (gridless) image size
                    img_width, img_height = pil_image.size
                    grid_info["cell_width"] = img_width / grid_info["nx"]
                    grid_info["cell_height"] = img_height / grid_info["ny"]

                    # Update edges for current image
                    grid_info["x_edges"] = [
                        i * grid_info["cell_width"] for i in range(grid_info["nx"] + 1)
                    ]
                    grid_info["y_edges"] = [
                        i * grid_info["cell_height"] for i in range(grid_info["ny"] + 1)
                    ]

                    return grid_info
                else:
                    if self.config.debug:
                        print(f"    ❌ Failed to detect grid on gridded variant")
                    return None

            except Exception as e:
                if self.config.debug:
                    print(f"    ❌ Error loading gridded variant: {e}")
                return None

        # Step 3: All same type variants, no filename grid → use morphological + brightness detection on current
        else:
            if self.config.debug:
                print(
                    f"🔄 STEP 3: Same type variants, using enhanced detection on current image"
                )

            grid_info = self._compare_detection_methods(
                image_path, img_info.filename, pil_image
            )
            return grid_info

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
        if self.config.test_mode:
            print(f"🧪 TEST MODE RESULTS (No images were saved)")
        print(f"════════════════════════════════════")
        print(f"Images found:      {stats.images_found}")
        print(f"Images processed:  {stats.images_processed}")
        print(f"Images skipped:    {stats.images_skipped}")
        print(f"Images failed:     {stats.images_failed}")

        # Tile breakdown by size
        total_tiles = stats.total_tiles_generated
        if total_tiles > 0:
            print(
                f"Total tiles:       {total_tiles} {'(would be generated)' if self.config.test_mode else '(generated)'}"
            )
            if stats.tiles_12x12 > 0:
                percentage = (stats.tiles_12x12 / total_tiles) * 100
                print(f"  • 12x12 tiles:   {stats.tiles_12x12} ({percentage:.1f}%)")
            if stats.tiles_20x20 > 0:
                percentage = (stats.tiles_20x20 / total_tiles) * 100
                print(f"  • 20x20 tiles:   {stats.tiles_20x20} ({percentage:.1f}%)")
            if stats.tiles_30x30 > 0:
                percentage = (stats.tiles_30x30 / total_tiles) * 100
                print(f"  • 30x30 tiles:   {stats.tiles_30x30} ({percentage:.1f}%)")
        else:
            print(
                f"Total tiles:       0 {'(would be generated)' if self.config.test_mode else '(generated)'}"
            )
        if duration:
            print(f"Duration:          {duration}")
        if not self.config.test_mode:
            print(f"Output directory:  {self.output_dir}")
        print(f"════════════════════════════════════")

    def clear_hash_cache_on_startup(self):
        """Clear hash cache at startup if requested"""
        duplicate_cache_file = self.output_dir / "duplicate_detection_progress.json"

        if duplicate_cache_file.exists():
            try:
                # Load current cache data
                with open(duplicate_cache_file, "r") as f:
                    data = json.load(f)

                # Check if hash cache exists and remove it
                if "hash_cache" in data:
                    cache_size = len(data["hash_cache"])
                    del data["hash_cache"]

                    # Save back without hash cache
                    with open(duplicate_cache_file, "w") as f:
                        json.dump(data, f, indent=2)

                    print(f"🧹 Cleared hash cache ({cache_size} entries) at startup")
                else:
                    print("ℹ️  No existing hash cache found")

            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Hash cache startup cleanup warning: {e}")
        else:
            print("ℹ️  No duplicate cache file found (fresh start)")

    def _cleanup_hash_cache(self):
        """Clean up hash cache to avoid false duplicates in future runs"""
        # The hash cache is stored in duplicate_detection_progress.json (not the main progress file)
        duplicate_cache_file = self.output_dir / "duplicate_detection_progress.json"

        if duplicate_cache_file.exists():
            try:
                # Load current cache data
                with open(duplicate_cache_file, "r") as f:
                    data = json.load(f)

                # Check if hash cache exists and remove it
                hash_cache_existed = "hash_cache" in data
                if hash_cache_existed:
                    cache_size = len(data["hash_cache"])
                    del data["hash_cache"]

                    # Save back without hash cache
                    with open(duplicate_cache_file, "w") as f:
                        json.dump(data, f, indent=2)

                    print(
                        f"✅ Cleared hash cache ({cache_size} entries) for future runs"
                    )
                else:
                    print(f"ℹ️  No hash cache found to clear")

            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Hash cache cleanup warning: {e}")
        else:
            print(f"ℹ️  No duplicate cache file found for hash cache cleanup")

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
