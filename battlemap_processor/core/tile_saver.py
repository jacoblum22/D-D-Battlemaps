"""
Tile Saver for saving optimally placed tiles as images

This module takes the placed tiles from OptimalTilePlacer and saves them
as 512x512 images to organized subfolders based on source.
"""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs
from PIL import Image

from .optimal_tile_placer import TilePlacement

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SavedTileInfo:
    """Information about a saved tile"""

    tile_placement: TilePlacement
    filename: str
    filepath: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of the tile saving operation"""

    source_name: str
    output_folder: str
    skipped: bool
    skipped_reason: Optional[str]
    saved_tiles: List[SavedTileInfo]
    total_files_found: int = 0  # If skipped, how many files were already there


class TileSaver:
    """
    Saves optimally placed tiles as 512x512 images
    """

    def __init__(self, output_size: int = 512, image_format: str = "PNG"):
        self.output_size = output_size
        self.image_format = image_format.upper()

    def save_tiles(
        self,
        source_image: Image.Image,
        grid_info: Dict[str, Any],
        placed_tiles: List[TilePlacement],
        source_path_or_url: str,
        output_folder: str = "generated_images",
        base_filename: Optional[str] = None,
        overwrite: bool = False,
        debug: bool = False,
    ) -> ProcessingResult:
        """
        Save all placed tiles as individual images in organized subfolders

        Args:
            source_image: The original battlemap image
            grid_info: Grid detection results with x_edges, y_edges
            placed_tiles: List of TilePlacement objects from OptimalTilePlacer
            source_path_or_url: Original source (file path, Google Drive URL, etc.)
            output_folder: Base folder to save images to
            base_filename: Base name for files (if None, derives from source)
            overwrite: If True, replace existing work; if False, skip if exists
            debug: Enable debug output

        Returns:
            ProcessingResult with information about what was saved or skipped
        """
        if not placed_tiles:
            logger.warning("No tiles to save!")
            return ProcessingResult(
                source_name="unknown",
                output_folder=output_folder,
                skipped=True,
                skipped_reason="No tiles to save",
                saved_tiles=[],
            )

        # Extract source name and create subfolder
        source_name = self.extract_source_name(source_path_or_url)
        subfolder_path = Path(output_folder) / source_name

        if debug:
            print(f"\n=== Tile Saving ===")
            print(f"Source: {source_path_or_url}")
            print(f"Source name: {source_name}")
            print(f"Output subfolder: {subfolder_path}")

        # Check for existing work
        existing_info = self.check_existing_work(str(subfolder_path))
        if self.should_skip_processing(existing_info, overwrite):
            if debug:
                print(f"⏭️  Skipping - {existing_info['reason']}")
                print(f"   Found {existing_info['file_count']} existing files")
                if not overwrite:
                    print("   Use overwrite=True to replace existing files")

            return ProcessingResult(
                source_name=source_name,
                output_folder=str(subfolder_path),
                skipped=True,
                skipped_reason=existing_info["reason"],
                saved_tiles=[],
                total_files_found=existing_info["file_count"],
            )

        # Prepare output folder
        if overwrite and subfolder_path.exists():
            if debug:
                print(f"🗑️  Clearing existing folder due to overwrite=True")
            shutil.rmtree(subfolder_path)

        subfolder_path.mkdir(parents=True, exist_ok=True)

        if debug:
            print(f"💾 Saving {len(placed_tiles)} tiles to '{subfolder_path}'")

        # Determine base filename
        if base_filename is None:
            base_filename = source_name

        # Save tiles using existing logic
        saved_tiles = self._save_tiles_to_folder(
            source_image,
            grid_info,
            placed_tiles,
            str(subfolder_path),
            base_filename,
            debug,
        )

        successful_saves = sum(1 for info in saved_tiles if info.success)
        if debug:
            print(f"✅ Saved {successful_saves}/{len(placed_tiles)} tiles successfully")

        return ProcessingResult(
            source_name=source_name,
            output_folder=str(subfolder_path),
            skipped=False,
            skipped_reason=None,
            saved_tiles=saved_tiles,
        )

    def extract_source_name(self, source_path_or_url: str) -> str:
        """
        Extract a meaningful folder name from various source types

        Handles:
        - Google Drive URLs
        - Zip file paths/URLs
        - Regular file paths
        - Other URLs
        """
        source = source_path_or_url.strip()

        # Handle Google Drive URLs
        if "drive.google.com" in source:
            # Try to extract file ID from various Google Drive URL formats
            patterns = [
                r"/file/d/([a-zA-Z0-9_-]+)",  # /file/d/{id}/view
                r"[?&]id=([a-zA-Z0-9_-]+)",  # ?id={id}
            ]

            for pattern in patterns:
                match = re.search(pattern, source)
                if match:
                    file_id = match.group(1)
                    return (
                        f"gdrive_{file_id[:16]}"  # Truncate for reasonable folder name
                    )

            # Fallback for Google Drive URLs
            return f"gdrive_{hash(source) % 100000:05d}"

        # Handle file paths (local or URL)
        try:
            if source.startswith(("http://", "https://")):
                # Extract filename from URL
                parsed = urlparse(source)
                filename = Path(parsed.path).name
            else:
                # Local file path
                filename = Path(source).name

            if filename:
                # Remove extension and clean up
                name = Path(filename).stem
                # Replace problematic characters for folder names
                name = re.sub(r'[<>:"/\\|?*]', "_", name)
                # Limit length
                if len(name) > 50:
                    name = name[:50]
                return name if name else "unknown"

        except Exception as e:
            logger.warning(f"Could not extract name from source: {e}")

        # Ultimate fallback
        return f"source_{hash(source) % 100000:05d}"

    def check_existing_work(self, output_folder: str) -> Dict[str, Any]:
        """
        Check if this source has already been processed

        Returns:
            Dict with 'exists', 'file_count', 'reason' keys
        """
        folder_path = Path(output_folder)

        if not folder_path.exists():
            return {"exists": False, "file_count": 0, "reason": "Folder does not exist"}

        # Count image files in the folder
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        image_files = [
            f
            for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        file_count = len(image_files)

        if file_count == 0:
            return {
                "exists": False,
                "file_count": 0,
                "reason": "Folder exists but contains no image files",
            }

        return {
            "exists": True,
            "file_count": file_count,
            "reason": f"Folder already contains {file_count} image files",
        }

    def should_skip_processing(
        self, existing_info: Dict[str, Any], overwrite: bool
    ) -> bool:
        """
        Determine if processing should be skipped based on existing work and overwrite setting
        """
        return existing_info["exists"] and not overwrite

    def _save_tiles_to_folder(
        self,
        source_image: Image.Image,
        grid_info: Dict[str, Any],
        placed_tiles: List[TilePlacement],
        output_folder: str,
        base_filename: str,
        debug: bool = False,
    ) -> List[SavedTileInfo]:
        """
        Internal method to save tiles to a specific folder (original save_tiles logic)
        """
        saved_tiles = []
        output_path = Path(output_folder)
        x_edges = grid_info["x_edges"]
        y_edges = grid_info["y_edges"]

        for i, tile in enumerate(placed_tiles):
            try:
                # Generate filename
                filename = self._generate_filename(
                    base_filename, i + 1, tile, self.image_format.lower()
                )
                filepath = output_path / filename

                # Extract and resize tile from source image
                tile_image = self._extract_tile_image(
                    source_image, tile, x_edges, y_edges
                )

                # Save the image
                tile_image.save(filepath, format=self.image_format)

                saved_info = SavedTileInfo(
                    tile_placement=tile,
                    filename=filename,
                    filepath=str(filepath),
                    success=True,
                )
                saved_tiles.append(saved_info)

                if debug:
                    print(f"  ✅ Saved tile {i+1}: {filename}")
                    print(f"     Position: ({tile.start_col}, {tile.start_row})")
                    print(
                        f"     Quality: {tile.good_count} good, {tile.boring_count} boring ({tile.boring_percentage:.1f}%)"
                    )

            except Exception as e:
                error_msg = f"Failed to save tile {i+1}: {str(e)}"
                logger.error(error_msg)

                saved_info = SavedTileInfo(
                    tile_placement=tile,
                    filename="",
                    filepath="",
                    success=False,
                    error_message=error_msg,
                )
                saved_tiles.append(saved_info)

                if debug:
                    print(f"  ❌ {error_msg}")

        return saved_tiles

    def _generate_filename(
        self, base_name: str, tile_number: int, tile: TilePlacement, extension: str
    ) -> str:
        """Generate a descriptive filename for a tile"""
        # Format: basename_tile_001_pos_12_8_good_135_boring_15.3pct.png
        filename = (
            f"{base_name}_tile_{tile_number:03d}_"
            f"pos_{tile.start_col}_{tile.start_row}_"
            f"good_{tile.good_count}_"
            f"boring_{tile.boring_percentage:.1f}pct.{extension}"
        )
        return filename

    def _extract_tile_image(
        self,
        source_image: Image.Image,
        tile: TilePlacement,
        x_edges: List[int],
        y_edges: List[int],
    ) -> Image.Image:
        """
        Extract a tile from the source image and resize to output_size

        This is similar to TileExtractor._create_tile_from_grid_coords but
        works with TilePlacement objects instead.
        """
        # Calculate pixel boundaries
        start_x = x_edges[tile.start_col]
        start_y = y_edges[tile.start_row]
        end_x = x_edges[tile.start_col + tile.size]
        end_y = y_edges[tile.start_row + tile.size]

        # Crop the tile from source image
        tile_crop = source_image.crop((start_x, start_y, end_x, end_y))

        # Resize to target size
        tile_resized = tile_crop.resize(
            (self.output_size, self.output_size), Image.Resampling.LANCZOS
        )

        return tile_resized

    def get_save_stats(self, processing_result: ProcessingResult) -> Dict[str, Any]:
        """Get statistics about the save operation from ProcessingResult"""
        if processing_result.skipped:
            return {
                "source_name": processing_result.source_name,
                "skipped": True,
                "skipped_reason": processing_result.skipped_reason,
                "existing_files": processing_result.total_files_found,
                "tiles_saved": 0,
                "success_rate": 0.0,
                "total_good_squares": 0,
                "estimated_size_mb": 0.0,
            }

        saved_tiles = processing_result.saved_tiles
        if not saved_tiles:
            return {
                "source_name": processing_result.source_name,
                "skipped": False,
                "tiles_saved": 0,
                "success_rate": 0.0,
                "total_good_squares": 0,
                "estimated_size_mb": 0.0,
            }

        successful_saves = sum(1 for info in saved_tiles if info.success)
        failed_saves = len(saved_tiles) - successful_saves
        success_rate = successful_saves / len(saved_tiles) * 100 if saved_tiles else 0

        total_good_squares = sum(
            info.tile_placement.good_count for info in saved_tiles if info.success
        )

        # Estimate total file size (rough estimate)
        estimated_size_mb = successful_saves * 0.5  # Assume ~0.5MB per 512x512 PNG

        return {
            "source_name": processing_result.source_name,
            "skipped": False,
            "tiles_saved": successful_saves,
            "failed_saves": failed_saves,
            "success_rate": success_rate,
            "total_good_squares": total_good_squares,
            "estimated_size_mb": estimated_size_mb,
            "output_folder": processing_result.output_folder,
        }
