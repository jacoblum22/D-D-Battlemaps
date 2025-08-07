"""
Boring Tile Detection for filtering out uninteresting tiles

Currently detects:
- Entirely black or very dark tiles
"""

import numpy as np
from PIL import Image
from typing import List, Tuple
import cv2


class BoringTileDetector:
    """Detects and filters out boring tiles from battlemap images"""

    def __init__(self):
        # Parameters for boring detection - more strict now
        self.dark_pixel_threshold = 5  # Only truly black pixels (was 30)
        self.max_dark_fraction = 0.98  # Must be almost entirely black (was 0.85)
        self.min_variance_threshold = 5.0  # Min variance for texture detection

    def is_boring_tile(self, tile_image: Image.Image) -> bool:
        """
        Check if a tile is boring (should be filtered out)

        Args:
            tile_image: PIL Image of the tile

        Returns:
            True if tile is boring, False otherwise
        """
        # Convert to numpy array
        img_array = np.array(tile_image)

        # Check if mostly dark/black
        if self._is_too_dark(img_array):
            return True

        # Add more boring checks here later
        # if self._is_solid_color(img_array):
        #     return True
        # if self._is_water_like(img_array):
        #     return True

        return False

    def _is_too_dark(self, img_array: np.ndarray) -> bool:
        """Check if image is mostly black/very dark"""
        # Convert to grayscale if needed
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Count dark pixels
        dark_pixels = np.sum(gray <= self.dark_pixel_threshold)
        total_pixels = gray.size
        dark_fraction = dark_pixels / total_pixels

        return bool(dark_fraction >= self.max_dark_fraction)

    def _is_solid_color(self, img_array: np.ndarray) -> bool:
        """Check if image is mostly one solid color (future use)"""
        # Calculate variance across all channels
        if img_array.ndim == 3:
            # For color images, check variance in each channel
            variances = [float(np.var(img_array[:, :, c])) for c in range(3)]
            max_variance = max(variances)
        else:
            # For grayscale
            max_variance = float(np.var(img_array))

        return bool(max_variance < self.min_variance_threshold)

    def analyze_tile_content(self, tile_image: Image.Image) -> dict:
        """
        Analyze tile content and return detailed statistics

        Args:
            tile_image: PIL Image of the tile

        Returns:
            Dict with analysis results
        """
        img_array = np.array(tile_image)

        # Convert to grayscale for analysis
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Calculate statistics
        dark_pixels = np.sum(gray <= self.dark_pixel_threshold)
        total_pixels = gray.size
        dark_fraction = dark_pixels / total_pixels

        mean_brightness = float(gray.mean())
        brightness_variance = float(gray.var())

        # Color variance (if color image)
        if img_array.ndim == 3:
            color_variances = [float(img_array[:, :, c].var()) for c in range(3)]
        else:
            color_variances = [brightness_variance]

        return {
            "is_boring": self.is_boring_tile(tile_image),
            "dark_fraction": dark_fraction,
            "mean_brightness": mean_brightness,
            "brightness_variance": brightness_variance,
            "color_variances": color_variances,
            "total_pixels": total_pixels,
            "dark_pixels": int(dark_pixels),
        }

    def filter_boring_tiles(
        self, tiles: List, verbose: bool = False
    ) -> Tuple[List, List]:
        """
        Filter out boring tiles from a list

        Args:
            tiles: List of tile objects with .image attribute
            verbose: Print filtering details

        Returns:
            (good_tiles, boring_tiles) tuple
        """
        good_tiles = []
        boring_tiles = []

        for i, tile in enumerate(tiles):
            if self.is_boring_tile(tile.image):
                boring_tiles.append(tile)
                if verbose:
                    analysis = self.analyze_tile_content(tile.image)
                    print(
                        f"  Tile {i}: BORING - {analysis['dark_fraction']:.1%} dark pixels"
                    )
            else:
                good_tiles.append(tile)
                if verbose:
                    analysis = self.analyze_tile_content(tile.image)
                    print(
                        f"  Tile {i}: GOOD - {analysis['dark_fraction']:.1%} dark pixels"
                    )

        return good_tiles, boring_tiles
