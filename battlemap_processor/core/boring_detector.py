"""
Boring Tile Detection for filtering out uninteresting tiles

Currently detects:
- Entirely black or very dark tiles
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Any, Union
import cv2


class BoringTileDetector:
    """Detects and filters out boring tiles from battlemap images"""

    def __init__(self):
        # Parameters for boring detection - more strict now
        self.dark_pixel_threshold = 5  # Only truly black pixels (was 30)
        self.max_dark_fraction = 0.98  # Must be almost entirely black (was 0.85)
        self.min_variance_threshold = 5.0  # Min variance for texture detection

    def _extract_image_features(self, img_array: np.ndarray) -> dict:
        """Extract common image features used by multiple methods"""
        # Convert to grayscale if needed
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Calculate basic statistics
        total_pixels = gray.size
        dark_pixels = np.sum(gray <= self.dark_pixel_threshold)
        dark_fraction = dark_pixels / total_pixels
        mean_brightness = float(gray.mean())
        brightness_variance = float(gray.var())

        # Color variance (if color image)
        if img_array.ndim == 3:
            color_variances = [float(img_array[:, :, c].var()) for c in range(3)]
        else:
            color_variances = [brightness_variance]

        return {
            "gray": gray,
            "total_pixels": total_pixels,
            "dark_pixels": int(dark_pixels),
            "dark_fraction": dark_fraction,
            "mean_brightness": mean_brightness,
            "brightness_variance": brightness_variance,
            "color_variances": color_variances,
        }

    def is_boring_tile(self, tile_image: Union[Image.Image, np.ndarray]) -> bool:
        """
        Check if a tile is boring (should be filtered out)

        Args:
            tile_image: PIL Image or numpy array of the tile

        Returns:
            True if tile is boring, False otherwise
        """
        # Convert to numpy array if needed
        if isinstance(tile_image, Image.Image):
            img_array = np.array(tile_image)
        else:
            img_array = tile_image

        # Extract features once
        features = self._extract_image_features(img_array)

        # Check if mostly dark/black
        if features["dark_fraction"] >= self.max_dark_fraction:
            return True

        # Add more boring checks here later
        # if self._is_solid_color(features):
        #     return True
        # if self._is_water_like(features):
        #     return True

        return False

    def _is_too_dark(self, img_array: np.ndarray) -> bool:
        """Check if image is mostly black/very dark"""
        features = self._extract_image_features(img_array)
        return features["dark_fraction"] >= self.max_dark_fraction

    def _is_solid_color(self, img_array: np.ndarray) -> bool:
        """Check if image is mostly one solid color (future use)"""
        features = self._extract_image_features(img_array)
        max_variance = max(features["color_variances"])
        return bool(max_variance < self.min_variance_threshold)

    def analyze_tile_content(self, tile_image: Union[Image.Image, np.ndarray]) -> dict:
        """
        Analyze tile content and return detailed statistics

        Args:
            tile_image: PIL Image or numpy array of the tile

        Returns:
            Dict with analysis results
        """
        # Convert to numpy array if needed
        if isinstance(tile_image, Image.Image):
            img_array = np.array(tile_image)
        else:
            img_array = tile_image

        # Extract features once
        features = self._extract_image_features(img_array)

        # Determine if boring using the same logic as is_boring_tile
        is_boring = features["dark_fraction"] >= self.max_dark_fraction

        # Return results without redundant calculations
        return {
            "is_boring": is_boring,
            "dark_fraction": features["dark_fraction"],
            "mean_brightness": features["mean_brightness"],
            "brightness_variance": features["brightness_variance"],
            "color_variances": features["color_variances"],
            "total_pixels": features["total_pixels"],
            "dark_pixels": features["dark_pixels"],
        }

    def filter_boring_tiles(
        self, tiles: List[Any], verbose: bool = False
    ) -> Tuple[List[Any], List[Any]]:
        """
        Filter out boring tiles from a list

        Args:
            tiles: List of tile objects with .image attribute (PIL.Image)
            verbose: Print filtering details

        Returns:
            (good_tiles, boring_tiles) tuple

        Raises:
            AttributeError: If any tile object lacks .image attribute
        """
        good_tiles = []
        boring_tiles = []

        for i, tile in enumerate(tiles):
            # Validate tile has required attribute
            if not hasattr(tile, "image"):
                raise AttributeError(
                    f"Tile {i} does not have required 'image' attribute"
                )

            if verbose:
                # When verbose, analyze once and use the result for both classification and output
                analysis = self.analyze_tile_content(tile.image)
                is_boring = analysis["is_boring"]

                if is_boring:
                    boring_tiles.append(tile)
                    print(
                        f"  Tile {i}: BORING - {analysis['dark_fraction']:.1%} dark pixels"
                    )
                else:
                    good_tiles.append(tile)
                    print(
                        f"  Tile {i}: GOOD - {analysis['dark_fraction']:.1%} dark pixels"
                    )
            else:
                # When not verbose, use the more efficient single check
                if self.is_boring_tile(tile.image):
                    boring_tiles.append(tile)
                else:
                    good_tiles.append(tile)

        return good_tiles, boring_tiles
