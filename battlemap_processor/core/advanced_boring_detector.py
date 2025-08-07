"""
Advanced Boring Detection for large uniform areas

This module detects both:
1. Individual black squares (existing functionality)
2. Large connected regions of uniform/low-interest content (new)
"""

import numpy as np
from PIL import Image
import cv2
from typing import Dict, Tuple, Set, List, Union, Optional
from collections import deque


class AdvancedBoringDetector:
    """
    Enhanced boring detection that identifies:
    - Individual black squares
    - Large uniform regions (water, plains, etc.)
    """

    def __init__(self):
        # Parameters for individual black square detection
        self.dark_pixel_threshold = 5  # Only truly black pixels
        self.max_dark_fraction = 0.98  # Must be almost entirely black

        # Parameters for uniform square detection (calibrated for large uniform areas)
        self.max_texture_variance = 35.0  # Low texture variance (smooth areas)
        # Removed saturation check - blue water is saturated but should be considered uniform
        self.max_brightness_variance = 45.0  # Consistent lighting across the square
        self.max_color_variance = 45.0  # Consistent color across RGB channels

        # Parameters for large uniform region detection
        self.min_region_size = (
            25  # Min squares in region to mark as boring (5x5, was 4x4=16)
        )
        self.color_similarity_threshold = (
            25.0  # Max color distance for grouping squares
        )

        # Parameters for gap filling (relaxed criteria for isolated squares)
        self.gap_fill_relaxation_factor = (
            1.7  # Multiply thresholds by this for gap filling (+70%, was +30%)
        )
        self.max_gap_size = (
            20  # Maximum isolated region size to consider for gap filling (was 3)
        )

    def analyze_image_regions(
        self, img: Image.Image, grid_info: Dict, debug: bool = False
    ) -> Dict[Tuple[int, int], str]:
        """
        Analyze the entire image for boring regions

        Returns:
            Dictionary mapping (col, row) -> reason
            where reason is: 'black', 'large_uniform_region', or 'good'
        """
        nx, ny = grid_info["nx"], grid_info["ny"]
        x_edges = grid_info["x_edges"]
        y_edges = grid_info["y_edges"]

        if debug:
            print(f"\n=== DEBUG: Analyzing {nx}x{ny} = {nx*ny} squares ===")

        # Step 1: Analyze each square individually
        square_analysis = {}
        square_features = {}  # Store features for region analysis
        uniform_count = 0
        black_count = 0
        debug_sample_count = 0

        for row in range(ny):
            for col in range(nx):
                # Extract square
                x0, x1 = x_edges[col], x_edges[col + 1]
                y0, y1 = y_edges[row], y_edges[row + 1]
                square = img.crop((x0, y0, x1, y1))

                # Check if square is black
                if self._is_black_square(np.array(square)):
                    square_analysis[(col, row)] = "black"
                    black_count += 1
                else:
                    # Mark as potentially uniform for region analysis
                    square_analysis[(col, row)] = "good"
                    features = self._extract_square_features(square)
                    square_features[(col, row)] = features

                    if features["is_uniform"]:
                        uniform_count += 1
                        if (
                            debug and uniform_count <= 3
                        ):  # Show first few uniform squares
                            print(
                                f"  Uniform square at ({col},{row}): "
                                f"texture_var={features['texture_variance']:.1f}, "
                                f"sat={features['mean_saturation']:.1f}"
                            )

        if debug:
            print(f"  Black squares: {black_count}")
            print(f"  Uniform squares: {uniform_count}")
            print(f"  Good squares: {len(square_features)}")

        # Step 2: Find connected regions of uniform squares
        uniform_squares = {
            pos for pos, features in square_features.items() if features["is_uniform"]
        }

        if debug:
            print(f"  Squares eligible for region analysis: {len(uniform_squares)}")

        uniform_regions = self._find_uniform_regions(uniform_squares, square_features)

        # Step 3: Mark large uniform regions as boring
        large_regions = 0
        total_uniform_squares_in_large_regions = 0

        for i, region in enumerate(uniform_regions):
            if len(region) >= self.min_region_size:
                large_regions += 1
                total_uniform_squares_in_large_regions += len(region)
                if debug:
                    print(f"  Large region {i+1}: {len(region)} squares")
                for pos in region:
                    if square_analysis[pos] == "good":  # Don't override black squares
                        square_analysis[pos] = "large_uniform_region"

        if debug:
            print(f"  Found {len(uniform_regions)} total regions")
            print(f"  Large regions (≥{self.min_region_size} squares): {large_regions}")
            print(
                f"  Squares marked as boring from large regions: {total_uniform_squares_in_large_regions}"
            )

        # Step 4: Gap filling - find isolated non-boring squares in boring regions
        gap_filled = self._fill_gaps_in_boring_regions(
            img, grid_info, square_analysis, debug=debug
        )

        if debug and gap_filled > 0:
            print(f"  Gap filling: converted {gap_filled} additional squares to boring")

        return square_analysis

    def _is_black_square(self, img_array: np.ndarray) -> bool:
        """Check if square is mostly black (existing logic)"""
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        dark_pixels = np.sum(gray <= self.dark_pixel_threshold)
        total_pixels = gray.size
        dark_fraction = dark_pixels / total_pixels

        return bool(dark_fraction >= self.max_dark_fraction)

    def _is_uniform_square(
        self, img_array: np.ndarray, debug_pos: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        Check if square is uniform (combines water-like detection with general uniformity)

        A square is considered uniform if it has:
        - Low texture variance (no big changes in contrast)
        - Low saturation (not highly colorful)
        - Low brightness variance (consistent lighting)
        - Low color variance in RGB channels
        """
        # Texture analysis using grayscale
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        texture_variance = float(gray.var())

        # Color analysis using HSV
        mean_sat = 0.0
        brightness_variance = 0.0
        max_color_variance = 0.0

        if img_array.ndim == 3:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

            # Check saturation (water/sky typically low saturation)
            saturation = hsv[:, :, 1]
            mean_sat = float(saturation.mean())

            # Check brightness uniformity
            value = hsv[:, :, 2]
            brightness_variance = float(value.var())

            # Check color variance in RGB channels
            color_variances = [float(img_array[:, :, c].var()) for c in range(3)]
            max_color_variance = max(color_variances)

        # Check individual criteria
        criteria_met = {
            "texture": texture_variance <= self.max_texture_variance,
            "brightness": brightness_variance <= self.max_brightness_variance,
            "color": max_color_variance <= self.max_color_variance,
        }

        is_uniform = all(criteria_met.values())

        # Debug output for failed squares
        if debug_pos and not is_uniform:
            failed_criteria = [k for k, v in criteria_met.items() if not v]
            print(f"    Square {debug_pos} failed criteria: {failed_criteria}")
            print(
                f"      texture_var={texture_variance:.1f} (max={self.max_texture_variance})"
            )
            print(
                f"      brightness_var={brightness_variance:.1f} (max={self.max_brightness_variance})"
            )
            print(
                f"      max_color_var={max_color_variance:.1f} (max={self.max_color_variance})"
            )
            print(f"      saturation={mean_sat:.1f} (removed from criteria)")

        return is_uniform

    def _extract_square_features(self, square: Image.Image) -> Dict:
        """Extract features from a square for region analysis"""
        img_array = np.array(square)

        # Color features
        mean_color = img_array.mean(axis=(0, 1))

        # Check if square meets uniformity criteria
        is_uniform = self._is_uniform_square(img_array)

        # Texture features
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        texture_variance = float(gray.var())

        # HSV features
        hsv = (
            cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            if img_array.ndim == 3
            else img_array
        )
        mean_hue = float(hsv[:, :, 0].mean()) if img_array.ndim == 3 else 0.0
        mean_sat = float(hsv[:, :, 1].mean()) if img_array.ndim == 3 else 0.0

        return {
            "mean_color": mean_color,
            "texture_variance": texture_variance,
            "mean_hue": mean_hue,
            "mean_saturation": mean_sat,
            "is_uniform": is_uniform,
        }

    def _find_uniform_regions(
        self,
        uniform_squares: Set[Tuple[int, int]],
        square_features: Dict[Tuple[int, int], Dict],
    ) -> List[Set[Tuple[int, int]]]:
        """Find connected regions of similar uniform squares"""
        visited = set()
        regions = []

        for start_pos in uniform_squares:
            if start_pos in visited:
                continue

            # Flood fill to find connected similar squares
            region = set()
            queue = deque([start_pos])
            start_features = square_features[start_pos]

            while queue:
                pos = queue.popleft()
                if pos in visited or pos not in uniform_squares:
                    continue

                # Check if this square is similar to the start square
                if self._are_squares_similar(start_features, square_features[pos]):
                    visited.add(pos)
                    region.add(pos)

                    # Add neighbors to queue
                    col, row = pos
                    neighbors = [
                        (col - 1, row),
                        (col + 1, row),
                        (col, row - 1),
                        (col, row + 1),
                    ]
                    for neighbor in neighbors:
                        if neighbor in uniform_squares and neighbor not in visited:
                            queue.append(neighbor)

            if len(region) > 1:  # Only keep regions with multiple squares
                regions.append(region)

        return regions

    def _are_squares_similar(self, features1: Dict, features2: Dict) -> bool:
        """Check if two squares are similar enough to be in the same uniform region"""
        # Both squares must individually be uniform
        if not (features1["is_uniform"] and features2["is_uniform"]):
            return False

        # Color similarity
        color_distance = np.linalg.norm(
            features1["mean_color"] - features2["mean_color"]
        )
        if color_distance > self.color_similarity_threshold:
            return False

        return True

    def _fill_gaps_in_boring_regions(
        self,
        img: Image.Image,
        grid_info: Dict,
        square_analysis: Dict[Tuple[int, int], str],
        debug: bool = False,
    ) -> int:
        """
        Fill gaps in boring regions by testing isolated non-boring squares with relaxed criteria

        Returns:
            Number of squares converted from good to boring
        """
        nx, ny = grid_info["nx"], grid_info["ny"]
        x_edges = grid_info["x_edges"]
        y_edges = grid_info["y_edges"]

        # Find isolated "good" squares (small regions surrounded by boring squares)
        good_squares = {
            pos for pos, reason in square_analysis.items() if reason == "good"
        }
        isolated_regions = self._find_isolated_regions(
            good_squares, square_analysis, nx, ny
        )

        gap_filled_count = 0

        for region in isolated_regions:
            if len(region) <= self.max_gap_size:  # Only consider small isolated regions
                # Check if this region is surrounded by boring squares
                if self._is_surrounded_by_boring(region, square_analysis, nx, ny):
                    # Test ALL squares in the region with relaxed criteria first
                    all_squares_pass = True
                    region_squares = []

                    for col, row in region:
                        # Extract square
                        x0, x1 = x_edges[col], x_edges[col + 1]
                        y0, y1 = y_edges[row], y_edges[row + 1]
                        square = img.crop((x0, y0, x1, y1))
                        region_squares.append((col, row, square))

                        # Test with relaxed criteria
                        if not self._is_uniform_square_relaxed(np.array(square)):
                            all_squares_pass = False
                            break  # No need to test remaining squares

                    # Only fill the region if ALL squares pass the relaxed criteria
                    if all_squares_pass:
                        for col, row, square in region_squares:
                            square_analysis[(col, row)] = "large_uniform_region"
                            gap_filled_count += 1
                            if debug:
                                print(f"    Gap filled: ({col},{row})")

        return gap_filled_count

    def _find_isolated_regions(
        self,
        good_squares: Set[Tuple[int, int]],
        square_analysis: Dict[Tuple[int, int], str],
        nx: int,
        ny: int,
    ) -> List[Set[Tuple[int, int]]]:
        """Find connected regions of good squares"""
        visited = set()
        regions = []

        for start_pos in good_squares:
            if start_pos in visited:
                continue

            # Flood fill to find connected good squares
            region = set()
            queue = deque([start_pos])

            while queue:
                pos = queue.popleft()
                if pos in visited or pos not in good_squares:
                    continue

                visited.add(pos)
                region.add(pos)

                # Add neighbors to queue
                col, row = pos
                neighbors = [
                    (col - 1, row),
                    (col + 1, row),
                    (col, row - 1),
                    (col, row + 1),
                ]
                for neighbor in neighbors:
                    neighbor_col, neighbor_row = neighbor
                    if (
                        0 <= neighbor_col < nx
                        and 0 <= neighbor_row < ny
                        and neighbor in good_squares
                        and neighbor not in visited
                    ):
                        queue.append(neighbor)

            if region:
                regions.append(region)

        return regions

    def _is_surrounded_by_boring(
        self,
        region: Set[Tuple[int, int]],
        square_analysis: Dict[Tuple[int, int], str],
        nx: int,
        ny: int,
    ) -> bool:
        """Check if a region is mostly surrounded by boring squares"""
        surrounding_squares = set()

        # Find all squares adjacent to the region
        for col, row in region:
            neighbors = [(col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1)]
            for neighbor in neighbors:
                neighbor_col, neighbor_row = neighbor
                if (
                    0 <= neighbor_col < nx
                    and 0 <= neighbor_row < ny
                    and neighbor not in region
                ):
                    surrounding_squares.add(neighbor)

        if not surrounding_squares:
            return False

        # Count how many surrounding squares are boring
        boring_neighbors = sum(
            1
            for pos in surrounding_squares
            if square_analysis.get(pos, "good") in ["black", "large_uniform_region"]
        )

        # Require at least 100% of surrounding squares to be boring (completely surrounded)
        return boring_neighbors / len(surrounding_squares) >= 1.0

    def _is_uniform_square_relaxed(self, img_array: np.ndarray) -> bool:
        """Check uniformity with relaxed criteria for gap filling"""
        # Use relaxed thresholds
        relaxed_texture_threshold = (
            self.max_texture_variance * self.gap_fill_relaxation_factor
        )
        relaxed_brightness_threshold = (
            self.max_brightness_variance * self.gap_fill_relaxation_factor
        )
        relaxed_color_threshold = (
            self.max_color_variance * self.gap_fill_relaxation_factor
        )

        # Texture analysis using grayscale
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        texture_variance = float(gray.var())
        if texture_variance > relaxed_texture_threshold:
            return False

        # Color analysis using HSV
        brightness_variance = 0.0
        max_color_variance = 0.0

        if img_array.ndim == 3:
            # Check brightness uniformity
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            value = hsv[:, :, 2]
            brightness_variance = float(value.var())
            if brightness_variance > relaxed_brightness_threshold:
                return False

            # Check color variance in RGB channels
            color_variances = [float(img_array[:, :, c].var()) for c in range(3)]
            max_color_variance = max(color_variances)
            if max_color_variance > relaxed_color_threshold:
                return False

        return True

    def get_boring_stats(self, square_analysis: Dict[Tuple[int, int], str]) -> Dict:
        """Get statistics about boring regions"""
        stats: Dict[str, Union[int, float]] = {
            "total_squares": len(square_analysis),
            "black_squares": 0,
            "large_uniform_regions": 0,
            "good_squares": 0,
        }

        for reason in square_analysis.values():
            if reason == "black":
                stats["black_squares"] += 1
            elif reason == "large_uniform_region":
                stats["large_uniform_regions"] += 1
            else:
                stats["good_squares"] += 1

        total_boring = stats["black_squares"] + stats["large_uniform_regions"]
        stats["total_boring"] = total_boring
        stats["boring_percentage"] = float(
            total_boring / stats["total_squares"] * 100
            if stats["total_squares"] > 0
            else 0.0
        )

        return stats
