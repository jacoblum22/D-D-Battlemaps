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

        # Parameters for blur detection (integrated from test_per_tile_blur.py)
        self.enable_blur_detection = True  # Enable blur detection as final step
        self.base_blur_threshold = 100.0  # Base Laplacian variance threshold for blur
        self.use_brightness_adaptive = True  # Enable brightness-adaptive thresholding
        self.target_brightness = (
            67.6  # Target brightness for threshold=100 (from Casino actual test)
        )
        self.brightness_power = (
            1.5  # Power for brightness scaling (higher = more adaptive)
        )
        self.restrictive_multiplier = (
            0.6  # Multiplier for isolated blurry tiles (LOWER = harder to be blurry)
        )
        self.permissive_multiplier = (
            1.5  # Multiplier for isolated clear tiles (HIGHER = easier to be blurry)
        )
        self.min_neighbors_for_restrictive = (
            2  # Min non-blurry neighbors needed to make blurry tile more restrictive
        )
        self.min_neighbors_for_permissive = (
            3  # Min blurry neighbors needed to make clear tile more permissive
        )
        self.min_blur_group_size = 11  # Min blur group size to keep
        self.min_blur_percentage = 25.0  # Min percentage of tiles blurred to accept

        # Parameters for grayscale detection (integrated from test_grayscale_detection.py)
        self.enable_grayscale_detection = (
            True  # Enable grayscale detection for underground maps
        )
        self.max_saturation = 30.0  # Maximum HSV saturation for grayscale detection
        self.min_grayscale_percentage = (
            25.0  # Min percentage of non-boring tiles that must be grayscale to accept
        )

    def analyze_image_regions(
        self, img: Image.Image, grid_info: Dict, debug: bool = False
    ) -> Tuple[Dict[Tuple[int, int], str], Dict[Tuple[int, int], str]]:
        """
        Analyze the entire image for boring regions

        Returns:
            Tuple of:
            - Dictionary mapping (col, row) -> 'boring' or 'good'
            - Dictionary mapping (col, row) -> specific reason ('black', 'uniform', 'gap_fill', 'grayscale', 'blur', 'good')
        """
        nx, ny = grid_info["nx"], grid_info["ny"]
        x_edges = grid_info["x_edges"]
        y_edges = grid_info["y_edges"]

        if debug:
            print(f"\n=== DEBUG: Analyzing {nx}x{ny} = {nx*ny} squares ===")

        # Initialize tracking dictionaries
        square_analysis = {}  # 'boring' or 'good'
        boring_reasons = {}  # specific reason for boring classification
        square_features = {}  # Store features for region analysis
        uniform_count = 0
        black_count = 0
        debug_sample_count = 0

        # Initialize all squares as good
        for row in range(ny):
            for col in range(nx):
                square_analysis[(col, row)] = "good"
                boring_reasons[(col, row)] = "good"

        # Step 1: Analyze each square individually
        for row in range(ny):
            for col in range(nx):
                # Extract square
                x0, x1 = x_edges[col], x_edges[col + 1]
                y0, y1 = y_edges[row], y_edges[row + 1]
                square = img.crop((x0, y0, x1, y1))
                square_array = np.array(square)

                # Check if square is black
                if self._is_black_square(square_array):
                    square_analysis[(col, row)] = "boring"
                    boring_reasons[(col, row)] = "black"
                    black_count += 1
                else:
                    # Check uniformity first (cheap test)
                    if self._is_uniform_square(square_array):
                        # Only extract full features for uniform squares
                        features = self._extract_square_features_from_array(
                            square_array
                        )
                        square_features[(col, row)] = features
                        uniform_count += 1

                        if (
                            debug and uniform_count <= 3
                        ):  # Show first few uniform squares
                            print(
                                f"  Uniform square at ({col},{row}): "
                                f"texture_var={features['texture_variance']:.1f}, "
                                f"sat={features['mean_saturation']:.1f}"
                            )

                    # Keep as good (will be processed later if uniform)
                    # square_analysis and boring_reasons already initialized as "good"

        if debug:
            print(f"  Black squares: {black_count}")
            print(f"  Uniform squares: {uniform_count}")
            print(f"  Good squares: {len(square_features)}")

        # Step 2: Find connected regions of uniform squares
        uniform_squares = set(
            square_features.keys()
        )  # Only squares that passed uniformity check have features

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
                        square_analysis[pos] = "boring"
                        boring_reasons[pos] = "uniform"

        if debug:
            print(f"  Found {len(uniform_regions)} total regions")
            print(f"  Large regions (≥{self.min_region_size} squares): {large_regions}")
            print(
                f"  Squares marked as boring from large regions: {total_uniform_squares_in_large_regions}"
            )

        # Step 4: Gap filling - find isolated non-boring squares in boring regions
        gap_filled = self._fill_gaps_in_boring_regions(
            img, grid_info, square_analysis, boring_reasons, debug
        )

        if debug and gap_filled > 0:
            print(f"  Gap filling: converted {gap_filled} additional squares to boring")

        # Step 5: Grayscale detection - find grayscale tiles among remaining "good" tiles
        grayscale_count = 0
        if self.enable_grayscale_detection:
            grayscale_count = self._detect_grayscale_tiles(
                img, grid_info, square_analysis, boring_reasons, debug
            )
            if debug and grayscale_count > 0:
                print(
                    f"  Grayscale detection: marked {grayscale_count} additional squares as boring"
                )

        # Step 6: Blur detection - find blurred tiles among remaining "good" tiles
        # Skip blur detection if grayscale detection was successful (likely underground map)
        if self.enable_blur_detection and grayscale_count == 0:
            blur_count = self._detect_blurred_tiles(
                img, grid_info, square_analysis, boring_reasons, debug
            )
            if debug and blur_count > 0:
                print(
                    f"  Blur detection: marked {blur_count} additional squares as boring"
                )
        elif self.enable_blur_detection and grayscale_count > 0:
            if debug:
                print(
                    f"  Blur detection: skipped because grayscale detection was successful ({grayscale_count} squares)"
                )

        return square_analysis, boring_reasons

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
        return self._extract_square_features_from_array(img_array)

    def _extract_square_features_from_array(self, img_array: np.ndarray) -> Dict:
        """Extract features from a square array for region analysis (optimized version)"""
        # Color features
        mean_color = img_array.mean(axis=(0, 1))

        # Check if square meets uniformity criteria (already computed, but store result)
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
        boring_reasons: Dict[Tuple[int, int], str],
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
                            square_analysis[(col, row)] = "boring"
                            boring_reasons[(col, row)] = "gap_fill"
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
            if square_analysis.get(pos, "good") == "boring"
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

    def _detect_blurred_tiles(
        self,
        img: Image.Image,
        grid_info: Dict,
        square_analysis: Dict[Tuple[int, int], str],
        boring_reasons: Dict[Tuple[int, int], str],
        debug: bool = False,
    ) -> int:
        """
        Detect blurred tiles among remaining "good" tiles and mark them as boring

        Returns:
            Number of tiles marked as blurred
        """
        if debug:
            print(f"  Starting blur detection...")

        # Calculate adaptive threshold for the entire image
        adaptive_threshold = self._calculate_adaptive_blur_threshold(img, debug=debug)

        nx, ny = grid_info["nx"], grid_info["ny"]
        x_edges = grid_info["x_edges"]
        y_edges = grid_info["y_edges"]

        # Step 1: Initial blur detection pass (only on "good" tiles)
        blur_analysis = {}
        tile_variances = {}
        good_tiles_checked = 0

        for row in range(ny):
            for col in range(nx):
                # Skip tiles that are already marked as boring
                if square_analysis.get((col, row), "good") != "good":
                    blur_analysis[(col, row)] = False
                    continue

                good_tiles_checked += 1

                # Extract square
                x0, x1 = x_edges[col], x_edges[col + 1]
                y0, y1 = y_edges[row], y_edges[row + 1]
                square = img.crop((x0, y0, x1, y1))
                square_array = np.array(square)

                # Get variance and initial blur decision
                variance = self._get_blur_variance(square_array)
                is_blurred = variance < adaptive_threshold

                blur_analysis[(col, row)] = is_blurred
                tile_variances[(col, row)] = variance

        if debug:
            initial_blur_count = sum(1 for v in blur_analysis.values() if v)
            print(f"    Checked {good_tiles_checked} good tiles for blur")
            print(f"    Initial blur detections: {initial_blur_count}")

        # Step 2: Context-aware refinement pass
        refinement_count = self._refine_blur_decisions(
            blur_analysis,
            tile_variances,
            adaptive_threshold,
            square_analysis,
            nx,
            ny,
            debug=debug,
        )

        if debug and refinement_count > 0:
            print(f"    Context-aware refinement changed {refinement_count} tiles")

        # Step 3: Remove small isolated blur groups
        tiles_unmarked = self._remove_small_blur_groups(
            blur_analysis, nx, ny, debug=debug
        )

        if debug and tiles_unmarked > 0:
            print(f"    Removed small blur groups: {tiles_unmarked} tiles")

        # Step 4: Global validation - reject if too few squares are blurred
        blurred_count = self._apply_global_blur_validation(
            blur_analysis, square_analysis, debug=debug
        )

        # Step 5: Mark blurred tiles as boring in the main analysis
        blur_marked = 0
        for pos, is_blurred in blur_analysis.items():
            if is_blurred and square_analysis.get(pos, "good") == "good":
                square_analysis[pos] = (
                    "boring"  # Mark as boring (same as other boring types)
                )
                boring_reasons[pos] = "blur"
                blur_marked += 1

        return blur_marked

    def _calculate_adaptive_blur_threshold(
        self, img: Image.Image, debug: bool = False
    ) -> float:
        """Calculate brightness-adaptive threshold for the entire image"""
        if not self.use_brightness_adaptive:
            return self.base_blur_threshold

        # Convert image to numpy array and calculate mean brightness
        img_array = np.array(img)
        if img_array.ndim == 3:
            # Convert to grayscale for brightness calculation
            if img_array.shape[2] == 4:  # RGBA
                # Handle transparency by compositing over white background
                alpha = img_array[:, :, 3:4] / 255.0
                rgb_array = img_array[:, :, :3] * alpha + 255 * (1 - alpha)
                mean_brightness = np.mean(rgb_array)
            else:  # RGB
                mean_brightness = np.mean(img_array)
        else:  # Grayscale
            mean_brightness = np.mean(img_array)

        # Calculate brightness factor using power scaling
        if mean_brightness > 0:
            brightness_factor = (
                mean_brightness / self.target_brightness
            ) ** self.brightness_power
        else:
            brightness_factor = 0.1  # Very low threshold for completely black images

        adaptive_threshold = self.base_blur_threshold * brightness_factor

        if debug:
            print(
                f"    Brightness-adaptive threshold: brightness={mean_brightness:.1f}, "
                f"factor={brightness_factor:.3f}, threshold={adaptive_threshold:.1f}"
            )

        return float(adaptive_threshold)

    def _get_blur_variance(self, tile_array: np.ndarray) -> float:
        """Get the Laplacian variance for a tile (used for blur detection)"""
        if tile_array.ndim == 3:
            gray = cv2.cvtColor(tile_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile_array

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        return float(variance)

    def _refine_blur_decisions(
        self,
        blur_analysis: Dict[Tuple[int, int], bool],
        tile_variances: Dict[Tuple[int, int], float],
        adaptive_threshold: float,
        square_analysis: Dict[Tuple[int, int], str],
        nx: int,
        ny: int,
        debug: bool = False,
    ) -> int:
        """Apply context-aware refinement to blur decisions"""
        refinement_count = 0

        for row in range(ny):
            for col in range(nx):
                # Skip tiles that are already marked as boring
                if square_analysis.get((col, row), "good") != "good":
                    continue

                current_blur_state = blur_analysis.get((col, row), False)
                current_variance = tile_variances.get((col, row), 0)

                # Count blurry neighbors
                blurry_neighbors, total_neighbors = self._count_blurry_neighbors(
                    col, row, blur_analysis, square_analysis, nx, ny
                )
                non_blurry_neighbors = total_neighbors - blurry_neighbors

                refined_decision = None

                # Case 1: Currently blurry, but surrounded by 2+ non-blurry neighbors
                if (
                    current_blur_state
                    and non_blurry_neighbors >= self.min_neighbors_for_restrictive
                ):
                    # Use more restrictive threshold (harder to be blurry)
                    restrictive_threshold = (
                        adaptive_threshold * self.restrictive_multiplier
                    )
                    refined_decision = current_variance < restrictive_threshold

                # Case 2: Currently not blurry, but surrounded by 3+ blurry neighbors
                elif (
                    not current_blur_state
                    and blurry_neighbors >= self.min_neighbors_for_permissive
                ):
                    # Use more permissive threshold (easier to be blurry)
                    permissive_threshold = (
                        adaptive_threshold * self.permissive_multiplier
                    )
                    refined_decision = current_variance < permissive_threshold

                # Apply refinement if decision changed
                if (
                    refined_decision is not None
                    and refined_decision != current_blur_state
                ):
                    blur_analysis[(col, row)] = refined_decision
                    refinement_count += 1

        return refinement_count

    def _count_blurry_neighbors(
        self,
        col: int,
        row: int,
        blur_analysis: Dict[Tuple[int, int], bool],
        square_analysis: Dict[Tuple[int, int], str],
        nx: int,
        ny: int,
    ) -> Tuple[int, int]:
        """Count how many neighboring tiles are blurry"""
        neighbors = [
            (col - 1, row),  # left
            (col + 1, row),  # right
            (col, row - 1),  # top
            (col, row + 1),  # bottom
        ]

        blurry_count = 0
        valid_neighbors = 0

        for n_col, n_row in neighbors:
            if 0 <= n_col < nx and 0 <= n_row < ny:
                # Only count neighbors that are "good" tiles (not already boring)
                if square_analysis.get((n_col, n_row), "good") == "good":
                    valid_neighbors += 1
                    if blur_analysis.get((n_col, n_row), False):
                        blurry_count += 1

        return blurry_count, valid_neighbors

    def _remove_small_blur_groups(
        self,
        blur_analysis: Dict[Tuple[int, int], bool],
        nx: int,
        ny: int,
        debug: bool = False,
    ) -> int:
        """Remove blur marking from groups smaller than min_blur_group_size"""
        # Find all blurred tiles
        blurred_tiles = {pos for pos, is_blurred in blur_analysis.items() if is_blurred}
        visited = set()
        tiles_unmarked = 0

        # Find connected groups of blurred tiles
        for start_pos in blurred_tiles:
            if start_pos in visited:
                continue

            # Flood fill to find connected group
            group = set()
            queue = deque([start_pos])

            while queue:
                pos = queue.popleft()
                if pos in visited or pos not in blurred_tiles:
                    continue

                visited.add(pos)
                group.add(pos)

                # Add neighbors to queue
                col, row = pos
                neighbors = [
                    (col - 1, row),
                    (col + 1, row),
                    (col, row - 1),
                    (col, row + 1),
                ]
                for neighbor in neighbors:
                    n_col, n_row = neighbor
                    if (
                        0 <= n_col < nx
                        and 0 <= n_row < ny
                        and neighbor in blurred_tiles
                        and neighbor not in visited
                    ):
                        queue.append(neighbor)

            # If group is too small, remove blur marking
            if len(group) < self.min_blur_group_size:
                tiles_unmarked += len(group)
                for pos in group:
                    blur_analysis[pos] = False

        return tiles_unmarked

    def _apply_global_blur_validation(
        self,
        blur_analysis: Dict[Tuple[int, int], bool],
        square_analysis: Dict[Tuple[int, int], str],
        debug: bool = False,
    ) -> int:
        """Apply global validation - reject if too few squares are blurred"""
        # Only consider "good" tiles for the percentage calculation
        good_tiles = [
            pos for pos in blur_analysis if square_analysis.get(pos, "good") == "good"
        ]
        blurred_tiles = [pos for pos in good_tiles if blur_analysis[pos]]

        if len(good_tiles) > 0:
            blur_percentage = len(blurred_tiles) / len(good_tiles) * 100

            if blur_percentage < self.min_blur_percentage:
                if debug:
                    print(
                        f"    Global validation: {blur_percentage:.1f}% of good tiles blurred is too low, "
                        f"rejecting all blur detections"
                    )
                # Reject all blur detections
                for pos in good_tiles:
                    blur_analysis[pos] = False
                return 0
            else:
                if debug:
                    print(
                        f"    Global validation: {blur_percentage:.1f}% of good tiles blurred is acceptable"
                    )
                return len(blurred_tiles)
        else:
            if debug:
                print("    Global validation: No good tiles to validate")
            return 0

    def _detect_grayscale_tiles(
        self,
        img: Image.Image,
        grid_info: Dict,
        square_analysis: Dict[Tuple[int, int], str],
        boring_reasons: Dict[Tuple[int, int], str],
        debug: bool = False,
    ) -> int:
        """
        Detect grayscale tiles among remaining "good" tiles and mark them as boring

        This implements saturation-based grayscale detection for underground maps.
        Only runs on tiles that haven't been marked as boring by previous steps.

        Returns:
            Number of tiles marked as grayscale
        """
        if debug:
            print(f"  Starting grayscale detection...")

        nx, ny = grid_info["nx"], grid_info["ny"]
        x_edges = grid_info["x_edges"]
        y_edges = grid_info["y_edges"]

        # Step 1: Analyze only "good" tiles for grayscale content
        grayscale_analysis = {}
        good_tiles_checked = 0

        for row in range(ny):
            for col in range(nx):
                # Skip tiles that are already marked as boring
                if square_analysis.get((col, row), "good") != "good":
                    grayscale_analysis[(col, row)] = False
                    continue

                good_tiles_checked += 1

                # Extract square
                x0, x1 = x_edges[col], x_edges[col + 1]
                y0, y1 = y_edges[row], y_edges[row + 1]
                square = img.crop((x0, y0, x1, y1))
                square_array = np.array(square)

                # Check if square is grayscale using saturation
                is_grayscale = self._is_grayscale_square(square_array)
                grayscale_analysis[(col, row)] = is_grayscale

        if debug:
            print(
                f"    Analyzed {good_tiles_checked} non-boring tiles for grayscale content"
            )

        # Step 2: Global validation - apply 25% filter
        good_tiles = [
            pos for pos, analysis in square_analysis.items() if analysis == "good"
        ]
        grayscale_tiles = [
            pos for pos in good_tiles if grayscale_analysis.get(pos, False)
        ]

        if good_tiles:
            grayscale_percentage = len(grayscale_tiles) / len(good_tiles) * 100

            if grayscale_percentage < self.min_grayscale_percentage:
                if debug:
                    print(
                        f"    Global validation: {grayscale_percentage:.1f}% of non-boring tiles are grayscale - too low, "
                        f"rejecting all grayscale detections"
                    )
                # Reject all grayscale detections
                return 0
            else:
                if debug:
                    print(
                        f"    Global validation: {grayscale_percentage:.1f}% of non-boring tiles are grayscale - acceptable"
                    )

                # Mark grayscale tiles as boring
                for pos in grayscale_tiles:
                    square_analysis[pos] = "boring"
                    boring_reasons[pos] = "grayscale"

                return len(grayscale_tiles)
        else:
            if debug:
                print("    Global validation: No non-boring tiles to analyze")
            return 0

    def _is_grayscale_square(self, square_array: np.ndarray) -> bool:
        """
        Check if a square is grayscale using saturation-based detection

        Returns:
            bool: True if square is grayscale (low saturation)
        """
        # Primary criterion: Low saturation (HSV analysis)
        if square_array.ndim == 3:
            hsv = cv2.cvtColor(square_array, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            mean_saturation = float(saturation.mean())
        else:
            mean_saturation = 0.0  # Grayscale images have zero saturation

        # Check saturation threshold
        return mean_saturation <= self.max_saturation

    def get_boring_stats(self, square_analysis: Dict[Tuple[int, int], str]) -> Dict:
        """Get statistics about boring regions"""
        stats: Dict[str, Union[int, float]] = {
            "total_squares": len(square_analysis),
            "boring_squares": 0,
            "good_squares": 0,
        }

        for reason in square_analysis.values():
            if reason == "boring":
                stats["boring_squares"] += 1
            else:
                stats["good_squares"] += 1

        stats["total_boring"] = stats["boring_squares"]
        stats["boring_percentage"] = float(
            stats["boring_squares"] / stats["total_squares"] * 100
            if stats["total_squares"] > 0
            else 0.0
        )

        return stats
