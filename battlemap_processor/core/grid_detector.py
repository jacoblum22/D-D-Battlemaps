"""
Grid Detection for D&D Battlemap images

Based on morphological blackhat operations to detect grid lines.
Uses the approach from the original code which works very well.
"""

import numpy as np
import cv2
from typing import Optional, Dict, Tuple, List
from PIL import Image


class GridDetector:
    """Detects grid structure in battlemap images"""

    def __init__(self):
        # Grid detection parameters (from original code)
        self.min_cell_px = 100  # Minimum cell size in pixels
        self.max_cell_px = 180  # Maximum cell size in pixels
        self.step_px = 10  # Test every 10 pixels
        self.edge_tolerance_px = 2  # Pixel slack at edges
        self.max_blackhat_kernel_size = 31  # Maximum morphological kernel size
        self.min_blackhat_kernel_size = 5  # Minimum morphological kernel size
        self.kernel_size_ratio = 0.02  # Kernel size as fraction of min image dimension
        self.sample_radius = 2  # Sampling radius around grid lines

    def _get_adaptive_kernel_size(self, image_height: int, image_width: int) -> int:
        """Calculate adaptive kernel size based on image dimensions"""
        min_dimension = min(image_height, image_width)
        adaptive_size = int(min_dimension * self.kernel_size_ratio)

        # Clamp to reasonable bounds
        kernel_size = max(
            self.min_blackhat_kernel_size,
            min(adaptive_size, self.max_blackhat_kernel_size),
        )

        return kernel_size

    def detect_grid(self, pil_img: Image.Image) -> Optional[Dict]:
        """
        Detect grid structure in a battlemap image

        Args:
            pil_img: PIL Image in RGB format

        Returns:
            Dict with grid info if detected, None otherwise
            Dict contains: nx, ny, cell_width, cell_height, x_edges, y_edges
        """
        # Convert to grayscale
        rgb_array = np.array(pil_img)
        if rgb_array.ndim == 3:
            gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb_array

        H, W = gray.shape

        # Generate candidate cell sizes
        upper_limit = min(self.max_cell_px, min(W, H))
        cell_sizes = list(range(self.min_cell_px, upper_limit + 1, self.step_px))

        # Find candidates that fit the image edges
        edge_candidates = []
        for cell_size in cell_sizes:
            nx = int(round(W / cell_size))
            ny = int(round(H / cell_size))

            if nx < 2 or ny < 2:
                continue

            # Check if grid fits within tolerance
            width_error = abs(W - nx * cell_size)
            height_error = abs(H - ny * cell_size)

            if (
                width_error <= self.edge_tolerance_px
                and height_error <= self.edge_tolerance_px
            ):
                edge_candidates.append((cell_size, nx, ny, width_error, height_error))

        if not edge_candidates:
            return None

        # Use morphological operations to score candidates
        v_map, h_map = self._create_blackhat_maps(gray)

        # Score each candidate
        scored_candidates = []
        for cell_size, nx, ny, w_err, h_err in edge_candidates:
            score = self._score_grid_candidate(v_map, h_map, W, H, nx, ny)

            # Add small penalty for edge errors and very dense grids
            penalty = -0.05 * (w_err + h_err) - 0.001 * (nx + ny)
            final_score = score + penalty

            scored_candidates.append(
                (final_score, score, cell_size, nx, ny, w_err, h_err)
            )

        # Choose best candidate
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        _, raw_score, best_size, nx, ny, w_err, h_err = scored_candidates[0]

        # Calculate actual cell dimensions
        cell_width = W / float(nx)
        cell_height = H / float(ny)

        # Generate grid edges
        x_edges = self._generate_grid_edges(W, nx)
        y_edges = self._generate_grid_edges(H, ny)

        return {
            "nx": nx,
            "ny": ny,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "score": raw_score,
            "size_px": best_size,
        }

    def _create_blackhat_maps(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create morphological blackhat maps to highlight grid lines"""
        # Get adaptive kernel size based on image dimensions
        H, W = gray.shape
        kernel_size = self._get_adaptive_kernel_size(H, W)

        # Vertical lines kernel (tall and thin)
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        # Horizontal lines kernel (wide and thin)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))

        # Apply blackhat operation
        v_map = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kv).astype(np.float32)
        h_map = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kh).astype(np.float32)

        # Normalize to [0, 1]
        v_map = (v_map - v_map.min()) / (np.ptp(v_map) + 1e-6)
        h_map = (h_map - h_map.min()) / (np.ptp(h_map) + 1e-6)

        return v_map, h_map

    def _score_grid_candidate(
        self, v_map: np.ndarray, h_map: np.ndarray, W: int, H: int, nx: int, ny: int
    ) -> float:
        """Score a grid candidate based on alignment and contrast"""
        # Get standard deviations for normalization
        _, v_std = v_map.mean(), v_map.std() + 1e-6
        _, h_std = h_map.mean(), h_map.std() + 1e-6

        # Calculate alignment scores (how well grid lines align with bright areas)
        align_x = self._alignment_score(v_map, W, nx, "vertical") / v_std
        align_y = self._alignment_score(h_map, H, ny, "horizontal") / h_std

        # Calculate contrast scores (grid lines vs mid-cell areas)
        contrast_x = self._contrast_score(v_map, W, nx, "vertical") / v_std
        contrast_y = self._contrast_score(h_map, H, ny, "horizontal") / h_std

        return align_x + align_y + contrast_x + contrast_y

    def _alignment_score(
        self, intensity_map: np.ndarray, size: int, n_cells: int, direction: str
    ) -> float:
        """Score how well expected grid positions align with high intensity areas.

        Uses vectorized NumPy operations for performance on large grids.
        """
        if n_cells < 2:
            return -1e9

        step = size / n_cells
        # Pre-allocate array and use vectorized operations
        positions = np.round(np.arange(n_cells + 1) * step).astype(int)

        # Vectorized sampling for better performance
        if direction == "vertical":
            # Sample columns around vertical grid lines using vectorized operations
            height = intensity_map.shape[0]

            # Calculate column ranges for all positions at once
            col_starts = np.maximum(0, positions - self.sample_radius)
            col_ends = np.minimum(
                intensity_map.shape[1], positions + self.sample_radius + 1
            )

            # Extract samples for all positions using list comprehension with vectorized slicing
            samples = np.array(
                [
                    intensity_map[:, col_starts[i] : col_ends[i]].mean()
                    for i in range(len(positions))
                ]
            )
        else:  # horizontal
            # Sample rows around horizontal grid lines using vectorized operations
            width = intensity_map.shape[1]

            # Calculate row ranges for all positions at once
            row_starts = np.maximum(0, positions - self.sample_radius)
            row_ends = np.minimum(
                intensity_map.shape[0], positions + self.sample_radius + 1
            )

            # Extract samples for all positions using list comprehension with vectorized slicing
            samples = np.array(
                [
                    intensity_map[row_starts[i] : row_ends[i], :].mean()
                    for i in range(len(positions))
                ]
            )

        return float(np.mean(samples))

    def _contrast_score(
        self, intensity_map: np.ndarray, size: int, n_cells: int, direction: str
    ) -> float:
        """Score contrast between grid lines and mid-cell areas.

        Uses vectorized NumPy operations for performance on large grids.
        """
        if n_cells < 2:
            return -1e9

        step = size / n_cells

        # Grid line positions and mid-cell positions using pre-allocated arrays
        on_positions = np.round(np.arange(n_cells + 1) * step).astype(int)
        mid_positions = np.round((np.arange(n_cells) + 0.5) * step).astype(int)

        # Vectorized sampling for better performance
        if direction == "vertical":
            # Calculate column ranges for all positions at once
            on_col_starts = np.maximum(0, on_positions - self.sample_radius)
            on_col_ends = np.minimum(
                intensity_map.shape[1], on_positions + self.sample_radius + 1
            )

            mid_col_starts = np.maximum(0, mid_positions - self.sample_radius)
            mid_col_ends = np.minimum(
                intensity_map.shape[1], mid_positions + self.sample_radius + 1
            )

            # Extract samples using vectorized operations
            on_samples = np.array(
                [
                    intensity_map[:, on_col_starts[i] : on_col_ends[i]].mean()
                    for i in range(len(on_positions))
                ]
            )

            mid_samples = np.array(
                [
                    intensity_map[:, mid_col_starts[i] : mid_col_ends[i]].mean()
                    for i in range(len(mid_positions))
                ]
            )
        else:  # horizontal
            # Calculate row ranges for all positions at once
            on_row_starts = np.maximum(0, on_positions - self.sample_radius)
            on_row_ends = np.minimum(
                intensity_map.shape[0], on_positions + self.sample_radius + 1
            )

            mid_row_starts = np.maximum(0, mid_positions - self.sample_radius)
            mid_row_ends = np.minimum(
                intensity_map.shape[0], mid_positions + self.sample_radius + 1
            )

            # Extract samples using vectorized operations
            on_samples = np.array(
                [
                    intensity_map[on_row_starts[i] : on_row_ends[i], :].mean()
                    for i in range(len(on_positions))
                ]
            )

            mid_samples = np.array(
                [
                    intensity_map[mid_row_starts[i] : mid_row_ends[i], :].mean()
                    for i in range(len(mid_positions))
                ]
            )

        # Return difference (grid lines should be brighter than mid-cells)
        return float(np.mean(on_samples) - np.mean(mid_samples))

    def _generate_grid_edges(self, length: int, n_cells: int) -> List[int]:
        """Generate integer grid edge positions that minimize drift"""
        edges = [0]
        step = length / float(n_cells)
        acc = 0.0

        for i in range(1, n_cells):
            acc += step
            edge = int(round(acc))
            # Ensure monotonic increase (no duplicates)
            if edge <= edges[-1]:
                edge = edges[-1] + 1
            edges.append(edge)

        edges.append(length)
        return edges
