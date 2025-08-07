"""
Tile Extractor for extracting grid-aligned tiles from battlemap images
"""

import logging
from typing import List, Dict, Any, TypedDict, Union
from PIL import Image
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)


class GridInfo(TypedDict):
    """Type definition for grid detection results"""

    nx: int  # Number of columns
    ny: int  # Number of rows
    x_edges: List[int]  # X coordinates of grid edges
    y_edges: List[int]  # Y coordinates of grid edges


@dataclass
class TileInfo:
    """Information about an extracted tile"""

    image: Image.Image
    grid_x: int  # Starting grid column
    grid_y: int  # Starting grid row
    pixel_x: int  # Starting pixel x
    pixel_y: int  # Starting pixel y
    squares_wide: int  # Number of grid squares wide
    squares_tall: int  # Number of grid squares tall


class TileExtractor:
    """Extracts grid-aligned tiles from battlemap images"""

    def __init__(self, tile_size: int = 512):
        self.tile_size = tile_size

    def _create_tile_from_grid_coords(
        self,
        pil_img: Image.Image,
        grid_info: Union[GridInfo, Dict[str, Any]],
        start_grid_x: int,
        start_grid_y: int,
        squares_per_tile: int,
    ) -> TileInfo:
        """
        Create a tile from grid coordinates (common logic for both methods)

        Args:
            pil_img: Source image
            grid_info: Grid detection results
            start_grid_x: Starting grid column
            start_grid_y: Starting grid row
            squares_per_tile: Number of grid squares per tile

        Returns:
            TileInfo object with the extracted and resized tile
        """
        x_edges = grid_info["x_edges"]
        y_edges = grid_info["y_edges"]

        end_grid_x = start_grid_x + squares_per_tile
        end_grid_y = start_grid_y + squares_per_tile

        # Convert to pixel coordinates
        start_pixel_x = x_edges[start_grid_x]
        start_pixel_y = y_edges[start_grid_y]
        end_pixel_x = x_edges[end_grid_x]
        end_pixel_y = y_edges[end_grid_y]

        # Extract and resize tile
        tile_crop = pil_img.crop(
            (start_pixel_x, start_pixel_y, end_pixel_x, end_pixel_y)
        )
        tile_resized = tile_crop.resize(
            (self.tile_size, self.tile_size), Image.Resampling.LANCZOS
        )

        # Create tile info
        return TileInfo(
            image=tile_resized,
            grid_x=start_grid_x,
            grid_y=start_grid_y,
            pixel_x=start_pixel_x,
            pixel_y=start_pixel_y,
            squares_wide=squares_per_tile,
            squares_tall=squares_per_tile,
        )

    def extract_tiles(
        self,
        pil_img: Image.Image,
        grid_info: Union[GridInfo, Dict[str, Any]],
        squares_per_tile: int = 12,
    ) -> List[TileInfo]:
        """
        Extract grid-aligned tiles from an image

        Args:
            pil_img: Source image
            grid_info: Grid detection results
            squares_per_tile: Number of grid squares per tile (width and height)

        Returns:
            List of TileInfo objects
        """
        nx_total = grid_info["nx"]
        ny_total = grid_info["ny"]

        # Check if we can extract any tiles of the requested size
        if nx_total < squares_per_tile or ny_total < squares_per_tile:
            logger.warning(
                f"Grid is {nx_total}x{ny_total}, too small for {squares_per_tile}x{squares_per_tile} tiles"
            )
            return []

        # Calculate how many tiles we can fit
        tiles_x = nx_total // squares_per_tile
        tiles_y = ny_total // squares_per_tile

        # Calculate centering offset (to center the tiling grid)
        remaining_x = nx_total - (tiles_x * squares_per_tile)
        remaining_y = ny_total - (tiles_y * squares_per_tile)

        # For odd remainders, use (remainder-1) for centering to avoid mid-cell seams
        offset_x = ((remaining_x - (remaining_x & 1)) // 2) if remaining_x > 0 else 0
        offset_y = ((remaining_y - (remaining_y & 1)) // 2) if remaining_y > 0 else 0

        logger.debug(
            f"Extracting {tiles_x}x{tiles_y} tiles with offset ({offset_x},{offset_y})"
        )

        tiles = []

        for row in range(tiles_y):
            for col in range(tiles_x):
                # Calculate grid coordinates
                start_grid_x = offset_x + (col * squares_per_tile)
                start_grid_y = offset_y + (row * squares_per_tile)

                # Use helper method for common crop/resize/TileInfo logic
                tile_info = self._create_tile_from_grid_coords(
                    pil_img, grid_info, start_grid_x, start_grid_y, squares_per_tile
                )
                tiles.append(tile_info)

        return tiles

    def extract_tiles_with_overlap(
        self,
        pil_img: Image.Image,
        grid_info: Union[GridInfo, Dict[str, Any]],
        squares_per_tile: int = 12,
        overlap_squares: int = 2,
    ) -> List[TileInfo]:
        """
        Extract overlapping grid-aligned tiles (for future use)

        This method allows tiles to overlap by a specified number of squares,
        which can be useful for ensuring we don't miss interesting features
        that span tile boundaries.
        """
        nx_total = grid_info["nx"]
        ny_total = grid_info["ny"]
        x_edges = grid_info["x_edges"]
        y_edges = grid_info["y_edges"]

        if nx_total < squares_per_tile or ny_total < squares_per_tile:
            return []

        tiles = []

        # Step size is reduced by overlap
        step_x = max(1, squares_per_tile - overlap_squares)
        step_y = max(1, squares_per_tile - overlap_squares)

        # Generate all possible tile positions
        grid_x_positions = list(range(0, nx_total - squares_per_tile + 1, step_x))
        grid_y_positions = list(range(0, ny_total - squares_per_tile + 1, step_y))

        # Make sure we include the last possible position
        if grid_x_positions[-1] != nx_total - squares_per_tile:
            grid_x_positions.append(nx_total - squares_per_tile)
        if grid_y_positions[-1] != ny_total - squares_per_tile:
            grid_y_positions.append(ny_total - squares_per_tile)

        for row_idx, start_grid_y in enumerate(grid_y_positions):
            for col_idx, start_grid_x in enumerate(grid_x_positions):
                # Use helper method for common crop/resize/TileInfo logic
                tile_info = self._create_tile_from_grid_coords(
                    pil_img, grid_info, start_grid_x, start_grid_y, squares_per_tile
                )
                tiles.append(tile_info)

        return tiles
