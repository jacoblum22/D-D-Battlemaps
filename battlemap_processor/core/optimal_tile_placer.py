"""
Optimal Tile Placer for battlemap images

This module finds optimal placements for 12x12 tiles while:
1. Avoiding overlaps between tiles
2. Ensuring no tile has more than 50% boring squares
3. Maximizing coverage of non-boring squares
"""

import logging
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class TilePlacement:
    """Information about a placed tile"""
    start_col: int  # Starting grid column
    start_row: int  # Starting grid row
    size: int  # Tile size (12x12 squares)
    boring_count: int  # Number of boring squares in tile
    good_count: int  # Number of good squares in tile
    boring_percentage: float  # Percentage of boring squares


class OptimalTilePlacer:
    """
    Places 12x12 tiles optimally on a battlemap to maximize non-boring square coverage
    """

    def __init__(self, tile_size: int = 12, max_boring_percentage: float = 40.0):
        self.tile_size = tile_size
        self.max_boring_percentage = max_boring_percentage
        self.total_squares_per_tile = tile_size * tile_size

    def find_optimal_placements(
        self,
        grid_info: Dict[str, Any],
        square_analysis: Dict[Tuple[int, int], str],
        debug: bool = False,
    ) -> List[TilePlacement]:
        """
        Find optimal tile placements using a greedy algorithm

        Args:
            grid_info: Grid detection results with nx, ny, x_edges, y_edges
            square_analysis: Dict mapping (col, row) -> reason ('black', 'large_uniform_region', 'good')
            debug: Enable debug output

        Returns:
            List of TilePlacement objects representing optimal tile positions
        """
        nx, ny = grid_info["nx"], grid_info["ny"]
        
        if debug:
            print(f"\n=== Optimal Tile Placement ===")
            print(f"Grid size: {nx}x{ny} squares")
            print(f"Tile size: {self.tile_size}x{self.tile_size} squares")
            print(f"Max boring percentage: {self.max_boring_percentage}%")

        # Generate all possible tile positions
        possible_positions = self._generate_possible_positions(nx, ny, debug)
        
        if not possible_positions:
            if debug:
                print("❌ No valid tile positions possible!")
            return []

        # Evaluate each position
        position_scores = self._evaluate_positions(
            possible_positions, square_analysis, debug
        )

        # Filter out positions with too many boring squares
        valid_positions = [
            (pos, score, analysis) for pos, score, analysis in position_scores
            if analysis.boring_percentage <= self.max_boring_percentage
        ]

        if debug:
            print(f"Valid positions (≤{self.max_boring_percentage}% boring): {len(valid_positions)}")

        if not valid_positions:
            if debug:
                print("❌ No positions meet the boring square criteria!")
            return []

        # Sort by good square count (descending) to prioritize high-value tiles
        valid_positions.sort(key=lambda x: x[1], reverse=True)

        # Greedily place tiles without overlaps
        placed_tiles = self._greedy_placement(valid_positions, debug)

        if debug:
            total_good_covered = sum(tile.good_count for tile in placed_tiles)
            total_squares_covered = len(placed_tiles) * self.total_squares_per_tile
            print(f"\n=== Final Results ===")
            print(f"Tiles placed: {len(placed_tiles)}")
            print(f"Total squares covered: {total_squares_covered}")
            print(f"Good squares covered: {total_good_covered}")

        return placed_tiles

    def _generate_possible_positions(
        self, nx: int, ny: int, debug: bool = False
    ) -> List[Tuple[int, int]]:
        """Generate all possible top-left positions for tiles"""
        positions = []
        
        # Calculate maximum possible starting positions
        max_start_col = nx - self.tile_size
        max_start_row = ny - self.tile_size

        if max_start_col < 0 or max_start_row < 0:
            if debug:
                print(f"❌ Grid too small: need at least {self.tile_size}x{self.tile_size}, got {nx}x{ny}")
            return positions

        for row in range(max_start_row + 1):
            for col in range(max_start_col + 1):
                positions.append((col, row))

        if debug:
            print(f"Generated {len(positions)} possible tile positions")

        return positions

    def _evaluate_positions(
        self,
        positions: List[Tuple[int, int]],
        square_analysis: Dict[Tuple[int, int], str],
        debug: bool = False,
    ) -> List[Tuple[Tuple[int, int], int, TilePlacement]]:
        """
        Evaluate each tile position and return scores

        Returns:
            List of (position, good_square_count, tile_analysis) tuples
        """
        evaluations = []

        for i, (start_col, start_row) in enumerate(positions):
            analysis = self._analyze_tile_position(
                start_col, start_row, square_analysis
            )
            evaluations.append(((start_col, start_row), analysis.good_count, analysis))

            if debug and i < 5:  # Show first few evaluations
                print(f"  Position ({start_col},{start_row}): "
                      f"{analysis.good_count} good, {analysis.boring_count} boring "
                      f"({analysis.boring_percentage:.1f}%)")

        return evaluations

    def _analyze_tile_position(
        self,
        start_col: int,
        start_row: int,
        square_analysis: Dict[Tuple[int, int], str],
    ) -> TilePlacement:
        """Analyze a specific tile position for boring square count"""
        boring_count = 0
        good_count = 0

        # Check all squares in the tile
        for row in range(start_row, start_row + self.tile_size):
            for col in range(start_col, start_col + self.tile_size):
                reason = square_analysis.get((col, row), "good")
                
                if reason in ["black", "large_uniform_region"]:
                    boring_count += 1
                else:
                    good_count += 1

        boring_percentage = (boring_count / self.total_squares_per_tile) * 100

        return TilePlacement(
            start_col=start_col,
            start_row=start_row,
            size=self.tile_size,
            boring_count=boring_count,
            good_count=good_count,
            boring_percentage=boring_percentage,
        )

    def _greedy_placement(
        self,
        valid_positions: List[Tuple[Tuple[int, int], int, TilePlacement]],
        debug: bool = False,
    ) -> List[TilePlacement]:
        """
        Use greedy algorithm to place tiles without overlaps

        Args:
            valid_positions: List of (position, score, analysis) sorted by score descending

        Returns:
            List of placed tiles
        """
        placed_tiles = []
        occupied_squares: Set[Tuple[int, int]] = set()

        for i, ((start_col, start_row), score, analysis) in enumerate(valid_positions):
            # Check if this tile would overlap with any already placed tiles
            tile_squares = self._get_tile_squares(start_col, start_row)
            
            if not tile_squares.intersection(occupied_squares):
                # No overlap, place the tile
                placed_tiles.append(analysis)
                occupied_squares.update(tile_squares)
                
                if debug:
                    print(f"  Placed tile {len(placed_tiles)} at ({start_col},{start_row}): "
                          f"{score} good squares ({analysis.boring_percentage:.1f}% boring)")
            elif debug and i < 10:  # Show first few rejected tiles
                print(f"  Rejected tile at ({start_col},{start_row}): overlaps existing tile")

        return placed_tiles

    def _get_tile_squares(self, start_col: int, start_row: int) -> Set[Tuple[int, int]]:
        """Get all squares covered by a tile at the given position"""
        squares = set()
        for row in range(start_row, start_row + self.tile_size):
            for col in range(start_col, start_col + self.tile_size):
                squares.add((col, row))
        return squares

    def get_placement_stats(
        self, 
        placed_tiles: List[TilePlacement],
        square_analysis: Dict[Tuple[int, int], str]
    ) -> Dict[str, Any]:
        """Get statistics about the tile placement"""
        if not placed_tiles:
            return {
                "tiles_placed": 0,
                "total_squares_covered": 0,
                "good_squares_covered": 0,
                "boring_squares_covered": 0,
                "coverage_efficiency": 0.0,
                "avg_boring_percentage": 0.0,
            }

        total_squares_covered = len(placed_tiles) * self.total_squares_per_tile
        good_squares_covered = sum(tile.good_count for tile in placed_tiles)
        boring_squares_covered = sum(tile.boring_count for tile in placed_tiles)
        
        # Calculate total good squares in the image
        total_good_squares = sum(
            1 for reason in square_analysis.values() if reason == "good"
        )
        
        coverage_efficiency = (
            good_squares_covered / total_good_squares * 100
            if total_good_squares > 0 else 0.0
        )

        avg_boring_percentage = (
            sum(tile.boring_percentage for tile in placed_tiles) / len(placed_tiles)
        )

        return {
            "tiles_placed": len(placed_tiles),
            "total_squares_covered": total_squares_covered,
            "good_squares_covered": good_squares_covered,
            "boring_squares_covered": boring_squares_covered,
            "coverage_efficiency": coverage_efficiency,
            "avg_boring_percentage": avg_boring_percentage,
            "total_good_squares_in_image": total_good_squares,
        }
