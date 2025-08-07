"""
Test script for Optimal Tile Placement

Tests the OptimalTilePlacer that finds the best positions for 12x12 tiles
while avoiding boring squares and overlaps.

Visualizes the results using matplotlib with color-coded overlays.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add the project root to path so we can import our modules
project_root = os.path.dirname(
    os.path.abspath(__file__)
)  # Test is at project root level
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from battlemap_processor.core.grid_detector import GridDetector
from battlemap_processor.core.advanced_boring_detector import AdvancedBoringDetector
from battlemap_processor.core.optimal_tile_placer import OptimalTilePlacer


def test_optimal_tile_placement(image_path, show_popup=True):
    """Test optimal tile placement on an image"""

    print(f"\n=== Testing Optimal Tile Placement on {Path(image_path).name} ===")

    # Load image
    try:
        img = Image.open(image_path)
        print(f"Loaded image: {img.size[0]}x{img.size[1]} pixels")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Detect grid
    grid_detector = GridDetector()
    print("Detecting grid...")
    grid_info = grid_detector.detect_grid(img)

    if grid_info is None:
        print("❌ No grid detected!")
        return

    print(f"✅ Grid detected: {grid_info['nx']}x{grid_info['ny']} squares")

    # Run boring detection
    boring_detector = AdvancedBoringDetector()
    print("Running boring detection...")
    square_analysis = boring_detector.analyze_image_regions(img, grid_info, debug=False)

    # Get boring stats
    boring_stats = boring_detector.get_boring_stats(square_analysis)
    print(f"Boring squares: {boring_stats['total_boring']} ({boring_stats['boring_percentage']:.1f}%)")

    # Run optimal tile placement
    tile_placer = OptimalTilePlacer(tile_size=12)  # Uses default max_boring_percentage=40.0
    print("Finding optimal tile placements...")
    placed_tiles = tile_placer.find_optimal_placements(
        grid_info, square_analysis, debug=True
    )

    # Get placement stats
    placement_stats = tile_placer.get_placement_stats(placed_tiles, square_analysis)

    print(f"\n=== Tile Placement Results ===")
    print(f"Tiles placed: {placement_stats['tiles_placed']}")
    print(f"Total squares covered: {placement_stats['total_squares_covered']}")
    print(f"Good squares covered: {placement_stats['good_squares_covered']} / {placement_stats['total_good_squares_in_image']} ({placement_stats['coverage_efficiency']:.1f}%)")
    print(f"Average boring percentage per tile: {placement_stats['avg_boring_percentage']:.1f}%")

    if show_popup:
        visualize_tile_placement(img, grid_info, square_analysis, placed_tiles, placement_stats)


def visualize_tile_placement(img, grid_info, square_analysis, placed_tiles, placement_stats):
    """Create a visual popup showing the tile placement results"""

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(
        f'Optimal Tile Placement - {placement_stats["tiles_placed"]} tiles placed '
        f'({placement_stats["coverage_efficiency"]:.1f}% good square coverage)',
        fontsize=16,
        fontweight="bold",
    )

    # Show original image with boring squares highlighted
    ax1.imshow(img)
    ax1.set_title("Image with Boring Squares Highlighted")
    ax1.axis("off")

    # Draw grid lines
    nx, ny = grid_info["nx"], grid_info["ny"]
    x_edges = grid_info["x_edges"]
    y_edges = grid_info["y_edges"]

    # Validate edge arrays
    if len(x_edges) < nx + 1 or len(y_edges) < ny + 1:
        print(f"❌ Invalid edge arrays")
        return

    # Draw light grid lines
    for x in x_edges:
        ax1.axvline(x, color="white", alpha=0.3, linewidth=0.5)
    for y in y_edges:
        ax1.axhline(y, color="white", alpha=0.3, linewidth=0.5)

    # Create overlay for boring squares (similar to advanced boring detection test)
    overlay1 = np.array(img)
    
    # Highlight boring squares in red
    for row in range(ny):
        for col in range(nx):
            reason = square_analysis.get((col, row), "good")
            
            if reason in ["black", "large_uniform_region"]:
                # Get square boundaries
                x0, x1 = x_edges[col], x_edges[col + 1]
                y0, y1 = y_edges[row], y_edges[row + 1]
                
                # Red overlay for boring squares
                overlay1[y0:y1, x0:x1] = overlay1[y0:y1, x0:x1] * 0.6 + np.array([255, 0, 0]) * 0.4

    ax1.imshow(overlay1)

    # Show image with tile placements
    ax2.imshow(img)
    ax2.set_title("Image with Placed Tiles")
    ax2.axis("off")

    # Create overlay for tile visualization
    overlay2 = np.array(img)

    # First, lightly highlight boring squares
    for row in range(ny):
        for col in range(nx):
            reason = square_analysis.get((col, row), "good")
            
            if reason in ["black", "large_uniform_region"]:
                x0, x1 = x_edges[col], x_edges[col + 1]
                y0, y1 = y_edges[row], y_edges[row + 1]
                # Light red tint for boring squares
                overlay2[y0:y1, x0:x1] = overlay2[y0:y1, x0:x1] * 0.8 + np.array([255, 0, 0]) * 0.2

    ax2.imshow(overlay2)

    # Draw tile boundaries
    if len(placed_tiles) > 0:
        tile_colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(placed_tiles)))  # Different colors for each tile
    else:
        tile_colors = []
    
    for i, tile in enumerate(placed_tiles):
        # Get tile boundaries in pixels
        x0 = x_edges[tile.start_col]
        y0 = y_edges[tile.start_row]
        x1 = x_edges[tile.start_col + tile.size]
        y1 = y_edges[tile.start_row + tile.size]
        
        # Create rectangle for tile boundary
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=3,
            edgecolor=tile_colors[i],
            facecolor='none',
            alpha=0.8
        )
        ax2.add_patch(rect)
        
        # Add tile number label
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        ax2.text(
            center_x, center_y, str(i + 1),
            fontsize=14, fontweight='bold',
            ha='center', va='center',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=tile_colors[i], alpha=0.8)
        )

    # Create legend for first plot
    legend_elements1 = [
        patches.Patch(color="red", alpha=0.4, label="Boring squares"),
        patches.Patch(color="lightgray", label="Good squares"),
    ]
    ax1.legend(handles=legend_elements1, loc="upper right", bbox_to_anchor=(1, 1))

    # Create legend for second plot
    legend_elements2 = [
        patches.Patch(color="red", alpha=0.2, label="Boring squares (light)"),
        patches.Patch(color="cyan", label="Tile boundaries"),
    ]
    ax2.legend(handles=legend_elements2, loc="upper right", bbox_to_anchor=(1, 1))

    # Add stats text box
    stats_text = (
        f"Tiles Placed: {placement_stats['tiles_placed']}\n"
        f"Squares Covered: {placement_stats['total_squares_covered']}\n"
        f"Good Squares Covered: {placement_stats['good_squares_covered']}\n"
        f"Coverage Efficiency: {placement_stats['coverage_efficiency']:.1f}%\n"
        f"Avg Boring %: {placement_stats['avg_boring_percentage']:.1f}%"
    )
    
    ax2.text(
        0.02, 0.02, stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


def main():
    """Main test function"""

    # Test images
    test_images_dir = Path("test_images")

    if len(sys.argv) > 1:
        # Test specific image
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"Error: Image '{image_path}' not found!")
            return

        test_optimal_tile_placement(image_path, show_popup=True)
    else:
        # Test all images in test_images directory
        if not test_images_dir.exists():
            print(f"Test images directory '{test_images_dir}' not found!")
            print("Usage: python test_optimal_tile_placement.py [image_path]")
            return

        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]:
            image_files.extend(test_images_dir.glob(ext))

        if not image_files:
            print(f"No image files found in '{test_images_dir}'!")
            return

        print(f"Found {len(image_files)} test images")

        for image_path in sorted(image_files):
            test_optimal_tile_placement(image_path, show_popup=True)

            # Wait for user input to continue to next image
            if len(image_files) > 1:
                input("\nPress Enter to continue to next image (or Ctrl+C to stop)...")


if __name__ == "__main__":
    main()
