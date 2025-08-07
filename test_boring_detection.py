"""
Test script for boring square detection with inline visualization
"""

import os
import sys
import argparse
from PIL import Image, ImageDraw
import numpy as np

# Try to import matplotlib for display
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visual display disabled.")

# Add the project root to path so we can import our modules
project_root = os.path.dirname(
    os.path.abspath(__file__)
)  # Test is at project root level
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from battlemap_processor.core.grid_detector import GridDetector
from battlemap_processor.core.boring_detector import BoringTileDetector


def test_boring_detection(image_path):
    """Test boring square detection and show results"""
    print(f"Testing boring square detection on: {image_path}")

    # Load image
    try:
        img = Image.open(image_path).convert("RGB")
        print(f"Image size: {img.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Detect grid
    print("Running grid detection...")
    detector = GridDetector()
    grid_info = detector.detect_grid(img)

    if not grid_info:
        print("✗ No grid detected - cannot proceed with square analysis")
        return

    nx, ny = grid_info["nx"], grid_info["ny"]
    print(f"✓ Grid detected: {nx}x{ny} squares")

    # Analyze each individual square
    print(f"Analyzing {nx * ny} individual grid squares...")
    boring_detector = BoringTileDetector()

    x_edges = grid_info["x_edges"]
    y_edges = grid_info["y_edges"]

    boring_count = 0
    good_count = 0
    square_analysis = {}  # (col, row) -> is_boring

    for row in range(ny):
        for col in range(nx):
            # Extract individual square
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]

            square = img.crop((x0, y0, x1, y1))

            # Analyze this square
            is_boring = boring_detector.is_boring_tile(square)
            square_analysis[(col, row)] = is_boring

            if is_boring:
                boring_count += 1
            else:
                good_count += 1

    print(f"\n📊 Results:")
    print(f"   Grid size: {nx} cols × {ny} rows = {nx * ny} squares")
    print(f"   Good squares: {good_count}")
    print(f"   Boring squares: {boring_count}")
    print(f"   Boring percentage: {boring_count/(boring_count+good_count):.1%}")

    # Show visual representation
    if HAS_MATPLOTLIB:
        show_boring_visualization(img, grid_info, square_analysis)
    else:
        print("\n(Visual display not available - matplotlib not installed)")
        print("Run: pip install matplotlib")


def show_boring_visualization(img, grid_info, square_analysis):
    """Display the image with boring squares highlighted"""
    print("Showing visualization...")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Original image
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Image with boring squares highlighted
    vis_img = np.array(img).copy()

    x_edges = grid_info["x_edges"]
    y_edges = grid_info["y_edges"]

    # Highlight boring squares with red overlay
    for (col, row), is_boring in square_analysis.items():
        if is_boring:
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]

            # Add red tint to boring squares
            vis_img[y0:y1, x0:x1, 0] = np.minimum(255, vis_img[y0:y1, x0:x1, 0] + 100)

    ax2.imshow(vis_img)
    ax2.set_title("Boring Squares (Red Tint)")
    ax2.axis("off")

    # Draw grid lines on both images
    for ax in [ax1, ax2]:
        for x in x_edges:
            ax.axvline(x, color="gray", alpha=0.3, linewidth=0.5)
        for y in y_edges:
            ax.axhline(y, color="gray", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test boring square detection on a battlemap"
    )
    parser.add_argument("image_path", help="Path to the image file")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        sys.exit(1)

    print("D&D Battlemap Boring Square Detection Test")
    print("=" * 50)
    print()

    test_boring_detection(args.image_path)
