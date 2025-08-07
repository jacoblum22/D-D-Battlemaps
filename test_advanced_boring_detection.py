"""
Test script for Advanced Boring Detection

Tests the new AdvancedBoringDetector that identi    # Define colors for different types of boring squares
    colors = {
        'black': [255, 0, 0],                    # Red for black squares
        'large_uniform_region': [0, 150, 255]   # Blue for large uniform regions
    }
1. Individual black squares
2. Large connected regions of uniform content (water, sky, etc.)
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from battlemap_processor.core.grid_detector import GridDetector
from battlemap_processor.core.advanced_boring_detector import AdvancedBoringDetector


def test_advanced_boring_detection(image_path, show_popup=True):
    """Test advanced boring detection on an image"""

    print(f"\n=== Testing Advanced Boring Detection on {Path(image_path).name} ===")

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

    # Run advanced boring detection
    boring_detector = AdvancedBoringDetector()

    print("Running advanced boring analysis...")
    square_analysis = boring_detector.analyze_image_regions(img, grid_info, debug=False)

    # Get statistics
    stats = boring_detector.get_boring_stats(square_analysis)

    print(f"\n=== Analysis Results ===")
    print(f"Total squares: {stats['total_squares']}")
    print(f"Black squares: {stats['black_squares']}")
    print(f"Large uniform regions: {stats['large_uniform_regions']}")
    print(f"Good squares: {stats['good_squares']}")
    print(f"Total boring: {stats['total_boring']} ({stats['boring_percentage']:.1f}%)")

    if show_popup:
        visualize_advanced_boring_detection(img, grid_info, square_analysis, stats)


def visualize_advanced_boring_detection(img, grid_info, square_analysis, stats):
    """Create a visual popup showing the advanced boring detection results"""

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f'Advanced Boring Detection Results - {stats["total_boring"]} boring squares ({stats["boring_percentage"]:.1f}%)',
        fontsize=14,
        fontweight="bold",
    )

    # Show original image
    ax1.imshow(img)
    ax1.set_title("Original Image with Grid")
    ax1.axis("off")

    # Draw grid lines
    nx, ny = grid_info["nx"], grid_info["ny"]
    x_edges = grid_info["x_edges"]
    y_edges = grid_info["y_edges"]

    for x in x_edges:
        ax1.axvline(x, color="white", alpha=0.5, linewidth=0.5)
    for y in y_edges:
        ax1.axhline(y, color="white", alpha=0.5, linewidth=0.5)

    # Create overlay for boring squares
    overlay = np.array(img)

    # Define colors for different types of boring squares
    colors = {
        "black": [255, 0, 0],  # Red for black squares
        "large_uniform_region": [0, 150, 255],  # Blue for large uniform regions
    }

    # Apply colored overlays
    for row in range(ny):
        for col in range(nx):
            reason = square_analysis.get((col, row), "good")

            if reason in colors:
                # Get square boundaries
                x0, x1 = x_edges[col], x_edges[col + 1]
                y0, y1 = y_edges[row], y_edges[row + 1]

                # Create colored overlay (semi-transparent)
                color = colors[reason]
                overlay[y0:y1, x0:x1] = (
                    overlay[y0:y1, x0:x1] * 0.6 + np.array(color) * 0.4
                )

    # Show overlay
    ax2.imshow(overlay)
    ax2.set_title("Boring Square Detection")
    ax2.axis("off")

    # Create legend
    legend_elements = []
    if stats["black_squares"] > 0:
        legend_elements.append(
            patches.Patch(
                color="red", label=f'Black squares ({stats["black_squares"]})'
            )
        )
    if stats["large_uniform_regions"] > 0:
        legend_elements.append(
            patches.Patch(
                color="#0096FF",
                label=f'Large uniform regions ({stats["large_uniform_regions"]})',
            )
        )
    if stats["good_squares"] > 0:
        legend_elements.append(
            patches.Patch(
                color="lightgray", label=f'Good squares ({stats["good_squares"]})'
            )
        )

    if legend_elements:
        ax2.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

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

        test_advanced_boring_detection(image_path, show_popup=True)
    else:
        # Test all images in test_images directory
        if not test_images_dir.exists():
            print(f"Test images directory '{test_images_dir}' not found!")
            print("Usage: python test_advanced_boring_detection.py [image_path]")
            return

        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]:
            image_files.extend(test_images_dir.glob(ext))

        if not image_files:
            print(f"No image files found in '{test_images_dir}'!")
            return

        print(f"Found {len(image_files)} test images")

        for image_path in sorted(image_files):
            test_advanced_boring_detection(image_path, show_popup=True)

            # Wait for user input to continue to next image
            if len(image_files) > 1:
                input("\nPress Enter to continue to next image (or Ctrl+C to stop)...")


if __name__ == "__main__":
    main()
