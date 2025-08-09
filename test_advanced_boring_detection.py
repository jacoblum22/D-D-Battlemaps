"""
Test script for Advanced Boring Detection

Tests the new AdvancedBoringDetector that identifies two types of boring squares:
1. Individual black squares
2. Large connected regions of uniform content (water, sky, etc.)

Visualizes the results using matplotlib with color-coded overlays.

Supports:
- Local image files
- Local directories
- Google Drive shared folders (recursive analysis)
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import tempfile

# Add the project root to path so we can import our modules
project_root = os.path.dirname(
    os.path.abspath(__file__)
)  # Test is at project root level
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from battlemap_processor.core.grid_detector import GridDetector
from battlemap_processor.core.advanced_boring_detector import AdvancedBoringDetector
from battlemap_processor.core.optimal_tile_placer import OptimalTilePlacer

# Try to import Google Drive handler
try:
    from battlemap_processor.core.google_drive_handler import GoogleDriveHandler
    from battlemap_processor.core.smart_image_selector import SmartImageSelector

    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    print(
        "⚠️ Google Drive functionality not available. Install: pip install google-api-python-client google-auth-oauthlib"
    )


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

    # Skip images with transparency (tokens, UI elements, etc.)
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        # Check if image actually has transparent pixels
        if img.mode == "RGBA":
            img_array = np.array(img)
            alpha_channel = img_array[:, :, 3]
            has_transparency = np.any(alpha_channel < 255)
        elif img.mode == "LA":
            img_array = np.array(img)
            alpha_channel = img_array[:, :, 1]
            has_transparency = np.any(alpha_channel < 255)
        else:
            has_transparency = True  # Has transparency info

        if has_transparency:
            print(
                "Image has transparent parts - skipping analysis (likely token or UI element)"
            )
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
    square_analysis, boring_reasons = boring_detector.analyze_image_regions(
        img, grid_info, debug=False
    )

    # Get statistics
    stats = boring_detector.get_boring_stats(square_analysis)

    print(f"\n=== Analysis Results ===")
    print(f"Total squares: {stats['total_squares']}")
    print(f"Boring squares: {stats['boring_squares']}")
    print(f"Good squares: {stats['good_squares']}")
    print(f"Total boring: {stats['total_boring']} ({stats['boring_percentage']:.1f}%)")

    # Run tile placement analysis (same as pipeline)
    print("\nRunning tile placement analysis...")
    tile_placer = (
        OptimalTilePlacer()
    )  # Use default settings (12x12 tiles, 40% max boring)
    placed_tiles = tile_placer.find_optimal_placements(
        grid_info=grid_info,
        square_analysis=square_analysis,
        debug=True,  # Enable debug to see what's happening
    )

    print(f"\n=== Tile Placement Results ===")
    print(f"Tiles placed: {len(placed_tiles)}")
    if placed_tiles:
        avg_boring = sum(tile.boring_percentage for tile in placed_tiles) / len(
            placed_tiles
        )
        print(f"Average boring per tile: {avg_boring:.1f}%")
        print(
            f"Total squares covered: {len(placed_tiles) * 144}"
        )  # 12x12 = 144 squares per tile

    if show_popup and (stats["boring_squares"] > 0 or placed_tiles):
        visualize_advanced_boring_detection(
            img, grid_info, square_analysis, boring_reasons, stats, placed_tiles
        )
    elif stats["boring_squares"] == 0 and not placed_tiles:
        print("No boring squares detected and no tiles placed - skipping visualization")

    return stats


def visualize_advanced_boring_detection(
    img, grid_info, square_analysis, boring_reasons, stats, placed_tiles=None
):
    """Create a visual popup showing the advanced boring detection results with color-coded boring types and tile placements"""

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

    # Validate edge arrays to prevent bounds errors
    if len(x_edges) < nx + 1:
        print(f"❌ Invalid x_edges: expected {nx + 1} edges, got {len(x_edges)}")
        return
    if len(y_edges) < ny + 1:
        print(f"❌ Invalid y_edges: expected {ny + 1} edges, got {len(y_edges)}")
        return

    for x in x_edges:
        ax1.axvline(x, color="white", alpha=0.5, linewidth=0.5)
    for y in y_edges:
        ax1.axhline(y, color="white", alpha=0.5, linewidth=0.5)

    # Create overlay for boring squares - convert to RGB if needed
    img_array = np.array(img)
    if img_array.ndim == 3 and img_array.shape[2] == 4:  # RGBA
        # Convert RGBA to RGB by compositing over white background
        rgb_array = np.zeros(
            (img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8
        )
        alpha = img_array[:, :, 3:4] / 255.0
        rgb_array = (img_array[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        overlay = rgb_array
    else:
        overlay = img_array.copy()

    # Define colors for different types of boring detection
    colors = {
        "black": [255, 0, 0],  # Red for black squares (better visibility)
        "uniform": [0, 100, 255],  # Blue for uniform regions
        "gap_fill": [255, 165, 0],  # Orange for gap filling
        "grayscale": [128, 128, 128],  # Gray for grayscale detection
        "blur": [128, 0, 128],  # Purple for blur detection
        "good": None,  # No overlay for good squares
    }

    # Count squares by type for statistics
    type_counts = {}
    for reason in boring_reasons.values():
        type_counts[reason] = type_counts.get(reason, 0) + 1

    # Apply colored overlays
    for row in range(ny):
        for col in range(nx):
            reason = boring_reasons.get((col, row), "good")

            if reason in colors and colors[reason] is not None:
                # Get square boundaries
                x0, x1 = x_edges[col], x_edges[col + 1]
                y0, y1 = y_edges[row], y_edges[row + 1]

                # Create colored overlay (semi-transparent)
                color = np.array(colors[reason])
                overlay[y0:y1, x0:x1] = (
                    overlay[y0:y1, x0:x1] * 0.6 + color * 0.4
                ).astype(np.uint8)

    # Show overlay
    ax2.imshow(overlay)
    ax2.set_title("Color-Coded Boring Detection + Tile Placements")
    ax2.axis("off")

    # Draw tile placement rectangles if available
    if placed_tiles:
        for i, tile in enumerate(placed_tiles):
            # Convert grid coordinates to pixel coordinates
            x_start = x_edges[tile.start_col]
            y_start = y_edges[tile.start_row]
            x_end = x_edges[min(tile.start_col + tile.size, nx)]
            y_end = y_edges[min(tile.start_row + tile.size, ny)]

            width = x_end - x_start
            height = y_end - y_start

            # Create green rectangle outline for tile placement
            rect = patches.Rectangle(
                (x_start, y_start),
                width,
                height,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
                alpha=0.8,
            )
            ax2.add_patch(rect)

            # Add tile number label
            ax2.text(
                x_start + width / 2,
                y_start + height / 2,
                str(i + 1),
                color="lime",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            )

    # Create legend with counts
    legend_elements = []
    color_map = {
        "black": "#FF0000",  # Red for black squares (better visibility)
        "uniform": "#0064FF",
        "gap_fill": "#FFA500",
        "grayscale": "#808080",
        "blur": "#800080",
        "good": "#90EE90",  # Light green for good squares
    }

    for reason, count in type_counts.items():
        if reason in color_map:
            legend_elements.append(
                patches.Patch(
                    color=color_map[reason],
                    label=f'{reason.replace("_", " ").title()} ({count})',
                )
            )

    # Add tile placement to legend if tiles were placed
    if placed_tiles:
        legend_elements.append(
            patches.Patch(color="lime", label=f"Tile Placements ({len(placed_tiles)})")
        )

    if legend_elements:
        ax2.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

    # Print detailed statistics
    print(f"\n=== Detailed Boring Detection Breakdown ===")
    total_squares = sum(type_counts.values())
    for reason, count in sorted(type_counts.items()):
        percentage = count / total_squares * 100 if total_squares > 0 else 0
        print(
            f"{reason.replace('_', ' ').title()}: {count} squares ({percentage:.1f}%)"
        )

    # Print tile placement details
    if placed_tiles:
        print(f"\n=== Tile Placement Details ===")
        for i, tile in enumerate(placed_tiles, 1):
            print(
                f"Tile {i}: Position ({tile.start_col}, {tile.start_row}), "
                f"Boring: {tile.boring_percentage:.1f}% ({tile.boring_count}/{tile.boring_count + tile.good_count})"
            )
    else:
        print(f"\n=== No Tiles Placed ===")
        print(
            "No suitable tile positions found (all potential positions exceed boring threshold)"
        )

    plt.tight_layout()
    plt.show()


def analyze_google_drive_folder(drive_url, show_popup=True):
    """Analyze all images in a Google Drive folder recursively"""

    if not GOOGLE_DRIVE_AVAILABLE:
        print("❌ Google Drive functionality not available!")
        print(
            "Please install required packages: pip install google-api-python-client google-auth-oauthlib"
        )
        return

    print(f"\n=== Analyzing Google Drive Folder ===")
    print(f"URL: {drive_url}")

    try:
        # Initialize Google Drive handler
        if not GOOGLE_DRIVE_AVAILABLE:
            raise ImportError("Google Drive handler not available")

        handler = GoogleDriveHandler()

        # Extract folder ID from URL
        folder_id = handler.extract_file_id(drive_url)
        if not folder_id:
            print("❌ Could not extract folder ID from URL")
            return

        print(f"Folder ID: {folder_id}")

        # Verify it's a folder
        if not handler.is_folder(folder_id):
            print("❌ The provided URL does not point to a folder")
            return

        # Get folder info
        folder_info = handler.get_file_info(folder_id)
        print(f"Folder name: {folder_info.get('name', 'Unknown')}")

        # List all images in the folder recursively
        print("🔍 Scanning for images...")
        images = handler.list_images_in_folder(folder_id, recursive=True)

        if not images:
            print("No images found in the folder")
            return

        print(f"Found {len(images)} total images")

        # Apply smart image selection to group and prioritize variants
        print("🧠 Applying smart image selection...")
        selector = SmartImageSelector()

        # Convert to format expected by SmartImageSelector
        image_list_for_selector = []
        for img in images:
            image_list_for_selector.append(
                {
                    "path": img["id"],  # Use Google Drive ID as path for now
                    "filename": img["name"],
                }
            )

        # Group images by variants (instead of just selecting optimal ones)
        variant_groups = selector.group_image_variants(image_list_for_selector)

        print(
            f"Smart selection: Found {len(variant_groups)} unique images with {len(images)} total files"
        )
        if len(variant_groups) < len(images):
            total_variants = sum(len(variants) for variants in variant_groups.values())
            print(
                f"Will process {len(variant_groups)} images, with fallback for {total_variants - len(variant_groups)} variants"
            )

        # Create prioritized processing list
        def prioritize_variants(variants):
            """Sort variants by selection priority (best first)"""
            if len(variants) <= 1:
                return variants

            # Apply the same logic as select_best_variant but return sorted list
            # Priority 1: Check for filename subsequence (prefer longer names)
            for i, variant_a in enumerate(variants):
                for j, variant_b in enumerate(variants):
                    if i != j:
                        from pathlib import Path

                        base_a = Path(variant_a.filename).stem
                        base_b = Path(variant_b.filename).stem

                        # If A is a subsequence of B, B should come first (longer name)
                        if selector._is_subsequence(base_a, base_b):
                            # Move variant_b to front
                            result = [variant_b] + [
                                v for v in variants if v != variant_b
                            ]
                            return result
                        elif selector._is_subsequence(base_b, base_a):
                            # Move variant_a to front
                            result = [variant_a] + [
                                v for v in variants if v != variant_a
                            ]
                            return result

            # Priority 2 & 3: Apply dimension-based rules
            has_dimensions = any(v.has_dimensions for v in variants)

            if has_dimensions:
                # Prefer gridless when dimensions available
                gridless_variants = [v for v in variants if v.is_gridless]
                other_variants = [v for v in variants if not v.is_gridless]

                # Within gridless, prefer ones with dimensions
                gridless_with_dims = [v for v in gridless_variants if v.has_dimensions]
                gridless_without_dims = [
                    v for v in gridless_variants if not v.has_dimensions
                ]

                return gridless_with_dims + gridless_without_dims + other_variants
            else:
                # Prefer gridded when no dimensions available
                gridded_variants = [v for v in variants if v.is_gridded]
                other_variants = [v for v in variants if not v.is_gridded]

                return gridded_variants + other_variants

        # Process each variant group with fallback logic
        total_groups = len(variant_groups)
        processed_groups = 0

        # Create temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Created temporary directory: {temp_dir}")
            processed_count = 0
            skipped_count = 0
            error_count = 0
            fallback_count = 0

            for group_index, (base_name, variants) in enumerate(
                variant_groups.items(), 1
            ):
                print(
                    f"\n=== Processing Group {group_index}/{total_groups}: {base_name} ==="
                )

                # Prioritize variants within this group
                prioritized_variants = prioritize_variants(variants)
                print(
                    f"Found {len(prioritized_variants)} variants: {[v.filename for v in prioritized_variants]}"
                )

                # Try each variant until one succeeds
                processed_this_group = False

                for variant_index, variant in enumerate(prioritized_variants):
                    # Find the original image info by filename
                    image_info = None
                    for img in images:
                        if img["name"] == variant.filename:
                            image_info = img
                            break

                    if not image_info:
                        print(f"⚠️ Could not find image info for {variant.filename}")
                        continue

                    variant_label = (
                        "primary"
                        if variant_index == 0
                        else f"fallback #{variant_index}"
                    )
                    print(f"\n--- Trying {variant_label}: {image_info['name']} ---")

                    # Generate temporary file path
                    temp_path = os.path.join(
                        temp_dir,
                        f"temp_image_{group_index}_{variant_index}.{image_info['name'].split('.')[-1]}",
                    )

                    try:
                        # Download the image
                        print(f"📥 Downloading to: {temp_path}")
                        if not handler.download_file(image_info["id"], temp_path):
                            print(f"❌ Failed to download {image_info['name']}")
                            continue

                        # Analyze the image
                        result = test_advanced_boring_detection(
                            temp_path, show_popup=show_popup
                        )

                        if result is not None:
                            processed_count += 1
                            processed_this_group = True
                            if variant_index > 0:
                                fallback_count += 1
                                print(f"✅ Success with fallback variant!")
                            else:
                                print(f"✅ Success with primary choice!")
                        else:
                            print(
                                f"⏭️ Skipped {variant.filename} (transparency/no grid)"
                            )
                            skipped_count += 1

                        # Clean up the downloaded file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            print(f"🗑️ Deleted temporary file: {temp_path}")
                        else:
                            print(
                                f"⚠️ Temporary file not found for deletion: {temp_path}"
                            )

                        # If we successfully processed this variant, break out of variant loop
                        if processed_this_group:
                            break

                    except Exception as e:
                        print(f"❌ Error processing {image_info['name']}: {e}")
                        error_count += 1
                        # Clean up on error
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            print(f"🗑️ Deleted temporary file after error: {temp_path}")
                        continue

                # Update group counter
                if processed_this_group:
                    processed_groups += 1
                else:
                    print(f"❌ All variants for {base_name} failed")

            # Print summary
            print(f"\n=== Google Drive Analysis Complete ===")
            print(f"📁 Temporary directory will be cleaned up: {temp_dir}")
            print(f"Total images found: {len(images)}")
            print(f"Image groups processed: {processed_groups}/{total_groups}")
            print(f"Successfully processed: {processed_count}")
            print(f"Fallback successes: {fallback_count}")
            print(f"Skipped (transparency/no grid): {skipped_count}")
            print(f"Errors: {error_count}")

        print(f"🧹 Temporary directory cleaned up automatically")

    except Exception as e:
        print(f"❌ Error accessing Google Drive: {e}")
        print("Make sure you have:")
        print("1. google_drive_credentials.json in the project root")
        print("2. Completed OAuth authentication")
        return


def cleanup_leftover_temp_files():
    """Find and clean up any leftover temporary files from previous runs"""
    print("\n🧹 Searching for leftover temporary files...")

    import glob
    import shutil

    temp_base = tempfile.gettempdir()
    print(f"Searching in: {temp_base}")

    # Look for our temp image files
    pattern = os.path.join(temp_base, "**", "temp_image_*.*")
    leftover_files = glob.glob(pattern, recursive=True)

    if not leftover_files:
        print("✅ No leftover temporary files found")
        return

    print(f"Found {len(leftover_files)} leftover temporary files:")

    # Group by directory
    directories_to_clean = set()
    for file_path in leftover_files:
        print(f"  📄 {file_path}")
        directories_to_clean.add(os.path.dirname(file_path))

    # Ask user before cleaning
    print(f"\nFound files in {len(directories_to_clean)} directories:")
    for dir_path in directories_to_clean:
        print(f"  📁 {dir_path}")

    response = input("\nDelete these leftover files? (y/N): ").strip().lower()
    if response != "y":
        print("Cleanup cancelled")
        return

    # Clean up files and empty directories
    files_deleted = 0
    dirs_deleted = 0

    for file_path in leftover_files:
        try:
            os.remove(file_path)
            files_deleted += 1
            print(f"🗑️ Deleted: {file_path}")
        except Exception as e:
            print(f"❌ Failed to delete {file_path}: {e}")

    # Try to remove empty directories
    for dir_path in directories_to_clean:
        try:
            # Only remove if directory is empty
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                dirs_deleted += 1
                print(f"🗑️ Removed empty directory: {dir_path}")
            else:
                print(f"📁 Directory not empty, keeping: {dir_path}")
        except Exception as e:
            print(f"❌ Failed to remove directory {dir_path}: {e}")

    print(
        f"\n✅ Cleanup complete: {files_deleted} files and {dirs_deleted} directories removed"
    )


def main():
    """Main test function"""

    # Test images
    test_images_dir = Path("test_images")

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # Check if it's a Google Drive URL
        if "drive.google.com" in arg:
            analyze_google_drive_folder(
                arg, show_popup=True
            )  # Show popups to display matplotlib visualizations
        else:
            # Test specific image
            image_path = arg
            if not Path(image_path).exists():
                print(f"Error: Image '{image_path}' not found!")
                return

            test_advanced_boring_detection(image_path, show_popup=True)
    else:
        # Test all images in test_images directory
        if not test_images_dir.exists():
            print(f"Test images directory '{test_images_dir}' not found!")
            print(
                "Usage: python test_advanced_boring_detection.py [image_path_or_drive_url]"
            )
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
