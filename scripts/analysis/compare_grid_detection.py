#!/usr/bin/env python3
"""
Grid Detection Comparison Script

Compares the original morphological grid detection with the new brightness-based detection.
"""

import sys
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Import original grid detector
from battlemap_processor.core.grid_detector import GridDetector


def detect_grid_brightness_method(image_path, min_grid_size=100, max_grid_size=150):
    """
    Brightness-based grid detection method using analyze_brightness.py
    """
    try:
        # Run the analyze_brightness script and capture output
        import os

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, "analyze_brightness.py", image_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            return {
                "method": "Brightness Analysis",
                "success": False,
                "error": f"Script failed: {result.stderr}",
            }

        output = result.stdout

        # Parse the output to extract grid information
        grid_size = None
        confidence = 0.0
        cols = 0
        rows = 0
        vertical_detections = 0
        horizontal_detections = 0

        # Look for grid size
        grid_size_match = re.search(r"Square grid size: (\d+(?:\.\d+)?) pixels", output)
        if grid_size_match:
            grid_size = float(grid_size_match.group(1))

        # Look for confidence
        confidence_match = re.search(r"Confidence: (\d+(?:\.\d+)?)%", output)
        if confidence_match:
            confidence = float(confidence_match.group(1)) / 100.0

        # Look for grid dimensions
        cols_match = re.search(r"Grid columns: (\d+)", output)
        if cols_match:
            cols = int(cols_match.group(1))

        rows_match = re.search(r"Grid rows: (\d+)", output)
        if rows_match:
            rows = int(rows_match.group(1))

        # Look for detections count
        vertical_match = re.search(r"Found (\d+) vertical brightness changes", output)
        if vertical_match:
            vertical_detections = int(vertical_match.group(1))

        horizontal_match = re.search(
            r"Found (\d+) horizontal brightness changes", output
        )
        if horizontal_match:
            horizontal_detections = int(horizontal_match.group(1))

        # If grid size was found, we have a successful detection
        success = grid_size is not None and grid_size > 0

        return {
            "method": "Brightness Analysis",
            "grid_size": grid_size,
            "confidence": confidence,
            "cols": cols,
            "rows": rows,
            "vertical_detections": vertical_detections,
            "horizontal_detections": horizontal_detections,
            "success": success,
        }

    except subprocess.TimeoutExpired:
        return {
            "method": "Brightness Analysis",
            "success": False,
            "error": "Analysis timeout",
        }
    except Exception as e:
        return {"method": "Brightness Analysis", "success": False, "error": str(e)}


def convert_score_to_confidence(detection_score, grid_quality_factors=None):
    """
    Convert the original morphological detection score to a confidence percentage.

    Uses a logarithmic-inspired approach since the original method:
    1. Already filtered candidates and picked the best one
    2. Morphological scores have diminishing returns (1->2 more significant than 10->11)
    3. Any successful detection should have reasonable baseline confidence

    New approach:
    - Base confidence: 55% (method successfully found the best grid candidate)
    - Logarithmic boost based on score quality
    - Cap at 95% to leave room for improvement
    """
    if detection_score is None:
        return 0.0

    if detection_score <= 0:
        return 0.0

    # Base confidence for any successful detection
    base_confidence = 55.0

    # Logarithmic boost - scores have diminishing returns
    # log(1 + score) gives: score 1→69%, score 2→75%, score 5→85%, score 10→92%
    log_boost = np.log(1 + detection_score) * 15.0

    # Additional linear boost for very high scores
    if detection_score > 5:
        linear_boost = (detection_score - 5) * 1.0
    else:
        linear_boost = 0

    confidence = base_confidence + log_boost + linear_boost

    # Cap at 95% (never 100% confident)
    return min(95.0, max(0.0, confidence))


def visualize_comparison(image_path, result1, result2):
    """
    Create a side-by-side visualization comparing both grid detection methods
    """
    try:
        # Load the image
        img = Image.open(image_path)
        if img.mode == "RGBA":
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1])
            img = rgb_img

        img_array = np.array(img)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Original method
        ax1.imshow(img_array)
        ax1.set_title(
            "Original Morphological Detection", fontsize=14, fontweight="bold"
        )
        ax1.axis("off")

        if result1.get("success"):
            # Draw grid overlay for original method
            confidence1 = result1.get("confidence", 0) * 100

            # Calculate grid positions
            cell_width = result1["cell_width"]
            cell_height = result1["cell_height"]

            # Draw vertical lines
            for i in range(result1["nx"] + 1):
                x = i * cell_width
                ax1.axvline(x=x, color="red", alpha=0.7, linewidth=0.5, linestyle="-")

            # Draw horizontal lines
            for i in range(result1["ny"] + 1):
                y = i * cell_height
                ax1.axhline(y=y, color="red", alpha=0.7, linewidth=0.5, linestyle="-")

            # Add text overlay
            ax1.text(
                10,
                30,
                f'Grid: {result1["nx"]}x{result1["ny"]}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10,
                fontweight="bold",
            )
            ax1.text(
                10,
                60,
                f"Confidence: {confidence1:.1f}%",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax1.text(
                img.width // 2,
                img.height // 2,
                "NO GRID DETECTED",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="red",
                    alpha=0.8,
                    edgecolor="white",
                ),
                color="white",
            )

        # Plot 2: New brightness method
        ax2.imshow(img_array)
        ax2.set_title("New Brightness-Based Detection", fontsize=14, fontweight="bold")
        ax2.axis("off")

        if result2.get("success"):
            # Draw grid overlay for new method
            confidence2 = result2["confidence"] * 100
            grid_size = result2["grid_size"]

            # Draw vertical lines
            for i in range(result2["cols"] + 1):
                x = i * grid_size
                ax2.axvline(x=x, color="blue", alpha=0.7, linewidth=0.5, linestyle="-")

            # Draw horizontal lines
            for i in range(result2["rows"] + 1):
                y = i * grid_size
                ax2.axhline(y=y, color="blue", alpha=0.7, linewidth=0.5, linestyle="-")

            # Add text overlay
            ax2.text(
                10,
                30,
                f'Grid: {result2["cols"]}x{result2["rows"]}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10,
                fontweight="bold",
            )
            ax2.text(
                10,
                60,
                f"Confidence: {confidence2:.1f}%",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax2.text(
                img.width // 2,
                img.height // 2,
                "NO GRID DETECTED",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="red",
                    alpha=0.8,
                    edgecolor="white",
                ),
                color="white",
            )

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"ERROR: Error creating visualization: {e}")


def compare_grid_methods(image_path):
    """
    Compare both grid detection methods on the same image
    """
    print(f"Comparing grid detection methods on: {Path(image_path).name}")
    print("=" * 80)

    # Method 1: Original morphological detection
    print("\nMETHOD 1: Original Morphological Detection")
    print("-" * 50)

    try:
        detector = GridDetector()
        img = Image.open(image_path)

        # Convert RGBA to RGB if needed
        if img.mode == "RGBA":
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1])
            img = rgb_img

        result1 = detector.detect_grid(img, str(image_path))

        if result1:
            confidence1 = convert_score_to_confidence(result1["score"])
            print(f"SUCCESS")
            print(f"   Grid dimensions: {result1['nx']}x{result1['ny']}")
            print(
                f"   Cell size: {result1['cell_width']:.1f}x{result1['cell_height']:.1f} pixels"
            )
            print(
                f"   Confidence: {confidence1:.1f}% (raw score: {result1['score']:.3f})"
            )
            print(f"   Size category: {result1['size_px']} pixels")

            # Store confidence and success flag for comparison
            result1["confidence"] = confidence1 / 100.0
            result1["success"] = True

            if "filename_dimensions" in result1:
                print(f"   Filename dims: {result1['filename_dimensions']}")
                print(f"   Filename match: {result1.get('filename_match', 'N/A')}")
        else:
            print(f"FAILED - No grid detected")
            result1 = {"success": False}

    except Exception as e:
        print(f"ERROR: {e}")
        result1 = {"success": False, "error": str(e)}

    # Method 2: New brightness-based detection
    print("\nMETHOD 2: New Brightness-Based Detection")
    print("-" * 50)

    result2 = detect_grid_brightness_method(image_path)

    if result2["success"]:
        print(f"SUCCESS")
        print(f"   Grid dimensions: {result2['cols']}x{result2['rows']}")
        print(f"   Cell size: {result2['grid_size']:.0f} pixels (square)")
        print(f"   Confidence: {result2['confidence']:.1%}")
        print(
            f"   Detections: {result2['vertical_detections']} vertical, {result2['horizontal_detections']} horizontal"
        )
    else:
        print(f"FAILED")
        if "error" in result2:
            print(f"   Error: {result2['error']}")

    # Comparison summary
    print("\n🔄 COMPARISON SUMMARY")
    print("-" * 50)

    if result1.get("success") and result2.get("success"):
        # Both methods succeeded
        orig_grid = f"{result1['nx']}x{result1['ny']}"
        new_grid = f"{result2['cols']}x{result2['rows']}"

        orig_size = (result1["cell_width"] + result1["cell_height"]) / 2
        new_size = result2["grid_size"]

        orig_confidence = result1.get("confidence", 0) * 100
        new_confidence = result2["confidence"] * 100

        print(f"📐 Grid dimensions - Original: {orig_grid}, New: {new_grid}")
        print(
            f"📏 Average cell size - Original: {orig_size:.0f}px, New: {new_size:.0f}px"
        )
        print(f"🎯 Confidence comparison:")
        print(f"   Original method: {orig_confidence:.1f}%")
        print(f"   New method: {new_confidence:.1f}%")

        # Determine which method is more confident
        if orig_confidence > new_confidence + 5:
            print(
                f"   🏆 Original method is more confident (+{orig_confidence - new_confidence:.1f}%)"
            )
        elif new_confidence > orig_confidence + 5:
            print(
                f"   🏆 New method is more confident (+{new_confidence - orig_confidence:.1f}%)"
            )
        else:
            print(f"   🤝 Similar confidence levels")

        # Check agreement
        if orig_grid == new_grid and abs(orig_size - new_size) < 10:
            print("METHODS AGREE - Both detected similar grids!")
        elif abs(orig_size - new_size) < 20:
            print(
                "WARNING: METHODS PARTIALLY AGREE - Similar cell sizes, different grid counts"
            )
        else:
            print("ERROR: METHODS DISAGREE - Significantly different results")

    elif result1.get("success"):
        orig_confidence = result1.get("confidence", 0) * 100
        print("Only original method succeeded")
        print(f"   Detected: {result1['nx']}x{result1['ny']} grid")
        print(f"   Confidence: {orig_confidence:.1f}%")

    elif result2.get("success"):
        print("Only new method succeeded")
        print(f"   Detected: {result2['cols']}x{result2['rows']} grid")
        print(f"   Confidence: {result2['confidence']:.1%}")

    else:
        print("ERROR: Both methods failed to detect a grid")

    # Show visual comparison
    print("\nOpening visual comparison...")
    visualize_comparison(image_path, result1, result2)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_grid_detection.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)

    compare_grid_methods(image_path)
