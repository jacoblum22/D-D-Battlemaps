#!/usr/bin/env python3
"""
Image Brightness Analysis Tool

Analyzes the brightness patterns in an ima                if diff_amount >= min_drop:
                    potential_lines.append(i)
                    # Only show first few detections to reduce noise
                    if len(potential_lines) <= 5:  # Reduced from 10 to 5
                        peak_dip = "Peak" if detect_peaks else "Dip"
                        vs_text = "vs" if not detect_peaks else "vs"
                        print(f"   {peak_dip} at {i}: {current_value:.1f} {vs_text} avg {local_average:.1f} (diff: {diff_amount:.1f})")help debug     # Detect brightness changes for horizontal lines (rows)
    horizontal_dips = find_brightness_changes(row_averages, window_size=20, min_drop=4.0)

    if len(horizontal_dips) > 0: detection.
Creates graphs showing average brightness for each column and row of pixels.
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def analyze_image_brightness(image_path: str, show_plots: bool = True):
    """Analyze brightness patterns in an image"""
    print(f"🔍 Analyzing brightness patterns in: {image_path}")

    # Load image
    img = Image.open(image_path)
    print(f"📐 Image size: {img.width}x{img.height} pixels")

    # Convert to grayscale for brightness analysis
    gray_img = img.convert("L")
    img_array = np.array(gray_img)

    print(f"🎨 Brightness range: {img_array.min()} - {img_array.max()}")
    print(f"📊 Average brightness: {img_array.mean():.1f}")

    # Calculate column averages (vertical patterns - for vertical grid lines)
    column_averages = np.mean(img_array, axis=0)

    # Calculate row averages (horizontal patterns - for horizontal grid lines)
    row_averages = np.mean(img_array, axis=1)

    # Create plots only if show_plots is True
    if show_plots:
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Brightness Analysis: {Path(image_path).name}", fontsize=16)

        # Plot 1: Original image
        ax1.imshow(gray_img, cmap="gray")
        ax1.set_title("Original Image (Grayscale)")
        ax1.axis("off")

        # Plot 2: Column averages (shows vertical grid lines)
        ax2.plot(column_averages)
        ax2.set_title("Column Brightness Averages (Vertical Grid Detection)")
        ax2.set_xlabel("Column (X position)")
        ax2.set_ylabel("Average Brightness")
        ax2.grid(True, alpha=0.3)

    # Add some analysis for vertical lines
    column_std = np.std(column_averages)
    column_mean = np.mean(column_averages)
    print(f"📊 Column brightness variation (std): {column_std:.2f}")
    print(f"🔍 Testing both dark and bright grid line detection...")

    # Find potential grid lines using simple dip detection
    from scipy.signal import find_peaks

    def find_brightness_changes(
        brightness_values,
        window_size=20,
        min_drop=4.0,
        group_distance=30,
        detect_peaks=False,
    ):
        """
        Find local minima (dips) or maxima (peaks) in brightness:
        For each position, check if it's at least min_drop points different than
        the average brightness within window_size pixels around it.
        Then group nearby detections and keep only the most extreme point from each group.

        Args:
            detect_peaks: If True, detect brightness peaks instead of dips
        """
        potential_lines = []

        detection_type = "peaks" if detect_peaks else "dips"
        comparison = "above" if detect_peaks else "below"
        # Reduced logging - only show summary
        print(
            f"🔍 Looking for brightness {detection_type} (>= {min_drop} points {comparison} local average)"
        )
        print(f"   Using window size of {window_size} pixels around each point")

        for i in range(len(brightness_values)):
            # Define the window around this position
            start_idx = max(0, i - window_size)
            end_idx = min(len(brightness_values), i + window_size + 1)

            # Calculate local average (excluding the current position)
            window_values = []
            for j in range(start_idx, end_idx):
                if j != i:  # Exclude current position from average
                    window_values.append(brightness_values[j])

            if len(window_values) > 0:
                local_average = np.mean(window_values)
                current_value = brightness_values[i]

                # Check if current value is significantly different from local average
                if detect_peaks:
                    # For peaks, current value should be higher than average
                    diff_amount = current_value - local_average
                else:
                    # For dips, current value should be lower than average
                    diff_amount = local_average - current_value

                if diff_amount >= min_drop:
                    potential_lines.append(i)
                    if len(potential_lines) <= 10:  # Debug first 10 detections
                        peak_dip = "Peak" if detect_peaks else "Dip"
                        vs_text = "vs" if not detect_peaks else "vs"
                        print(
                            f"   {peak_dip} at {i}: {current_value:.1f} {vs_text} avg {local_average:.1f} (diff: {diff_amount:.1f})"
                        )

        print(f"   Found {len(potential_lines)} raw {detection_type}")

        # Group nearby detections and keep only the most extreme point from each group
        if len(potential_lines) == 0:
            return np.array(potential_lines)

        potential_lines = sorted(potential_lines)
        grouped_lines = []
        current_group = [potential_lines[0]]

        for i in range(1, len(potential_lines)):
            # If this point is within group_distance of the last point in current group
            if potential_lines[i] - current_group[-1] <= group_distance:
                current_group.append(potential_lines[i])
            else:
                # Finish current group and find the most extreme point
                if detect_peaks:
                    # For peaks, find the highest value
                    extreme_idx = max(current_group, key=lambda x: brightness_values[x])
                else:
                    # For dips, find the lowest value
                    extreme_idx = min(current_group, key=lambda x: brightness_values[x])
                grouped_lines.append(extreme_idx)
                current_group = [potential_lines[i]]

        # Handle the last group
        if current_group:
            if detect_peaks:
                extreme_idx = max(current_group, key=lambda x: brightness_values[x])
            else:
                extreme_idx = min(current_group, key=lambda x: brightness_values[x])
            grouped_lines.append(extreme_idx)

        print(
            f"   After grouping nearby points (within {group_distance} pixels): {len(grouped_lines)} final {detection_type}"
        )
        return np.array(grouped_lines)

    def detect_grid_pattern(detected_points, max_dimension, tolerance=3):
        """
        Detect the most common spacing between detected points and generate predicted grid lines
        """
        if len(detected_points) < 2:
            return None, []

        # Calculate spacings between consecutive points
        spacings = []
        for i in range(len(detected_points) - 1):
            spacing = detected_points[i + 1] - detected_points[i]
            spacings.append(spacing)

        if not spacings:
            return None, []

        print(f"   Spacings between consecutive points: {spacings[:10]}...")

        # Group similar spacings (within tolerance)
        spacing_groups = {}
        for spacing in spacings:
            # Find if this spacing fits in an existing group
            found_group = False
            for group_center in spacing_groups:
                if abs(spacing - group_center) <= tolerance:
                    spacing_groups[group_center].append(spacing)
                    found_group = True
                    break

            if not found_group:
                spacing_groups[spacing] = [spacing]

        # Find the group with the most spacings (most common pattern)
        if not spacing_groups:
            return None, []

        most_common_group = max(
            spacing_groups.keys(), key=lambda k: len(spacing_groups[k])
        )
        avg_spacing = np.mean(spacing_groups[most_common_group])
        count = len(spacing_groups[most_common_group])

        print(f"   Most common spacing: {avg_spacing:.1f} pixels ({count} occurrences)")

        # Generate predicted grid lines starting from the first detected point
        if len(detected_points) > 0:
            start_pos = detected_points[0]
            predicted_lines = []

            # Go backwards from start
            pos = start_pos - avg_spacing
            while pos >= 0:
                predicted_lines.append(pos)
                pos -= avg_spacing

            # Add the start position
            predicted_lines.append(start_pos)

            # Go forwards from start
            pos = start_pos + avg_spacing
            while pos <= max_dimension:
                predicted_lines.append(pos)
                pos += avg_spacing

            predicted_lines = sorted(predicted_lines)
            print(f"   Generated {len(predicted_lines)} predicted grid lines")
            return avg_spacing, predicted_lines

        return None, []

    def determine_square_grid(
        vertical_dips,
        horizontal_dips,
        img_width,
        img_height,
        tolerance=3,
        min_grid_size=100,
        max_grid_size=150,
    ):
        """
        Determine the optimal square grid size from both vertical and horizontal detections.
        Only consider grid sizes between min_grid_size and max_grid_size pixels.
        Generate grid lines starting from 0 that perfectly span the image.
        Returns grid_size, vertical_lines, horizontal_lines, confidence_score
        """
        all_spacings = []

        # Get spacings from vertical detections
        if len(vertical_dips) >= 2:
            v_spacings = []
            for i in range(len(vertical_dips) - 1):
                spacing = vertical_dips[i + 1] - vertical_dips[i]
                if min_grid_size <= spacing <= max_grid_size:  # Filter by size range
                    v_spacings.append(spacing)
            all_spacings.extend(v_spacings)
            # Only show summary for cleaner output
            if len(v_spacings) > 0:
                print(
                    f"   Vertical spacings (filtered): {len(v_spacings)} values in range {min_grid_size}-{max_grid_size}px"
                )

        # Get spacings from horizontal detections
        if len(horizontal_dips) >= 2:
            h_spacings = []
            for i in range(len(horizontal_dips) - 1):
                spacing = horizontal_dips[i + 1] - horizontal_dips[i]
                if min_grid_size <= spacing <= max_grid_size:  # Filter by size range
                    h_spacings.append(spacing)
            all_spacings.extend(h_spacings)
            # Only show summary for cleaner output
            if len(h_spacings) > 0:
                print(
                    f"   Horizontal spacings (filtered): {len(h_spacings)} values in range {min_grid_size}-{max_grid_size}px"
                )

        if not all_spacings:
            print(
                f"   No spacings found in range {min_grid_size}-{max_grid_size} pixels"
            )
            return None, [], [], 0.0

        # Group similar spacings (within tolerance)
        spacing_groups = {}
        for spacing in all_spacings:
            found_group = False
            for group_center in spacing_groups:
                if abs(spacing - group_center) <= tolerance:
                    spacing_groups[group_center].append(spacing)
                    found_group = True
                    break

            if not found_group:
                spacing_groups[spacing] = [spacing]

        # Find the most common spacing
        most_common_group = max(
            spacing_groups.keys(), key=lambda k: len(spacing_groups[k])
        )
        optimal_grid_size = np.mean(spacing_groups[most_common_group])
        matching_spacings = spacing_groups[most_common_group]

        print(
            f"   Optimal square grid size: {optimal_grid_size:.1f} pixels ({len(matching_spacings)} samples)"
        )

        # CONSTRAINT: Grid must align with ALL edges - find valid divisors
        def find_common_divisors(width, height, min_size=50, max_size=200):
            """Find divisors that evenly divide both width and height in the given range"""
            divisors = []
            for size in range(min_size, min(max_size + 1, min(width, height) + 1)):
                if width % size == 0 and height % size == 0:
                    divisors.append(size)
            return divisors

        # Find valid grid sizes that perfectly fit the image
        valid_grid_sizes = find_common_divisors(
            img_width, img_height, min_grid_size, max_grid_size
        )

        if not valid_grid_sizes:
            print(
                f"   ERROR: No valid grid sizes found that divide both {img_width}x{img_height}"
            )
            return None, [], [], 0.0

        # Choose the valid grid size closest to our detected optimal size
        best_grid_size = min(valid_grid_sizes, key=lambda x: abs(x - optimal_grid_size))
        size_difference = abs(best_grid_size - optimal_grid_size)

        print(f"   Valid grid sizes: {valid_grid_sizes}")
        print(
            f"   Chosen grid size: {best_grid_size} pixels (diff: {size_difference:.1f} from optimal)"
        )

        # Penalty for large difference between detected and chosen size
        size_accuracy = max(0.0, 1.0 - (size_difference / 20.0))  # 20px tolerance

        # Calculate confidence score with the chosen valid grid size
        confidence_score = calculate_grid_confidence(
            all_spacings,
            matching_spacings,
            vertical_dips,
            horizontal_dips,
            best_grid_size,
            img_width,
            img_height,
        )

        # Apply size accuracy penalty to confidence
        confidence_score = confidence_score * size_accuracy

        # Debug: Show confidence breakdown
        total_detected = len(vertical_dips) + len(horizontal_dips)
        expected_vertical = img_width // best_grid_size  # Exact division now
        expected_horizontal = img_height // best_grid_size  # Exact division now
        total_expected = expected_vertical + expected_horizontal
        coverage_ratio = total_detected / total_expected if total_expected > 0 else 0

        print(
            f"   Grid coverage: {total_detected}/{total_expected} lines ({coverage_ratio:.1%})"
        )
        print(
            f"   Expected: {expected_vertical}V + {expected_horizontal}H, Detected: {len(vertical_dips)}V + {len(horizontal_dips)}H"
        )

        # Generate grid lines - now perfectly aligned with all edges
        vertical_lines = []
        for i in range(expected_vertical + 1):  # +1 to include the right edge
            vertical_lines.append(i * best_grid_size)

        horizontal_lines = []
        for i in range(expected_horizontal + 1):  # +1 to include the bottom edge
            horizontal_lines.append(i * best_grid_size)

        # Only show summary to reduce noise
        print(
            f"   Generated {len(vertical_lines)} vertical and {len(horizontal_lines)} horizontal grid lines"
        )
        print(f"   Confidence score: {confidence_score:.1%}")
        return best_grid_size, vertical_lines, horizontal_lines, confidence_score

    def calculate_grid_confidence(
        all_spacings,
        matching_spacings,
        vertical_dips,
        horizontal_dips,
        grid_size,
        img_width,
        img_height,
    ):
        """
        Calculate a confidence score (0-1) for the detected grid based on multiple factors.
        Heavily emphasizes the number of grid lines detected.
        """
        if not all_spacings or not matching_spacings:
            return 0.0

        # Factor 1: Ratio of matching spacings to total spacings
        matching_ratio = len(matching_spacings) / len(all_spacings)

        # Factor 2: Consistency of matching spacings (lower std deviation = higher confidence)
        if len(matching_spacings) > 1:
            spacing_std = float(np.std(matching_spacings))
            # Normalize: std of 0 = 1.0, std of 10 = 0.0
            clustering_score = max(0.0, 1.0 - (spacing_std / 10.0))
        else:
            clustering_score = 0.5

        # Factor 3: Alignment of detected points with predicted grid
        alignment_score = 0.0
        total_points = 0

        # Check vertical alignment
        if len(vertical_dips) > 0:
            for dip in vertical_dips:
                closest_grid_line = round(dip / grid_size) * grid_size
                distance = abs(dip - closest_grid_line)
                # Points within 5 pixels of grid line get full score
                point_score = max(0, 1.0 - (distance / 5.0))
                alignment_score += point_score
            total_points += len(vertical_dips)

        # Check horizontal alignment
        if len(horizontal_dips) > 0:
            for dip in horizontal_dips:
                closest_grid_line = round(dip / grid_size) * grid_size
                distance = abs(dip - closest_grid_line)
                point_score = max(0, 1.0 - (distance / 5.0))
                alignment_score += point_score
            total_points += len(horizontal_dips)

        if total_points > 0:
            alignment_score = alignment_score / total_points

        # Factor 4: Expected vs detected grid lines (HEAVILY WEIGHTED)
        expected_vertical_lines = max(1, int(img_width / grid_size))
        expected_horizontal_lines = max(1, int(img_height / grid_size))
        total_expected = expected_vertical_lines + expected_horizontal_lines

        # Calculate coverage ratio - how many grid lines we actually detected
        coverage_ratio = total_points / total_expected

        # SEVERE penalty for too few detected lines
        # Need at least 30% of expected lines for decent confidence
        if coverage_ratio < 0.3:
            line_coverage_score = (
                coverage_ratio / 0.3 * 0.3
            )  # Max 0.3 if below threshold
        else:
            line_coverage_score = (
                0.3 + (coverage_ratio - 0.3) * 0.7 / 0.7
            )  # Scale from 0.3 to 1.0

        line_coverage_score = min(1.0, line_coverage_score)

        # Combine all factors with heavy emphasis on line coverage
        confidence = (
            (matching_ratio * 0.15)  # Reduced weight
            + (clustering_score * 0.15)  # Reduced weight
            + (alignment_score * 0.20)  # Slightly reduced weight
            + (line_coverage_score * 0.50)  # MAJOR weight on having enough grid lines
        )

        return min(1.0, confidence)  # Cap at 1.0

    # Test both dark lines (dips) and bright lines (peaks)
    print(f"\n📉 Testing dark grid lines (brightness dips):")
    dark_vertical_dips = find_brightness_changes(
        column_averages, window_size=20, min_drop=4.0, detect_peaks=False
    )
    dark_horizontal_dips = find_brightness_changes(
        row_averages, window_size=20, min_drop=4.0, detect_peaks=False
    )

    print(f"\n📈 Testing bright grid lines (brightness peaks):")
    bright_vertical_peaks = find_brightness_changes(
        column_averages, window_size=20, min_drop=4.0, detect_peaks=True
    )
    bright_horizontal_peaks = find_brightness_changes(
        row_averages, window_size=20, min_drop=4.0, detect_peaks=True
    )

    # Determine optimal grid for both methods
    dark_grid_size, dark_v_lines, dark_h_lines, dark_confidence = determine_square_grid(
        dark_vertical_dips, dark_horizontal_dips, img.width, img.height
    )

    bright_grid_size, bright_v_lines, bright_h_lines, bright_confidence = (
        determine_square_grid(
            bright_vertical_peaks, bright_horizontal_peaks, img.width, img.height
        )
    )

    # Choose the method with higher confidence
    print(f"\n🎯 Results comparison:")
    if dark_grid_size and bright_grid_size:
        print(
            f"   Dark lines: {dark_grid_size:.1f}px grid, {dark_confidence:.1%} confidence"
        )
        print(
            f"   Bright lines: {bright_grid_size:.1f}px grid, {bright_confidence:.1%} confidence"
        )

        if bright_confidence > dark_confidence:
            print(f"   🏆 Using bright grid line detection (higher confidence)")
            final_grid_size = bright_grid_size
            final_confidence = bright_confidence
            vertical_dips = bright_vertical_peaks
            horizontal_dips = bright_horizontal_peaks
            vertical_grid_lines = bright_v_lines
            horizontal_grid_lines = bright_h_lines
            detection_method = "bright"
        else:
            print(f"   🏆 Using dark grid line detection (higher confidence)")
            final_grid_size = dark_grid_size
            final_confidence = dark_confidence
            vertical_dips = dark_vertical_dips
            horizontal_dips = dark_horizontal_dips
            vertical_grid_lines = dark_v_lines
            horizontal_grid_lines = dark_h_lines
            detection_method = "dark"
    elif dark_grid_size:
        print(
            f"   Dark lines: {dark_grid_size:.1f}px grid, {dark_confidence:.1%} confidence"
        )
        print(f"   Bright lines: No grid detected")
        print(f"   🏆 Using dark grid line detection (only option)")
        final_grid_size = dark_grid_size
        final_confidence = dark_confidence
        vertical_dips = dark_vertical_dips
        horizontal_dips = dark_horizontal_dips
        vertical_grid_lines = dark_v_lines
        horizontal_grid_lines = dark_h_lines
        detection_method = "dark"
    elif bright_grid_size:
        print(f"   Dark lines: No grid detected")
        print(
            f"   Bright lines: {bright_grid_size:.1f}px grid, {bright_confidence:.1%} confidence"
        )
        print(f"   🏆 Using bright grid line detection (only option)")
        final_grid_size = bright_grid_size
        final_confidence = bright_confidence
        vertical_dips = bright_vertical_peaks
        horizontal_dips = bright_horizontal_peaks
        vertical_grid_lines = bright_v_lines
        horizontal_grid_lines = bright_h_lines
        detection_method = "bright"
    else:
        print(f"   Dark lines: No grid detected")
        print(f"   Bright lines: No grid detected")
        print(f"   ERROR: No grid detected with either method")
        final_grid_size = None
        final_confidence = 0
        vertical_dips = np.array([])
        horizontal_dips = np.array([])
        vertical_grid_lines = []
        horizontal_grid_lines = []
        detection_method = "none"

    # Detect brightness changes for vertical and horizontal lines
    # (Keeping old logic below for reference, but using results from above)
    if len(vertical_dips) > 0:
        if show_plots:
            ax2.plot(
                vertical_dips,
                column_averages[vertical_dips],
                "ro",
                markersize=4,
                alpha=0.7,
            )
        print(
            f"🔍 Found {len(vertical_dips)} vertical brightness changes at columns: {vertical_dips[:10]}..."
        )

    # Add vertical grid lines to column graph
    if show_plots:
        if vertical_grid_lines and final_grid_size:
            for col in vertical_grid_lines:
                ax2.axvline(x=col, color="blue", alpha=0.3, linewidth=1, linestyle="--")
            ax2.set_title(
                f"Column Averages ({len(vertical_dips)} {detection_method} detections, {final_grid_size:.0f}px grid, {final_confidence:.1%})"
            )
        elif len(vertical_dips) > 0:
            ax2.set_title(
                f"Column Averages ({len(vertical_dips)} brightness changes detected)"
            )
        else:
            ax2.set_title("Column Averages (no brightness changes detected)")

    # Print results regardless of show_plots
    if vertical_grid_lines and final_grid_size:
        estimated_cols = round(img.width / final_grid_size)
        print(f"📏 Square grid size: {final_grid_size:.0f} pixels")
        print(f"📐 Grid columns: {estimated_cols}")
        print(f"🎯 Confidence: {final_confidence:.1%}")
    elif len(vertical_dips) > 0:
        pass  # Already printed above
    else:
        if detection_method == "none":
            print("🔍 No vertical brightness changes found")

    if show_plots:
        # Plot 3: Row averages (shows horizontal grid lines)
        ax3.plot(row_averages)
        ax3.set_title("Row Brightness Averages (Horizontal Grid Detection)")
        ax3.set_xlabel("Row (Y position)")
        ax3.set_ylabel("Average Brightness")
        ax3.grid(True, alpha=0.3)

    # Detect brightness changes for horizontal lines (rows)
    if len(horizontal_dips) > 0:
        if show_plots:
            ax3.plot(
                horizontal_dips,
                row_averages[horizontal_dips],
                "ro",
                markersize=4,
                alpha=0.7,
            )
        print(
            f"🔍 Found {len(horizontal_dips)} horizontal brightness changes at rows: {horizontal_dips[:10]}..."
        )

    # Add horizontal grid lines to row graph
    if show_plots:
        if horizontal_grid_lines and final_grid_size:
            for row in horizontal_grid_lines:
                ax3.axvline(x=row, color="blue", alpha=0.3, linewidth=1, linestyle="--")
            ax3.set_title(
                f"Row Averages ({len(horizontal_dips)} {detection_method} detections, {final_grid_size:.0f}px grid, {final_confidence:.1%})"
            )
        elif len(horizontal_dips) > 0:
            ax3.set_title(
                f"Row Averages ({len(horizontal_dips)} brightness changes detected)"
            )
        else:
            ax3.set_title("Row Averages (no brightness changes detected)")

    # Print results regardless of show_plots
    if horizontal_grid_lines and final_grid_size:
        estimated_rows = round(img.height / final_grid_size)
        print(f"📐 Grid rows: {estimated_rows}")
    elif len(horizontal_dips) > 0:
        pass  # Already printed above
    else:
        if detection_method == "none":
            print(f"🔍 No significant horizontal brightness changes found")

    # Add analysis for horizontal lines
    row_std = np.std(row_averages)
    row_mean = np.mean(row_averages)
    print(f"📊 Row brightness variation (std): {row_std:.2f}")

    if show_plots:
        # Plot 4: Brightness histogram
        ax4.hist(img_array.flatten(), bins=50, alpha=0.7, color="blue")
        ax4.set_title("Brightness Histogram")
        ax4.set_xlabel("Brightness Value")
        ax4.set_ylabel("Pixel Count")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Configure matplotlib to wait indefinitely for user interaction (no timeout)
        # This prevents the automatic closing that was causing subprocess issues
        plt.rcParams["figure.max_open_warning"] = (
            0  # Disable warning about too many open figures
        )

        # Show the plot and wait for user to close it manually
        plt.show(block=True)  # block=True ensures it waits for user interaction

    # Additional detailed analysis
    print(f"\n📊 DETAILED ANALYSIS:")
    print(f"────────────────────")

    # Try different grid size estimates based on common pixel sizes
    common_cell_sizes = [100, 110, 120, 130, 140, 150, 160, 170, 180]

    print(f"\n🔍 Grid size estimates based on common cell sizes:")
    for cell_size in common_cell_sizes:
        cols = img.width / cell_size
        rows = img.height / cell_size

        # Check if this gives reasonable grid dimensions
        if 10 <= cols <= 100 and 10 <= rows <= 100:
            col_error = abs(cols - round(cols))
            row_error = abs(rows - round(rows))
            total_error = col_error + row_error

            print(
                f"  📐 {cell_size}px cells: {cols:.1f}x{rows:.1f} → {round(cols)}x{round(rows)} (error: {total_error:.3f})"
            )

    return {
        "column_averages": column_averages,
        "row_averages": row_averages,
        "vertical_lines": vertical_dips if len(vertical_dips) > 0 else [],
        "horizontal_lines": horizontal_dips if len(horizontal_dips) > 0 else [],
        "image_size": (img.width, img.height),
    }


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze brightness patterns in an image"
    )
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip matplotlib display (for automated processing)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"ERROR: Image file not found: {args.image_path}")
        sys.exit(1)

    try:
        analyze_image_brightness(args.image_path, show_plots=not args.no_display)
        print(f"\nSUCCESS: Analysis complete!")
    except Exception as e:
        print(f"ERROR: Error analyzing image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
