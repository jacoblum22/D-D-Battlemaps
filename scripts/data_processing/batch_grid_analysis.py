#!/usr/bin/env python3
"""
Batch Grid Analysis Script

Downloads images from Google Drive folders and runs grid detection comparison on each image.
Recursively searches through folders, processes each image, and cleans up temporary files.
"""

import os
import sys
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

# Import the existing Google Drive handler
try:
    from battlemap_processor.core.google_drive_handler import GoogleDriveHandler
except ImportError:
    print(
        "ERROR: Could not import GoogleDriveHandler. Make sure battlemap_processor is available."
    )
    sys.exit(1)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class GridAnalysisBatch:
    """Batch processor for grid detection analysis on Google Drive images"""

    def __init__(self):
        self.drive_handler: Optional[GoogleDriveHandler] = None
        self.temp_dir: Optional[Path] = None
        self.results: List[Dict[str, Any]] = []

    def initialize_drive_handler(self) -> bool:
        """Initialize the Google Drive handler"""
        try:
            self.drive_handler = GoogleDriveHandler()
            print("OK: Google Drive handler initialized")
            return True
        except Exception as e:
            print(f"ERROR: Failed to initialize Google Drive handler: {e}")
            return False

    def extract_folder_id_from_url(self, url: str) -> Optional[str]:
        """Extract folder ID from Google Drive URL"""
        # Match various Google Drive URL formats
        patterns = [
            r"drive\.google\.com/drive/folders/([a-zA-Z0-9-_]+)",
            r"drive\.google\.com/drive/u/\d+/folders/([a-zA-Z0-9-_]+)",
            r"drive\.google\.com/open\?id=([a-zA-Z0-9-_]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # If no pattern matches, assume the URL is just the folder ID
        if re.match(r"^[a-zA-Z0-9-_]+$", url.strip()):
            return url.strip()

        return None

    def is_image_file(self, filename: str) -> bool:
        """Check if file is a supported image format"""
        return Path(filename).suffix.lower() in IMAGE_EXTENSIONS

    def run_grid_analysis(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Run grid detection comparison on an image"""
        try:
            print(f"Analyzing: {image_path.name}")

            # Set environment for proper Unicode handling
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            # Run the comparison script
            result = subprocess.run(
                [sys.executable, "compare_grid_detection.py", str(image_path)],
                capture_output=True,
                text=True,
                timeout=60,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            if result.returncode != 0:
                print(f"ERROR: Analysis failed: {result.stderr}")
                return None

            # Parse the output to extract key information
            output = result.stdout
            analysis_result = {
                "filename": image_path.name,
                "success": True,
                "original_method": self._parse_method_result(
                    output, "METHOD 1: Original Morphological Detection"
                ),
                "brightness_method": self._parse_method_result(
                    output, "METHOD 2: New Brightness-Based Detection"
                ),
                "comparison": self._parse_comparison(output),
            }

            return analysis_result

        except subprocess.TimeoutExpired:
            print(f"⏰ Analysis timeout for {image_path.name}")
            return {"filename": image_path.name, "success": False, "error": "timeout"}
        except Exception as e:
            print(f"ERROR: Analysis error for {image_path.name}: {e}")
            return {"filename": image_path.name, "success": False, "error": str(e)}

    def _parse_method_result(self, output: str, method_header: str) -> Dict[str, Any]:
        """Parse the results for a specific method from the output"""
        method_result: Dict[str, Any] = {"success": False}

        # Find the method section
        method_start = output.find(method_header)
        if method_start == -1:
            return method_result

        # Extract the method section (until next major section or end)
        method_section = output[method_start : method_start + 1000]  # Reasonable limit

        if "SUCCESS" in method_section:
            method_result["success"] = True

            # Extract grid dimensions
            grid_match = re.search(r"Grid dimensions: (\d+)x(\d+)", method_section)
            if grid_match:
                method_result["grid_cols"] = int(grid_match.group(1))
                method_result["grid_rows"] = int(grid_match.group(2))

            # Extract cell size
            cell_match = re.search(
                r"Cell size: ([\d.]+)(?:x[\d.]+)? pixels", method_section
            )
            if cell_match:
                method_result["cell_size"] = float(cell_match.group(1))

            # Extract confidence
            conf_match = re.search(r"Confidence: ([\d.]+)%", method_section)
            if conf_match:
                method_result["confidence"] = float(conf_match.group(1))

        return method_result

    def _parse_comparison(self, output: str) -> Dict[str, Any]:
        """Parse the comparison summary from the output"""
        comparison = {}

        if "METHODS AGREE" in output:
            comparison["agreement"] = "agree"
        elif "METHODS PARTIALLY AGREE" in output:
            comparison["agreement"] = "partial"
        elif "METHODS DISAGREE" in output:
            comparison["agreement"] = "disagree"
        elif "Only original method succeeded" in output:
            comparison["agreement"] = "original_only"
        elif "Only new method succeeded" in output:
            comparison["agreement"] = "brightness_only"
        elif "Both methods failed" in output:
            comparison["agreement"] = "both_failed"
        else:
            comparison["agreement"] = "unknown"

        return comparison

    def cleanup_temp_files(self, image_path: Path):
        """Clean up temporary files created during analysis"""
        try:
            # Remove analysis plots
            analysis_plot = (
                image_path.parent / f"{image_path.stem}_brightness_analysis.png"
            )
            if analysis_plot.exists():
                analysis_plot.unlink()

            # Remove any other temporary files that might be created
            # (matplotlib might create additional files)
            for ext in [".png", ".jpg", ".jpeg"]:
                temp_file = image_path.parent / f"{image_path.stem}_analysis{ext}"
                if temp_file.exists():
                    temp_file.unlink()

        except Exception as e:
            print(f"⚠️  Warning: Could not clean up some temp files: {e}")

    def process_folder_recursively(
        self, folder_id: str, folder_name: str = "Root", depth: int = 0
    ) -> List[Dict[str, Any]]:
        """Recursively process all images in a folder and its subfolders"""
        indent = "  " * depth
        print(f"{indent}Processing folder: {folder_name}")

        if not self.drive_handler:
            print(f"{indent}ERROR: Google Drive handler not initialized")
            return []

        try:
            # Get all images from folder recursively using existing handler
            # This will get all images from this folder and subfolders
            all_images = self.drive_handler.list_images_in_folder(
                folder_id, recursive=True
            )

            print(
                f"{indent}   Found {len(all_images)} images total (including subfolders)"
            )
            results = []

            # Process each image
            for image_info in all_images:
                # Create a display path for the image
                display_path = (
                    f"{folder_name}/{image_info.get('path', image_info['name'])}"
                )
                print(f"{indent}  Processing: {display_path}")

                # Create temporary file path
                temp_image_path = self.temp_dir / image_info["name"]

                # Download the image using existing handler
                try:
                    success = self.drive_handler.download_file(
                        image_info["id"], str(temp_image_path)
                    )
                    if not success:
                        print(f"{indent}    ERROR: Failed to download")
                        continue

                    # Run grid analysis
                    analysis_result = self.run_grid_analysis(temp_image_path)
                    if analysis_result:
                        analysis_result["folder_path"] = display_path
                        results.append(analysis_result)
                        self._print_quick_summary(analysis_result, indent + "    ")

                    # Clean up
                    self.cleanup_temp_files(temp_image_path)
                    if temp_image_path.exists():
                        temp_image_path.unlink()

                except Exception as e:
                    print(f"{indent}    ERROR: Failed to process: {e}")

        except Exception as e:
            print(f"{indent}ERROR: Error processing folder {folder_name}: {e}")
            results = []

        return results

    def _print_quick_summary(self, result: Dict[str, Any], indent: str):
        """Print a quick summary of analysis results"""
        if not result.get("success"):
            print(f"{indent}ERROR: Failed: {result.get('error', 'Unknown error')}")
            return

        orig = result.get("original_method", {})
        bright = result.get("brightness_method", {})
        comparison = result.get("comparison", {})

        if orig.get("success") and bright.get("success"):
            orig_grid = f"{orig.get('grid_cols', '?')}x{orig.get('grid_rows', '?')}"
            bright_grid = (
                f"{bright.get('grid_cols', '?')}x{bright.get('grid_rows', '?')}"
            )
            agreement = comparison.get("agreement", "unknown")

            if agreement == "agree":
                status = "OK: AGREE"
            elif agreement == "partial":
                status = "⚠️  PARTIAL"
            elif agreement == "disagree":
                status = "ERROR: DISAGREE"
            else:
                status = f"❓ {agreement.upper()}"

            print(
                f"{indent}{status} - Original: {orig_grid}, Brightness: {bright_grid}"
            )
        else:
            orig_status = "OK" if orig.get("success") else "ERROR"
            bright_status = "OK" if bright.get("success") else "ERROR"
            print(f"{indent}Original: {orig_status}, Brightness: {bright_status}")

    def save_detailed_results(
        self,
        results: List[Dict[str, Any]],
        output_file: str = "grid_analysis_results.json",
    ):
        """Save detailed analysis results to JSON file"""
        try:
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "total_images": len(results),
                        "results": results,
                        "summary": self._generate_summary(results),
                    },
                    f,
                    indent=2,
                )
            print(f"📊 Detailed results saved to: {output_file}")
        except Exception as e:
            print(f"ERROR: Error saving results: {e}")

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        total = len(results)
        if total == 0:
            return {"total": 0}

        successful = sum(1 for r in results if r.get("success"))
        agreements = [
            r.get("comparison", {}).get("agreement")
            for r in results
            if r.get("success")
        ]

        summary = {
            "total_images": total,
            "successful_analyses": successful,
            "failed_analyses": total - successful,
            "agreement_counts": {},
        }

        for agreement in [
            "agree",
            "partial",
            "disagree",
            "original_only",
            "brightness_only",
            "both_failed",
        ]:
            summary["agreement_counts"][agreement] = agreements.count(agreement)

        return summary

    def print_final_summary(self, results: List[Dict[str, Any]]):
        """Print final summary of all analyses"""
        print("\n" + "=" * 80)
        print("📊 FINAL SUMMARY")
        print("=" * 80)

        summary = self._generate_summary(results)

        print(f"Total images processed: {summary['total_images']}")
        print(f"Successful analyses: {summary['successful_analyses']}")
        print(f"Failed analyses: {summary['failed_analyses']}")

        if summary["successful_analyses"] > 0:
            print("\nAgreement breakdown:")
            for agreement, count in summary["agreement_counts"].items():
                if count > 0:
                    percentage = (count / summary["successful_analyses"]) * 100
                    print(
                        f"  {agreement.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
                    )

    def run(self, drive_url: str):
        """Main execution function"""
        print("Starting batch grid detection analysis...")
        print(f"Google Drive URL: {drive_url}")

        # Extract folder ID
        folder_id = self.extract_folder_id_from_url(drive_url)
        if not folder_id:
            print("❌ Could not extract folder ID from URL")
            return

        print(f"Folder ID: {folder_id}")

        # Initialize Google Drive handler
        if not self.initialize_drive_handler():
            return

        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="grid_analysis_"))
        print(f"Temporary directory: {self.temp_dir}")

        try:
            # Process all images recursively
            results = self.process_folder_recursively(folder_id)

            # Save results
            self.save_detailed_results(results)

            # Print summary
            self.print_final_summary(results)

        finally:
            # Clean up temporary directory
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory")


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python batch_grid_analysis.py <google_drive_folder_url>")
        print("\nExample:")
        print(
            "  python batch_grid_analysis.py 'https://drive.google.com/drive/folders/1abc123def456'"
        )
        print("  python batch_grid_analysis.py '1abc123def456'")
        sys.exit(1)

    drive_url = sys.argv[1]

    # Check if required scripts exist
    required_scripts = ["compare_grid_detection.py", "analyze_brightness.py"]
    for script in required_scripts:
        if not os.path.exists(script):
            print(f"❌ Required script not found: {script}")
            sys.exit(1)

    # Run the batch analysis
    batch_analyzer = GridAnalysisBatch()
    batch_analyzer.run(drive_url)


if __name__ == "__main__":
    main()
