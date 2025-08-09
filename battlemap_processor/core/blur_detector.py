"""
Blur Detector for Multi-Floor Buildings

Detects images that show blurred environments (indicating 2nd floor views)
which may be less suitable for training tile generation.
"""

import cv2
import logging
import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)


class BlurDetector:
    """
    Detects images with significant blur that may indicate 2nd floor views
    """

    def __init__(
        self, laplacian_threshold: float = 100.0, edge_density_threshold: float = 0.05
    ):
        """
        Initialize the blur detector

        Args:
            laplacian_threshold: Minimum Laplacian variance for sharp images
            edge_density_threshold: Minimum edge density for sharp images
        """
        self.laplacian_threshold = laplacian_threshold
        self.edge_density_threshold = edge_density_threshold

    def analyze_sharpness(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze the sharpness characteristics of an image

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with sharpness metrics
        """
        try:
            # Load image in grayscale for analysis
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                return {
                    "error": f"Could not load image: {image_path}",
                    "is_sharp": False,
                }

            # Calculate Laplacian variance (primary blur detection method)
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            laplacian_var = laplacian.var()

            # Calculate edge density using Canny edge detection
            edges = cv2.Canny(img, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Calculate standard deviation (measure of contrast)
            std_dev = float(np.std(img.astype(np.float64)))

            # Determine if image is sharp based on thresholds
            is_sharp_laplacian = laplacian_var >= self.laplacian_threshold
            is_sharp_edges = edge_density >= self.edge_density_threshold

            # Image is considered sharp if it passes both tests
            is_sharp = is_sharp_laplacian and is_sharp_edges

            return {
                "laplacian_variance": float(laplacian_var),
                "edge_density": float(edge_density),
                "standard_deviation": float(std_dev),
                "is_sharp_laplacian": is_sharp_laplacian,
                "is_sharp_edges": is_sharp_edges,
                "is_sharp": is_sharp,
                "laplacian_threshold": self.laplacian_threshold,
                "edge_density_threshold": self.edge_density_threshold,
                "image_shape": img.shape,
            }

        except (cv2.error, OSError) as e:
            # Log specific expected errors with stack trace
            logger.error(f"Error analyzing image {image_path}: {str(e)}", exc_info=True)
            return {"error": f"Error analyzing image: {str(e)}", "is_sharp": False}
        except Exception as e:
            # Log unexpected errors and re-raise for debugging
            logger.critical(
                f"Unexpected error analyzing image {image_path}: {str(e)}",
                exc_info=True,
            )
            raise

    def is_blurred_floor(self, image_path: str) -> Tuple[bool, str]:
        """
        Determine if an image appears to be a blurred 2nd floor view

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (is_blurred_floor, reason)
        """
        analysis = self.analyze_sharpness(image_path)

        if "error" in analysis:
            return False, analysis["error"]

        if not analysis["is_sharp"]:
            # Image is blurred - likely a 2nd floor view
            reasons = []
            if not analysis["is_sharp_laplacian"]:
                reasons.append(
                    f"Low Laplacian variance: {analysis['laplacian_variance']:.1f} < {self.laplacian_threshold}"
                )
            if not analysis["is_sharp_edges"]:
                reasons.append(
                    f"Low edge density: {analysis['edge_density']:.4f} < {self.edge_density_threshold}"
                )

            return True, "Detected blurred 2nd floor view: " + "; ".join(reasons)
        else:
            return (
                False,
                f"Sharp base floor: Laplacian={analysis['laplacian_variance']:.1f}, Edges={analysis['edge_density']:.4f}",
            )

    def should_skip_blurred_image(self, image_path: str) -> Tuple[bool, str]:
        """
        Determine if an image should be skipped due to blur (2nd floor indicator)

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (should_skip, reason)
        """
        is_blurred, reason = self.is_blurred_floor(image_path)
        return is_blurred, reason


def main():
    """Test the blur detector"""
    detector = BlurDetector(laplacian_threshold=100.0, edge_density_threshold=0.05)

    test_images = [
        "test_images/Casino Base.webp",
        "test_images/Casino 2nd Floor.webp",
        "test_images/Dungeon.webp",
        "test_images/Harbor [99x99].webp",
    ]

    print("🔍 Testing Blur Detector for Multi-Floor Buildings")
    print("=" * 60)

    for image_path in test_images:
        if Path(image_path).exists():
            should_skip, reason = detector.should_skip_blurred_image(image_path)
            analysis = detector.analyze_sharpness(image_path)

            print(f"\n📁 {Path(image_path).name}")
            print(f"   Should skip: {should_skip}")
            print(f"   Reason: {reason}")
            if "error" not in analysis:
                print(f"   Laplacian variance: {analysis['laplacian_variance']:.2f}")
                print(f"   Edge density: {analysis['edge_density']:.4f}")
                print(f"   Is sharp: {analysis['is_sharp']}")
        else:
            print(f"\n📁 {image_path} - FILE NOT FOUND")


if __name__ == "__main__":
    main()
