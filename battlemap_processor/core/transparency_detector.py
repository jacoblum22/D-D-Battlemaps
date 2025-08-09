"""
Transparency Detector for Battlemap Processing

Detects images with significant transparency that should be skipped in the pipeline.
"""

import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)


class TransparencyDetector:
    """
    Detects images with significant transparency
    """

    def __init__(self, transparency_threshold: float = 10.0):
        """
        Initialize the transparency detector

        Args:
            transparency_threshold: Minimum percentage of transparent pixels to consider image transparent (0-100)

        Raises:
            ValueError: If transparency_threshold is not between 0 and 100
        """
        # Validate transparency threshold upfront
        if not isinstance(transparency_threshold, (int, float)):
            raise ValueError(
                f"transparency_threshold must be a number, got {type(transparency_threshold)}"
            )

        if not (0 <= transparency_threshold <= 100):
            raise ValueError(
                f"transparency_threshold must be between 0 and 100, got {transparency_threshold}"
            )

        self.transparency_threshold = float(transparency_threshold)

    def has_transparency(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if an image has significant transparency

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (has_transparency, analysis_info)
        """
        try:
            # Load image safely with context manager to ensure file is closed
            with Image.open(image_path) as pil_img:
                original_mode = pil_img.mode

                # Check if image has any form of transparency
                has_alpha = self._has_alpha_support(pil_img)

                if not has_alpha:
                    return False, {
                        "original_mode": original_mode,
                        "has_alpha_channel": False,
                        "transparency_percentage": 0.0,
                        "reason": "no_alpha_support",
                    }

                # Normalize to RGBA for consistent alpha channel handling
                # This handles LA, P with transparency, and other modes properly
                rgba_img = pil_img.convert("RGBA")

                # Convert to numpy array to analyze alpha values
                img_array = np.array(rgba_img)

                # Extract alpha channel (guaranteed to be index 3 after RGBA conversion)
                alpha_channel = img_array[:, :, 3]

                # Calculate transparency statistics
                total_pixels = alpha_channel.size
                transparent_pixels = np.sum(alpha_channel < 255)  # Any transparency
                fully_transparent = np.sum(alpha_channel == 0)  # Fully transparent

                transparency_percentage = (transparent_pixels / total_pixels) * 100

                analysis_info = {
                    "original_mode": original_mode,
                    "has_alpha_channel": True,
                    "total_pixels": total_pixels,
                    "transparent_pixels": transparent_pixels,
                    "fully_transparent_pixels": fully_transparent,
                    "transparency_percentage": transparency_percentage,
                    "alpha_range": (int(alpha_channel.min()), int(alpha_channel.max())),
                    "threshold_used": self.transparency_threshold,
                }

                # Determine if transparency is significant
                has_significant_transparency = (
                    transparency_percentage >= self.transparency_threshold
                )

                if has_significant_transparency:
                    analysis_info["reason"] = (
                        f"transparency_above_threshold_{transparency_percentage:.1f}%"
                    )
                else:
                    analysis_info["reason"] = (
                        f"transparency_below_threshold_{transparency_percentage:.1f}%"
                    )

                return bool(has_significant_transparency), analysis_info

        except (OSError, IOError) as e:
            # File-related errors (missing file, corrupted image, etc.)
            logger.error(f"Error loading image {image_path}: {e}")
            return False, {
                "original_mode": "unknown",
                "has_alpha_channel": False,
                "transparency_percentage": 0.0,
                "reason": f"file_error_{str(e)}",
            }
        except Exception as e:
            # Unexpected errors - log and re-raise for debugging
            logger.critical(
                f"Unexpected error analyzing transparency in {image_path}: {e}",
                exc_info=True,
            )
            raise

    def _has_alpha_support(self, pil_img: Image.Image) -> bool:
        """
        Check if an image has alpha support in any form

        Args:
            pil_img: PIL Image object

        Returns:
            True if image supports transparency
        """
        # Direct alpha channel modes
        if pil_img.mode in ("RGBA", "LA"):
            return True

        # Palette mode with transparency
        if pil_img.mode == "P" and "transparency" in pil_img.info:
            return True

        # Images with transparency info
        if hasattr(pil_img, "info") and "transparency" in pil_img.info:
            return True

        return False

    def should_skip_image(self, image_path: str) -> Tuple[bool, str]:
        """
        Determine if an image should be skipped due to transparency

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (should_skip, reason)
        """
        has_transparency, info = self.has_transparency(image_path)

        if has_transparency:
            return (
                True,
                f"Image has {info['transparency_percentage']:.1f}% transparency (threshold: {self.transparency_threshold}%)",
            )
        else:
            return False, info["reason"]


def main():
    """Test the transparency detector"""
    detector = TransparencyDetector(transparency_threshold=10.0)

    test_images = [
        "test_images/Transparent Trees.webp",
        "test_images/Casino Base.webp",
        "test_images/Casino 2nd Floor.webp",
        "test_images/Dungeon.webp",
    ]

    print("🔍 Testing Transparency Detector")
    print("=" * 60)

    for image_path in test_images:
        if Path(image_path).exists():
            should_skip, reason = detector.should_skip_image(image_path)
            has_transparency, analysis = detector.has_transparency(image_path)

            print(f"\n📁 {Path(image_path).name}")
            print(f"   Should skip: {should_skip}")
            print(f"   Reason: {reason}")
            if analysis["has_alpha_channel"]:
                print(f"   Transparency: {analysis['transparency_percentage']:.2f}%")
                print(f"   Alpha range: {analysis['alpha_range']}")
        else:
            print(f"\n📁 {image_path} - FILE NOT FOUND")


if __name__ == "__main__":
    main()
