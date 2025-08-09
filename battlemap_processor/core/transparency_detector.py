"""
Transparency Detector for Battlemap Processing

Detects images with significant transparency that should be skipped in the pipeline.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any


class TransparencyDetector:
    """
    Detects images with significant transparency
    """

    def __init__(self, transparency_threshold: float = 10.0):
        """
        Initialize the transparency detector

        Args:
            transparency_threshold: Minimum percentage of transparent pixels to consider image transparent
        """
        self.transparency_threshold = transparency_threshold

    def has_transparency(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if an image has significant transparency

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (has_transparency, analysis_info)
        """
        try:
            # Load image with PIL to preserve alpha channel
            pil_img = Image.open(image_path)

            # Check if image has alpha channel
            if "A" not in pil_img.mode:
                return False, {
                    "has_alpha_channel": False,
                    "transparency_percentage": 0.0,
                    "reason": "no_alpha_channel",
                }

            # Convert to numpy array to analyze alpha values
            img_array = np.array(pil_img)
            if len(img_array.shape) != 3 or img_array.shape[2] != 4:
                return False, {
                    "has_alpha_channel": True,
                    "transparency_percentage": 0.0,
                    "reason": "invalid_alpha_format",
                }

            # Extract alpha channel
            alpha_channel = img_array[:, :, 3]

            # Calculate transparency statistics
            total_pixels = alpha_channel.size
            transparent_pixels = np.sum(alpha_channel < 255)  # Any transparency
            fully_transparent = np.sum(alpha_channel == 0)  # Fully transparent

            transparency_percentage = (transparent_pixels / total_pixels) * 100

            analysis_info = {
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

        except Exception as e:
            return False, {
                "has_alpha_channel": False,
                "transparency_percentage": 0.0,
                "reason": f"error_{str(e)}",
            }

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
