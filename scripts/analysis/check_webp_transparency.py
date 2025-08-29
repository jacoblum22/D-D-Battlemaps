#!/usr/bin/env python3
"""
Quick script to check WebP transparency properties
"""

import sys
from PIL import Image
import numpy as np
from pathlib import Path


def check_transparency(image_path):
    print(f"Checking: {image_path}")

    img = Image.open(image_path)
    print(f"Mode: {img.mode}")
    print(f"Size: {img.size}")
    print(f"Format: {img.format}")
    print(f"Info keys: {list(img.info.keys())}")
    print(f"Has 'transparency' in info: {'transparency' in img.info}")

    # Check if it actually has transparent pixels
    if img.mode == "RGBA":
        print("Image is in RGBA mode - checking alpha channel...")
        alpha = img.getchannel("A")
        alpha_array = np.array(alpha)
        total_pixels = alpha_array.size
        transparent_pixels = np.sum(alpha_array < 255)
        transparency_percentage = (transparent_pixels / total_pixels) * 100
        print(f"Transparency: {transparency_percentage:.1f}% transparent pixels")

        if transparency_percentage > 0:
            print(f"Min alpha: {alpha_array.min()}, Max alpha: {alpha_array.max()}")
    elif img.mode == "LA":
        print("Image is in LA mode - checking alpha channel...")
        alpha = img.getchannel("A")
        alpha_array = np.array(alpha)
        total_pixels = alpha_array.size
        transparent_pixels = np.sum(alpha_array < 255)
        transparency_percentage = (transparent_pixels / total_pixels) * 100
        print(f"Transparency: {transparency_percentage:.1f}% transparent pixels")
    else:
        print(f"Image mode {img.mode} - checking for transparency info...")

        if "transparency" in img.info:
            print(f"Transparency info: {img.info['transparency']}")
        else:
            print("No transparency detected in mode or info")

        # Convert to RGBA to see if there are actually transparent pixels
        try:
            rgba_img = img.convert("RGBA")
            alpha = rgba_img.getchannel("A")
            alpha_array = np.array(alpha)
            total_pixels = alpha_array.size
            transparent_pixels = np.sum(alpha_array < 255)
            transparency_percentage = (transparent_pixels / total_pixels) * 100
            print(
                f"After RGBA conversion: {transparency_percentage:.1f}% transparent pixels"
            )

            if transparency_percentage > 0:
                print(f"Min alpha: {alpha_array.min()}, Max alpha: {alpha_array.max()}")
        except Exception as e:
            print(f"Error converting to RGBA: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_webp_transparency.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    check_transparency(image_path)
