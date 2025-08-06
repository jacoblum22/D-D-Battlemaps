"""
Simple test script for debugging the battlemap processor
"""
import os
import sys
import argparse
from PIL import Image
import numpy as np

# Add the project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from battlemap_processor.core.grid_detector import GridDetector
from battlemap_processor.core.tile_extractor import TileExtractor

def test_grid_detection(image_path, generate_tiles=True):
    """Test grid detection on a single image"""
    print(f"Testing grid detection on: {image_path}")
    
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
        print(f"Image size: {img.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Test grid detection
    detector = GridDetector()
    print("Running grid detection...")
    grid_info = detector.detect_grid(img)
    
    if grid_info:
        print("✓ Grid detected!")
        print(f"  Grid size: {grid_info['nx']}x{grid_info['ny']} cells")
        print(f"  Cell size: {grid_info['cell_width']:.1f}x{grid_info['cell_height']:.1f} pixels")
        print(f"  Detection score: {grid_info['score']:.3f}")
        print(f"  Estimated cell size: {grid_info['size_px']} pixels")
        
        # Test tile extraction only if requested
        if generate_tiles:
            test_tile_extraction(img, grid_info)
        else:
            print("\nTile generation skipped (--no-generation flag used)")
        
    else:
        print("✗ No grid detected")
        print("This could mean:")
        print("  - No visible grid in the image")
        print("  - Grid cell size outside 100-180 pixel range")
        print("  - Grid lines too faint or irregular")

def test_tile_extraction(img, grid_info):
    """Test tile extraction"""
    print("\nTesting tile extraction...")
    
    extractor = TileExtractor(tile_size=512)
    
    # Test only 12x12 squares per tile
    squares = 12
    print(f"  Testing {squares}x{squares} squares per tile...")
    tiles = extractor.extract_tiles(img, grid_info, squares)
    print(f"    Result: {len(tiles)} tiles extracted")
    
    if tiles:
        print(f"    Sample tile info:")
        tile = tiles[0]
        print(f"      Grid position: ({tile.grid_x}, {tile.grid_y})")
        print(f"      Pixel position: ({tile.pixel_x}, {tile.pixel_y})")
        print(f"      Squares: {tile.squares_wide}x{tile.squares_tall}")
        print(f"      Output size: {tile.image.size}")
        
        # Create generated_images folder if it doesn't exist
        os.makedirs("generated_images", exist_ok=True)
        
        # Save a sample tile for inspection
        sample_path = f"generated_images/sample_tile_{squares}x{squares}.png"
        tiles[0].image.save(sample_path)
        print(f"      Saved sample: {sample_path}")
        
        # Save a few more tiles if available
        for i in range(1, min(4, len(tiles))):
            tile = tiles[i]
            sample_path = f"generated_images/sample_tile_{squares}x{squares}_{i+1}.png"
            tile.image.save(sample_path)
            print(f"      Saved sample {i+1}: {sample_path}")
    
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test grid detection on a battlemap image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--no-generation", action="store_true", 
                       help="Skip tile generation and only show grid detection results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        sys.exit(1)
    
    print("D&D Battlemap Grid Detection Test")
    print("=" * 40)
    
    if args.no_generation:
        print("Mode: Grid detection only (no tile generation)")
    else:
        print("Mode: Grid detection + sample tile generation")
    
    print()
    
    test_grid_detection(args.image_path, generate_tiles=not args.no_generation)
