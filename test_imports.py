"""
Quick module test to verify imports work correctly
"""
try:
    print("Testing imports...")
    
    print("  ✓ PIL/Pillow")
    from PIL import Image
    
    print("  ✓ NumPy") 
    import numpy as np
    
    print("  ✓ OpenCV")
    import cv2
    print(f"    OpenCV version: {cv2.__version__}")
    
    print("  ✓ GridDetector")
    from battlemap_processor.core.grid_detector import GridDetector
    
    print("  ✓ TileExtractor")
    from battlemap_processor.core.tile_extractor import TileExtractor
    
    print("  ✓ InputHandler")
    from battlemap_processor.core.input_handler import InputHandler
    
    print("  ✓ ImageProcessor") 
    from battlemap_processor.core.image_processor import ImageProcessor
    
    print("\n✓ All modules imported successfully!")
    print("The system is ready to process battlemap images.")
    
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    print("Please check your virtual environment and dependencies.")
except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
