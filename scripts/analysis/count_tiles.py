#!/usr/bin/env python3
"""
Tile Counting Script

This script analyzes battlemap images to count the total number of tiles that would be generated
across multiple tile sizes without displaying any matplotlib. Based on test_smart_grid_comprehensive.py.

Usage:
    python count_tiles.py
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
import time
import concurrent.futures
import multiprocessing
from functools import partial

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import pipeline components - using the same as test_smart_grid_comprehensive.py
from battlemap_processor.core.smart_image_selector import SmartImageSelector
from battlemap_processor.core.grid_detector import GridDetector
from battlemap_processor.core.advanced_boring_detector import AdvancedBoringDetector
from battlemap_processor.core.optimal_tile_placer import OptimalTilePlacer
from battlemap_processor.core.image_source_handler import ImageSourceHandler
from PIL import Image
import subprocess
import re


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value"""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    response = input(full_prompt).strip()
    return response if response else (default or "")


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user"""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{default_str}]: ").strip().lower()

    if not response:
        return default
    return response.startswith("y")


def collect_sources() -> List[str]:
    """Collect input sources from user - same as run_battlemap_pipeline.py"""
    print("\n📂 Input Sources Configuration")
    print("════════════════════════════════")
    print("Enter your image sources (Google Drive URLs, zip files, local directories).")
    print("Examples:")
    print("  • https://drive.google.com/drive/folders/1ABC123...")
    print("  • C:\\\\Users\\\\username\\\\Pictures\\\\Battlemaps")
    print("Type 'done' when finished adding sources.")

    sources = []
    while True:
        source = get_user_input(f"\nSource {len(sources) + 1} (or 'done')")

        if source.lower() == "done":
            if not sources:
                print("⚠️  You must add at least one source!")
                continue
            break

        if source:
            sources.append(source)
            print(f"✅ Added: {source}")

    return sources


def convert_score_to_confidence(detection_score):
    """Convert morphological detection score to confidence percentage"""
    if detection_score is None:
        return 0.0

    if detection_score <= 0:
        return 0.0

    # Base confidence for any successful detection
    base_confidence = 55.0

    # Logarithmic boost - scores have diminishing returns
    import math

    log_boost = math.log(1 + detection_score) * 15.0

    # Additional linear boost for very high scores
    if detection_score > 5:
        linear_boost = (detection_score - 5) * 1.0
    else:
        linear_boost = 0

    confidence = base_confidence + log_boost + linear_boost

    # Cap at 95% (never 100% confident)
    return min(95.0, max(0.0, confidence)) / 100.0  # Return as decimal (0-1)


def brightness_to_grid_info(image_path: str) -> Optional[Dict]:
    """Run brightness-based grid detection using analyze_brightness.py with --no-display"""
    try:
        # Run the analyze_brightness script with --no-display flag
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, "analyze_brightness.py", "--no-display", image_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            return None

        output = result.stdout

        # Parse the output to extract grid information
        grid_size = None
        confidence = 0.0
        cols = 0
        rows = 0

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

        # If grid size was found, we have a successful detection
        if grid_size and grid_size > 0:
            # Load image to get dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size

                # Calculate cell dimensions
                if cols > 0 and rows > 0:
                    cell_width = img_width / cols
                    cell_height = img_height / rows
                else:
                    # Estimate grid dimensions from image size and grid_size
                    cols = int(img_width / grid_size)
                    rows = int(img_height / grid_size)
                    cell_width = img_width / cols if cols > 0 else grid_size
                    cell_height = img_height / rows if rows > 0 else grid_size

                # Generate grid edges
                x_edges = [i * cell_width for i in range(cols + 1)]
                y_edges = [i * cell_height for i in range(rows + 1)]

                return {
                    "nx": cols,
                    "ny": rows,
                    "cell_width": cell_width,
                    "cell_height": cell_height,
                    "x_edges": x_edges,
                    "y_edges": y_edges,
                    "score": confidence,
                    "size_px": grid_size,
                    "detection_method": "brightness_analysis",
                    "filename_match": False,
                }

        return None

    except Exception as e:
        print(f"    ❌ Error in brightness analysis: {e}")
        return None


def compare_detection_methods(image_path: str, filename: str, pil_image: Image.Image):
    """Compare morphological and brightness-based detection"""
    grid_detector = GridDetector()

    # Run morphological detection
    try:
        morphological_result = grid_detector.detect_grid(pil_image, filename)
    except Exception as e:
        print(f"    ❌ Morphological detection error: {e}")
        morphological_result = None

    # Run brightness-based detection
    brightness_result = brightness_to_grid_info(image_path)

    # Compare results
    if morphological_result and brightness_result:
        # Convert morphological raw score to confidence percentage
        morph_raw_score = morphological_result.get("score", 0.0)
        morph_score = convert_score_to_confidence(morph_raw_score)
        bright_score = brightness_result.get("score", 0.0)

        if morph_score == bright_score:
            # Update morphological result with converted confidence
            morphological_result["score"] = morph_score
            return morphological_result
        elif bright_score > morph_score:
            return brightness_result
        else:
            # Update morphological result with converted confidence
            morphological_result["score"] = morph_score
            return morphological_result

    elif morphological_result:
        # Convert morphological raw score to confidence percentage
        morph_raw_score = morphological_result.get("score", 0.0)
        morph_score = convert_score_to_confidence(morph_raw_score)

        # Update morphological result with converted confidence
        morphological_result["score"] = morph_score
        return morphological_result

    elif brightness_result:
        return brightness_result

    else:
        return None


def enhanced_grid_detection_fixed(
    img_info, pil_image: Image.Image, image_path: str, image_handler
):
    """Enhanced grid detection with proper path handling for gridded variants"""
    grid_detector = GridDetector()

    # Step 1: Grid in filename → use filename dimensions directly (skip visual detection)
    if img_info.has_dimensions:
        print(f"    🔄 STEP 1: Grid dimensions found in filename")
        
        # Extract dimensions directly from filename and create grid info
        filename_dims = grid_detector.extract_dimensions_from_filename(img_info.filename)
        if filename_dims:
            nx, ny = filename_dims
            img_width, img_height = pil_image.size
            
            cell_width = img_width / float(nx)
            cell_height = img_height / float(ny)
            
            # Generate grid edges
            x_edges = [i * cell_width for i in range(nx + 1)]
            y_edges = [i * cell_height for i in range(ny + 1)]
            
            grid_info = {
                "nx": nx,
                "ny": ny,
                "cell_width": cell_width,
                "cell_height": cell_height,
                "x_edges": x_edges,
                "y_edges": y_edges,
                "score": 0.95,  # High confidence for filename dimensions
                "size_px": None,
                "filename_dimensions": filename_dims,
                "filename_match": True,
                "detection_method": "filename_direct",
            }
            
            print(f"    📐 Using filename dimensions: {nx}x{ny} squares")
            return grid_info
        else:
            print(f"    ⚠️  has_dimensions=True but could not extract dimensions from filename")
            # Fall back to visual detection
            grid_info = compare_detection_methods(image_path, img_info.filename, pil_image)
            return grid_info

    # Step 2: Only one variant OR no both_variants, no filename grid → morphological + brightness detection
    elif not img_info.has_both_variants:
        print(f"    🔄 STEP 2: Single variant, comparing detection methods")
        grid_info = compare_detection_methods(image_path, img_info.filename, pil_image)
        return grid_info

    # Step 4: Mixed variants, no filename grid → detect on gridded, apply to gridless
    elif (
        img_info.has_both_variants
        and img_info.gridded_variant_path
        and not img_info.is_gridded
    ):
        print(f"    🔄 STEP 4: Mixed variants, detect on gridded, apply to gridless")

        # Download the gridded variant properly
        try:
            # Use the image handler to download the gridded variant
            if img_info.gridded_variant_path.startswith("gdrive://"):
                # Create a temporary ImageInfo for the gridded variant
                from battlemap_processor.core.image_source_handler import ImageInfo

                gridded_img_info = ImageInfo(
                    path=img_info.gridded_variant_path,
                    filename=img_info.gridded_variant_filename
                    or "gridded_variant.webp",
                    source_type="google_drive",
                    relative_path=img_info.gridded_variant_filename
                    or "gridded_variant.webp",
                    size_bytes=None,
                    source_url=img_info.gridded_variant_path,
                    has_dimensions=False,
                    is_gridless=False,
                    is_gridded=True,
                    gridded_variant_path=None,
                    gridded_variant_filename=None,
                    has_both_variants=False,
                )

                # Download the gridded variant
                gridded_image_path = image_handler.download_single_image(
                    gridded_img_info, debug=False
                )

                if not gridded_image_path:
                    print(f"    ❌ Failed to download gridded variant")
                    return None

            else:
                # Local path
                gridded_image_path = img_info.gridded_variant_path

            with Image.open(gridded_image_path) as gridded_pil:
                gridded_pil = gridded_pil.convert("RGB")
                gridded_filename = (
                    img_info.gridded_variant_filename
                    or os.path.basename(gridded_image_path)
                )
                grid_info = compare_detection_methods(
                    gridded_image_path, gridded_filename, gridded_pil
                )

            if grid_info:
                print(
                    f"    📐 Detected grid: {grid_info['nx']}x{grid_info['ny']} squares"
                )

                # Update grid dimensions for current (gridless) image size
                img_width, img_height = pil_image.size
                grid_info["cell_width"] = img_width / grid_info["nx"]
                grid_info["cell_height"] = img_height / grid_info["ny"]

                # Update edges for current image
                grid_info["x_edges"] = [
                    i * grid_info["cell_width"] for i in range(grid_info["nx"] + 1)
                ]
                grid_info["y_edges"] = [
                    i * grid_info["cell_height"] for i in range(grid_info["ny"] + 1)
                ]

                return grid_info
            else:
                print(f"    ❌ Failed to detect grid on gridded variant")
                return None

        except Exception as e:
            print(f"    ❌ Error loading gridded variant: {e}")
            return None

    # Step 3: All same type variants, no filename grid → use morphological + brightness detection on current
    else:
        print(
            f"    🔄 STEP 3: Same type variants, using enhanced detection on current image"
        )
        grid_info = compare_detection_methods(image_path, img_info.filename, pil_image)
        return grid_info


def process_single_image_for_multiprocessing(args) -> Optional[Dict]:
    """Process a single image (download, grid detection, boring analysis) - for multiprocessing
    
    Args:
        args: Tuple of (img_info_data, image_handler_config)
    """
    img_info_data, temp_dir = args
    
    # Import here to avoid issues with multiprocessing
    from battlemap_processor.core.image_source_handler import ImageSourceHandler, ImageInfo  
    from battlemap_processor.core.advanced_boring_detector import AdvancedBoringDetector
    from PIL import Image
    
    try:
        # Recreate ImageInfo object from data
        img_info = ImageInfo(
            path=img_info_data['path'],
            filename=img_info_data['filename'], 
            source_type=img_info_data['source_type'],
            relative_path=img_info_data['relative_path'],
            size_bytes=img_info_data.get('size_bytes'),
            source_url=img_info_data.get('source_url'),
            has_dimensions=img_info_data.get('has_dimensions', False),
            is_gridless=img_info_data.get('is_gridless', False),
            is_gridded=img_info_data.get('is_gridded', False),
            gridded_variant_path=img_info_data.get('gridded_variant_path'),
            gridded_variant_filename=img_info_data.get('gridded_variant_filename'),
            has_both_variants=img_info_data.get('has_both_variants', False)
        )
        
        # Create image handler for this process
        image_handler = ImageSourceHandler(temp_dir=temp_dir)
        
        # Download/access the image
        image_path = image_handler.download_single_image(img_info, debug=False)
        if not image_path:
            return None
        
        # Load the image
        pil_image = Image.open(image_path).convert("RGB")
        
        # Enhanced grid detection 
        grid_info = enhanced_grid_detection_fixed(img_info, pil_image, image_path, image_handler)
        if not grid_info:
            return None
        
        # Boring analysis
        boring_detector = AdvancedBoringDetector()
        square_analysis, boring_reasons = boring_detector.analyze_image_regions(
            pil_image, grid_info, debug=False
        )
        
        # Return processed data (pickle-safe)
        return {
            'img_name': img_info.filename,
            'grid_info': grid_info,
            'square_analysis': square_analysis,
            'boring_reasons': boring_reasons,
            'success': True
        }
        
    except Exception as e:
        return {
            'img_name': img_info_data.get('filename', 'unknown'),
            'error': str(e),
            'success': False
        }


def process_tile_size_for_multiprocessing(args) -> Dict:
    """Process tile placement for a single tile size - designed for multiprocessing
    
    Args:
        args: Tuple of (tile_size, processed_images_data)
    """
    tile_size, processed_images_data = args
    
    # Import here to avoid issues with multiprocessing
    from battlemap_processor.core.optimal_tile_placer import OptimalTilePlacer
    
    # Create tile placer for this size
    tile_placer = OptimalTilePlacer(tile_size=tile_size, max_boring_percentage=50.0)
    
    results_per_image = []
    total_tiles = 0
    
    for img_data in processed_images_data:
        img_name = img_data['img_name']
        grid_info = img_data['grid_info']
        square_analysis = img_data['square_analysis']
        
        try:
            # Run tile placement (CPU-intensive operation)
            placed_tiles = tile_placer.find_optimal_placements(
                grid_info=grid_info,
                square_analysis=square_analysis,
                debug=False,
            )
            
            tile_count = len(placed_tiles)
            total_tiles += tile_count
            
            results_per_image.append({
                'image': img_name,
                'tiles': tile_count
            })
                
        except Exception as e:
            print(f"  ❌ {img_name}: Error in tile placement: {e}")
            results_per_image.append({
                'image': img_name,
                'tiles': 0
            })
    
    return {
        'tile_size': tile_size,
        'total_tiles': total_tiles,
        'results_per_image': results_per_image,
        'processed_images': len(processed_images_data)
    }


def main():
    """Main function with OPTIMIZED structure for efficiency"""
    print("🧮 Battlemap Tile Counter (OPTIMIZED)")
    print("══════════════════════════════════════")
    print("Count tiles across multiple sizes without displaying matplotlib")
    print("Based on test_smart_grid_comprehensive.py logic")
    print()

    # Collect sources
    sources = collect_sources()

    # Configuration
    print("\n⚙️  Configuration")
    print("═══════════════")

    max_images = None
    if get_yes_no("Limit number of images for testing?", default=False):
        max_images = int(get_user_input("Maximum images to process", "20"))

    print("\n📊 Tile size analysis will be run for: 12x12, 14x14, 16x16, 18x18, 20x20")

    if not get_yes_no("Proceed with counting?", default=True):
        print("❌ Cancelled.")
        return

    # Tile sizes to test
    tile_sizes = [12, 14, 16, 18, 20]

    print(f"\n🚀 Starting OPTIMIZED tile counting across {len(tile_sizes)} sizes...")
    print("═" * 60)

    start_time = time.time()

    # STEP 1: Collect all images with FOLDER-AWARE smart selection
    print(f"\n📂 Collecting images with folder-aware smart selection...")
    collection_start = time.time()
    image_handler = ImageSourceHandler()
    
    all_images = []
    for source_idx, source in enumerate(sources):
        print(f"📁 Finding images in source {source_idx + 1}/{len(sources)}...")
        
        try:
            # First, get ALL images without smart selection
            all_images_in_source = image_handler.find_images_from_source(
                source=source,
                debug=False,  # Disable debug for initial collection
                list_only=True,
                use_smart_selection=False,  # Get everything first
            )
            
            if not all_images_in_source:
                print(f"  ⚠️  No images found")
                continue
                
            print(f"  � Found {len(all_images_in_source)} total images")
            
            # Group images by their parent folder path (keep full ImageInfo objects)
            from collections import defaultdict
            folder_groups = defaultdict(list)
            
            for img in all_images_in_source:
                # Extract folder path from relative_path
                folder_path = str(Path(img.relative_path).parent)
                folder_groups[folder_path].append(img)
            
            print(f"  📁 Organized into {len(folder_groups)} folders")
            
            # Apply smart selection within each folder
            from battlemap_processor.core.smart_image_selector import SmartImageSelector
            selector = SmartImageSelector()
            
            selected_images = []
            for folder_path, folder_images in folder_groups.items():
                print(f"    📂 Processing folder: {folder_path} ({len(folder_images)} images)")
                
                # Convert ImageInfo objects to the format expected by SmartImageSelector
                folder_images_dict = []
                for img in folder_images:
                    folder_images_dict.append({
                        'path': getattr(img, 'path', ''),
                        'filename': img.filename
                    })
                
                # Apply smart selection to this folder's images only
                folder_selected = selector.select_optimal_images(folder_images_dict)
                
                # Convert back to ImageInfo objects by matching with original images
                for selected_dict in folder_selected:
                    # Find the original ImageInfo object
                    original_img = next(img for img in folder_images 
                                      if img.filename == selected_dict['filename'])
                    
                    # Update the ImageInfo object with smart selection metadata
                    original_img.has_dimensions = selected_dict.get('has_dimensions', False)
                    original_img.has_both_variants = selected_dict.get('has_both_variants', False)
                    original_img.gridded_variant_path = selected_dict.get('gridded_variant_path')
                    original_img.gridded_variant_filename = selected_dict.get('gridded_variant_filename')
                    
                    selected_images.append(original_img)
                    print(f"      🎯 Selected: {selected_dict['filename']} (reason: {selected_dict.get('selection_reason', 'unknown')}, variants: {selected_dict.get('total_variants', 1)})")
                    if selected_dict.get('has_both_variants', False):
                        print(f"          └─ Gridded variant: {selected_dict.get('gridded_variant_filename', 'unknown')}")
            
            all_images.extend(selected_images)
            print(f"  ✅ Selected {len(selected_images)} optimal images from {len(folder_groups)} folders")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            # Fallback to original approach
            try:
                images = image_handler.find_images_from_source(
                    source=source,
                    debug=True,
                    list_only=True,
                    use_smart_selection=True,
                )
                if images:
                    all_images.extend(images)
                    print(f"  📊 Fallback: Found {len(images)} images")
            except Exception as fallback_error:
                print(f"  ❌ Fallback also failed: {fallback_error}")

    if not all_images:
        print("⚠️  No images found in any source")
        return

    print(f"\n📊 Total images found: {len(all_images)}")

    # Apply image limit if specified
    if max_images and len(all_images) > max_images:
        all_images = all_images[:max_images]
        print(f"🔢 Limited to {max_images} images for testing")

    collection_time = time.time() - collection_start
    print(f"⏱️  Image collection took: {collection_time:.1f} seconds")

    # STEP 2: Process all images with MULTIPROCESSING (expensive operations)
    print(f"\n🔄 Processing images with MULTIPROCESSING (download, grid detection, boring analysis)...")
    processing_start = time.time()
    
    # Determine optimal number of workers for image processing
    num_image_workers = min(len(all_images), max(1, multiprocessing.cpu_count() // 2))
    print(f"  � Using {num_image_workers} CPU cores for parallel image processing...")
    
    # Prepare data for multiprocessing (pickle-safe)
    image_processing_args = []
    for img_info in all_images:
        img_info_data = {
            'path': getattr(img_info, 'path', ''),
            'filename': img_info.filename,
            'source_type': img_info.source_type,
            'relative_path': img_info.relative_path,
            'size_bytes': getattr(img_info, 'size_bytes', None),
            'source_url': getattr(img_info, 'source_url', None),
            'has_dimensions': getattr(img_info, 'has_dimensions', False),
            'is_gridless': getattr(img_info, 'is_gridless', False),
            'is_gridded': getattr(img_info, 'is_gridded', False),
            'gridded_variant_path': getattr(img_info, 'gridded_variant_path', None),
            'gridded_variant_filename': getattr(img_info, 'gridded_variant_filename', None),
            'has_both_variants': getattr(img_info, 'has_both_variants', False)
        }
        image_processing_args.append((img_info_data, image_handler.temp_dir))
    
    # Process images in parallel
    processed_images = []
    failed_images = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_image_workers) as executor:
        # Submit all image processing tasks
        future_to_img = {
            executor.submit(process_single_image_for_multiprocessing, args): args[0]['filename']
            for args in image_processing_args
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_img):
            img_name = future_to_img[future]
            try:
                result = future.result()
                if result and result.get('success', False):
                    processed_images.append(result)
                    print(f"  ✅ {img_name}: Grid {result['grid_info']['nx']}x{result['grid_info']['ny']}")
                else:
                    failed_images += 1
                    error_msg = result.get('error', 'Unknown error') if result else 'No result'
                    print(f"  ❌ {img_name}: {error_msg}")
            except Exception as e:
                failed_images += 1
                print(f"  ❌ {img_name}: Process error: {e}")
    
    if not processed_images:
        print("❌ No images were successfully processed")
        return
    
    processing_time = time.time() - processing_start
    print(f"\n📊 Successfully processed: {len(processed_images)} images (parallel)")
    print(f"❌ Failed to process: {failed_images} images")
    print(f"⏱️  Parallel image processing took: {processing_time:.1f} seconds")

    # STEP 3: Run tile placement for each size with TRUE PARALLELISM (multiprocessing)
    print(f"\n🎯 Running tile placement analysis for {len(tile_sizes)} sizes with MULTIPROCESSING...")
    placement_start = time.time()
    
    # Prepare data for multiprocessing (pickle-safe)
    processed_images_data = []
    for processed_img in processed_images:
        processed_images_data.append({
            'img_name': processed_img['img_name'],
            'grid_info': processed_img['grid_info'],
            'square_analysis': processed_img['square_analysis']
        })
    
    # Prepare arguments for each process
    process_args = [(tile_size, processed_images_data) for tile_size in tile_sizes]
    
    # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
    # Reserve some cores for tile placement since we already used some for image processing
    remaining_cores = max(1, multiprocessing.cpu_count() - num_image_workers)
    num_workers = min(len(tile_sizes), remaining_cores)
    print(f"  🚀 Using {num_workers} CPU cores for parallel tile placement...")
    
    size_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tile size processing tasks
        future_to_size = {
            executor.submit(process_tile_size_for_multiprocessing, args): args[0] 
            for args in process_args
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_size):
            tile_size = future_to_size[future]
            try:
                result = future.result()
                size_results[tile_size] = result
                print(f"  ✅ Completed {tile_size}x{tile_size}: {result['total_tiles']} total tiles")
            except Exception as e:
                print(f"  ❌ Error processing {tile_size}x{tile_size}: {e}")
                size_results[tile_size] = {
                    'tile_size': tile_size,
                    'total_tiles': 0,
                    'results_per_image': [],
                    'processed_images': len(processed_images)
                }

    placement_time = time.time() - placement_start
    end_time = time.time()
    duration = end_time - start_time

    # Display results with better formatting
    print(f"\n📊 Tile Counting Results")
    print("═" * 80)
    print(f"{'Tile Size':<12} {'Total Tiles':<12} {'Per Image Details':<40}")
    print("-" * 80)

    # Sort results by tile size for consistent display
    for tile_size in sorted(tile_sizes):
        result = size_results[tile_size]
        total_tiles = result['total_tiles']
        
        # Create per-image details string
        image_details = []
        for img_result in result['results_per_image']:
            image_details.append(f"{img_result['image']}: {img_result['tiles']}")
        
        details_str = " | ".join(image_details)
        if len(details_str) > 37:
            details_str = details_str[:34] + "..."
        
        print(f"{tile_size}x{tile_size:<8} {total_tiles:<12} {details_str:<40}")

    print("-" * 80)
    print()
    print(f"⏱️  Performance Summary:")
    print(f"   Image collection: {collection_time:.1f}s")
    print(f"   Image processing: {processing_time:.1f}s") 
    print(f"   Tile placement (multiprocessing): {placement_time:.1f}s")
    print(f"   Total time: {duration:.1f}s")
    print(f"� Processed {len(processed_images)} images successfully")
    
    # Show detailed breakdown for multi-image datasets
    if len(processed_images) > 1:
        print(f"\n📋 Detailed Per-Image Breakdown:")
        for tile_size in sorted(tile_sizes):
            result = size_results[tile_size]
            print(f"\n  {tile_size}x{tile_size} tiles:")
            for img_result in result['results_per_image']:
                print(f"    {img_result['image']}: {img_result['tiles']} tiles")

    if max_images:
        print(f"\n🔢 Note: Results based on {max_images} images limit")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
