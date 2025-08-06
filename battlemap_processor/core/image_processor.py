"""
Image Processor for handling tile output and organization
"""
import os
from typing import List, Dict, Any
from .tile_extractor import TileInfo

class ImageProcessor:
    """Handles saving and organizing extracted tiles"""
    
    def __init__(self):
        pass
    
    def save_tiles(self, tiles: List[TileInfo], source_name: str, output_dir: str) -> int:
        """
        Save extracted tiles to the output directory
        
        Args:
            tiles: List of TileInfo objects to save
            source_name: Name of the source image (for filename generation)
            output_dir: Output directory path
            
        Returns:
            Number of tiles successfully saved
        """
        if not tiles:
            return 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean source name for use in filenames
        base_name = self._clean_filename(source_name)
        if base_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')):
            base_name = os.path.splitext(base_name)[0]
        
        saved_count = 0
        
        for i, tile in enumerate(tiles):
            try:
                # Generate filename with grid position info
                filename = f"{base_name}_tile_{tile.grid_x:02d}_{tile.grid_y:02d}_{tile.squares_wide}x{tile.squares_tall}.png"
                
                # Handle filename collisions
                output_path = os.path.join(output_dir, filename)
                counter = 1
                while os.path.exists(output_path):
                    name_part, ext = os.path.splitext(filename)
                    filename = f"{name_part}_{counter:02d}{ext}"
                    output_path = os.path.join(output_dir, filename)
                    counter += 1
                
                # Save the tile
                tile.image.save(output_path, format='PNG', optimize=True)
                saved_count += 1
                
            except Exception as e:
                print(f"    Error saving tile {i}: {e}")
                continue
        
        return saved_count
    
    def save_tiles_organized(self, tiles: List[TileInfo], source_name: str, 
                           output_dir: str, organize_by: str = "source") -> int:
        """
        Save tiles with organized folder structure
        
        Args:
            tiles: List of TileInfo objects to save
            source_name: Name of the source image  
            output_dir: Base output directory
            organize_by: Organization method ("source", "size", "none")
            
        Returns:
            Number of tiles successfully saved
        """
        if not tiles:
            return 0
        
        # Determine output subdirectory
        if organize_by == "source":
            # Create subfolder for each source image
            base_name = self._clean_filename(source_name)
            if base_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')):
                base_name = os.path.splitext(base_name)[0]
            subdir = os.path.join(output_dir, base_name)
        elif organize_by == "size":
            # Create subfolder based on tile size
            squares = tiles[0].squares_wide  # Assuming all tiles same size
            subdir = os.path.join(output_dir, f"{squares}x{squares}_tiles")  
        else:
            # No organization, save directly to output_dir
            subdir = output_dir
        
        return self.save_tiles(tiles, source_name, subdir)
    
    def _clean_filename(self, filename: str) -> str:
        """Clean a filename to be filesystem-safe"""
        # Remove or replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove extra spaces and dots
        filename = filename.strip('. ')
        
        # Ensure it's not empty
        if not filename:
            filename = "unnamed"
        
        return filename
    
    def create_summary_report(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Create a summary report of the processing results
        
        Args:
            results: Dictionary with processing statistics
            output_dir: Output directory to save report
        """
        report_path = os.path.join(output_dir, "processing_summary.txt")
        
        try:
            with open(report_path, 'w') as f:
                f.write("D&D Battlemap Processing Summary\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Total images processed: {results.get('images_processed', 0)}\n")
                f.write(f"Images with grids detected: {results.get('grids_detected', 0)}\n")
                f.write(f"Total tiles extracted: {results.get('tiles_extracted', 0)}\n")
                f.write(f"Output directory: {output_dir}\n\n")
                
                if 'image_details' in results:
                    f.write("Per-image results:\n")
                    f.write("-" * 20 + "\n")
                    for img_name, details in results['image_details'].items():
                        f.write(f"\n{img_name}:\n")
                        f.write(f"  Grid detected: {details.get('grid_detected', False)}\n")
                        if details.get('grid_detected'):
                            f.write(f"  Grid size: {details.get('grid_nx')}x{details.get('grid_ny')}\n")
                            f.write(f"  Tiles extracted: {details.get('tiles_extracted', 0)}\n")
                
        except Exception as e:
            print(f"Error creating summary report: {e}")
