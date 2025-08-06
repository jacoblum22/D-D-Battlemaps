"""
Input Handler for processing Google Drive links and zip files
"""
import os
import io
import re
import zipfile
import tempfile
import shutil
from typing import Dict, Optional, List
from PIL import Image

# Google Drive imports (will be used when we implement GDrive support)
# from google.oauth2.credentials import Credentials
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload

class InputHandler:
    """Handles input from Google Drive links and local zip files"""
    
    def __init__(self):
        self.supported_image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
        self.temp_dir = None
    
    def get_images_from_source(self, source: str) -> Dict[str, Image.Image]:
        """
        Get images from either a Google Drive link or local zip file
        
        Args:
            source: Google Drive URL or local file path
            
        Returns:
            Dict mapping image names to PIL Image objects
        """
        if self._is_google_drive_link(source):
            return self._get_images_from_gdrive(source)
        elif self._is_zip_file(source):
            return self._get_images_from_zip(source)
        elif os.path.isdir(source):
            return self._get_images_from_directory(source)
        elif self._is_image_file(source):
            return self._get_single_image(source)
        else:
            raise ValueError(f"Unsupported source type: {source}")
    
    def _is_google_drive_link(self, source: str) -> bool:
        """Check if source is a Google Drive link"""
        return 'drive.google.com' in source and '/folders/' in source
    
    def _is_zip_file(self, source: str) -> bool:
        """Check if source is a zip file"""
        return source.lower().endswith('.zip') and os.path.isfile(source)
    
    def _is_image_file(self, source: str) -> bool:
        """Check if source is an image file"""
        ext = os.path.splitext(source)[1].lower()
        return ext in self.supported_image_extensions and os.path.isfile(source)
    
    def _get_single_image(self, source: str) -> Dict[str, Image.Image]:
        """Load a single image file"""
        try:
            img = Image.open(source).convert('RGB')
            name = os.path.basename(source)
            return {name: img}
        except Exception as e:
            print(f"Error loading image {source}: {e}")
            return {}
    
    def _get_images_from_directory(self, source: str) -> Dict[str, Image.Image]:
        """Get all images from a directory recursively"""
        images = {}
        
        for root, dirs, files in os.walk(source):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.supported_image_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        img = Image.open(file_path).convert('RGB')
                        # Create a unique name including folder structure
                        rel_path = os.path.relpath(file_path, source)
                        images[rel_path] = img
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue
        
        return images
    
    def _get_images_from_zip(self, zip_path: str) -> Dict[str, Image.Image]:
        """Extract and load images from a zip file"""
        images = {}
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    # Skip directories and non-image files
                    if member.endswith('/') or not any(member.lower().endswith(ext) for ext in self.supported_image_extensions):
                        continue
                    
                    try:
                        with zf.open(member) as f:
                            img_data = f.read()
                            img = Image.open(io.BytesIO(img_data)).convert('RGB')
                            # Use just the filename, not full path
                            name = os.path.basename(member)
                            images[name] = img
                    except Exception as e:
                        print(f"Error loading {member} from zip: {e}")
                        continue
        
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid zip file")
        except Exception as e:
            print(f"Error processing zip file {zip_path}: {e}")
        
        return images
    
    def _get_images_from_gdrive(self, gdrive_url: str) -> Dict[str, Image.Image]:
        """
        Get images from Google Drive folder (placeholder for now)
        
        For now, this is a placeholder. We'll implement this after
        the core functionality is working.
        """
        print("Google Drive integration not yet implemented")
        print("For now, please download the files and use a local zip file or directory")
        return {}
    
    def _extract_folder_id_from_url(self, url: str) -> Optional[str]:
        """Extract folder ID from Google Drive URL"""
        match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
        return match.group(1) if match else None
    
    def cleanup(self):
        """Clean up any temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
