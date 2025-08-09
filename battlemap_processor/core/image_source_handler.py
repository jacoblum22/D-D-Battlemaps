"""
Image Source Handler for processing Google Drive links and zip files

This module can find and extract images from:
- Google Drive shared links (files and folders)
- Zip files (local or remote)
- Local directories

It searches recursively through all folders and returns information about found images.
"""

import logging
import os
import re
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse, parse_qs
import requests
from dataclasses import dataclass

# Try to import Google Drive handler
try:
    from .google_drive_handler import GoogleDriveHandler
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    GoogleDriveHandler = None

# Always import smart image selector
from .smart_image_selector import SmartImageSelector

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Information about a found image"""
    path: str  # Full path to the image file
    filename: str  # Just the filename
    source_type: str  # 'google_drive', 'zip_file', 'local'
    relative_path: str  # Path relative to source root
    size_bytes: Optional[int] = None  # File size if available
    source_url: Optional[str] = None  # Original source URL


class ImageSourceHandler:
    """
    Handles finding images from various sources including Google Drive and zip files
    """

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Initialize smart image selector
        self.smart_selector = SmartImageSelector()
        
        # Initialize Google Drive handler if available
        self.google_drive_handler = None
        if GOOGLE_DRIVE_AVAILABLE and GoogleDriveHandler:
            try:
                self.google_drive_handler = GoogleDriveHandler()
            except Exception as e:
                logger.warning(f"Could not initialize Google Drive handler: {e}")
                self.google_drive_handler = None

    def find_images_from_source(
        self, source: str, debug: bool = False, list_only: bool = True, 
        use_smart_selection: bool = True
    ) -> List[ImageInfo]:
        """
        Find all images from a source (Google Drive URL, zip file, or local path)

        Args:
            source: Google Drive URL, zip file path/URL, or local directory path
            debug: Enable debug output
            list_only: If True, only list files without downloading (when possible)
            use_smart_selection: If True, apply smart image selection for gridded/gridless variants

        Returns:
            List of ImageInfo objects for all found images
        """
        if debug:
            print(f"\n=== Finding images from source ===")
            print(f"Source: {source}")
            if list_only:
                print("Mode: List only (no downloads)")
            if use_smart_selection:
                print("Smart selection: Enabled (will prefer optimal variants)")

        source = source.strip()

        try:
            # Get initial image list
            images = self._get_images_from_source_internal(source, debug, list_only)
            
            # Apply smart selection if requested
            if use_smart_selection and images:
                if debug:
                    print(f"\n🧠 Applying smart image selection...")
                    print(f"Found {len(images)} total images")
                
                # Convert ImageInfo to dict format for smart selector
                image_dicts = [
                    {
                        'path': img.path,
                        'filename': img.filename
                    }
                    for img in images
                ]
                
                # Apply smart selection
                selected_dicts = self.smart_selector.select_optimal_images(image_dicts)
                
                if debug:
                    print(f"Selected {len(selected_dicts)} optimal images:")
                    for sel in selected_dicts:
                        reason = sel.get('selection_reason', 'unknown')
                        variants = sel.get('total_variants', 1)
                        print(f"  📁 {sel['filename']} (reason: {reason}, variants: {variants})")
                
                # Create new ImageInfo objects for selected images
                selected_paths = {sel['path'] for sel in selected_dicts}
                images = [img for img in images if img.path in selected_paths]
            
            return images

        except Exception as e:
            logger.error(f"Error processing source '{source}': {e}")
            if debug:
                print(f"❌ Error: {e}")
            return []

    def _get_images_from_source_internal(
        self, source: str, debug: bool = False, list_only: bool = True
    ) -> List[ImageInfo]:
        """
        Internal method to get images without smart selection

        Args:
            source: Google Drive URL, zip file path/URL, or local directory path
            debug: Enable debug output
            list_only: If True, only list files without downloading (when possible)

        Returns:
            List of ImageInfo objects for all found images
        """
        try:
            # Determine source type and handle accordingly
            if self._is_google_drive_url(source):
                if debug:
                    print("Detected: Google Drive URL")
                return self._handle_google_drive(source, debug, list_only)
            
            elif self._is_zip_source(source):
                if debug:
                    print("Detected: Zip file")
                return self._handle_zip_file(source, debug, list_only)
            
            elif os.path.isdir(source):
                if debug:
                    print("Detected: Local directory")
                return self._handle_local_directory(source, debug, list_only)
            
            elif os.path.isfile(source):
                if debug:
                    print("Detected: Local file")
                return self._handle_single_file(source, debug, list_only)
            
            else:
                # Try as URL for zip or image
                if debug:
                    print("Detected: URL (attempting as zip or image)")
                return self._handle_url(source, debug, list_only)

        except Exception as e:
            logger.error(f"Error processing source '{source}': {e}")
            if debug:
                print(f"❌ Error: {e}")
            return []

    def _is_google_drive_url(self, url: str) -> bool:
        """Check if URL is a Google Drive link"""
        return 'drive.google.com' in url.lower()

    def _is_zip_source(self, source: str) -> bool:
        """Check if source appears to be a zip file"""
        return (
            source.lower().endswith('.zip') or
            (source.startswith(('http://', 'https://')) and '.zip' in source.lower())
        )

    def _handle_google_drive(
        self, url: str, debug: bool = False, list_only: bool = True
    ) -> List[ImageInfo]:
        """Handle Google Drive URLs using the Google Drive API"""
        if debug:
            print("🔗 Processing Google Drive URL with API...")

        # Check if Google Drive handler is available
        if not self.google_drive_handler:
            if debug:
                print("❌ Google Drive API not available - falling back to basic method")
            return self._handle_google_drive_fallback(url, debug, list_only)

        try:
            # Extract file/folder ID from URL
            file_id = self.google_drive_handler.extract_file_id(url)
            if not file_id:
                if debug:
                    print("❌ Could not extract Google Drive ID from URL")
                return []

            if debug:
                print(f"📁 Google Drive ID: {file_id}")

            # Get file info to determine if it's a file or folder
            file_info = self.google_drive_handler.get_file_info(file_id)
            if not file_info:
                if debug:
                    print("❌ Could not get file information")
                return []

            if debug:
                print(f"📄 Name: {file_info.get('name', 'Unknown')}")
                print(f"📦 Type: {file_info.get('mimeType', 'Unknown')}")

            images = []

            # Handle folders
            if self.google_drive_handler.is_folder(file_id):
                if debug:
                    print("📁 Processing Google Drive folder...")
                
                # List images in folder (true list-only for folders!)
                image_files = self.google_drive_handler.list_images_in_folder(file_id)
                
                if debug:
                    print(f"�️ Found {len(image_files)} images in folder")

                for img_file in image_files:
                    # For list-only mode, create virtual paths
                    if list_only:
                        path = f"gdrive://{img_file['id']}"
                        size_bytes = img_file.get('size', 0)
                    else:
                        # For full mode, download the file
                        temp_file = os.path.join(self.temp_dir, f"gdrive_{img_file['id']}_{img_file['name']}")
                        if self.google_drive_handler.download_file(img_file['id'], temp_file):
                            path = temp_file
                            size_bytes = os.path.getsize(temp_file)
                        else:
                            if debug:
                                print(f"❌ Failed to download {img_file['name']}")
                            continue

                    images.append(ImageInfo(
                        path=path,
                        filename=img_file['name'],
                        source_type="google_drive",
                        relative_path=img_file['path'],
                        size_bytes=size_bytes if not list_only else img_file.get('size', 0),
                        source_url=url
                    ))

            # Handle individual files
            else:
                if debug:
                    print("📄 Processing individual Google Drive file...")

                # Check if it's an image
                mime_type = file_info.get('mimeType', '')
                if not any(img_type in mime_type for img_type in ['image/', 'png', 'jpg', 'jpeg', 'webp']):
                    if debug:
                        print(f"❌ File is not an image: {mime_type}")
                    return []

                filename = file_info.get('name', f"gdrive_file_{file_id}")
                
                if list_only:
                    # List-only mode for individual files
                    path = f"gdrive://{file_id}"
                    size_bytes = int(file_info.get('size', 0)) if file_info.get('size') else 0
                else:
                    # Full mode - download the file
                    temp_file = os.path.join(self.temp_dir, f"gdrive_{file_id}_{filename}")
                    if self.google_drive_handler.download_file(file_id, temp_file):
                        path = temp_file
                        size_bytes = os.path.getsize(temp_file)
                    else:
                        if debug:
                            print(f"❌ Failed to download {filename}")
                        return []

                images.append(ImageInfo(
                    path=path,
                    filename=filename,
                    source_type="google_drive",
                    relative_path=filename,
                    size_bytes=size_bytes,
                    source_url=url
                ))

            return images

        except Exception as e:
            logger.error(f"Error handling Google Drive URL with API: {e}")
            if debug:
                print(f"❌ Google Drive API error: {e}")
            return []

    def _handle_google_drive_fallback(
        self, url: str, debug: bool = False, list_only: bool = True
    ) -> List[ImageInfo]:
        """Fallback Google Drive handling without API"""
        if debug:
            print("🔗 Processing Google Drive URL (fallback method)...")
            print("⚠️  Limited functionality without Google Drive API")

        try:
            # Extract file/folder ID from URL  
            file_id = self._extract_google_drive_id(url)
            if not file_id:
                if debug:
                    print("❌ Could not extract Google Drive ID from URL")
                return []

            if debug:
                print(f"📁 Google Drive ID: {file_id}")
                print("⚠️  Note: This method only works for individual files, not folders")

            # Try direct download (only works for individual files)
            download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            
            temp_file = os.path.join(self.temp_dir, f"gdrive_fallback_{file_id}")
            
            if debug:
                print(f"💾 Attempting download to: {temp_file}")

            response = self.session.get(download_url, stream=True)
            
            if response.status_code == 200:
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                if self._is_image_file(temp_file):
                    if debug:
                        print("🖼️ Downloaded file is an image")
                    
                    size_bytes = os.path.getsize(temp_file) if not list_only else None
                    filename = f"gdrive_image_{file_id}"
                    
                    # For list_only, clean up immediately
                    if list_only:
                        os.remove(temp_file)
                        temp_file = f"gdrive://{file_id}"
                    
                    return [ImageInfo(
                        path=temp_file,
                        filename=filename,
                        source_type="google_drive",
                        relative_path=filename,
                        size_bytes=size_bytes,
                        source_url=url
                    )]
                else:
                    if debug:
                        print("❓ Downloaded file is not an image")
            else:
                if debug:
                    print(f"❌ Download failed with status code: {response.status_code}")

        except Exception as e:
            logger.error(f"Error handling Google Drive URL (fallback): {e}")
            if debug:
                print(f"❌ Fallback error: {e}")

        return []

    def _extract_google_drive_id(self, url: str) -> Optional[str]:
        """Extract file/folder ID from Google Drive URL"""
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',  # /file/d/{id}/view
            r'/folders/([a-zA-Z0-9_-]+)',  # /folders/{id}
            r'[?&]id=([a-zA-Z0-9_-]+)',   # ?id={id}
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None

    def _handle_zip_file(
        self, source: str, debug: bool = False, list_only: bool = True
    ) -> List[ImageInfo]:
        """Handle zip files (local or remote)"""
        if debug:
            print("📦 Processing zip file...")

        try:
            temp_zip = None
            
            if source.startswith(('http://', 'https://')):
                # Remote zip file
                if debug:
                    print("🌐 Downloading remote zip file...")
                
                temp_zip = os.path.join(self.temp_dir, "remote_archive.zip")
                response = self.session.get(source, stream=True)
                
                if response.status_code == 200:
                    with open(temp_zip, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    zip_path = temp_zip
                else:
                    if debug:
                        print(f"❌ Failed to download zip: {response.status_code}")
                    return []
            else:
                zip_path = source

            # List images in zip without extraction
            if list_only:
                images = self._list_images_in_zip(zip_path, "zip_file", source, debug)
            else:
                images = self._extract_and_find_images_from_zip(
                    zip_path, "zip_file", source, debug
                )

            # Clean up temp zip if we downloaded it
            if temp_zip and os.path.exists(temp_zip):
                os.remove(temp_zip)

            return images

        except Exception as e:
            logger.error(f"Error handling zip file: {e}")
            if debug:
                print(f"❌ Zip file error: {e}")
            return []

    def _list_images_in_zip(
        self, zip_path: str, source_type: str, source_url: str, debug: bool = False
    ) -> List[ImageInfo]:
        """List images in zip file without extracting them"""
        images = []
        
        try:
            if debug:
                print(f"📋 Listing contents of zip file...")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    # Skip directories
                    if file_info.is_dir():
                        continue
                    
                    filename = file_info.filename
                    
                    # Check if it's an image file
                    if self._is_image_file(filename):
                        # Create virtual path for list-only mode
                        virtual_path = f"zip://{zip_path}#{filename}"
                        
                        images.append(ImageInfo(
                            path=virtual_path,
                            filename=os.path.basename(filename),
                            source_type=source_type,
                            relative_path=filename,
                            size_bytes=file_info.file_size,
                            source_url=source_url
                        ))

            if debug:
                print(f"🖼️ Found {len(images)} images in zip (list-only)")

        except Exception as e:
            logger.error(f"Error listing zip contents: {e}")
            if debug:
                print(f"❌ Zip listing error: {e}")

        return images

    def _extract_and_find_images_from_zip(
        self, zip_path: str, source_type: str, source_url: str, debug: bool = False
    ) -> List[ImageInfo]:
        """Extract zip and find all images recursively"""
        images = []
        
        try:
            # Create temp extraction directory
            extract_dir = os.path.join(self.temp_dir, f"extracted_{os.path.basename(zip_path)}")
            os.makedirs(extract_dir, exist_ok=True)

            if debug:
                print(f"📂 Extracting to: {extract_dir}")

            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            if debug:
                print("✅ Zip extracted successfully")

            # Find images recursively in extracted directory
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if self._is_image_file(file):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, extract_dir)
                        
                        images.append(ImageInfo(
                            path=file_path,
                            filename=file,
                            source_type=source_type,
                            relative_path=relative_path,
                            size_bytes=os.path.getsize(file_path),
                            source_url=source_url
                        ))

            if debug:
                print(f"🖼️ Found {len(images)} images in zip")

        except Exception as e:
            logger.error(f"Error extracting zip: {e}")
            if debug:
                print(f"❌ Zip extraction error: {e}")

        return images

    def _handle_local_directory(
        self, directory: str, debug: bool = False, list_only: bool = True
    ) -> List[ImageInfo]:
        """Handle local directory"""
        if debug:
            print("📁 Processing local directory...")

        images = []
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if self._is_image_file(file):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, directory)
                        
                        # Only get file size if not in list_only mode
                        size_bytes = None
                        if not list_only:
                            size_bytes = os.path.getsize(file_path)
                        
                        images.append(ImageInfo(
                            path=file_path,
                            filename=file,
                            source_type="local",
                            relative_path=relative_path,
                            size_bytes=size_bytes,
                            source_url=directory
                        ))

            if debug:
                print(f"🖼️ Found {len(images)} images in directory")

        except Exception as e:
            logger.error(f"Error processing directory: {e}")
            if debug:
                print(f"❌ Directory error: {e}")

        return images

    def _handle_single_file(
        self, file_path: str, debug: bool = False, list_only: bool = True
    ) -> List[ImageInfo]:
        """Handle single local file"""
        if debug:
            print("📄 Processing single file...")

        if self._is_image_file(file_path):
            if debug:
                print("🖼️ File is an image")
            
            # Only get file size if not in list_only mode
            size_bytes = None
            if not list_only:
                size_bytes = os.path.getsize(file_path)
            
            return [ImageInfo(
                path=file_path,
                filename=os.path.basename(file_path),
                source_type="local",
                relative_path=os.path.basename(file_path),
                size_bytes=size_bytes,
                source_url=file_path
            )]
        else:
            if debug:
                print("❌ File is not an image")
            return []

    def _handle_url(
        self, url: str, debug: bool = False, list_only: bool = True
    ) -> List[ImageInfo]:
        """Handle generic URL (try as zip or image)"""
        if debug:
            print("🌐 Processing generic URL...")

        try:
            # Try downloading as zip first
            if '.zip' in url.lower():
                return self._handle_zip_file(url, debug, list_only)
            
            # For direct image URLs in list_only mode, we can just check headers
            if list_only:
                if debug:
                    print("🔍 Checking URL headers without downloading...")
                
                # HEAD request to check content type without downloading
                response = self.session.head(url)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if any(img_type in content_type for img_type in ['image/', 'png', 'jpg', 'jpeg', 'webp']):
                        filename = os.path.basename(urlparse(url).path) or "url_image"
                        size_bytes = int(response.headers.get('content-length', 0)) or None
                        
                        if debug:
                            print(f"🖼️ URL is an image (content-type: {content_type})")
                        
                        return [ImageInfo(
                            path=f"url://{url}",
                            filename=filename,
                            source_type="url",
                            relative_path=filename,
                            size_bytes=size_bytes,
                            source_url=url
                        )]
                    else:
                        if debug:
                            print(f"❓ URL content-type is not an image: {content_type}")
                        return []
            else:
                # Full download for non-list mode
                temp_file = os.path.join(self.temp_dir, "downloaded_file")
                response = self.session.get(url, stream=True)
                
                if response.status_code == 200:
                    with open(temp_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    if self._is_image_file(temp_file):
                        if debug:
                            print("🖼️ Downloaded file is an image")
                        return [ImageInfo(
                            path=temp_file,
                            filename=os.path.basename(urlparse(url).path) or "downloaded_image",
                            source_type="url",
                            relative_path=os.path.basename(urlparse(url).path) or "downloaded_image",
                            size_bytes=os.path.getsize(temp_file),
                            source_url=url
                        )]

        except Exception as e:
            logger.error(f"Error handling URL: {e}")
            if debug:
                print(f"❌ URL error: {e}")

        return []

    def _is_image_file(self, filename: str) -> bool:
        """Check if filename appears to be an image"""
        return Path(filename).suffix.lower() in self.image_extensions

    def cleanup_temp_files(self, debug: bool = False):
        """Clean up any temporary files created during processing"""
        if debug:
            print("🧹 Cleaning up temporary files...")
        
        # This is a basic cleanup - in a production system you might want
        # more sophisticated temp file management
        pass

    def download_single_image(self, image_info: ImageInfo, debug: bool = False) -> Optional[str]:
        """
        Download a single image to a temporary location for processing
        
        Args:
            image_info: ImageInfo object describing the image to download
            debug: Enable debug output
        
        Returns:
            Path to the downloaded/accessible image file, or None if failed
        """
        if debug:
            print(f"\n📥 Downloading single image: {image_info.filename}")
            print(f"   Source type: {image_info.source_type}")
            print(f"   Path: {image_info.path}")
        
        try:
            # Handle local files (already accessible)
            if image_info.source_type == "local":
                if os.path.exists(image_info.path):
                    if debug:
                        print(f"✅ Local file accessible at: {image_info.path}")
                    return image_info.path
                else:
                    if debug:
                        print(f"❌ Local file not found: {image_info.path}")
                    return None
            
            # Handle Google Drive files
            elif image_info.source_type == "google_drive":
                if image_info.path.startswith("gdrive://"):
                    # Extract Google Drive file ID
                    file_id = image_info.path.replace("gdrive://", "")
                    
                    if self.google_drive_handler:
                        # Use API to download
                        temp_file = os.path.join(self.temp_dir, f"pipeline_{file_id}_{image_info.filename}")
                        if self.google_drive_handler.download_file(file_id, temp_file):
                            if debug:
                                print(f"✅ Downloaded from Google Drive to: {temp_file}")
                            return temp_file
                        else:
                            if debug:
                                print(f"❌ Failed to download from Google Drive: {file_id}")
                            return None
                    else:
                        # Fallback method
                        download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
                        temp_file = os.path.join(self.temp_dir, f"pipeline_fallback_{file_id}_{image_info.filename}")
                        
                        response = self.session.get(download_url, stream=True)
                        if response.status_code == 200:
                            with open(temp_file, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            
                            if self._is_image_file(temp_file):
                                if debug:
                                    print(f"✅ Downloaded from Google Drive (fallback) to: {temp_file}")
                                return temp_file
                        
                        if debug:
                            print(f"❌ Failed to download from Google Drive (fallback): {file_id}")
                        return None
                else:
                    # Already downloaded file
                    if os.path.exists(image_info.path):
                        if debug:
                            print(f"✅ Google Drive file already downloaded: {image_info.path}")
                        return image_info.path
                    else:
                        if debug:
                            print(f"❌ Google Drive file not found: {image_info.path}")
                        return None
            
            # Handle zip files
            elif image_info.source_type == "zip_file":
                if image_info.path.startswith("zip://"):
                    # Parse zip path format: zip://path/to/archive.zip#internal/path
                    zip_path_info = image_info.path.replace("zip://", "")
                    if '#' in zip_path_info:
                        zip_path, internal_path = zip_path_info.split('#', 1)
                    else:
                        if debug:
                            print(f"❌ Invalid zip path format: {image_info.path}")
                        return None
                    
                    # Extract single file from zip
                    temp_file = os.path.join(self.temp_dir, f"pipeline_zip_{image_info.filename}")
                    
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            with zip_ref.open(internal_path) as source, open(temp_file, 'wb') as target:
                                shutil.copyfileobj(source, target)
                        
                        if debug:
                            print(f"✅ Extracted from zip to: {temp_file}")
                        return temp_file
                    
                    except (KeyError, zipfile.BadZipFile) as e:
                        if debug:
                            print(f"❌ Failed to extract from zip: {e}")
                        return None
                else:
                    # Already extracted file
                    if os.path.exists(image_info.path):
                        if debug:
                            print(f"✅ Zip file already extracted: {image_info.path}")
                        return image_info.path
                    else:
                        if debug:
                            print(f"❌ Extracted zip file not found: {image_info.path}")
                        return None
            
            # Handle URL files
            elif image_info.source_type == "url":
                if image_info.path.startswith("url://"):
                    # Extract actual URL
                    actual_url = image_info.path.replace("url://", "")
                    temp_file = os.path.join(self.temp_dir, f"pipeline_url_{image_info.filename}")
                    
                    response = self.session.get(actual_url, stream=True)
                    if response.status_code == 200:
                        with open(temp_file, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        if debug:
                            print(f"✅ Downloaded from URL to: {temp_file}")
                        return temp_file
                    else:
                        if debug:
                            print(f"❌ Failed to download from URL: {response.status_code}")
                        return None
                else:
                    # Already downloaded file
                    if os.path.exists(image_info.path):
                        if debug:
                            print(f"✅ URL file already downloaded: {image_info.path}")
                        return image_info.path
                    else:
                        if debug:
                            print(f"❌ Downloaded URL file not found: {image_info.path}")
                        return None
            
            else:
                if debug:
                    print(f"❌ Unknown source type: {image_info.source_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error downloading single image {image_info.filename}: {e}")
            if debug:
                print(f"❌ Download error: {e}")
            return None

    def get_source_stats(self, images: List[ImageInfo]) -> Dict[str, Any]:
        """Get statistics about found images"""
        if not images:
            return {
                "total_images": 0,
                "total_size_mb": 0.0,
                "source_types": {},
                "file_extensions": {},
            }

        total_size = sum(img.size_bytes or 0 for img in images)
        source_types = {}
        file_extensions = {}

        for img in images:
            # Count source types
            source_types[img.source_type] = source_types.get(img.source_type, 0) + 1
            
            # Count file extensions
            ext = Path(img.filename).suffix.lower()
            file_extensions[ext] = file_extensions.get(ext, 0) + 1

        return {
            "total_images": len(images),
            "total_size_mb": total_size / (1024 * 1024),
            "source_types": source_types,
            "file_extensions": file_extensions,
        }
