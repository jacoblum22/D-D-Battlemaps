"""
Duplicate Detection for Battlemap Pipeline

This module provides fast duplicate detection for both source links and image files:
1. Source link deduplication with smart Google Drive URL normalization
2. Fast file-based duplicate detection using MD5 hashes
3. Progress saving for large datasets
"""

import logging
import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class DuplicateInfo:
    """Information about detected duplicates"""

    original_path: str
    duplicate_paths: List[str]
    reason: str  # 'identical_hash', 'same_gdrive_folder', etc.


class SourceDeduplicator:
    """
    Handles deduplication of source URLs/paths with smart Google Drive detection
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def deduplicate_sources(
        self, sources: List[str]
    ) -> Tuple[List[str], List[DuplicateInfo]]:
        """
        Remove duplicate sources, detecting various formats of the same source

        Args:
            sources: List of source URLs/paths

        Returns:
            Tuple of (deduplicated_sources, duplicate_info_list)
        """
        if self.debug:
            print(f"\n🔍 Deduplicating {len(sources)} sources...")

        deduplicated = []
        duplicates = []
        seen_normalized = {}

        for source in sources:
            source = source.strip()
            if not source:
                continue

            # Normalize the source to detect duplicates
            normalized = self._normalize_source(source)

            if normalized in seen_normalized:
                # Found duplicate
                original_source = seen_normalized[normalized]

                # Check if we already have a duplicate entry for this normalized source
                existing_duplicate = None
                for dup in duplicates:
                    if dup.original_path == original_source:
                        existing_duplicate = dup
                        break

                if existing_duplicate:
                    existing_duplicate.duplicate_paths.append(source)
                else:
                    duplicates.append(
                        DuplicateInfo(
                            original_path=original_source,
                            duplicate_paths=[source],
                            reason=self._get_duplicate_reason(source),
                        )
                    )

                if self.debug:
                    print(f"  🔄 Duplicate detected: {source}")
                    print(f"     Original: {original_source}")
            else:
                # New source
                seen_normalized[normalized] = source
                deduplicated.append(source)

                if self.debug:
                    print(f"  ✅ Added: {source}")

        if self.debug:
            print(
                f"✅ Deduplication complete: {len(deduplicated)} unique sources, {len(duplicates)} duplicate groups"
            )

        return deduplicated, duplicates

    def _normalize_source(self, source: str) -> str:
        """
        Normalize a source URL/path to detect duplicates

        Args:
            source: Source URL or path

        Returns:
            Normalized representation for comparison
        """
        source = source.strip()

        # Handle Google Drive URLs
        if "drive.google.com" in source.lower():
            return self._normalize_google_drive_url(source)

        # Handle other URLs - normalize basic URL components
        if source.startswith(("http://", "https://")):
            parsed = urlparse(source.lower())
            # Remove query parameters and fragments for basic URL comparison
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")

        # Handle local paths - normalize path separators and resolve relative paths
        try:
            return str(Path(source).resolve()).lower()
        except (OSError, ValueError):
            # If path resolution fails, return as-is
            return source.lower()

    def _normalize_google_drive_url(self, url: str) -> str:
        """
        Normalize Google Drive URLs to detect the same folder/file in different formats

        Args:
            url: Google Drive URL

        Returns:
            Normalized representation based on the Drive ID
        """
        # Extract Google Drive ID using various patterns
        drive_id = self._extract_google_drive_id(url)

        if drive_id:
            # Determine if it's likely a folder or file based on URL patterns
            if "/folders/" in url or "/drive/folders/" in url:
                return f"gdrive_folder:{drive_id}"
            else:
                return f"gdrive_file:{drive_id}"

        # If we can't extract ID, fall back to basic URL normalization
        return url.lower().replace("www.", "").rstrip("/")

    def _extract_google_drive_id(self, url: str) -> Optional[str]:
        """Extract file/folder ID from Google Drive URL"""
        patterns = [
            r"/file/d/([a-zA-Z0-9_-]+)",  # /file/d/{id}/view
            r"/folders/([a-zA-Z0-9_-]+)",  # /folders/{id}
            r"/drive/folders/([a-zA-Z0-9_-]+)",  # /drive/folders/{id}
            r"[?&]id=([a-zA-Z0-9_-]+)",  # ?id={id}
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _get_duplicate_reason(self, source: str) -> str:
        """Get a human-readable reason for why this source is considered a duplicate"""
        if "drive.google.com" in source.lower():
            if "/folders/" in source or "/drive/folders/" in source:
                return "same_gdrive_folder"
            else:
                return "same_gdrive_file"
        elif source.startswith(("http://", "https://")):
            return "same_url"
        else:
            return "same_local_path"


class FileDeduplicator:
    """
    Fast file-based duplicate detection using MD5 hashes
    """

    def __init__(self, progress_file: Optional[str] = None, debug: bool = False):
        self.progress_file = progress_file
        self.debug = debug
        self.hash_cache = {}  # path -> hash mapping
        self.load_progress()

    def deduplicate_images(
        self, images: List, temp_dir: str
    ) -> Tuple[List, List[DuplicateInfo]]:
        """
        Remove duplicate images based on file hash comparison

        Args:
            images: List of ImageInfo objects
            temp_dir: Temporary directory for downloading files for hashing

        Returns:
            Tuple of (deduplicated_images, duplicate_info_list)
        """
        if self.debug:
            print(f"\n🔍 Checking {len(images)} images for duplicates...")

        deduplicated = []
        duplicates = []
        hash_to_image = {}  # hash -> first ImageInfo with that hash
        processed_count = 0

        # Import here to avoid circular imports
        from .image_source_handler import ImageSourceHandler

        # Create a temporary image handler for downloading files to hash
        temp_handler = ImageSourceHandler(temp_dir=temp_dir)

        for image in images:
            processed_count += 1

            if self.debug and processed_count % 100 == 0:
                print(f"  📊 Processed {processed_count}/{len(images)} images...")

            # Calculate hash for this image
            image_hash = self._get_image_hash(image, temp_handler)

            if image_hash is None:
                # Couldn't calculate hash - include the image anyway
                if self.debug:
                    print(f"  ⚠️  Could not hash {image.filename}, including anyway")
                deduplicated.append(image)
                continue

            if image_hash in hash_to_image:
                # Found duplicate
                original_image = hash_to_image[image_hash]

                # Check if we already have a duplicate entry for this hash
                existing_duplicate = None
                for dup in duplicates:
                    if dup.original_path == original_image.path:
                        existing_duplicate = dup
                        break

                if existing_duplicate:
                    existing_duplicate.duplicate_paths.append(image.path)
                else:
                    duplicates.append(
                        DuplicateInfo(
                            original_path=original_image.path,
                            duplicate_paths=[image.path],
                            reason="identical_hash",
                        )
                    )

                if self.debug:
                    print(
                        f"  🔄 Duplicate: {image.filename} matches {original_image.filename}"
                    )
            else:
                # New unique image
                hash_to_image[image_hash] = image
                deduplicated.append(image)

        # Save progress
        self.save_progress()

        if self.debug:
            print(
                f"✅ File deduplication complete: {len(deduplicated)} unique images, {len(duplicates)} duplicate groups"
            )

        return deduplicated, duplicates

    def _get_image_hash(self, image_info, image_handler) -> Optional[str]:
        """
        Get MD5 hash of an image file

        Args:
            image_info: ImageInfo object
            image_handler: ImageSourceHandler for downloading files

        Returns:
            MD5 hash string, or None if hashing failed
        """
        # Check cache first
        if image_info.path in self.hash_cache:
            return self.hash_cache[image_info.path]

        try:
            # For local files, hash directly
            if image_info.source_type == "local" and not image_info.path.startswith(
                ("gdrive://", "zip://", "url://")
            ):
                if Path(image_info.path).exists():
                    image_hash = self._calculate_file_hash(image_info.path)
                    self.hash_cache[image_info.path] = image_hash
                    return image_hash
                else:
                    return None

            # For remote/virtual files, download temporarily to hash
            temp_path = image_handler.download_single_image(image_info, debug=False)
            if temp_path and Path(temp_path).exists():
                image_hash = self._calculate_file_hash(temp_path)
                self.hash_cache[image_info.path] = image_hash

                # Clean up temp file if it's not the original
                if temp_path != image_info.path and temp_path.startswith(
                    image_handler.temp_dir
                ):
                    try:
                        Path(temp_path).unlink()
                    except (FileNotFoundError, PermissionError):
                        pass

                return image_hash
            else:
                return None

        except Exception as e:
            logger.warning(f"Error hashing image {image_info.filename}: {e}")
            return None

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def load_progress(self):
        """Load hash cache from progress file"""
        if not self.progress_file or not Path(self.progress_file).exists():
            return

        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                self.hash_cache = data.get("hash_cache", {})

            if self.debug:
                print(
                    f"📁 Loaded {len(self.hash_cache)} cached hashes from {self.progress_file}"
                )

        except Exception as e:
            logger.warning(f"Could not load duplicate detection progress: {e}")
            self.hash_cache = {}

    def save_progress(self):
        """Save hash cache to progress file"""
        if not self.progress_file:
            return

        try:
            data = {"hash_cache": self.hash_cache, "version": "1.0"}

            with open(self.progress_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not save duplicate detection progress: {e}")

    def clear_cache(self):
        """Clear the hash cache"""
        self.hash_cache = {}
        if self.progress_file and Path(self.progress_file).exists():
            try:
                Path(self.progress_file).unlink()
            except Exception as e:
                logger.warning(f"Could not delete progress file: {e}")


class DuplicateDetector:
    """
    Combined duplicate detector for sources and files
    """

    def __init__(self, output_dir: str, debug: bool = False):
        self.output_dir = Path(output_dir)
        self.debug = debug

        # Create progress file path
        self.progress_file = self.output_dir / "duplicate_detection_progress.json"

        # Initialize components
        self.source_deduplicator = SourceDeduplicator(debug=debug)
        self.file_deduplicator = FileDeduplicator(
            progress_file=str(self.progress_file), debug=debug
        )

    def deduplicate_sources(
        self, sources: List[str]
    ) -> Tuple[List[str], List[DuplicateInfo]]:
        """Deduplicate source URLs/paths"""
        return self.source_deduplicator.deduplicate_sources(sources)

    def deduplicate_images(
        self, images: List, temp_dir: str
    ) -> Tuple[List, List[DuplicateInfo]]:
        """Deduplicate images based on file content"""
        return self.file_deduplicator.deduplicate_images(images, temp_dir)

    def get_duplicate_report(
        self,
        source_duplicates: List[DuplicateInfo],
        image_duplicates: List[DuplicateInfo],
    ) -> Dict:
        """Generate a summary report of all detected duplicates"""
        return {
            "source_duplicates": {
                "count": len(source_duplicates),
                "details": [
                    {
                        "original": dup.original_path,
                        "duplicates": dup.duplicate_paths,
                        "reason": dup.reason,
                    }
                    for dup in source_duplicates
                ],
            },
            "image_duplicates": {
                "count": len(image_duplicates),
                "details": [
                    {
                        "original": dup.original_path,
                        "duplicates": dup.duplicate_paths,
                        "reason": dup.reason,
                    }
                    for dup in image_duplicates
                ],
            },
        }

    def clear_progress(self):
        """Clear all duplicate detection progress"""
        self.file_deduplicator.clear_cache()
