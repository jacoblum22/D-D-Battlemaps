"""
Google Drive API Handler for Battlemap Processor

This module provides Google Drive API integration to properly handle:
- Individual file downloads
- Folder content listing
- True list-only mode for folders
- Proper authentication and token management
"""

import os
import pickle
import json
import stat
import logging
from typing import List, Optional, Dict, Any, Union
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.credentials import Credentials as BaseCredentials
from googleapiclient.http import MediaIoBaseDownload
import io
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Scopes needed for Google Drive access (read-only for security)
# For read-only access: use drive.readonly
# For file-specific access: use drive.file
# For full access: use drive (avoid unless necessary)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveHandler:
    """
    Handles Google Drive API operations for the battlemap processor
    """

    def __init__(
        self,
        credentials_file: str = "google_drive_credentials.json",
        token_file: str = "google_drive_token.json",  # Changed from .pickle to .json
        lazy_auth: bool = True,
    ):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service: Optional[Any] = None
        self._authenticated = False

        # Defer authentication unless explicitly requested
        if not lazy_auth:
            self._authenticate()

    def _ensure_authenticated(self):
        """Ensure we're authenticated before making API calls"""
        if not self._authenticated:
            self._authenticate()

    def _save_credentials_securely(self, creds: Any) -> None:
        """Save credentials to JSON with secure file permissions"""
        try:
            # Convert credentials to JSON
            creds_data = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }

            # Write to file with secure permissions
            with open(self.token_file, "w") as token_file:
                json.dump(creds_data, token_file, indent=2)

            # Set restrictive permissions (owner read/write only)
            os.chmod(self.token_file, stat.S_IRUSR | stat.S_IWUSR)
            logger.info(f"Credentials saved securely to {self.token_file}")

        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise

    def _load_credentials_from_json(self) -> Optional[Credentials]:
        """Load credentials from JSON file"""
        try:
            with open(self.token_file, "r") as token_file:
                creds_data = json.load(token_file)

            return Credentials.from_authorized_user_info(creds_data, SCOPES)

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Could not load credentials from JSON: {e}")
            return None

    def _migrate_pickle_to_json(self) -> Optional[Credentials]:
        """Migrate old pickle token file to secure JSON format"""
        pickle_file = self.token_file.replace(".json", ".pickle")

        if not os.path.exists(pickle_file):
            return None

        logger.info("Migrating old pickle token file to secure JSON format...")

        try:
            # Load from pickle (last time!)
            with open(pickle_file, "rb") as token:
                creds = pickle.load(token)

            # Save in new format
            self._save_credentials_securely(creds)

            # Remove old pickle file
            os.remove(pickle_file)
            logger.info("Migration completed successfully")

            return creds

        except Exception as e:
            logger.warning(f"Failed to migrate pickle token file: {e}")
            return None

    def _authenticate(self):
        """Authenticate with Google Drive API using secure JSON token storage"""
        creds = None

        # Try to load existing token from JSON
        creds = self._load_credentials_from_json()

        # If no JSON token, try migrating from old pickle format
        if not creds:
            creds = self._migrate_pickle_to_json()

        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    # Save refreshed credentials
                    self._save_credentials_securely(creds)
                except Exception as e:
                    logger.warning(f"Failed to refresh token: {e}")
                    creds = None

            if not creds:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Google Drive credentials file not found: {self.credentials_file}\n"
                        "Please download your OAuth2 credentials from Google Cloud Console"
                    )

                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                creds = flow.run_local_server(port=0)

                # Save new credentials securely
                self._save_credentials_securely(creds)

        # Build the service
        self.service = build("drive", "v3", credentials=creds)
        self._authenticated = True
        logger.info("Google Drive API authenticated successfully")

    def extract_file_id(self, url: str) -> Optional[str]:
        """Extract file/folder ID from Google Drive URL"""
        import re

        # Handle different URL formats
        patterns = [
            r"/file/d/([a-zA-Z0-9-_]+)",  # File links
            r"/folders/([a-zA-Z0-9-_]+)",  # Folder links
            r"/uc\?id=([a-zA-Z0-9-_]+)",  # Direct download format
            r"id=([a-zA-Z0-9-_]+)",  # Direct ID parameter
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Get information about a file or folder"""
        self._ensure_authenticated()

        if not self.service:
            raise RuntimeError("Google Drive service not initialized")

        try:
            file_info = (
                self.service.files()
                .get(  # type: ignore
                    fileId=file_id, fields="id,name,mimeType,size,parents,trashed"
                )
                .execute()
            )

            return file_info
        except Exception as e:
            logger.error(f"Failed to get file info for {file_id}: {e}")
            return {}

    def is_folder(self, file_id: str) -> bool:
        """Check if the given ID is a folder"""
        file_info = self.get_file_info(file_id)
        return file_info.get("mimeType") == "application/vnd.google-apps.folder"

    def list_images_in_folder(
        self, folder_id: str, recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """List all image files in a folder"""
        self._ensure_authenticated()
        images = []

        # Image MIME types to look for
        image_mimes = [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/gif",
            "image/bmp",
            "image/tiff",
            "image/svg+xml",
        ]

        def _list_folder_contents(folder_id: str, path: str = ""):
            """Recursively list folder contents"""
            if not self.service:
                raise RuntimeError("Google Drive service not initialized")

            try:
                # Query for files in this folder
                query = f"'{folder_id}' in parents and trashed=false"

                page_token = None
                while True:
                    results = (
                        self.service.files()
                        .list(  # type: ignore
                            q=query,
                            fields="nextPageToken,files(id,name,mimeType,size,parents)",
                            pageSize=100,
                            pageToken=page_token,
                            supportsAllDrives=True,
                            includeItemsFromAllDrives=True,
                        )
                        .execute()
                    )

                    files = results.get("files", [])

                    for file in files:
                        file_path = f"{path}/{file['name']}" if path else file["name"]

                        # If it's an image, add it to results
                        if file["mimeType"] in image_mimes:
                            images.append(
                                {
                                    "id": file["id"],
                                    "name": file["name"],
                                    "mimeType": file["mimeType"],
                                    "size": (
                                        int(file.get("size", 0))
                                        if file.get("size")
                                        else 0
                                    ),
                                    "path": file_path,
                                }
                            )

                        # If it's a folder and we're doing recursive search
                        elif (
                            recursive
                            and file["mimeType"] == "application/vnd.google-apps.folder"
                        ):
                            _list_folder_contents(file["id"], file_path)

                    page_token = results.get("nextPageToken")
                    if not page_token:
                        break

            except Exception as e:
                logger.error(f"Failed to list folder contents for {folder_id}: {e}")

        _list_folder_contents(folder_id)
        return images

    def download_file(self, file_id: str, destination: str) -> bool:
        """Download a file from Google Drive"""
        self._ensure_authenticated()

        if not self.service:
            raise RuntimeError("Google Drive service not initialized")

        try:
            # Get file metadata
            file_info = self.get_file_info(file_id)
            if not file_info:
                return False

            # Create directory if needed - handle bare filenames safely
            dest_dir = os.path.dirname(destination)
            if dest_dir:  # Only create directory if there's actually a directory part
                os.makedirs(dest_dir, exist_ok=True)

            # Download the file
            request = self.service.files().get_media(fileId=file_id)  # type: ignore

            with open(destination, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.debug(
                            f"Download progress: {int(status.progress() * 100)}%"
                        )

            logger.info(f"Downloaded {file_info['name']} to {destination}")
            return True

        except Exception as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            return False

    def test_connection(self) -> bool:
        """Test if the Google Drive connection is working"""
        try:
            self._ensure_authenticated()
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

        if not self.service:
            logger.error("Google Drive service not initialized")
            return False

        try:
            # Try to get user info
            about = self.service.about().get(fields="user").execute()  # type: ignore
            user_email = about.get("user", {}).get("emailAddress", "Unknown")
            logger.info(
                f"Google Drive connection successful. Authenticated as: {user_email}"
            )
            return True
        except Exception as e:
            logger.error(f"Google Drive connection test failed: {e}")
            return False

    def delete_file(self, file_id: str) -> bool:
        """Delete a file from Google Drive - DISABLED in read-only mode"""
        logger.warning("Delete operation not available in read-only mode")
        logger.info(
            "To enable delete operations, change SCOPES to include 'drive' instead of 'drive.readonly'"
        )
        return False


def test_google_drive_handler():
    """Test function for Google Drive handler"""
    print("🧪 Testing Google Drive Handler")
    print("=" * 40)

    try:
        # Initialize handler
        handler = GoogleDriveHandler()

        # Test connection
        if handler.test_connection():
            print("✅ Google Drive API connection successful!")
        else:
            print("❌ Google Drive API connection failed!")
            return False

        # Test with a file ID (you can replace with your own)
        test_url = input(
            "\n📥 Enter a Google Drive URL to test (or press Enter to skip): "
        ).strip()

        if test_url:
            file_id = handler.extract_file_id(test_url)
            if file_id:
                print(f"📁 Extracted ID: {file_id}")

                file_info = handler.get_file_info(file_id)
                if file_info:
                    print(f"📄 Name: {file_info.get('name', 'Unknown')}")
                    print(f"📦 Type: {file_info.get('mimeType', 'Unknown')}")

                    if handler.is_folder(file_id):
                        print("📁 This is a folder. Listing images...")
                        images = handler.list_images_in_folder(file_id)
                        print(f"🖼️  Found {len(images)} images:")
                        for img in images[:5]:  # Show first 5
                            print(f"   - {img['name']} ({img['size']} bytes)")
                        if len(images) > 5:
                            print(f"   ... and {len(images) - 5} more")
                    else:
                        print("📄 This is a file.")
                else:
                    print("❌ Could not get file info")
            else:
                print("❌ Could not extract file ID from URL")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    test_google_drive_handler()
