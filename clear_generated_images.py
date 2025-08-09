#!/usr/bin/env python3
"""
Clear Generated Images Folder

This script safely clears the generated_images folder and all its contents.
Useful for starting fresh with a new pipeline run.
Handles Windows permission issues and stubborn files.
"""

import os
import shutil
import stat
import time
from pathlib import Path


def force_remove_file(file_path):
    """
    Forcefully remove a file by changing permissions first
    """
    try:
        # Change file permissions to make it writable
        os.chmod(file_path, stat.S_IWRITE)
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"  ❌ Could not force-remove {file_path}: {e}")
        return False


def force_remove_directory(dir_path):
    """
    Forcefully remove a directory and all its contents
    """
    def handle_remove_readonly(func, path, exc):
        """Error handler for removing read-only files"""
        if os.path.exists(path):
            # Make the file writable and try again
            os.chmod(path, stat.S_IWRITE)
            func(path)
    
    try:
        shutil.rmtree(dir_path, onerror=handle_remove_readonly)
        return True
    except Exception as e:
        print(f"  ❌ Could not force-remove directory {dir_path}: {e}")
        return False


def clear_generated_images_robust(folder_name: str = "generated_images"):
    """
    Robustly clear the generated images folder with Windows permission handling
    
    Args:
        folder_name: Name of the folder to clear (default: "generated_images")
    """
    folder_path = Path(folder_name)
    
    if not folder_path.exists():
        print(f"📁 Folder '{folder_name}' doesn't exist - nothing to clear.")
        return
    
    # Get some stats before deletion
    total_files = 0
    total_size = 0
    failed_files = []
    
    print(f"🔍 Scanning '{folder_name}' folder...")
    
    for root, dirs, files in os.walk(folder_path):
        total_files += len(files)
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                pass  # File might be inaccessible
    
    print(f"📊 Found {total_files} files ({total_size / (1024*1024):.1f} MB) in '{folder_name}'")
    
    # Confirm deletion
    response = input(f"🗑️  Are you sure you want to delete everything in '{folder_name}'? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("❌ Operation cancelled.")
        return
    
    print(f"🧹 Clearing '{folder_name}' folder...")
    
    # Method 1: Try the simple approach first
    try:
        shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Successfully cleared '{folder_name}' folder (simple method)!")
        return
    except Exception as e:
        print(f"⚠️  Simple deletion failed: {e}")
        print("� Trying robust deletion method...")
    
    # Method 2: Robust deletion - handle files individually
    success_count = 0
    failed_count = 0
    
    # Walk the directory tree from bottom up (deepest first)
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Remove all files in this directory
        for file in files:
            file_path = os.path.join(root, file)
            print(f"  🗑️  Removing file: {os.path.relpath(file_path, folder_path)}")
            
            if force_remove_file(file_path):
                success_count += 1
            else:
                failed_count += 1
                failed_files.append(file_path)
        
        # Remove all subdirectories in this directory
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            print(f"  📁 Removing directory: {os.path.relpath(dir_path, folder_path)}")
            
            if not force_remove_directory(dir_path):
                failed_count += 1
                failed_files.append(dir_path)
    
    # Finally, try to remove the main directory
    try:
        if folder_path.exists():
            force_remove_directory(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Recreated empty '{folder_name}' folder.")
    except Exception as e:
        print(f"⚠️  Could not fully clear main directory: {e}")
    
    # Report results
    print(f"\n📊 Deletion Summary:")
    print(f"  ✅ Successfully removed: {success_count} items")
    if failed_count > 0:
        print(f"  ❌ Failed to remove: {failed_count} items")
        print(f"\n🔧 Suggested solutions for stubborn files:")
        print(f"  1. Close any programs that might be using the files")
        print(f"  2. Wait a moment and try again (files might be temporarily locked)")
        print(f"  3. Restart your computer if the issue persists")
        print(f"  4. Try running the script as Administrator")
        
        # Show first few failed files
        print(f"\n📝 Failed files/folders:")
        for failed_file in failed_files[:5]:
            print(f"  - {failed_file}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    else:
        print(f"✅ All files successfully cleared!")


def clear_generated_images(folder_name: str = "generated_images"):
    """
    Clear the generated images folder and all its contents
    
    Args:
        folder_name: Name of the folder to clear (default: "generated_images")
    """
    # Use the robust method
    clear_generated_images_robust(folder_name)


def main():
    """Main function"""
    print("🧹 Generated Images Folder Cleaner")
    print("═══════════════════════════════════")
    
    # Clear default folder
    clear_generated_images("generated_images")
    
    # Ask if they want to clear any other folders
    while True:
        other_folder = input("\n📂 Clear another folder? Enter folder name (or press Enter to finish): ").strip()
        if not other_folder:
            break
        clear_generated_images(other_folder)
    
    print("\n👍 All done!")


if __name__ == "__main__":
    main()
