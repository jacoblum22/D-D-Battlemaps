#!/usr/bin/env python3
"""
Clear Generated Images and Captions

This script safely clears the generated_images folder and associated caption files.
Useful for starting fresh with a new pipeline run.
Handles Windows permission issues and stubborn files.
"""

import os
import shutil
import stat
import time
from pathlib import Path
from typing import Union, List


def force_remove_file(file_path: str) -> bool:
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


def force_remove_directory(dir_path: Union[str, Path]) -> bool:
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


def get_caption_files() -> List[Path]:
    """
    Get list of caption files that should be cleared
    """
    caption_files = []
    
    # Root-level caption files
    root_caption_patterns = [
        "train_captions.txt",
        "val_captions.txt", 
        "test_captions.txt",
        "validation_prompts.txt",
        "phase4_captions.json",
        "*_captions.txt",
        "*_captions.json"
    ]
    
    for pattern in root_caption_patterns:
        if '*' in pattern:
            # Handle wildcard patterns
            import glob
            matches = glob.glob(pattern)
            for match in matches:
                caption_files.append(Path(match))
        else:
            file_path = Path(pattern)
            if file_path.exists():
                caption_files.append(file_path)
    
    # Dataset caption directories
    dataset_dirs = [
        "dataset_v1/captions",
        "dataset_v1_training/captions",
        "dataset_v1_archive"
    ]
    
    for dataset_dir in dataset_dirs:
        dir_path = Path(dataset_dir)
        if dir_path.exists():
            caption_files.append(dir_path)
    
    return caption_files


def clear_images_and_captions_robust(
    images_folder: str = "generated_images", 
    clear_captions: bool = True,
    clear_progress: bool = True
) -> None:
    """
    Robustly clear the generated images folder and caption files
    
    Args:
        images_folder: Name of the images folder to clear (default: "generated_images")
        clear_captions: Whether to clear caption files (default: True)
        clear_progress: Whether to clear progress files (default: True)
    """
    images_path = Path(images_folder)
    
    # Collect items to clear
    items_to_clear = []
    
    # Add images folder if it exists
    if images_path.exists():
        items_to_clear.append((images_path, "images folder"))
    
    # Add caption files if requested
    caption_files = []
    if clear_captions:
        caption_files = get_caption_files()
        for caption_file in caption_files:
            items_to_clear.append((caption_file, "caption file/folder"))
    
    # Add progress files if requested
    progress_files = []
    if clear_progress:
        progress_patterns = [
            "pipeline_progress.json",
            "duplicate_detection_progress.json",
            "processed_images.json",
            "*_progress.json"
        ]
        
        for pattern in progress_patterns:
            if '*' in pattern:
                import glob
                matches = glob.glob(pattern)
                for match in matches:
                    progress_file = Path(match)
                    if progress_file.exists():
                        progress_files.append(progress_file)
                        items_to_clear.append((progress_file, "progress file"))
            else:
                progress_file = Path(pattern)
                if progress_file.exists():
                    progress_files.append(progress_file)
                    items_to_clear.append((progress_file, "progress file"))
    
    if not items_to_clear:
        print(f"📁 No generated content found - nothing to clear.")
        return
    
    # Get stats before deletion
    total_files = 0
    total_size = 0
    
    print(f"🔍 Scanning items to clear...")
    
    for item_path, item_type in items_to_clear:
        if item_path.is_file():
            total_files += 1
            try:
                total_size += item_path.stat().st_size
            except OSError:
                pass
        elif item_path.is_dir():
            for root, _dirs, files in os.walk(item_path):
                total_files += len(files)
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except OSError:
                        pass
    
    print(f"📊 Found {total_files} files ({total_size / (1024*1024):.1f} MB) to clear:")
    
    # Show what will be cleared
    for item_path, item_type in items_to_clear:
        print(f"  • {item_path} ({item_type})")
    
    # Confirm deletion
    response = input(f"\n🗑️  Are you sure you want to delete all the above items? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("❌ Operation cancelled.")
        return
    
    print(f"🧹 Clearing generated content...")
    
    success_count = 0
    failed_count = 0
    failed_items = []
    
    # Clear each item
    for item_path, item_type in items_to_clear:
        print(f"\n🗑️  Clearing {item_type}: {item_path}")
        
        try:
            if item_path.is_file():
                if force_remove_file(str(item_path)):
                    success_count += 1
                    print(f"  ✅ Removed file")
                else:
                    failed_count += 1
                    failed_items.append(str(item_path))
                    
            elif item_path.is_dir():
                if force_remove_directory(item_path):
                    success_count += 1
                    print(f"  ✅ Removed directory")
                else:
                    failed_count += 1
                    failed_items.append(str(item_path))
            else:
                print(f"  ℹ️  Item doesn't exist (already cleared?)")
                
        except Exception as e:
            print(f"  ❌ Error clearing {item_path}: {e}")
            failed_count += 1
            failed_items.append(str(item_path))
    
    # Recreate the images folder
    try:
        images_path.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Recreated empty '{images_folder}' folder.")
    except Exception as e:
        print(f"\n⚠️  Could not recreate images folder: {e}")
    
    # Report results
    print("\n📊 Clearing Summary:")
    print(f"  ✅ Successfully cleared: {success_count} items")
    if failed_count > 0:
        print(f"  ❌ Failed to clear: {failed_count} items")
        print("\n🔧 Suggested solutions for stubborn files:")
        print("  1. Close any programs that might be using the files")
        print("  2. Wait a moment and try again (files might be temporarily locked)")
        print("  3. Restart your computer if the issue persists")
        print("  4. Try running the script as Administrator")
        
        # Show first few failed items
        print("\n📝 Failed items:")
        for failed_item in failed_items[:5]:
            print(f"  - {failed_item}")
        if len(failed_items) > 5:
            print(f"  ... and {len(failed_items) - 5} more")
    else:
        print("✅ All items successfully cleared!")


def clear_generated_images(folder_name: str = "generated_images"):
    """
    Clear the generated images folder and all its contents
    
    Args:
        folder_name: Name of the folder to clear (default: "generated_images")
    """
    # Use the robust method with captions and progress files
    clear_images_and_captions_robust(folder_name, clear_captions=True, clear_progress=True)


def main() -> None:
    """Main function"""
    print("🧹 Generated Content Cleaner")
    print("════════════════════════════")
    print("This script will clear:")
    print("  • Generated images folder")
    print("  • Caption files (train_captions.txt, val_captions.txt, etc.)")
    print("  • Progress files (pipeline_progress.json, etc.)")
    print("  • Dataset caption folders")
    
    # Ask what to clear
    print("\nWhat would you like to clear?")
    print("1. Everything (images + captions + progress) [DEFAULT]")
    print("2. Images only")
    print("3. Captions only") 
    print("4. Progress files only")
    print("5. Custom selection")
    
    choice = input("\nEnter choice (1-5, or press Enter for default): ").strip()
    
    if choice == "2":
        # Images only
        folder_path = Path("generated_images")
        if folder_path.exists():
            clear_images_and_captions_robust("generated_images", clear_captions=False, clear_progress=False)
        else:
            print("📁 No generated_images folder found.")
            
    elif choice == "3":
        # Captions only
        caption_files = get_caption_files()
        if caption_files:
            print(f"🔍 Found {len(caption_files)} caption files/folders to clear:")
            for cf in caption_files:
                print(f"  • {cf}")
            
            confirm = input("\nClear these caption files? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                for cf in caption_files:
                    try:
                        if cf.is_file():
                            cf.unlink()
                            print(f"  ✅ Removed: {cf}")
                        elif cf.is_dir():
                            shutil.rmtree(cf)
                            print(f"  ✅ Removed: {cf}")
                    except Exception as e:
                        print(f"  ❌ Failed to remove {cf}: {e}")
            else:
                print("❌ Caption clearing cancelled.")
        else:
            print("📁 No caption files found.")
            
    elif choice == "4":
        # Progress files only
        import glob
        progress_patterns = [
            "pipeline_progress.json",
            "duplicate_detection_progress.json", 
            "processed_images.json",
            "*_progress.json"
        ]
        
        progress_files = []
        for pattern in progress_patterns:
            if '*' in pattern:
                matches = glob.glob(pattern)
                progress_files.extend([Path(m) for m in matches if Path(m).exists()])
            else:
                pf = Path(pattern)
                if pf.exists():
                    progress_files.append(pf)
        
        if progress_files:
            print(f"🔍 Found {len(progress_files)} progress files:")
            for pf in progress_files:
                print(f"  • {pf}")
                
            confirm = input("\nClear these progress files? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                for pf in progress_files:
                    try:
                        pf.unlink()
                        print(f"  ✅ Removed: {pf}")
                    except Exception as e:
                        print(f"  ❌ Failed to remove {pf}: {e}")
            else:
                print("❌ Progress clearing cancelled.")
        else:
            print("📁 No progress files found.")
            
    elif choice == "5":
        # Custom selection
        clear_images = input("Clear images folder? (y/N): ").strip().lower() in ['y', 'yes']
        clear_captions = input("Clear caption files? (y/N): ").strip().lower() in ['y', 'yes']
        clear_progress = input("Clear progress files? (y/N): ").strip().lower() in ['y', 'yes']
        
        if clear_images or clear_captions or clear_progress:
            clear_images_and_captions_robust(
                "generated_images", 
                clear_captions=clear_captions, 
                clear_progress=clear_progress
            )
        else:
            print("❌ Nothing selected to clear.")
    else:
        # Default: clear everything
        clear_generated_images("generated_images")
    
    print("\n👍 All done!")


if __name__ == "__main__":
    main()
