#!/usr/bin/env python3
"""
Interactive Battlemap Pipeline Script

This script provides an interactive interface for running the complete battlemap
processing pipeline. It allows users to:

1. Configure pipeline settings
2. Add multiple input sources (Google Drive, zip files, local directories)
3. Run the pipeline with progress tracking
4. Resume from saved progress
5. View processing statistics

Usage:
    python run_battlemap_pipeline.py
"""

import sys
import os
from pathlib import Path
from typing import List, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import pipeline components
from battlemap_processor.core.battlemap_pipeline import (
    BattlemapPipeline,
    PipelineConfig,
)


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


def get_int_input(
    prompt: str, default: Optional[int] = None, allow_none: bool = False
) -> Optional[int]:
    """Get integer input from user"""
    if default is not None:
        full_prompt = f"{prompt} [{default}]: "
    elif allow_none:
        full_prompt = f"{prompt} [unlimited]: "
    else:
        full_prompt = f"{prompt}: "

    response = input(full_prompt).strip()

    if not response:
        return default

    if allow_none and response.lower() in ["none", "unlimited", "no limit"]:
        return None

    try:
        return int(response)
    except ValueError:
        print("❌ Invalid number. Please enter a valid integer.")
        return get_int_input(prompt, default, allow_none)


def get_float_input(prompt: str, default: float) -> float:
    """Get float input from user"""
    response = input(f"{prompt} [{default}]: ").strip()

    if not response:
        return default

    try:
        return float(response)
    except ValueError:
        print("❌ Invalid number. Please enter a valid decimal number.")
        return get_float_input(prompt, default)


def collect_sources() -> List[str]:
    """Collect input sources from user"""
    print("\n📂 Input Sources Configuration")
    print("════════════════════════════════")
    print("Enter your image sources (Google Drive URLs, zip files, local directories).")
    print("Examples:")
    print("  • https://drive.google.com/drive/folders/1ABC123...")
    print("  • https://example.com/battlemaps.zip")
    print("  • /path/to/local/directory")
    print("  • C:\\\\Users\\\\username\\\\Pictures\\\\Battlemaps")
    print(
        "\n💡 TIP: You can enter multiple Google Drive URLs separated by ', ' (comma + space)"
    )
    print(
        "Example: https://drive.google.com/.../folder1, https://drive.google.com/.../folder2"
    )
    print("\nType 'done' when finished adding sources.")

    sources = []
    while True:
        source = get_user_input(f"\nSource {len(sources) + 1} (or 'done')")

        if source.lower() == "done":
            if not sources:
                print("⚠️  You must add at least one source!")
                continue
            break

        if source:
            # Check if input contains multiple comma-separated URLs
            if ", " in source:
                # Split on comma + space and clean each URL
                multiple_sources = [
                    url.strip() for url in source.split(", ") if url.strip()
                ]

                if multiple_sources:
                    print(
                        f"\n🔍 Detected {len(multiple_sources)} sources in your input:"
                    )
                    for i, url in enumerate(multiple_sources, 1):
                        # Truncate long URLs for display
                        display_url = url if len(url) <= 60 else url[:57] + "..."
                        print(f"  {i}. {display_url}")

                    # Ask for confirmation
                    confirm = get_yes_no(
                        f"\nAdd all {len(multiple_sources)} sources?", default=True
                    )

                    if confirm:
                        for url in multiple_sources:
                            sources.append(url)
                        print(f"✅ Added {len(multiple_sources)} sources")
                    else:
                        print("❌ Batch input cancelled")
                else:
                    print("⚠️  No valid URLs found in comma-separated input")
            else:
                # Single source - handle as before
                sources.append(source)
                print(f"✅ Added: {source}")

    return sources


def configure_pipeline() -> PipelineConfig:
    """Configure pipeline settings interactively"""
    print("\\n⚙️  Pipeline Configuration")
    print("═══════════════════════════")

    # Collect sources
    sources = collect_sources()

    print("\\n🎯 Processing Settings")
    print("─────────────────────")

    # Smart selection
    use_smart_selection = get_yes_no(
        "Use smart image selection (prefer gridless over gridded variants)?",
        default=True,
    )

    # Limits for proof-of-concept
    print("\\n🔢 Processing Limits")
    print("─────────────────────")
    print(
        "Set limits to create a manageable dataset (or use unlimited for full processing):"
    )

    max_images = get_int_input(
        "Maximum images to process (or 'unlimited')", default=20, allow_none=True
    )

    max_tiles_per_image = get_int_input(
        "Maximum tiles per image (or 'unlimited')",
        default=None,  # Changed to unlimited by default
        allow_none=True,
    )

    # Detection settings
    print("\\n🔍 Detection Settings")
    print("──────────────────────")

    tile_size = get_int_input("Tile size (grid squares)", default=12)
    boring_threshold = get_float_input(
        "Boring threshold (max fraction of boring squares per tile)", default=0.5
    )

    # Output settings
    print("\\n💾 Output Settings")
    print("───────────────────")

    output_dir = get_user_input("Output directory", default="generated_images")
    tile_output_size = get_int_input("Tile output size (pixels)", default=512)

    # Processing settings
    print("\\n🛠️  Processing Settings")
    print("────────────────────────")

    save_progress = get_yes_no("Save progress for resuming?", default=True)
    debug = get_yes_no("Enable debug output?", default=True)

    # Create config
    config = PipelineConfig(
        sources=sources,
        use_smart_selection=use_smart_selection,
        max_images=max_images,
        max_tiles_per_image=max_tiles_per_image,
        tile_size=tile_size or 12,  # Provide default if None
        boring_threshold=boring_threshold,
        output_dir=output_dir,
        tile_output_size=tile_output_size or 512,  # Provide default if None
        save_progress=save_progress,
        debug=debug,
    )

    return config


def show_config_summary(config: PipelineConfig):
    """Show a summary of the pipeline configuration"""
    print("\\n📋 Configuration Summary")
    print("═════════════════════════")
    print(f"Sources:              {len(config.sources)} source(s)")
    for i, source in enumerate(config.sources, 1):
        source_display = source if len(source) <= 60 else source[:57] + "..."
        print(f"  {i}. {source_display}")
    print(
        f"Smart selection:      {'✅ Enabled' if config.use_smart_selection else '❌ Disabled'}"
    )
    print(f"Max images:           {config.max_images or 'Unlimited'}")
    print(f"Max tiles per image:  {config.max_tiles_per_image or 'Unlimited'}")
    print(f"Tile size:            {config.tile_size}x{config.tile_size} squares")
    print(f"Boring threshold:     {config.boring_threshold:.1%}")
    print(f"Output directory:     {config.output_dir}")
    print(
        f"Tile output size:     {config.tile_output_size}x{config.tile_output_size} pixels"
    )
    print(f"Save progress:        {'✅ Yes' if config.save_progress else '❌ No'}")
    print(f"Debug output:         {'✅ Enabled' if config.debug else '❌ Disabled'}")


def check_for_existing_progress(output_dir: str) -> bool:
    """Check if there's existing progress that can be resumed"""
    progress_file = Path(output_dir) / "pipeline_progress.json"
    return progress_file.exists()


def main():
    """Main interactive function"""
    print("🗺️  Battlemap Processing Pipeline")
    print("═══════════════════════════════════")
    print("Generate training tiles from battlemap images")
    print("Supports Google Drive, zip files, and local directories")

    try:
        # Check for existing progress first
        default_output = "generated_images"
        if check_for_existing_progress(default_output):
            resume = get_yes_no(
                f"\\n📁 Found existing progress in '{default_output}'. Resume from where you left off?",
                default=True,
            )

            if resume:
                print("\\n▶️  Resuming from saved progress...")
                # Load existing config and run
                config = PipelineConfig(
                    sources=[], output_dir=default_output
                )  # Will be loaded from progress
                pipeline = BattlemapPipeline(config)
                if pipeline.resume_from_progress():
                    print("✅ Progress loaded successfully")
                    print("\\n🚀 Continuing pipeline processing...")
                    stats = pipeline.run()
                    print("\\n🎉 Pipeline completed!")
                    return
                else:
                    print("❌ Could not load progress. Starting fresh...")

        # Configure new pipeline
        config = configure_pipeline()

        # Show summary and confirm
        show_config_summary(config)

        if not get_yes_no("\\nProceed with this configuration?", default=True):
            print("❌ Configuration cancelled.")
            return

        # Create and run pipeline
        print("\\n🚀 Starting pipeline...")
        pipeline = BattlemapPipeline(config)

        try:
            stats = pipeline.run()
            print("\\n🎉 Pipeline completed successfully!")

        except KeyboardInterrupt:
            print("\\n\\n⏸️  Pipeline interrupted by user.")
            if config.save_progress:
                print(
                    "💾 Progress has been saved. You can resume later by running this script again."
                )
            sys.exit(0)

        except Exception as e:
            print(f"\\n❌ Pipeline error: {e}")
            if config.save_progress:
                print(
                    "💾 Progress has been saved. You can resume later after fixing the issue."
                )
            sys.exit(1)

    except KeyboardInterrupt:
        print("\\n\\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
