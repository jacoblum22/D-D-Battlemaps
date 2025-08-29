"""
Smart Image Selector for Battlemap Processing

This module automatically selects the best version of battlemap images when
multiple versions exist (gridded vs gridless). It follows these rules:

1. If dimensions are in filename → prefer gridless version
2. If no dimensions in filename → prefer gridded version
3. Falls back to any available version if only one exists

This helps ensure we use visual grid detection when needed, but avoid
interfering grids when we have filename dimensions.
"""

import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ImageVariant:
    """Represents a variant of an image (gridded/gridless)"""

    path: str
    filename: str
    base_name: str  # Name without grid indicators
    has_dimensions: bool
    is_gridless: bool
    is_gridded: bool
    dimensions: Optional[Tuple[int, int]] = None


class SmartImageSelector:
    """
    Selects the best version of battlemap images based on grid/gridless variants
    """

    def __init__(self):
        # Patterns to identify gridless versions (case insensitive)
        # More flexible patterns that handle various separators or no separators
        self.gridless_patterns = [
            r"gridless",
            r"grid[\s_-]*less",
            r"[\s_-]gl[\s_-]*$",  # GL at end with separator before
            r"[\s_-]gl$",  # GL at end with separator
            r"^gl$",  # GL alone
            r"(?<=\w)GL_",  # GL after word character and before underscore
            r"DayGL(?=_)",  # Specific pattern like "DayGL_"
            r"MapGL$",  # Pattern like "ForestMapGL"
            r"GL(?=\.|$)",  # GL at very end before extension or end
            r"[\s_-]GL[\s_-]",  # GL with separators
            r"(?<=[a-z])GL(?=[A-Z])",  # GL between lowercase and uppercase (camelCase)
            r"(?<=[a-zA-Z])GL(?=[A-Z])",  # GL between any letter and uppercase
            r"no[\s_-]*grid",
            r"nogrid",
        ]

        # Patterns to identify gridded versions (case insensitive)
        # More flexible patterns that handle various separators or no separators
        self.gridded_patterns = [
            r"gridded",
            r"grid[\s_-]*on",
            r"with[\s_-]*grid",
            r"withgrid",
            r"[\s_-]grid[\s_-]*$",  # Grid at end with separator before
            r"[\s_-]grid$",  # Grid at end with separator
            r"^grid$",  # Grid alone
            r"(?<=\w)Grid_",  # Grid after word character and before underscore
            r"DayGrid(?=_)",  # Specific pattern like "DayGrid_"
            r"DayGrid(?=[A-Z])",  # Pattern like "DayGridDivinity"
            r"MapGrid$",  # Pattern like "ForestMapGrid"
            r"Grid(?=\.|$)",  # Grid at very end before extension or end
            r"[\s_-]Grid[\s_-]",  # Grid with separators
            r"(?<=[a-z])Grid(?=[A-Z])",  # Grid between lowercase and uppercase (camelCase)
            r"(?<=[a-zA-Z])Grid(?=[A-Z])",  # Grid between any letter and uppercase
        ]

        # Dimension extraction patterns - ordered from most specific to least specific
        # Now includes patterns for flexible separators and no separators
        self.dimension_patterns = [
            # Explicit brackets/parentheses (most specific)
            r"\((\d+)[x×](\d+)\)",
            r"\[(\d+)[x×](\d+)\]",
            r"\{(\d+)[x×](\d+)\}",
            # With underscores (specific)
            r"_(\d+)[x×](\d+)_",
            # With flexible separators (spaces, hyphens, underscores, or combinations)
            r"[\s_-]+(\d+)[\s_-]*[x×][\s_-]*(\d+)[\s_-]+",  # separators before and after
            r"[\s_-]+(\d+)[x×](\d+)[\s_-]+",  # separators around, no spaces around x
            # Word boundaries (existing)
            r"\b(\d+)[x×](\d+)\b",
            r"\b(\d+)\s*[x×]\s*(\d+)\b",
            # No separators (least specific - could match unwanted things)
            r"([1-9]\d?)[x×]([1-9]\d?)(?![0-9])",  # 1-99 x 1-99, must not be followed by digit
        ]

    def extract_dimensions_from_filename(
        self, filename: str
    ) -> Optional[Tuple[int, int]]:
        """Extract dimensions from filename (same as GridDetector)"""
        if not filename:
            return None

        name_without_ext = Path(filename).stem

        for pattern in self.dimension_patterns:
            match = re.search(pattern, name_without_ext, re.IGNORECASE)
            if match:
                try:
                    width = int(match.group(1))
                    height = int(match.group(2))
                    if 1 <= width <= 100 and 1 <= height <= 100:
                        return (width, height)
                except (ValueError, IndexError):
                    continue

        return None

    def is_gridless(self, filename: str) -> bool:
        """Check if filename indicates a gridless version"""
        name = Path(filename).stem.lower()
        return any(
            re.search(pattern, name, re.IGNORECASE)
            for pattern in self.gridless_patterns
        )

    def is_gridded(self, filename: str) -> bool:
        """Check if filename indicates a gridded version"""
        name = Path(filename).stem.lower()
        return any(
            re.search(pattern, name, re.IGNORECASE) for pattern in self.gridded_patterns
        )

    def get_base_name(self, filename: str) -> str:
        """
        Extract the base name without grid indicators or dimensions

        E.g., "Forest_20x25_Gridless.png" → "Forest"
             "Cave_15x10_GL.png" → "Cave"
             "Dungeon_Gridded.jpg" → "Dungeon"
             "GoblinCampBaseDayGL_Watermark.png" → "GoblinCampBaseDay_Watermark"
        """
        name = Path(filename).stem
        original_name = name

        # Strategy: Remove dimension patterns first, then grid indicators

        # Step 1: Remove dimension patterns
        for pattern in self.dimension_patterns:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE)

        # Step 2: Remove gridless indicators
        for pattern in self.gridless_patterns:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE)

        # Step 3: Remove gridded indicators
        for pattern in self.gridded_patterns:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE)

        # Step 4: Clean up separators without being too aggressive
        # Only collapse multiple separators, don't remove all
        name = re.sub(r"[\s_-]{2,}", "_", name)  # Only collapse 2+ separators
        name = name.strip("_")  # Remove leading/trailing separators

        # If we removed everything or got too short, use original
        if not name or len(name) < 2:
            return original_name

        return name

    def analyze_image(self, image_path: str, filename: str) -> ImageVariant:
        """Analyze a single image and return its variant info"""
        base_name = self.get_base_name(filename)
        dimensions = self.extract_dimensions_from_filename(filename)
        has_dimensions = dimensions is not None
        is_gridless = self.is_gridless(filename)
        is_gridded = self.is_gridded(filename)

        return ImageVariant(
            path=image_path,
            filename=filename,
            base_name=base_name,
            has_dimensions=has_dimensions,
            is_gridless=is_gridless,
            is_gridded=is_gridded,
            dimensions=dimensions,
        )

    def group_image_variants(
        self, image_list: List[Dict]
    ) -> Dict[str, List[ImageVariant]]:
        """
        Group images by their base name to find variants

        Args:
            image_list: List of dicts with 'path' and 'filename' keys

        Returns:
            Dict mapping base_name to list of ImageVariant objects
        """
        groups = defaultdict(list)

        for img_info in image_list:
            path = img_info.get("path", "")
            filename = img_info.get("filename", "")

            if not filename:
                continue

            variant = self.analyze_image(path, filename)
            groups[variant.base_name].append(variant)

        # Merge groups where one base is a subsequence of another
        groups_dict = dict(groups)
        self._merge_subsequence_groups(groups_dict)

        return groups_dict

    def _merge_subsequence_groups(self, groups: Dict[str, List[ImageVariant]]) -> None:
        """
        Merge groups where one base name is a subsequence of another.
        This handles cases like:
        - "GoblinCampBaseDay" and "GoblinCampBaseDayGL"
        - "Harbor" and "HarborGridded"
        - "BaseDayDivinityIsland" and "BaseDayDivinityIslandTrees"
        """
        # Convert to list to avoid modifying dict during iteration
        bases = list(groups.keys())

        for i, base1 in enumerate(bases):
            if base1 not in groups:  # Already merged
                continue

            for j, base2 in enumerate(bases[i + 1 :], i + 1):
                if base2 not in groups:  # Already merged
                    continue

                # Check substring first (more specific than subsequence)
                if base1 in base2:
                    # base1 is substring of base2, merge into base1 (shorter is more fundamental)
                    groups[base1].extend(groups[base2])
                    del groups[base2]
                elif base2 in base1:
                    # base2 is substring of base1, merge into base2
                    groups[base2].extend(groups[base1])
                    del groups[base1]
                    break  # base1 no longer exists
                # If not substring, check subsequence
                elif self._is_subsequence(base1, base2):
                    # base1 is subsequence of base2, merge into base1 (shorter is more fundamental)
                    groups[base1].extend(groups[base2])
                    del groups[base2]
                elif self._is_subsequence(base2, base1):
                    # base2 is subsequence of base1, merge into base2
                    groups[base2].extend(groups[base1])
                    del groups[base1]
                    break  # base1 no longer exists

    def _is_subsequence(self, s1: str, s2: str) -> bool:
        """
        Check if s1 is a subsequence of s2 (not just substring).
        A subsequence maintains order but doesn't need to be contiguous.
        """
        if len(s1) >= len(s2):
            return False

        # Convert to lowercase for comparison
        s1, s2 = s1.lower(), s2.lower()

        i = 0  # pointer for s1
        for char in s2:
            if i < len(s1) and s1[i] == char:
                i += 1

        return i == len(s1)  # All characters of s1 found in order

    def select_best_variant(self, variants: List[ImageVariant]) -> ImageVariant:
        """
        Select the best variant using improved rules:
        1. If dimensions in filename → use gridless version
        2. If no dimensions in filename → use gridless for output (detect on gridded if available)
        3. If only one variant type → use what's available

        The key insight: Always prefer gridless for final output, but detect on gridded when needed
        """
        if not variants:
            raise ValueError("No variants provided")

        if len(variants) == 1:
            return variants[0]

        # Rule 1: If any variant has dimensions in filename, prefer gridless with dimensions
        has_dimensions = any(v.has_dimensions for v in variants)

        if has_dimensions:
            # Always prefer gridless when dimensions are available
            gridless_variants = [v for v in variants if v.is_gridless]
            if gridless_variants:
                # If multiple gridless, prefer one with dimensions
                with_dims = [v for v in gridless_variants if v.has_dimensions]
                return with_dims[0] if with_dims else gridless_variants[0]

            # If no explicit gridless but dimensions exist, prefer the one with dimensions
            # This handles cases where gridless isn't explicitly marked but dimensions exist
            with_dims = [v for v in variants if v.has_dimensions]
            if with_dims:
                return with_dims[0]

        # Rule 2: No dimensions in filename - prefer gridless for output
        # The pipeline will detect on gridded variant if available
        gridless_variants = [v for v in variants if v.is_gridless]
        if gridless_variants:
            return gridless_variants[0]

        # Rule 3: Only gridded variants available - use gridded
        gridded_variants = [v for v in variants if v.is_gridded]
        if gridded_variants:
            return gridded_variants[0]

        # Subsequence check for remaining cases (from original logic)
        for i, variant_a in enumerate(variants):
            for j, variant_b in enumerate(variants):
                if i != j:
                    base_a = Path(variant_a.filename).stem
                    base_b = Path(variant_b.filename).stem

                    # If A is a subsequence of B, prefer B (longer name)
                    if self._is_subsequence(base_a, base_b):
                        return variant_b
                    # If B is a subsequence of A, prefer A (longer name)
                    elif self._is_subsequence(base_b, base_a):
                        return variant_a

        # Fall back to first variant
        return variants[0]

    def select_optimal_images(self, image_list: List[Dict]) -> List[Dict]:
        """
        Select the optimal version of each image according to the rules

        Args:
            image_list: List of dicts with 'path' and 'filename' keys

        Returns:
            List of selected image dicts with additional metadata
        """
        # Group variants by base name
        variant_groups = self.group_image_variants(image_list)

        selected_images = []

        for base_name, variants in variant_groups.items():
            best_variant = self.select_best_variant(variants)

            # Find gridded variant for potential grid detection
            gridded_variant = None
            for v in variants:
                if v.is_gridded:
                    gridded_variant = v
                    break

            # Create result dict with additional metadata
            result = {
                "path": best_variant.path,
                "filename": best_variant.filename,
                "base_name": base_name,
                "dimensions": best_variant.dimensions,
                "has_dimensions": best_variant.has_dimensions,
                "is_gridless": best_variant.is_gridless,
                "is_gridded": best_variant.is_gridded,
                "selection_reason": self._get_selection_reason(variants, best_variant),
                "total_variants": len(variants),
                "variant_types": [self._get_variant_type(v) for v in variants],
                # Add gridded variant info for potential grid detection
                "gridded_variant_path": gridded_variant.path if gridded_variant else None,
                "gridded_variant_filename": gridded_variant.filename if gridded_variant else None,
                "has_both_variants": len([v for v in variants if v.is_gridless]) > 0 and len([v for v in variants if v.is_gridded]) > 0,
            }

            selected_images.append(result)

        return selected_images

    def _get_variant_type(self, variant: ImageVariant) -> str:
        """Get a human-readable variant type"""
        types = []
        if variant.is_gridless:
            types.append("gridless")
        if variant.is_gridded:
            types.append("gridded")
        if variant.has_dimensions and variant.dimensions:
            types.append(f"dims({variant.dimensions[0]}x{variant.dimensions[1]})")
        return "+".join(types) if types else "plain"

    def _get_selection_reason(
        self, variants: List[ImageVariant], selected: ImageVariant
    ) -> str:
        """Get a human-readable selection reason"""
        if len(variants) == 1:
            return "only_variant"

        has_dimensions = any(v.has_dimensions for v in variants)
        has_gridless = any(v.is_gridless for v in variants)
        has_gridded = any(v.is_gridded for v in variants)

        if has_dimensions:
            if selected.is_gridless:
                return "gridless_with_filename_dims"
            elif selected.has_dimensions:
                return "has_filename_dims"
        else:
            # No filename dimensions
            if selected.is_gridless and has_gridded:
                return "gridless_for_output_detect_on_gridded"
            elif selected.is_gridless:
                return "gridless_only_option"
            elif selected.is_gridded:
                return "gridded_only_option"

        return "fallback"


def test_smart_selector():
    """Test the smart image selector"""
    print("🧪 Testing Smart Image Selector")
    print("=" * 50)

    selector = SmartImageSelector()

    # Test cases - these should group together
    test_images = [
        # Case 1: Forest variants - should prefer gridless (has dimensions)
        {
            "path": "/maps/Forest_20x25_Gridless.png",
            "filename": "Forest_20x25_Gridless.png",
        },
        {"path": "/maps/Forest_20x25_Grid.png", "filename": "Forest_20x25_Grid.png"},
        # Case 2: Dungeon variants - should prefer gridded (no dimensions)
        {"path": "/maps/Dungeon_Gridless.jpg", "filename": "Dungeon_Gridless.jpg"},
        {"path": "/maps/Dungeon_Gridded.jpg", "filename": "Dungeon_Gridded.jpg"},
        # Case 3: Single image - no choice
        {"path": "/maps/Harbor.webp", "filename": "Harbor.webp"},
        # Case 4: Cave variants - should prefer gridless (has dimensions)
        {"path": "/maps/Cave_15x10_GL.png", "filename": "Cave_15x10_GL.png"},
        {"path": "/maps/Cave_15x10_Gridded.png", "filename": "Cave_15x10_Gridded.png"},
        {"path": "/maps/Cave_NoGrid.png", "filename": "Cave_NoGrid.png"},
    ]

    # Test base name extraction first
    print("🔍 Base Name Extraction Test:")
    for img in test_images:
        filename = img["filename"]
        base_name = selector.get_base_name(filename)
        variant = selector.analyze_image(img["path"], filename)
        print(f"  {filename}")
        print(
            f"    → Base: '{base_name}' | Gridless: {variant.is_gridless} | Gridded: {variant.is_gridded} | Dims: {variant.dimensions}"
        )

    print("\n📊 Grouping Test:")
    groups = selector.group_image_variants(test_images)
    for base_name, variants in groups.items():
        print(f"\n� Group '{base_name}':")
        for v in variants:
            print(
                f"  - {v.filename} (gridless={v.is_gridless}, gridded={v.is_gridded}, dims={v.dimensions})"
            )

    print("\n�📋 Selection Results:")
    selected = selector.select_optimal_images(test_images)
    for img in selected:
        print(f"\n🎯 Selected: {img['filename']}")
        print(f"   Base name: {img['base_name']}")
        print(f"   Reason: {img['selection_reason']}")
        print(
            f"   Variants: {img['total_variants']} ({', '.join(img['variant_types'])})"
        )
        if img["dimensions"]:
            print(f"   Dimensions: {img['dimensions'][0]}x{img['dimensions'][1]}")

    print("\n✅ Smart selector test completed!")


if __name__ == "__main__":
    test_smart_selector()
