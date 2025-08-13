"""
Controlled Vocabulary Captioning System for D&D Battlemaps

This module provides a system for captioning battlemap images using a controlled
vocabulary and iterative improvement process.
"""

import os
import json
import random
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

import openai
from dotenv import load_dotenv
from PIL import Image


@dataclass
class Caption:
    """Structured caption for a battlemap image."""

    description: str
    terrain: List[str]
    features: List[str]
    scene_type: str
    style: str
    grid: str
    extras: List[str]
    attributes: Optional[Dict[str, List[str]]] = None
    image_path: str = ""
    raw_response: str = ""

    def __post_init__(self):
        """Initialize attributes if not provided."""
        if self.attributes is None:
            self.attributes = {}


@dataclass
class VocabularyStats:
    """Statistics about vocabulary usage and OOV terms."""

    total_images: int
    oov_terms: Dict[str, int]  # term -> frequency
    category_usage: Dict[str, Dict[str, int]]  # category -> term -> frequency
    successful_captions: int
    failed_captions: int
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0


class ControlledVocabularyCaptioner:
    """Main class for captioning battlemap images with controlled vocabulary."""

    # Controlled vocabulary lists
    TERRAIN = [
        "forest",
        "desert",
        "swamp",
        "snow",
        "ice",
        "coastline",
        "river",
        "lake",
        "waterfall",
        "lava",
        "cave",
        "cliffs",
        "grassland",
        "ruins",
        "dungeon",
        "sewer",
        "street",
        "plaza",
        "rooftop",
        "interior",
        "corridor",
        "garden",
        "path",
    ]

    FEATURES = [
        "altar",
        "anvil",
        "archway",
        "barricade",
        "barrel",
        "bed",
        "bench",
        "bloodstain",
        "boat",
        "bones",
        "bookshelf",
        "boulder",
        "brazier",
        "bridge",
        "broken wall",
        "buoy",
        "bush",
        "campfire",
        "candlestick",
        "cart",
        "chain",
        "chair",
        "chandelier",
        "chasm",
        "chest",
        "cobweb",
        "crates",
        "crystals",
        "debris",
        "desk",
        "dock",
        "door",
        "fence",
        "forge",
        "fountain",
        "gate",
        "gear",
        "grate",
        "ladder",
        "lantern",
        "lever",
        "mushrooms",
        "pipe",
        "pit",
        "pulley",
        "rail tracks",
        "rope",
        "roots",
        "rubble",
        "rug",
        "secret door",
        "stairs",
        "statue",
        "table",
        "tapestry",
        "tent",
        "tile mosaic",
        "torch",
        "trap",
        "tree",
        "vines",
        "wall",
        "well",
        "window",
        "workbench",
    ]

    SCENE_TYPES = [
        "tavern",
        "inn",
        "kitchen",
        "library",
        "study",
        "laboratory",
        "workshop",
        "armory",
        "barracks",
        "chapel",
        "crypt",
        "throne room",
        "market",
        "warehouse",
        "prison",
        "courtyard",
        "garden",
        "street",
        "sewer",
        "cave",
        "mine",
        "tower",
        "ruins",
        "wilderness",
        "road",
        "clearing",
        "camp",
        "bridge",
        "dock",
        "plaza",
        "dungeon",
    ]

    # Synonym mapping for normalization
    CANONICAL_TERMS = {
        "tables": "table",
        "wooden table": "table",
        "long table": "table",
        "bookshelves": "bookshelf",
        "shelves": "bookshelf",
        "staircase": "stairs",
        "stairway": "stairs",
        "cobwebs": "cobweb",
        "webs": "cobweb",
        "barrels": "barrel",
        "boxes": "crates",
        "crate": "crates",
        "chairs": "chair",
        "wooden chair": "chair",
        "doors": "door",
        "wooden door": "door",
        "trees": "tree",
        "wooden tree": "tree",
        "bushes": "bush",
        "shrubs": "bush",
        "boulders": "boulder",
        "rock": "boulder",  # Normalize rock to boulder
        "rocks": "boulder",
        "stone": "boulder",
        "torches": "torch",
        "torch sconce": "torch",
        "windows": "window",
        "glass window": "window",
        "wells": "well",
        "water well": "well",
        "bridges": "bridge",
        "wooden bridge": "bridge",
        "fences": "fence",
        "wooden fence": "fence",
        "gates": "gate",
        "wooden gate": "gate",
        "statues": "statue",
        "stone statue": "statue",
        "flowers": "bush",  # Normalize flower to bush
        "flower": "bush",
        "plants": "bush",
        "plant": "bush",
        # Gear synonyms
        "cog": "gear",
        "cogs": "gear",
        "cogwheel": "gear",
        "gearwheel": "gear",
        "gears": "gear",
        # Edge case aliases for consistent terminology
        "cracked wall": "broken wall",
        "damaged wall": "broken wall",
        "footprint": "tracks",
        "footprints": "tracks",
        "railway": "rail tracks",
        "railroad": "rail tracks",
        "rail road": "rail tracks",
        "train tracks": "rail tracks",
        "crystal": "crystals",
        "mushroom": "mushrooms",
        # Original normalization mappings
        "walls": "wall",
        "stone wall": "wall",
        "wooden wall": "wall",
        "chains": "chain",
        "metal chain": "chain",
        "iron chain": "chain",
        "lamp": "lantern",
        "lamps": "lantern",
        "lanterns": "lantern",
        "oil lamp": "lantern",
        "torch lamp": "lantern",
        "paths": "path",
        "stone path": "path",
        "dirt path": "path",
        "walkway": "path",
        "pathway": "path",
        # Fix case and plural issues
        "dungeons": "dungeon",
        "Dungeon": "dungeon",
        "interiors": "interior",
        "Interior": "interior",
        "plazas": "plaza",
        "Plaza": "plaza",
        "chasms": "chasm",
        "Chasm": "chasm",
        "corridors": "corridor",
        "Corridor": "corridor",
    }

    # Attributes mapping for descriptive terms
    ATTRIBUTE_MAPPINGS = {
        "glow": ("lighting", "glow"),
        "bioluminescent": ("lighting", "bioluminescent"),
        "dim": ("lighting", "dim"),
        "bright": ("lighting", "bright"),
        "glowing": ("lighting", "glow"),
        "broken": ("condition", "broken"),
        "damaged": ("condition", "damaged"),
        "ruined": ("condition", "ruined"),
        "cracked": ("condition", "cracked"),
        "raised": ("elevation", "raised"),
        "sunken": ("elevation", "sunken"),
        "elevated": ("elevation", "raised"),
        "roofed": ("coverage", "roofed"),
        "covered": ("coverage", "covered"),
        "roof": ("coverage", "roofed"),
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the captioner with OpenAI API."""
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)

        # Convert vocabulary lists to sets for faster lookup
        self.terrain_set = set(self.TERRAIN)
        self.features_set = set(self.FEATURES)
        self.scene_types_set = set(self.SCENE_TYPES)

    def get_caption_prompt(self) -> str:
        """Generate the structured prompt for OpenAI."""
        terrain_str = ", ".join(self.TERRAIN)
        features_str = ", ".join(self.FEATURES)
        scene_types_str = ", ".join(self.SCENE_TYPES)

        return f"""You are an expert in fantasy cartography and tabletop RPGs.
Describe each image as structured JSON for training an image generator.

Approved vocabulary (lowercase, singular nouns):

terrain = [{terrain_str}]
(Use terrain for the general environment/setting type)

features = [{features_str}]
(Use features for specific objects, structures, and details visible in the scene)

scene_type = [{scene_types_str}]
(Use scene_type for the specific type of location/room/area this represents)

Output this exact JSON:
{{
  "description": "<1–2 natural sentences; do not repeat the words 'battle map'>",
  "terrain": ["<1-2 from terrain list - the general environment>"],
  "features": ["<3–6 from features list - specific objects/details>"],
  "scene_type": "<1 from scene_type list - the specific location type>",
  "style": "<art style or mapping tool if identifiable, else 'digital painting'>",
  "grid": "<yes or no>",
  "extras": ["<free-text items not in the lists, optional>"],
  "attributes": {{"<category>": ["<descriptive terms>"]}}
}}

Rules:
- Use only the approved words in terrain/features/scene_type. Use attributes for descriptive terms (e.g., glow → lighting, broken → condition).
- Terrain = broad environment/setting (forest, interior, etc.), Features = discrete objects/props, Scene_type = functional purpose of the space
- For outdoor areas without specific buildings, use scene_type like "wilderness", "road", "clearing"
- If a term can belong to multiple fields, prefer: scene_type > terrain > features when it names the primary identity of the location (e.g., dungeon, tavern, sewer)
- Lowercase; singular nouns; sort arrays alphabetically; no duplicates.
- AVOID "unknown" scene_type - always try to find the closest match from the scene_type list
- If a term is missing from the vocabulary, choose the closest related term; only use extras when nothing is reasonably close
- Move descriptive words to attributes (e.g., glow → lighting, broken → condition)
- Features should be 3-6 items maximum.
- Grid should be "yes" if you can see a grid overlay, "no" otherwise.
- Description should be natural and engaging, not just a list of features.

Analyze this image:"""

    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def caption_single_image(
        self, image_path: str
    ) -> Tuple[Optional[Caption], int, float]:
        """Caption a single image using OpenAI GPT-4o. Returns (caption, tokens_used, cost_usd)."""
        try:
            # Encode image
            base64_image = self.encode_image_base64(image_path)

            # Create the request
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.get_caption_prompt()},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistency
            )

            # Calculate cost (GPT-4o pricing as of 2024)
            # Input: $5.00 per 1M tokens, Output: $15.00 per 1M tokens
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens

            cost_usd = (input_tokens * 5.00 / 1_000_000) + (
                output_tokens * 15.00 / 1_000_000
            )

            # Parse the response
            content = response.choices[0].message.content
            if content is None:
                print(f"Warning: Empty response for {image_path}")
                return None, total_tokens, cost_usd

            content = content.strip()

            # Extract JSON from the response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                print(f"Warning: No JSON found in response for {image_path}")
                return None, total_tokens, cost_usd

            json_str = content[json_start:json_end]

            try:
                caption_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error for {image_path}: {e}")
                print(f"Raw response: {content}")
                return None, total_tokens, cost_usd

            # Create Caption object
            caption = Caption(
                description=caption_data.get("description", ""),
                terrain=caption_data.get("terrain", []),
                features=caption_data.get("features", []),
                scene_type=caption_data.get("scene_type", ""),
                style=caption_data.get("style", ""),
                grid=caption_data.get("grid", ""),
                extras=caption_data.get("extras", []),
                attributes=caption_data.get("attributes", {}),
                image_path=image_path,
                raw_response=content,
            )

            return caption, total_tokens, cost_usd

        except Exception as e:
            print(f"Error captioning {image_path}: {e}")
            return None, 0, 0.0

    def normalize_terms(
        self, items: List[str], allowed_set: Set[str]
    ) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """Normalize terms using canonical mapping and extract OOV terms and attributes."""
        normalized = []
        oov_terms = []
        attributes = {}

        for item in items:
            if not item:
                continue

            # Clean and normalize
            item = item.strip().lower()

            # Skip compound descriptive phrases that shouldn't be normalized
            if len(item.split()) > 2:  # Skip long descriptive phrases
                continue

            # Check if it's an attribute term
            if item in self.ATTRIBUTE_MAPPINGS:
                category, value = self.ATTRIBUTE_MAPPINGS[item]
                if category not in attributes:
                    attributes[category] = []
                if value not in attributes[category]:
                    attributes[category].append(value)
                continue

            # Apply canonical mapping
            item = self.CANONICAL_TERMS.get(item, item)

            # Remove common adjectives
            for adj in [
                "wooden ",
                "stone ",
                "metal ",
                "iron ",
                "old ",
                "ancient ",
                "broken ",
                "cobblestone ",
            ]:
                if item.startswith(adj):
                    item = item[len(adj) :]
                    break

            # Check if it's in the allowed vocabulary
            if item in allowed_set and item not in normalized:
                normalized.append(item)
            elif (
                item not in allowed_set and item and len(item.split()) <= 2
            ):  # Only add simple terms as OOV
                oov_terms.append(item)

        # Sort and limit
        normalized = sorted(list(set(normalized)))
        if len(normalized) > 6:  # Cap features at 6
            normalized = normalized[:6]

        return normalized, oov_terms, attributes

    def normalize_caption(self, caption: Caption) -> Tuple[Caption, List[str]]:
        """Normalize a caption and extract OOV terms."""
        all_oov = []
        combined_attributes = caption.attributes.copy() if caption.attributes else {}

        # Normalize terrain
        terrain, terrain_oov, terrain_attrs = self.normalize_terms(
            caption.terrain, self.terrain_set
        )
        all_oov.extend(terrain_oov)
        self._merge_attributes(combined_attributes, terrain_attrs)

        # Normalize features
        features, features_oov, features_attrs = self.normalize_terms(
            caption.features, self.features_set
        )
        all_oov.extend(features_oov)
        self._merge_attributes(combined_attributes, features_attrs)

        # Normalize scene_type
        scene_type = caption.scene_type.strip().lower()
        scene_type = self.CANONICAL_TERMS.get(scene_type, scene_type)
        if scene_type not in self.scene_types_set:
            all_oov.append(scene_type)
            scene_type = "unknown"

        # Create normalized caption
        normalized = Caption(
            description=caption.description,
            terrain=terrain,
            features=features,
            scene_type=scene_type,
            style=caption.style,
            grid=caption.grid.lower() if caption.grid else "unknown",
            extras=caption.extras + all_oov,  # Move OOV to extras
            attributes=combined_attributes,
            image_path=caption.image_path,
            raw_response=caption.raw_response,
        )

        return normalized, all_oov

    def _merge_attributes(
        self, target: Dict[str, List[str]], source: Dict[str, List[str]]
    ):
        """Merge attributes dictionaries."""
        for category, values in source.items():
            if category not in target:
                target[category] = []
            for value in values:
                if value not in target[category]:
                    target[category].append(value)

    def load_processed_images(self) -> Set[str]:
        """Load list of previously processed images from caption files."""
        processed = set()

        # Check common caption files
        caption_files = [
            "captions_batch.json",
            "test_captions.json",
            "validation_captions.json",
            "phase1_captions.json",
            "phase2_captions.json",
            "phase3_captions.json",
        ]

        for filename in caption_files:
            if Path(filename).exists():
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        captions = json.load(f)
                    for caption in captions:
                        if isinstance(caption, dict) and "image_path" in caption:
                            # Normalize path separators
                            img_path = caption["image_path"].replace("\\", "/")
                            processed.add(img_path)
                            # Also add the original format in case of path separator differences
                            processed.add(caption["image_path"])
                except (json.JSONDecodeError, KeyError):
                    print(f"Warning: Could not load processed images from {filename}")
                    continue

        return processed

    def save_processed_images(
        self, image_paths: List[str], filename: str = "processed_images.json"
    ):
        """Save list of processed images for future exclusion."""
        existing_processed = self.load_processed_images()

        # Add new images
        all_processed = existing_processed.union(set(image_paths))

        # Save as a list (sets aren't JSON serializable)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(sorted(list(all_processed)), f, indent=2, ensure_ascii=False)

        print(f"Saved {len(all_processed)} total processed images to {filename}")

    def get_processing_stats(self) -> Dict[str, float]:
        """Get statistics about processed vs unprocessed images."""
        processed = self.load_processed_images()

        # Count total available images
        all_images = self.find_images(
            "generated_images", max_images=10000, exclude_processed=False
        )

        return {
            "total_available": len(all_images),
            "already_processed": len(processed),
            "remaining_unprocessed": len(all_images) - len(processed),
            "processed_percentage": (
                round(len(processed) / len(all_images) * 100, 1) if all_images else 0
            ),
        }

    def find_images(
        self, root_dir: str, max_images: int = 100, exclude_processed: bool = True
    ) -> List[str]:
        """Find random sample of images from the generated_images directory."""
        image_paths = []
        root_path = Path(root_dir)

        # Load previously processed images if requested
        processed_images = set()
        if exclude_processed:
            processed_images = self.load_processed_images()
            print(
                f"Found {len(processed_images)} previously processed images to exclude"
            )

        # Collect all image files
        for folder in root_path.iterdir():
            if folder.is_dir() and not folder.name.startswith("."):
                # Look for gdrive subfolder
                gdrive_folder = (
                    folder / f"gdrive_{folder.name.split('_')[0]}"
                    if "gdrive_" not in folder.name
                    else folder
                )
                if not gdrive_folder.exists():
                    # Try other common patterns
                    subfolders = [f for f in folder.iterdir() if f.is_dir()]
                    if subfolders:
                        gdrive_folder = subfolders[0]  # Take the first subfolder

                if gdrive_folder.exists():
                    for img_file in gdrive_folder.glob("*.png"):
                        img_path = str(img_file)
                        if not exclude_processed or img_path not in processed_images:
                            image_paths.append(img_path)
                else:
                    # Look directly in the folder
                    for img_file in folder.glob("*.png"):
                        img_path = str(img_file)
                        if not exclude_processed or img_path not in processed_images:
                            image_paths.append(img_path)

        print(f"Found {len(image_paths)} unprocessed images available")

        # Random sample
        if len(image_paths) > max_images:
            image_paths = random.sample(image_paths, max_images)

        print(f"Selected {len(image_paths)} images to process")
        return image_paths

    def process_batch(
        self, image_paths: List[str], output_file: str = "captions_batch.json"
    ) -> VocabularyStats:
        """Process a batch of images and analyze vocabulary usage."""
        captions = []
        all_oov = []
        successful = 0
        failed = 0
        total_tokens = 0
        total_cost = 0.0

        print(f"Processing {len(image_paths)} images...")

        for i, img_path in enumerate(image_paths, 1):
            print(f"Processing {i}/{len(image_paths)}: {Path(img_path).name}")

            # Caption the image
            caption, tokens_used, cost_usd = self.caption_single_image(img_path)

            total_tokens += tokens_used
            total_cost += cost_usd

            if caption is None:
                failed += 1
                continue

            # Normalize and extract OOV
            normalized_caption, oov_terms = self.normalize_caption(caption)
            captions.append(normalized_caption)
            all_oov.extend(oov_terms)
            successful += 1

            # Show running cost estimate
            if i % 5 == 0 or i == len(image_paths):
                avg_cost = total_cost / i if i > 0 else 0
                estimated_total = avg_cost * len(image_paths)
                print(
                    f"  Running cost: ${total_cost:.4f} | Avg per image: ${avg_cost:.4f} | Est. total: ${estimated_total:.4f}"
                )

        # Save captions
        with open(output_file, "w", encoding="utf-8") as f:
            caption_dicts = [asdict(cap) for cap in captions]
            json.dump(caption_dicts, f, indent=2, ensure_ascii=False)

        # Save list of processed images for future exclusion
        processed_paths = [cap.image_path for cap in captions]
        self.save_processed_images(processed_paths)

        # Analyze vocabulary usage
        category_usage = {
            "terrain": defaultdict(int),
            "features": defaultdict(int),
            "scene_type": defaultdict(int),
        }

        for caption in captions:
            for term in caption.terrain:
                category_usage["terrain"][term] += 1
            for term in caption.features:
                category_usage["features"][term] += 1
            if caption.scene_type != "unknown":
                category_usage["scene_type"][caption.scene_type] += 1

        # Count OOV terms
        oov_counter = Counter(all_oov)

        stats = VocabularyStats(
            total_images=len(image_paths),
            oov_terms=dict(oov_counter),
            category_usage=dict(category_usage),
            successful_captions=successful,
            failed_captions=failed,
            total_tokens_used=total_tokens,
            total_cost_usd=total_cost,
        )

        return stats

    def analyze_oov_terms(
        self, stats: VocabularyStats, min_frequency: int = 3
    ) -> Dict[str, List[str]]:
        """Analyze OOV terms and suggest vocabulary additions."""
        suggestions = {
            "high_frequency": [],  # Terms that appear frequently
            "terrain_candidates": [],  # Could be terrain
            "features_candidates": [],  # Could be features
            "scene_candidates": [],  # Could be scene types
        }

        # Terrain-related keywords
        terrain_keywords = [
            "mountain",
            "hill",
            "valley",
            "beach",
            "marsh",
            "bog",
            "tundra",
            "volcano",
            "canyon",
        ]

        # Feature-related keywords
        feature_keywords = [
            "pillar",
            "column",
            "beam",
            "pipe",
            "machinery",
            "crystal",
            "gem",
            "stone",
            "rock",
        ]

        # Scene-related keywords
        scene_keywords = [
            "hall",
            "chamber",
            "room",
            "building",
            "structure",
            "area",
            "zone",
        ]

        for term, frequency in stats.oov_terms.items():
            if frequency >= min_frequency:
                suggestions["high_frequency"].append(f"{term} ({frequency})")

                # Categorize based on keywords
                if any(keyword in term for keyword in terrain_keywords):
                    suggestions["terrain_candidates"].append(f"{term} ({frequency})")
                elif any(keyword in term for keyword in feature_keywords):
                    suggestions["features_candidates"].append(f"{term} ({frequency})")
                elif any(keyword in term for keyword in scene_keywords):
                    suggestions["scene_candidates"].append(f"{term} ({frequency})")

        return suggestions

    def generate_report(
        self,
        stats: VocabularyStats,
        suggestions: Dict[str, List[str]],
        output_file: str = "vocabulary_analysis.txt",
    ):
        """Generate a comprehensive vocabulary analysis report."""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== VOCABULARY ANALYSIS REPORT ===\n\n")

            f.write(f"Total images processed: {stats.total_images}\n")
            f.write(f"Successful captions: {stats.successful_captions}\n")
            f.write(f"Failed captions: {stats.failed_captions}\n")
            f.write(
                f"Success rate: {stats.successful_captions/stats.total_images*100:.1f}%\n\n"
            )

            # Cost analysis
            f.write("=== COST ANALYSIS ===\n\n")
            f.write(f"Total tokens used: {stats.total_tokens_used:,}\n")
            f.write(f"Total cost: ${stats.total_cost_usd:.4f}\n")
            if stats.successful_captions > 0:
                f.write(
                    f"Average cost per image: ${stats.total_cost_usd/stats.successful_captions:.4f}\n"
                )
                f.write(
                    f"Average tokens per image: {stats.total_tokens_used/stats.successful_captions:.0f}\n"
                )

            # Cost projections
            f.write(f"\nCost projections:\n")
            projections = [100, 500, 1000, 5000, 10000]
            avg_cost_per_image = (
                stats.total_cost_usd / stats.successful_captions
                if stats.successful_captions > 0
                else 0
            )
            for count in projections:
                projected_cost = avg_cost_per_image * count
                f.write(f"  {count:,} images: ${projected_cost:.2f}\n")
            f.write("\n")

            f.write("=== VOCABULARY USAGE ===\n\n")

            for category, usage in stats.category_usage.items():
                f.write(f"{category.upper()}:\n")
                sorted_usage = sorted(usage.items(), key=lambda x: x[1], reverse=True)
                for term, count in sorted_usage:
                    f.write(f"  {term}: {count}\n")
                f.write("\n")

            f.write("=== OUT-OF-VOCABULARY ANALYSIS ===\n\n")

            f.write(f"Total unique OOV terms: {len(stats.oov_terms)}\n")
            f.write(f"Total OOV occurrences: {sum(stats.oov_terms.values())}\n\n")

            f.write("HIGH FREQUENCY OOV TERMS (consider adding to vocabulary):\n")
            for term in suggestions["high_frequency"]:
                f.write(f"  {term}\n")
            f.write("\n")

            if suggestions["terrain_candidates"]:
                f.write("TERRAIN CANDIDATES:\n")
                for term in suggestions["terrain_candidates"]:
                    f.write(f"  {term}\n")
                f.write("\n")

            if suggestions["features_candidates"]:
                f.write("FEATURES CANDIDATES:\n")
                for term in suggestions["features_candidates"]:
                    f.write(f"  {term}\n")
                f.write("\n")

            if suggestions["scene_candidates"]:
                f.write("SCENE TYPE CANDIDATES:\n")
                for term in suggestions["scene_candidates"]:
                    f.write(f"  {term}\n")
                f.write("\n")

            f.write("ALL OOV TERMS (sorted by frequency):\n")
            sorted_oov = sorted(
                stats.oov_terms.items(), key=lambda x: x[1], reverse=True
            )
            for term, count in sorted_oov:
                f.write(f"  {term}: {count}\n")

        print(f"Analysis report saved to {output_file}")


def main():
    """Main function to run the captioning process."""
    # Initialize captioner
    captioner = ControlledVocabularyCaptioner()

    # Find images to process
    root_dir = "generated_images"
    image_paths = captioner.find_images(root_dir, max_images=100)

    if not image_paths:
        print("No images found!")
        return

    # Process batch
    print("Starting captioning process...")
    stats = captioner.process_batch(image_paths, "captions_batch.json")

    # Analyze OOV terms
    suggestions = captioner.analyze_oov_terms(stats, min_frequency=2)

    # Generate report
    captioner.generate_report(stats, suggestions, "vocabulary_analysis.txt")

    print("\n=== QUICK SUMMARY ===")
    print(f"Processed: {stats.successful_captions}/{stats.total_images} images")
    print(f"OOV terms found: {len(stats.oov_terms)}")
    print(f"High frequency OOV: {len(suggestions['high_frequency'])}")
    print("\nCheck 'vocabulary_analysis.txt' for detailed report")
    print("Check 'captions_batch.json' for all captions")


if __name__ == "__main__":
    main()
