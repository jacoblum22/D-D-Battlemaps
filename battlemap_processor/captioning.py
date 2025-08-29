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
        "pillar",
        "pit",
        "pond",
        "pool",
        "pulley",
        "rail tracks",
        "rope",
        "roots",
        "rubble",
        "rug",
        "sarcophagus",
        "secret door",
        "skull",
        "stairs",
        "statue",
        "stump",
        "table",
        "tapestry",
        "tent",
        "tile mosaic",
        "tools",
        "torch",
        "trap",
        "tree",
        "vines",
        "wall",
        "waterfall",
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
        "stones": "boulder",
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
        # Water feature normalizations
        "pools": "pool",
        "water pool": "pool",
        "ponds": "pond",
        "water pond": "pond",
        "waterfalls": "waterfall",
        "water fall": "waterfall",
        # Tool normalizations
        "tool": "tools",
        "workshop tools": "tools",
        "crafting tools": "tools",
        # Death/dungeon feature normalizations
        "skulls": "skull",
        "human skull": "skull",
        "animal skull": "skull",
        "sarcophagi": "sarcophagus",
        "coffin": "sarcophagus",  # Normalize coffin to sarcophagus
        "coffins": "sarcophagus",
        # Tree feature normalizations
        "stumps": "stump",
        "tree stump": "stump",
        "log": "stump",  # Tree logs could be stumps
        "logs": "stump",
        # Additional feature alignment mappings
        "fire": "torch",
        "flame": "torch",
        "flames": "torch",
        "gem": "crystals",
        "gems": "crystals",
        "pillar": "pillar",  # Add pillar to features if not already there
        "column": "pillar",
        "columns": "pillar",
        "pillars": "pillar",
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
        "warm": ("lighting", "warm"),
        "broken": ("condition", "broken"),
        "damaged": ("condition", "damaged"),
        "ruined": ("condition", "ruined"),
        "cracked": ("condition", "cracked"),
        "twisted": ("condition", "twisted"),
        "organic": ("material", "organic"),
        "raised": ("elevation", "raised"),
        "sunken": ("elevation", "sunken"),
        "elevated": ("elevation", "raised"),
        "roofed": ("coverage", "roofed"),
        "covered": ("coverage", "covered"),
        "roof": ("coverage", "roofed"),
        "mystical": ("magical", "mystical"),
        "colorful": ("appearance", "colorful"),
        "stained": ("condition", "stained"),
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
  "description": "<1–2 natural sentences; describe as a top-down battlemap tile>",
  "terrain": ["<1-2 from terrain list - the general environment>"],
  "features": ["<3–6 from features list - specific objects/details>"],
  "scene_type": "<1 from scene_type list - the specific location type>",
  "style": "<art style or mapping tool if identifiable, else 'digital battlemap'>",
  "extras": ["<free-text items not in the lists, optional>"],
  "attributes": {{"<category>": ["<descriptive terms>"]}}
}}

EXAMPLES:

Forest clearing with pond:
{{
  "description": "A vibrant forest clearing surrounds a small pond, with lush foliage and colorful flowers creating a peaceful battlemap scene.",
  "terrain": ["forest"],
  "features": ["bench", "bush", "pond", "tree"],
  "scene_type": "clearing",
  "style": "digital battlemap",
  "extras": [],
  "attributes": {{"color": ["vibrant", "colorful"], "lighting": ["dappled", "natural"], "density": ["lush"], "condition": ["peaceful"]}}
}}

Lava landscape:
{{
  "description": "A fiery top-down view reveals molten lava flows through rocky terrain, with a carved stone face overlooking the scene.",
  "terrain": ["lava"],
  "features": ["boulder", "statue", "wall"],
  "scene_type": "wilderness", 
  "style": "digital battlemap",
  "extras": [],
  "attributes": {{"condition": ["carved", "rocky"], "lighting": ["fiery", "glowing"], "material": ["molten"]}}
}}

Urban street scene:
{{
  "description": "This urban battlemap shows a network of streets and buildings, with cobblestone paths and illuminated corners from above.",
  "terrain": ["street"],
  "features": ["lantern", "wall"],
  "scene_type": "street",
  "style": "digital battlemap",
  "extras": ["building"],
  "attributes": {{"lighting": ["illuminated", "bright"], "material": ["cobblestone"], "density": ["urban"]}}
}}

Rules:
- Use only the approved words in terrain/features/scene_type. Use attributes for descriptive terms (e.g., glow → lighting, broken → condition).
- Terrain = broad environment/setting (forest, interior, etc.), Features = discrete objects/props, Scene_type = functional purpose of the space
- For outdoor areas without specific buildings, use scene_type like "wilderness", "road", "clearing"
- If a term can belong to multiple fields, prefer: scene_type > terrain > features when it names the primary identity of the location (e.g., dungeon, tavern, sewer)
- Always describe from a top-down perspective appropriate for battlemap use
- Use "digital battlemap" as default style unless clearly identifiable otherwise

TERRAIN DISTINCTION RULES:
- "path" = outdoor walkways without walls (dirt paths, stone paths, forest trails)
- "corridor" = indoor passages with walls (dungeon hallways, building corridors)

WATER FEATURE DISTINCTIONS:
- "fountain" = artificial decorative water feature with pumps/spouts
- "pool" = artificial contained water (swimming pools, ritual pools)
- "pond" = natural small body of standing water
- "waterfall" = flowing water falling from height
- Use "lake" or "river" as terrain only for large water bodies that dominate the scene

- Lowercase; singular nouns; sort arrays alphabetically; no duplicates.
- AVOID "unknown" scene_type - always try to find the closest match from the scene_type list
- If a term is missing from the vocabulary, choose the closest related term; only use extras when nothing is reasonably close
- Move descriptive words to attributes (e.g., glow → lighting, broken → condition, twisted → condition, organic → material)
- Features should be 3-6 items maximum.
- Description should reference the top-down/battlemap perspective naturally (but this is optional - only ~30% need explicit style markers).
- Attributes should include 2-3 categories: lighting (dim, glowing, bright), condition (ruined, overgrown, cracked), color (vibrant, green, grey), density (dense, sparse, crowded), material (stone, wood, metal).
- Each attribute category should have 1-3 specific tags for richness.
- Ensure all major objects mentioned in description also appear in features list.

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
                model="gpt-4o-mini",
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

            # Calculate cost (GPT-4o-mini pricing as of 2024)
            # Input: $0.15 per 1M tokens, Output: $0.60 per 1M tokens
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens

            cost_usd = (input_tokens * 0.15 / 1_000_000) + (
                output_tokens * 0.60 / 1_000_000
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

            # Create Caption object with normalized path
            normalized_image_path = image_path.replace("\\", "/")
            caption = Caption(
                description=caption_data.get("description", ""),
                terrain=caption_data.get("terrain", []),
                features=caption_data.get("features", []),
                scene_type=caption_data.get("scene_type", ""),
                style=caption_data.get("style", ""),
                extras=caption_data.get("extras", []),
                attributes=caption_data.get("attributes", {}),
                image_path=normalized_image_path,
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

    def smart_reclassify_terms(self, caption: Caption) -> Caption:
        """Intelligently move misclassified terms to their correct categories."""

        # Create working copies
        terrain = list(caption.terrain)
        features = list(caption.features)
        scene_type = caption.scene_type
        extras = list(caption.extras)

        # Define categorization rules
        terrain_terms = set(self.TERRAIN)
        features_terms = set(self.FEATURES)
        scene_type_terms = set(self.SCENE_TYPES)

        # Function to move term between lists
        def move_term(term: str, from_list: List[str], to_list: List[str]) -> bool:
            """Move term from one list to another if found."""
            normalized_term = self.CANONICAL_TERMS.get(
                term.lower().strip(), term.lower().strip()
            )
            if normalized_term in from_list:
                from_list.remove(normalized_term)
                if normalized_term not in to_list:
                    to_list.append(normalized_term)
                return True
            return False

        # Check all lists for misplaced terms and move them
        all_terms_to_check = set()
        all_terms_to_check.update([t.lower().strip() for t in terrain])
        all_terms_to_check.update([t.lower().strip() for t in features])
        all_terms_to_check.update([scene_type.lower().strip()] if scene_type else [])
        all_terms_to_check.update([t.lower().strip() for t in extras])

        for term in all_terms_to_check:
            normalized_term = self.CANONICAL_TERMS.get(term, term)

            # Skip empty terms
            if not normalized_term:
                continue

            # If it's a terrain term, ensure it's in terrain
            if normalized_term in terrain_terms:
                # Remove from wrong places
                if normalized_term in features:
                    features.remove(normalized_term)
                if normalized_term == scene_type:
                    scene_type = "unknown"  # Will be fixed later
                if normalized_term in extras:
                    extras.remove(normalized_term)
                # Add to terrain if not already there
                if normalized_term not in terrain:
                    terrain.append(normalized_term)

            # If it's a features term, ensure it's in features
            elif normalized_term in features_terms:
                # Remove from wrong places
                if normalized_term in terrain:
                    terrain.remove(normalized_term)
                if normalized_term == scene_type:
                    scene_type = "unknown"  # Will be fixed later
                if normalized_term in extras:
                    extras.remove(normalized_term)
                # Add to features if not already there
                if normalized_term not in features:
                    features.append(normalized_term)

            # If it's a scene_type term, ensure it's the scene_type
            elif normalized_term in scene_type_terms:
                # Remove from wrong places
                if normalized_term in terrain:
                    terrain.remove(normalized_term)
                if normalized_term in features:
                    features.remove(normalized_term)
                if normalized_term in extras:
                    extras.remove(normalized_term)
                # Set as scene_type (prioritize over unknown)
                if scene_type == "unknown" or not scene_type:
                    scene_type = normalized_term

        # If scene_type is still unknown, try to infer from context
        if scene_type == "unknown" or not scene_type:
            # Simple heuristics for common cases
            if "interior" in terrain:
                if any(term in features for term in ["altar", "throne"]):
                    scene_type = "throne room"
                elif any(term in features for term in ["bed", "chest"]):
                    scene_type = "inn"
                elif any(term in features for term in ["bookshelf", "desk"]):
                    scene_type = "study"
                else:
                    scene_type = "dungeon"  # Default for interior
            elif any(t in terrain for t in ["forest", "grassland", "cliffs"]):
                scene_type = "wilderness"
            elif "street" in terrain:
                scene_type = "street"
            elif "sewer" in terrain:
                scene_type = "sewer"
            else:
                scene_type = "wilderness"  # Default fallback

        # Create new caption with reclassified terms
        return Caption(
            description=caption.description,
            terrain=sorted(list(set(terrain))),  # Remove duplicates and sort
            features=sorted(list(set(features))),
            scene_type=scene_type,
            style=caption.style,
            extras=sorted(list(set(extras))),
            attributes=caption.attributes,
            image_path=caption.image_path,
            raw_response=caption.raw_response,
        )

    def normalize_caption(self, caption: Caption) -> Tuple[Caption, List[str]]:
        """Normalize a caption and extract OOV terms."""
        # First, intelligently reclassify misplaced terms
        reclassified_caption = self.smart_reclassify_terms(caption)

        all_oov = []
        combined_attributes = (
            reclassified_caption.attributes.copy()
            if reclassified_caption.attributes
            else {}
        )

        # Normalize terrain
        terrain, terrain_oov, terrain_attrs = self.normalize_terms(
            reclassified_caption.terrain, self.terrain_set
        )
        all_oov.extend(terrain_oov)
        self._merge_attributes(combined_attributes, terrain_attrs)

        # Normalize features
        features, features_oov, features_attrs = self.normalize_terms(
            reclassified_caption.features, self.features_set
        )
        all_oov.extend(features_oov)
        self._merge_attributes(combined_attributes, features_attrs)

        # Normalize scene_type
        scene_type = reclassified_caption.scene_type.strip().lower()
        scene_type = self.CANONICAL_TERMS.get(scene_type, scene_type)
        if scene_type not in self.scene_types_set:
            all_oov.append(scene_type)
            scene_type = "unknown"

        # Create normalized caption with improved post-processing
        description = reclassified_caption.description

        # 1. STYLE MARKERS: Only add battlemap/top-down to ~30% of captions randomly
        import random

        add_style_markers = random.random() < 0.30

        if add_style_markers:
            has_battlemap = "battlemap" in description.lower()
            has_topdown = (
                "top-down" in description.lower() or "top down" in description.lower()
            )

            if not has_battlemap and not has_topdown:
                # Add both if neither is present
                description = (
                    description.rstrip(".") + " shown in this top-down battlemap view."
                )
            elif not has_battlemap:
                # Just add battlemap
                if (
                    "top-down" in description.lower()
                    or "top down" in description.lower()
                ):
                    description = description.replace(
                        "top-down", "top-down battlemap"
                    ).replace("top down", "top-down battlemap")
                else:
                    description = (
                        description.rstrip(".") + " as seen in this battlemap."
                    )
            elif not has_topdown:
                # Just add top-down
                description = description.replace("battlemap", "top-down battlemap")

        # 2. SCENE TYPE CONSISTENCY: Improve cave/dungeon classification
        if scene_type == "unknown" or not scene_type:
            # Apply improved scene type rules
            if "cave" in terrain:
                # Check if it's structured (dungeon) or natural (wilderness)
                structured_keywords = [
                    "temple",
                    "ruins",
                    "mine",
                    "chamber",
                    "hall",
                    "room",
                    "passage",
                    "corridor",
                ]
                natural_keywords = [
                    "cavern",
                    "grotto",
                    "natural",
                    "wild",
                    "exploration",
                ]

                description_lower = description.lower()
                has_structured = any(
                    keyword in description_lower for keyword in structured_keywords
                )
                has_natural = any(
                    keyword in description_lower for keyword in natural_keywords
                )

                if has_structured or any(
                    f in features
                    for f in ["altar", "chest", "throne", "statue", "pillar"]
                ):
                    scene_type = "dungeon"
                else:
                    scene_type = "wilderness"
            elif "interior" in terrain:
                if any(term in features for term in ["altar", "throne"]):
                    scene_type = "throne room"
                elif any(term in features for term in ["bed", "chest"]):
                    scene_type = "inn"
                elif any(term in features for term in ["bookshelf", "desk"]):
                    scene_type = "study"
                else:
                    scene_type = "dungeon"
            elif any(t in terrain for t in ["forest", "grassland", "cliffs"]):
                scene_type = "wilderness"
            elif "street" in terrain:
                scene_type = "street"
            elif "sewer" in terrain:
                scene_type = "sewer"
            else:
                scene_type = "wilderness"

        # 3. FEATURE-DESCRIPTION ALIGNMENT: Extract key nouns from description
        import re

        description_words = re.findall(r"\b\w+\b", description.lower())

        # Key nouns that should be in features if mentioned
        potential_features = {
            "lava": "lava",
            "water": "pool",
            "fire": "torch",
            "flame": "torch",
            "crystal": "crystals",
            "gem": "crystals",
            "rock": "boulder",
            "stone": "boulder",
            "altar": "altar",
            "throne": "throne",
            "chest": "chest",
            "statue": "statue",
            "pillar": "pillar",
            "column": "pillar",
            "torch": "torch",
            "lantern": "lantern",
            "tree": "tree",
            "bush": "bush",
            "flower": "bush",
            "plant": "bush",
            "wall": "wall",
            "door": "door",
            "gate": "gate",
            "fence": "fence",
            "pool": "pool",
            "pond": "pond",
            "fountain": "fountain",
            "well": "well",
            "bridge": "bridge",
            "stairs": "stairs",
            "ladder": "ladder",
            "barrel": "barrel",
            "crate": "crates",
            "table": "table",
            "chair": "chair",
        }

        # Add missing features that are mentioned in description
        additional_features = []
        for word in description_words:
            if word in potential_features:
                mapped_feature = potential_features[word]
                if (
                    mapped_feature in self.features_set
                    and mapped_feature not in features
                ):
                    additional_features.append(mapped_feature)

        # Merge additional features (keeping limit of 6 total)
        all_features = features + additional_features
        features = sorted(list(set(all_features)))[:6]

        # 4. ATTRIBUTE RICHNESS: Ensure 2-3 attribute categories with multiple tags
        # Extract more attributes from description
        additional_attributes = {}

        # Lighting attributes
        lighting_terms = {
            "glowing": "glowing",
            "bright": "bright",
            "dim": "dim",
            "dark": "dim",
            "illuminated": "bright",
            "torch-lit": "torch-lit",
            "moonlit": "moonlit",
            "shadowy": "dim",
            "fiery": "fiery",
            "radiant": "bright",
            "warm": "warm",
            "soft": "soft",
            "natural": "natural",
            "daylight": "daylight",
            "sunlit": "bright",
        }

        # Condition attributes
        condition_terms = {
            "ruined": "ruined",
            "broken": "broken",
            "cracked": "cracked",
            "overgrown": "overgrown",
            "mossy": "mossy",
            "flooded": "flooded",
            "treacherous": "treacherous",
            "jagged": "jagged",
            "smooth": "smooth",
            "weathered": "weathered",
            "ancient": "ancient",
            "worn": "worn",
            "mysterious": "mysterious",
            "peaceful": "peaceful",
            "rustic": "rustic",
            "well-maintained": "well-maintained",
            "scattered": "scattered",
            "twisted": "twisted",
            "natural": "natural",
            "lush": "lush",
            "dense": "dense",
            "sandy": "sandy",
        }

        # Color attributes
        color_terms = {
            "green": "green",
            "red": "red",
            "blue": "blue",
            "grey": "grey",
            "gray": "grey",
            "brown": "brown",
            "golden": "golden",
            "silver": "silver",
            "black": "black",
            "white": "white",
            "colorful": "colorful",
            "vibrant": "vibrant",
            "lush": "green",
        }

        # Density/scale attributes
        density_terms = {
            "dense": "dense",
            "thick": "dense",
            "sparse": "sparse",
            "crowded": "crowded",
            "sprawling": "sprawling",
            "vast": "vast",
            "narrow": "narrow",
            "wide": "wide",
            "bustling": "crowded",
            "lively": "crowded",
        }

        # Material attributes
        material_terms = {
            "stone": "stone",
            "wooden": "wood",
            "wood": "wood",
            "metal": "metal",
            "iron": "metal",
            "cobblestone": "stone",
            "rocky": "stone",
            "molten": "molten",
        }

        # Extract attributes from description
        for word in description_words:
            if word in lighting_terms:
                if "lighting" not in additional_attributes:
                    additional_attributes["lighting"] = []
                if lighting_terms[word] not in additional_attributes["lighting"]:
                    additional_attributes["lighting"].append(lighting_terms[word])

            if word in condition_terms:
                if "condition" not in additional_attributes:
                    additional_attributes["condition"] = []
                if condition_terms[word] not in additional_attributes["condition"]:
                    additional_attributes["condition"].append(condition_terms[word])

            if word in color_terms:
                if "color" not in additional_attributes:
                    additional_attributes["color"] = []
                if color_terms[word] not in additional_attributes["color"]:
                    additional_attributes["color"].append(color_terms[word])

            if word in density_terms:
                if "density" not in additional_attributes:
                    additional_attributes["density"] = []
                if density_terms[word] not in additional_attributes["density"]:
                    additional_attributes["density"].append(density_terms[word])

            if word in material_terms:
                if "material" not in additional_attributes:
                    additional_attributes["material"] = []
                if material_terms[word] not in additional_attributes["material"]:
                    additional_attributes["material"].append(material_terms[word])

        # Merge attributes
        for category, values in additional_attributes.items():
            if category not in combined_attributes:
                combined_attributes[category] = []
            for value in values:
                if value not in combined_attributes[category]:
                    combined_attributes[category].append(value)

        # Ensure at least 2-3 attribute categories with multiple tags if we have fewer
        if len(combined_attributes) < 2:
            # Add default attributes based on terrain/scene/features
            if "lava" in terrain and "lighting" not in combined_attributes:
                combined_attributes["lighting"] = ["fiery", "glowing"]
            if (
                any(t in terrain for t in ["forest", "grassland"])
                and "color" not in combined_attributes
            ):
                combined_attributes["color"] = ["green", "natural"]
            if "cave" in terrain and "lighting" not in combined_attributes:
                combined_attributes["lighting"] = ["dim", "shadowy"]
            if "interior" in terrain and "condition" not in combined_attributes:
                combined_attributes["condition"] = ["enclosed"]
            if "street" in terrain and "material" not in combined_attributes:
                combined_attributes["material"] = ["stone", "cobblestone"]
            if (
                any(f in features for f in ["boulder", "wall"])
                and "material" not in combined_attributes
            ):
                combined_attributes["material"] = ["stone"]
            if (
                any(f in features for f in ["tree", "bush"])
                and "color" not in combined_attributes
            ):
                combined_attributes["color"] = ["green"]

        # Enhance existing categories to have multiple tags where appropriate
        for category in combined_attributes:
            current_tags = combined_attributes[category]
            if len(current_tags) == 1:
                # Add complementary tags based on context
                if category == "lighting":
                    if "bright" in current_tags:
                        combined_attributes[category].append("natural")
                    elif "dim" in current_tags:
                        combined_attributes[category].append("shadowy")
                    elif "glow" in current_tags or "glowing" in current_tags:
                        combined_attributes[category].append("warm")
                elif category == "condition":
                    if "natural" in current_tags and any(
                        t in terrain for t in ["forest", "grassland"]
                    ):
                        combined_attributes[category].append("lush")
                    elif "rugged" in current_tags:
                        combined_attributes[category].append("rocky")
                elif category == "color" and "green" in current_tags:
                    if any(f in features for f in ["tree", "bush"]):
                        combined_attributes[category].append("vibrant")

        # Ensure we have at least 2 categories
        if len(combined_attributes) < 2:
            # Add a generic lighting attribute if missing
            if "lighting" not in combined_attributes:
                combined_attributes["lighting"] = ["natural"]
            # Add a generic condition attribute if still needed
            if len(combined_attributes) < 2 and "condition" not in combined_attributes:
                combined_attributes["condition"] = ["well-maintained"]

        normalized = Caption(
            description=description,
            terrain=terrain,
            features=features,
            scene_type=scene_type,
            style=reclassified_caption.style,
            extras=reclassified_caption.extras + all_oov,  # Move OOV to extras
            attributes=combined_attributes,
            image_path=reclassified_caption.image_path,
            raw_response=reclassified_caption.raw_response,
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
                            # Normalize path separators and only add the normalized version
                            img_path = caption["image_path"].replace("\\", "/")
                            processed.add(img_path)
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
                # Look for gdrive subfolders - process ALL gdrive folders in the directory
                if "gdrive_" in folder.name:
                    # This folder itself is a gdrive folder
                    for img_file in folder.glob("*.png"):
                        img_path = str(img_file)
                        if not exclude_processed or img_path not in processed_images:
                            image_paths.append(img_path)
                else:
                    # Look for ALL subfolders starting with "gdrive_"
                    gdrive_folders = [
                        f
                        for f in folder.iterdir()
                        if f.is_dir() and f.name.startswith("gdrive_")
                    ]

                    if gdrive_folders:
                        # Process images in ALL gdrive subfolders
                        for gdrive_folder in gdrive_folders:
                            for img_file in gdrive_folder.glob("*.png"):
                                img_path = str(img_file)
                                if (
                                    not exclude_processed
                                    or img_path not in processed_images
                                ):
                                    image_paths.append(img_path)
                    else:
                        # If no gdrive subfolders found, look directly in the folder
                        for img_file in folder.glob("*.png"):
                            img_path = str(img_file)
                            if (
                                not exclude_processed
                                or img_path not in processed_images
                            ):
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
