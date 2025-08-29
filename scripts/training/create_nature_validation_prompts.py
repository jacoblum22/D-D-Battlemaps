#!/usr/bin/env python3
"""
Create nature-themed validation prompts based on wilderness/grassland vocabulary
Uses vocabulary and terminology from captions_grassland.json
"""

import json
from pathlib import Path


def create_nature_validation_prompts():
    """Create nature-themed validation prompts using grassland vocabulary"""

    # Define validation prompts based on grassland/wilderness vocabulary
    validation_prompts = [
        # Grassland clearing scenes
        {
            "prompt": "A lush grassland clearing with scattered trees and vibrant greenery. terrain: grassland. features: boulder, tree, bush. scene_type: clearing. attributes: color(green, vibrant), lighting(natural), condition(lush). grid: yes.",
            "description": "Grassland clearing with trees",
            "seed": 12345,
        },
        {
            "prompt": "A serene grassland with winding paths through sparse vegetation. terrain: grassland, path. features: boulder, bush, rock. scene_type: wilderness. attributes: lighting(natural), condition(serene, natural). grid: yes.",
            "description": "Grassland with natural paths",
            "seed": 12346,
        },
        # Garden scenes
        {
            "prompt": "A vibrant garden featuring colorful flowers and lush foliage surrounding a tranquil pond. terrain: garden. features: flower, pond, bush, tree. scene_type: garden. attributes: color(vibrant, green), lighting(natural), density(lush). grid: no.",
            "description": "Garden with pond and flowers",
            "seed": 12347,
        },
        {
            "prompt": "A well-maintained garden with stone paths winding through dense greenery. terrain: garden, path. features: bush, tree, plant. scene_type: garden. attributes: condition(well-maintained), density(dense), lighting(bright). grid: yes.",
            "description": "Structured garden with paths",
            "seed": 12348,
        },
        # River and water scenes
        {
            "prompt": "A peaceful river flowing through lush grassland with rocky outcrops along the banks. terrain: river, grassland. features: boulder, tree, bridge. scene_type: wilderness. attributes: color(blue, green), condition(lush, natural), lighting(natural). grid: yes.",
            "description": "River through grassland",
            "seed": 12349,
        },
        {
            "prompt": "A cascading waterfall surrounded by vibrant vegetation and rocky cliffs. terrain: cliffs, river. features: waterfall, boulder, tree, bush. scene_type: wilderness. attributes: lighting(natural, bright), condition(rugged), color(green, grey). grid: no.",
            "description": "Waterfall in rocky terrain",
            "seed": 12350,
        },
        # Forest wilderness
        {
            "prompt": "A dense forest clearing with twisted roots and scattered bushes creating natural pathways. terrain: forest. features: tree, bush, roots, boulder. scene_type: clearing. attributes: density(dense), condition(natural, overgrown), lighting(natural). grid: yes.",
            "description": "Dense forest clearing",
            "seed": 12351,
        },
        {
            "prompt": "A tranquil forest path winding through lush greenery with dappled natural lighting. terrain: forest, path. features: tree, bush, plant. scene_type: wilderness. attributes: condition(tranquil, lush), lighting(natural, soft), density(dense). grid: yes.",
            "description": "Forest path with natural lighting",
            "seed": 12352,
        },
        # Rocky and cliff terrain
        {
            "prompt": "A rugged cliff terrain with sparse vegetation and scattered boulders overlooking a valley. terrain: cliffs, grassland. features: boulder, bush, rock. scene_type: wilderness. attributes: condition(rugged, natural), density(sparse), lighting(natural). grid: yes.",
            "description": "Rugged cliff wilderness",
            "seed": 12353,
        },
        {
            "prompt": "A rocky landscape with vibrant green patches and a natural stone platform. terrain: grassland, cliffs. features: boulder, plant, platform, rock. scene_type: wilderness. attributes: color(green, grey), condition(natural, rocky), density(sparse). grid: no.",
            "description": "Rocky terrain with vegetation",
            "seed": 12354,
        },
        # Water features and pools
        {
            "prompt": "A serene natural pool surrounded by lush foliage and flowering plants. terrain: grassland. features: pool, flower, bush, tree. scene_type: clearing. attributes: condition(serene, lush), color(green, blue), lighting(natural). grid: yes.",
            "description": "Natural pool with foliage",
            "seed": 12355,
        },
        {
            "prompt": "A mystical pond with glowing water surrounded by vibrant flowers and twisted trees. terrain: garden. features: pond, flower, tree, bush. scene_type: garden. attributes: lighting(glowing, natural), color(vibrant, green), condition(mystical). grid: no.",
            "description": "Mystical glowing pond",
            "seed": 12356,
        },
        # Rustic wilderness with campfire
        {
            "prompt": "A rustic wilderness clearing with a warm campfire surrounded by scattered rocks and lush vegetation. terrain: grassland. features: campfire, boulder, bush, tree. scene_type: clearing. attributes: lighting(warm, natural), condition(rustic, natural), color(green, brown). grid: yes.",
            "description": "Wilderness campfire clearing",
            "seed": 12357,
        },
        # Overgrown and wild areas
        {
            "prompt": "An overgrown grassland with twisted trees and dense foliage creating mysterious pathways. terrain: grassland. features: tree, bush, roots, path. scene_type: wilderness. attributes: condition(overgrown, twisted), density(dense), lighting(natural). grid: yes.",
            "description": "Overgrown wild grassland",
            "seed": 12358,
        },
        # Bridge and crossing scenes
        {
            "prompt": "A wooden bridge crossing over a rushing stream surrounded by vibrant greenery and rocky terrain. terrain: river, grassland. features: bridge, boulder, tree, waterfall. scene_type: wilderness. attributes: condition(flowing, lush), color(green, blue), lighting(natural). grid: yes.",
            "description": "Bridge over rushing stream",
            "seed": 12359,
        },
    ]

    # Create output file
    output_file = "validation_prompts.txt"

    # Save prompts as simple text file for easy reference
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Nature-Themed Validation Prompts\\n")
        f.write(
            "# Based on wilderness/grassland vocabulary from captions_grassland.json\\n"
        )
        f.write("# All prompts use nature-themed terminology and features\\n\\n")

        for i, prompt_data in enumerate(validation_prompts, 1):
            f.write(f"Prompt {i}: {prompt_data['description']}\\n")
            f.write(f"Seed: {prompt_data['seed']}\\n")
            f.write(f"Text: {prompt_data['prompt']}\\n")
            f.write("-" * 80 + "\\n\\n")

    # Also save as JSON for programmatic use
    with open("validation_prompts.json", "w", encoding="utf-8") as f:
        json.dump(validation_prompts, f, indent=2, ensure_ascii=False)

    print(f"Created {len(validation_prompts)} nature-themed validation prompts")
    print(f"Text file: {output_file}")
    print("JSON file: validation_prompts.json")
    print("\\nVocabulary used:")
    print("- Terrain: grassland, garden, river, forest, cliffs, path")
    print(
        "- Features: boulder, tree, bush, plant, pond, pool, bridge, waterfall, flower, rock, campfire, roots"
    )
    print("- Scene types: wilderness, clearing, garden")
    print(
        "- Attributes: natural lighting, lush/dense vegetation, vibrant colors, serene/tranquil atmosphere"
    )


if __name__ == "__main__":
    create_nature_validation_prompts()
