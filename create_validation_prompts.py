#!/usr/bin/env python3
"""
Step 4: Create fixed validation prompts for comparing base vs LoRA
Creates ~15 prompts that exercise content variety
"""

import json
from pathlib import Path


def create_validation_prompts():
    """Create fixed validation prompts for model comparison"""

    # Define validation prompts based on your dataset variety
    validation_prompts = [
        # Interior scenes
        {
            "prompt": "A cozy inn interior with multiple rooms connected by a central corridor. terrain: interior. features: bed, chair, desk, door, table. scene_type: inn. attributes: lighting(warm), condition(cozy). grid: yes.",
            "description": "Interior inn scene",
            "seed": 12345,
        },
        {
            "prompt": "A library filled with bookshelves and study areas. terrain: interior. features: bookshelf, chair, desk, table. scene_type: library. attributes: lighting(warm), condition(organized). grid: yes.",
            "description": "Interior library scene",
            "seed": 12346,
        },
        {
            "prompt": "A bustling tavern with tables and a well-stocked bar. terrain: interior. features: barrel, chair, table, torch. scene_type: tavern. attributes: lighting(warm). grid: yes.",
            "description": "Interior tavern scene",
            "seed": 12347,
        },
        # Wilderness scenes
        {
            "prompt": "A winding path cuts through a dense forest surrounded by towering trees. terrain: forest, path. features: boulder, bush, tree. scene_type: wilderness. attributes: lighting(natural), condition(lush). grid: yes.",
            "description": "Forest wilderness with path",
            "seed": 12348,
        },
        {
            "prompt": "A serene clearing in the wilderness with scattered bushes and trees. terrain: grassland. features: bush, tree, campfire. scene_type: clearing. attributes: lighting(sunlit). grid: no.",
            "description": "Wilderness clearing",
            "seed": 12349,
        },
        {
            "prompt": "A rugged mountain path with rocky cliffs and scattered boulders. terrain: cliffs. features: boulder, bush, rubble. scene_type: wilderness. attributes: condition(rugged). grid: yes.",
            "description": "Mountain cliffs wilderness",
            "seed": 12350,
        },
        # Coastal scenes
        {
            "prompt": "A serene coastline meets a lush forest with gentle waves. terrain: coastline, forest. features: boulder, tree, debris. scene_type: wilderness. attributes: lighting(serene). grid: yes.",
            "description": "Coastal forest scene",
            "seed": 12351,
        },
        {
            "prompt": "A wooden dock extends over turbulent waters with scattered crates. terrain: coastline. features: barrel, crates, dock, debris. scene_type: dock. attributes: condition(scattered). grid: yes.",
            "description": "Coastal dock scene",
            "seed": 12352,
        },
        # Cave scenes
        {
            "prompt": "A mysterious cave with glowing blue crystals illuminating rocky walls. terrain: cave. features: boulder, crystals, debris. scene_type: cave. attributes: lighting(glowing), condition(rugged). grid: yes.",
            "description": "Crystal cave scene",
            "seed": 12353,
        },
        {
            "prompt": "A dark cave passage with scattered rubble and cobwebs. terrain: cave. features: boulder, cobweb, rubble. scene_type: cave. attributes: lighting(dim), condition(rocky). grid: no.",
            "description": "Dark cave passage",
            "seed": 12354,
        },
        # Urban/market scenes
        {
            "prompt": "A bustling market street lined with stalls and vendors. terrain: street. features: barrel, bench, crates, table. scene_type: market. attributes: condition(dusty), lighting(vibrant). grid: yes.",
            "description": "Market street scene",
            "seed": 12355,
        },
        {
            "prompt": "A stone courtyard surrounded by walls with a central fountain. terrain: interior. features: bench, fountain, gate, stairs, wall. scene_type: courtyard. attributes: condition(well-maintained), lighting(natural). grid: yes.",
            "description": "Courtyard scene",
            "seed": 12356,
        },
        # Dungeon scenes
        {
            "prompt": "A dungeon chamber with stone walls and scattered debris. terrain: interior. features: wall, rubble, chain, torch. scene_type: dungeon. attributes: lighting(dim), condition(ancient). grid: yes.",
            "description": "Dungeon chamber",
            "seed": 12357,
        },
        # Special terrain
        {
            "prompt": "A snowy landscape with scattered trees and rocky outcrops. terrain: snow. features: boulder, bush, tree. scene_type: wilderness. attributes: condition(frosty), lighting(soft). grid: no.",
            "description": "Snowy wilderness",
            "seed": 12358,
        },
        {
            "prompt": "A bridge crosses over a deep chasm with rocky walls. terrain: cave, river. features: bridge, chain, chasm. scene_type: bridge. attributes: condition(narrow), lighting(dim). grid: yes.",
            "description": "Underground bridge",
            "seed": 12359,
        },
    ]

    # Create output directory
    dataset_dir = Path("dataset_v1")
    validation_dir = dataset_dir / "validation_prompts"
    validation_dir.mkdir(exist_ok=True)

    # Save prompts as JSON
    with open(validation_dir / "validation_prompts.json", "w", encoding="utf-8") as f:
        json.dump(validation_prompts, f, indent=2)

    # Save prompts as simple text file for easy reference
    with open(validation_dir / "validation_prompts.txt", "w", encoding="utf-8") as f:
        for i, prompt_data in enumerate(validation_prompts, 1):
            f.write(f"Prompt {i}: {prompt_data['description']}\n")
            f.write(f"Seed: {prompt_data['seed']}\n")
            f.write(f"Text: {prompt_data['prompt']}\n")
            f.write("-" * 80 + "\n")

    # Create inference script template
    inference_script = """#!/usr/bin/env python3
'''
Validation inference script template
Use this to generate images with base SD1.5 and your LoRA at each checkpoint
'''

import json
from pathlib import Path

def run_validation_inference(model_path, lora_path=None, output_dir="validation_outputs"):
    '''
    Run inference on validation prompts
    
    Args:
        model_path: Path to base SD1.5 model
        lora_path: Path to LoRA checkpoint (None for base model)
        output_dir: Directory to save generated images
    '''
    
    # Load validation prompts
    with open("dataset_v1/validation_prompts/validation_prompts.json") as f:
        prompts = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # TODO: Implement your inference pipeline here
    # For each prompt:
    #   1. Set the seed
    #   2. Generate image with prompt text
    #   3. Save with descriptive filename
    
    print(f"Would generate {len(prompts)} validation images")
    print(f"Model: {model_path}")
    print(f"LoRA: {lora_path or 'None (base model)'}")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    # Example usage:
    # Base model
    run_validation_inference("path/to/sd15", output_dir="val_base")
    
    # With LoRA checkpoint
    run_validation_inference("path/to/sd15", "path/to/lora.safetensors", "val_lora_500")
"""

    with open(
        validation_dir / "run_validation_inference.py", "w", encoding="utf-8"
    ) as f:
        f.write(inference_script)

    print(f"Created {len(validation_prompts)} validation prompts")
    print(f"Saved to {validation_dir}")
    print("Files created:")
    print("  - validation_prompts.json (structured data)")
    print("  - validation_prompts.txt (human readable)")
    print("  - run_validation_inference.py (inference template)")


if __name__ == "__main__":
    create_validation_prompts()
