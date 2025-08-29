# Training Results Documentation

This folder contains comprehensive documentation of the LoRA training results for the D&D Battlemap AI model.

## Contents

### Image Collections

- **comparison_images/** - Side-by-side comparisons showing training progression
  - `v1_200/` - Results from 200-step training run (initial adaptation)
  - `v2_6000/` - Results from 6,000-step training run (full training)

- **validation_prompt_images/** - Validation results using nature/wilderness prompts
  - Seed-based generation results (seed12345.png through seed12359.png)
  - Comprehensive testing of model's environment generation capabilities

### Training Methodology

The training followed a progressive approach:

1. **Run A (200 steps)**: Proof of concept to validate the pipeline
2. **Run B (6,000 steps)**: Full training run achieving style mastery

### Key Achievements

- ✅ Clear style adaptation from generic Stable Diffusion to battlemap aesthetic
- ✅ Maintained prompt following capability while adding style specialization
- ✅ Grid-aware generation (understands when to include/exclude grids)
- ✅ Diverse environment handling (taverns, forests, dungeons, villages, wilderness)

### Technical Configuration

**Hardware**: NVIDIA RTX 2000 Ada (16GB VRAM)
**Framework**: sd-scripts with LoRA adaptation
**Base Model**: Stable Diffusion 1.5 (runwayml/stable-diffusion-v1-5)
**Dataset**: 1,446 curated battlemap tiles with structured captions

For complete technical details, see the main [README.md](../../README.md) and [configuration files](../../configs/).

## Image References

All images demonstrate clear progression from base model capabilities to specialized battlemap generation, maintaining high quality and prompt adherence throughout the training process.
