# D&D Battlemap Processor

A Python tool for extracting grid-aligned tiles from D&D battlemap images. Automatically detects grid structures in battlemap images and extracts non-overlapping tiles suitable for training machine learning models or other applications.

## Features

- **Grid Detection**: Automatically detects grid patterns in battlemap images using morphological operations
- **Multiple Input Sources**: Supports zip files, directories, single images, and Google Drive folders (planned)
- **Grid-Aligned Extraction**: Extracts tiles that perfectly align with the detected grid structure
- **Configurable Tile Size**: Output tiles in any size (default 512x512 pixels)
- **Flexible Grid Squares**: Extract tiles containing any number of grid squares (default 12x12)
- **Quality Control**: Built-in filtering for dark or low-quality tiles (planned)

## Installation

1. **Clone or download this project**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python main.py --help
   ```

## Usage

### Basic Usage

Process a single image:
```bash
python main.py path/to/your/battlemap.jpg
```

Process a zip file of images:
```bash
python main.py path/to/your/maps.zip
```

Process a directory:
```bash
python main.py path/to/your/maps/folder
```

### Advanced Options

```bash
python main.py <source> [options]

Options:
  --squares N       Number of grid squares per tile (default: 12)
  --output DIR      Output directory (default: 'output')
  --tile-size SIZE  Output tile size in pixels (default: 512)
```

### Examples

Extract 14x14 square tiles:
```bash
python main.py maps.zip --squares 14
```

Save to custom directory with different tile size:
```bash
python main.py maps/ --output extracted_tiles --tile-size 256
```

## Testing Grid Detection

To test grid detection on a single image and see visualization:

```bash
python test_grid_detection.py path/to/battlemap.jpg
```

This will:
- Show the detected grid overlay on your image
- Display sample extracted tiles
- Print grid detection statistics

## How It Works

1. **Grid Detection**: Uses morphological blackhat operations to detect grid lines
   - Tests different cell sizes (100-180 pixels in 10px increments)  
   - Scores candidates based on alignment and contrast
   - Selects the best-fitting grid structure

2. **Tile Extraction**: 
   - Extracts non-overlapping tiles aligned to the detected grid
   - Each tile contains exactly N×N grid squares (configurable)
   - Tiles are resized to the target output size

3. **Quality Control** (planned):
   - Filter out very dark tiles
   - Remove "boring" tiles (solid colors, water, etc.)
   - Optimize tile selection for maximum coverage

## Output Structure

```
output/
├── battlemap1_tile_00_00_12x12.png
├── battlemap1_tile_00_12_12x12.png  
├── battlemap1_tile_12_00_12x12.png
└── ...
```

Filename format: `{source}_{tile}_{grid_x}_{grid_y}_{squares}x{squares}.png`

- `grid_x`, `grid_y`: Starting grid coordinates
- `squares`: Number of grid squares in the tile

## Requirements

- Python 3.7+
- OpenCV (cv2)
- PIL/Pillow
- NumPy
- Google API libraries (for Google Drive support)

## Troubleshooting

### "No grid detected"
- Make sure your image has a visible grid pattern
- Try images with grid cell sizes between 100-180 pixels
- Ensure the grid lines are darker than the background

### "Grid is too small for requested tile size"
- Reduce the `--squares` parameter
- Or use a smaller image/larger grid

### Import errors
- Install all requirements: `pip install -r requirements.txt`
- Make sure you're using Python 3.7+

## Future Features

- [ ] Google Drive integration
- [ ] Advanced quality filtering ("boring" tile detection)
- [ ] Deduplication using perceptual hashing
- [ ] Batch processing optimization
- [ ] Web interface
- [ ] Support for non-square grids

## Development

The project is organized into modular components:

- `battlemap_processor/core/input_handler.py`: Handle various input sources
- `battlemap_processor/core/grid_detector.py`: Grid detection algorithms  
- `battlemap_processor/core/tile_extractor.py`: Tile extraction logic
- `battlemap_processor/core/image_processor.py`: Output handling

To contribute or modify the code, start by understanding these core modules.

## LoRA Training Results

This project has been used to train a LoRA model for generating D&D battlemaps using Stable Diffusion 1.5.

### Training Setup

- **Base model**: runwayml/stable-diffusion-v1-5
- **Method**: LoRA (sd-scripts)
- **Dataset**: 1,446 images (train 1,250 · val 125 · test 71) with paired captions
- **Captions**: Include the trigger token `battlemaps` plus map attributes (e.g., terrain, scene_type, grid)
- **Resolution**: 512×512 (square crops)
- **Hardware**: 1× NVIDIA RTX 2000 Ada (16 GB VRAM)
- **Key flags**: `--gradient_checkpointing`, `--sdpa`, `--mixed_precision fp16`
- **Optimizer**: AdamW
- **Batch size**: 1

### Training Runs

#### Run A — "Quick Underfit" (200 steps)

**Goal**: Smoke test to verify the pipeline end-to-end.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/sd-scripts && source venv/bin/activate

accelerate launch --mixed_precision=fp16 train_network.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/workspace/battlemaps_ds/kohya/train" \
  --output_dir="/workspace/battlemaps_ds/output" \
  --logging_dir="/workspace/battlemaps_ds/logs" \
  --resolution=512,512 \
  --caption_extension=".txt" \
  --network_module=networks.lora \
  --network_dim=8 --network_alpha=4 \
  --learning_rate=1e-4 --text_encoder_lr=1e-6 \
  --optimizer_type=AdamW \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --sdpa \
  --max_data_loader_n_workers=1 \
  --max_train_steps=200 \
  --save_every_n_steps=100 \
  --save_model_as="safetensors" \
  --clip_skip=2
```

- **LoRA rank/alpha**: 8/4
- **Total steps**: 200 (≈ 1–2 minutes on this GPU)
- **Checkpoint**: at-step00000200.safetensors (also last.safetensors)
- **Result**: Learns a hint of the style; samples still drift toward base-model imagery.

#### Run B — "First Pass" (6,000 steps)

**Goal**: Materially adapt SD1.5 to the battlemap style.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/sd-scripts && source venv/bin/activate

accelerate launch --mixed_precision=fp16 train_network.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/workspace/battlemaps_ds/kohya/train" \
  --output_dir="/workspace/battlemaps_ds/output" \
  --logging_dir="/workspace/battlemaps_ds/logs" \
  --resolution=512,512 \
  --caption_extension=".txt" \
  --network_module=networks.lora \
  --network_dim=16 --network_alpha=8 \
  --learning_rate=1e-4 --text_encoder_lr=1e-6 \
  --optimizer_type=AdamW \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --sdpa \
  --max_data_loader_n_workers=1 \
  --max_train_steps=6000 \
  --save_every_n_steps=500 \
  --save_model_as="safetensors" \
  --clip_skip=2
```

- **LoRA rank/alpha**: 16/8
- **Total steps**: 6,000
- **Checkpoints**: Saved every 500 steps

### Sample Generation

Test prompts used for evaluation:

```python
prompts = [
    "battlemaps, top-down medieval tavern interior, wooden tables, chairs, barrel, warm lighting, grid: yes",
    "battlemaps, top-down forest path through ancient trees, clearing, stones, grid: no", 
    "battlemaps, top-down dungeon corridor, stone walls, torches, moss, grid: yes",
    "battlemaps, top-down coastal village with docks and boats, shoreline, grid: yes",
]
```

**Generation settings**:
- LoRA scale: 1.15
- Steps: 40
- Guidance scale: 6.5
- Size: 512×512
- Negative prompt: "photo, photorealistic, perspective view, UI panels, text, watermark, real human, camera, logo, poster, comic, noisy, blurry, low quality"

**Comparison images**: Generated samples can be found in the `comparison_images/` folder showing the progression from base model to trained LoRA outputs.

### Training Results Comparison

#### Run A (200 steps) vs Run B (6,000 steps)

The following images demonstrate the progression from the initial undertrained model (200 steps) to the more fully trained version (6,000 steps):

**Prompt 1: "battlemaps, top-down medieval tavern interior, wooden tables, chairs, barrel, warm lighting, grid: yes"**

| Run A (200 steps) | Run B (6,000 steps) |
|---|---|
| ![Tavern 200 steps](comparison_images/v1_200/bm_01.png) | ![Tavern 6000 steps](comparison_images/v2_6000/bm_01%20(1).png) |

**Prompt 2: "battlemaps, top-down forest path through ancient trees, clearing, stones, grid: no"**

| Run A (200 steps) | Run B (6,000 steps) |
|---|---|
| ![Forest 200 steps](comparison_images/v1_200/bm_02.png) | ![Forest 6000 steps](comparison_images/v2_6000/bm_02%20(1).png) |

**Prompt 3: "battlemaps, top-down dungeon corridor, stone walls, torches, moss, grid: yes"**

| Run A (200 steps) | Run B (6,000 steps) |
|---|---|
| ![Dungeon 200 steps](comparison_images/v1_200/bm_03.png) | ![Dungeon 6000 steps](comparison_images/v2_6000/bm_03%20(1).png) |

**Prompt 4: "battlemaps, top-down coastal village with docks and boats, shoreline, grid: yes"**

| Run A (200 steps) | Run B (6,000 steps) |
|---|---|
| ![Coastal 200 steps](comparison_images/v1_200/bm_04.png) | ![Coastal 6000 steps](comparison_images/v2_6000/bm_04%20(1).png) |

**Observations**:
- **Run A (200 steps)**: Shows initial adaptation with basic style hints but still drifts toward base model imagery
- **Run B (6,000 steps)**: Demonstrates clear mastery of the battlemap style with proper top-down perspective, grid awareness, and battlemap-specific details

### Caption Format

The dataset uses structured captions with the following format:

```
<description>. terrain: <terrain_types>. features: <feature_list>. scene_type: <scene>. attributes: <attributes>. grid: <yes|no>.
```

Example:
```
A medieval tavern interior with wooden tables and chairs. terrain: interior. features: table, chair, barrel, wall. scene_type: tavern. attributes: lighting(warm). grid: yes.
```
