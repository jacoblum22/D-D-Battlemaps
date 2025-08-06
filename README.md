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
