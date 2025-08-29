# Project Architecture

## Overview

The D&D Battlemap Processor is designed as a modular machine learning pipeline that transforms raw battlemap images into training datasets for AI models. The architecture follows a clear data flow from input processing through grid detection, tile extraction, captioning, and finally model training.

## Core Architecture

```
Input Images → Grid Detection → Tile Extraction → Caption Generation → Dataset Creation → Model Training
```

## Directory Structure

```
dnd-battlemaps/
├── battlemap_processor/           # Core processing library
│   ├── core/                     # Core processing modules
│   ├── input_handler.py          # Handle various input sources
│   ├── grid_detector.py          # Grid detection algorithms
│   ├── tile_extractor.py         # Tile extraction logic
│   └── image_processor.py        # Image processing utilities
├── scripts/                      # Processing and analysis scripts
│   ├── data_processing/          # Dataset creation and preparation
│   ├── analysis/                 # Data analysis and validation
│   ├── training/                 # Model training utilities
│   ├── testing/                  # Test scripts and validation
│   └── utilities/                # Helper scripts and tools
├── configs/                      # Configuration files
│   ├── kohya_training_config.toml
│   └── *.json.template
├── data/                         # Data files (gitignored)
│   ├── captions/                 # Generated captions
│   ├── datasets/                 # Dataset archives
│   └── *.txt                     # Dataset splits
├── docs/                         # Documentation and results
│   ├── training_results/         # Model training results
│   ├── SETUP.md                  # Installation guide
│   └── *.txt                     # Analysis reports
├── examples/                     # Usage examples
└── main.py                       # Primary entry point
```

## Core Components

### 1. BattlemapProcessor (Core Library)

**Location**: `battlemap_processor/`

The main processing engine that orchestrates the entire pipeline:

- **InputHandler**: Manages various input sources (files, directories, zip archives)
- **GridDetector**: Computer vision algorithms for detecting grid patterns
- **TileExtractor**: Extracts grid-aligned tiles from detected grids
- **ImageProcessor**: Handles image transformations and output

### 2. Grid Detection Algorithm

**Technology**: OpenCV morphological operations

**Process**:
1. Convert images to grayscale
2. Apply morphological blackhat operations
3. Test multiple grid cell sizes (100-180px)
4. Score candidates based on alignment and contrast
5. Select optimal grid structure

**Key Features**:
- Automatic grid size detection
- Robust to various image qualities
- Handles non-perfect grid alignment

### 3. Tile Extraction Pipeline

**Process**:
1. Use detected grid to define extraction regions
2. Extract non-overlapping NxN grid square tiles
3. Resize to target output dimensions (default 512x512)
4. Apply quality filtering (brightness, completeness)

### 4. Caption Generation System

**Technology**: OpenAI GPT API integration

**Features**:
- Structured caption format with metadata
- Terrain and feature classification
- Scene type identification
- Grid presence detection

**Caption Structure**:
```
<description>. terrain: <types>. features: <list>. scene_type: <type>. attributes: <attrs>. grid: <yes|no>.
```

### 5. Dataset Creation Pipeline

**Scripts**: `scripts/data_processing/`

**Process**:
1. Batch process raw battlemap images
2. Generate structured captions for each tile
3. Create train/validation/test splits
4. Package datasets for training platforms

### 6. Model Training Infrastructure

**Technology**: LoRA (Low-Rank Adaptation) with Stable Diffusion

**Configuration**: Kohya's sd-scripts framework

**Results**: Successfully trained models with documented validation

## Data Flow

### Phase 1: Image Processing
```
Raw Battlemaps → Grid Detection → Tile Extraction → Quality Filtering
```

### Phase 2: Dataset Creation
```
Filtered Tiles → Caption Generation → Vocabulary Analysis → Dataset Packaging
```

### Phase 3: Model Training
```
Packaged Dataset → LoRA Training → Model Validation → Result Documentation
```

## Technical Highlights

### Computer Vision
- Custom grid detection algorithms using morphological operations
- Robust handling of various battlemap styles and qualities
- Automatic quality assessment and filtering

### Machine Learning Pipeline
- Complete end-to-end ML workflow
- Automated dataset creation and validation
- Integration with modern training frameworks (Stable Diffusion/LoRA)

### Data Engineering
- Scalable batch processing capabilities
- Structured data formats with comprehensive metadata
- Version control for datasets and model artifacts

### Documentation and Validation
- Comprehensive training result documentation
- Validation image generation and comparison
- Detailed analysis of model performance across different prompts

## Scalability Considerations

The architecture is designed for scalability:

- **Modular Design**: Each component can be developed and tested independently
- **Batch Processing**: Efficient handling of large image collections
- **Configurable Parameters**: Easy adaptation to different use cases
- **Cloud Integration**: Ready for deployment on cloud training platforms

## Future Enhancements

Potential areas for expansion:

- **Web Interface**: Browser-based processing interface
- **Advanced Quality Filters**: ML-based "boring" tile detection
- **Multi-Grid Support**: Handle non-square grids and irregular patterns
- **Real-time Processing**: Stream processing capabilities
- **Model Serving**: Deployment pipeline for trained models
