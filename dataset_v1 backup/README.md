# D&D Battlemaps LoRA Training Dataset

## Dataset Structure
```
dataset_v1/
├── images/
│   ├── train/          # 1250 training images (PNG files)
│   ├── val/            # 125 validation images (PNG files)
│   └── test/           # 71 test images (PNG files)
├── captions/
│   ├── train/          # 1250 training caption files (TXT files)
│   ├── val/            # 125 validation caption files (TXT files)
│   └── test/           # 71 test caption files (TXT files)
├── train.txt           # Training split file list
├── val.txt             # Validation split file list
├── test.txt            # Test split file list
├── meta.jsonl          # Dataset metadata (JSONL format)
├── summary.json        # Dataset statistics and info
├── kohya_training_config.toml  # Training configuration
├── validation_prompts.txt      # Validation prompts
└── README.md           # This file
```

## Dataset Information
- **Total Images**: 1446
- **Image Format**: PNG, 512x512 pixels
- **Caption Format**: Individual .txt files per image
- **Split Method**: Group-aware (prevents tile leakage)
- **Target Model**: Stable Diffusion 1.5
- **Training Method**: LoRA fine-tuning

## Training Configuration
- Configuration file: `kohya_training_config.toml`
- Validation prompts: `validation_prompts.txt`
- Recommended: RTX 4090 or better for optimal training speed

## Split Statistics
- **Train**: 1250 images (1250 copied, 1250 captions)
- **Val**: 125 images (125 copied, 125 captions)
- **Test**: 71 images (71 copied, 71 captions)

## Notes
- Group-aware splitting prevents tiles from the same battlemap appearing in different splits
- Only images with captions are included in the dataset
- Caption dropout was applied during preprocessing for training robustness
- Vocabulary analysis was completed to ensure comprehensive coverage

## Files Included
- Dataset images and captions: ✓
- Metadata files: ✓
- Training configuration: ✓
- Validation prompts: ✓

Generated on: 2025-08-14 11:42:51
