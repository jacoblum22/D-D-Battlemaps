# Setup and Installation Guide

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dnd-battlemaps.git
cd dnd-battlemaps
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv battlemap_env

# Activate virtual environment
# On Windows:
battlemap_env\Scripts\activate
# On macOS/Linux:
source battlemap_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the environment template and configure your API keys:

```bash
# Copy template
cp .env.template .env

# Edit .env file with your actual API keys
# Required for captioning functionality:
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Verify Installation

Test that everything is working:

```bash
python main.py --help
```

You should see the help message for the battlemap processor.

## Google Drive Integration (Optional)

If you plan to use Google Drive integration:

1. Create a Google Cloud Console project
2. Enable the Google Drive API
3. Create service account credentials
4. Download the credentials JSON file
5. Rename it to `google_drive_credentials.json` and place in project root

## Testing the Installation

Run a simple test to make sure everything works:

```bash
# Test grid detection on a sample image
python scripts/testing/test_grid_detection.py path/to/test/image.jpg
```

## Troubleshooting

### Common Issues

**Import Errors**
- Make sure you're in the correct virtual environment
- Verify all dependencies are installed: `pip list`

**"No grid detected" Error**
- Ensure your test image has a visible grid pattern
- Grid cell size should be between 100-180 pixels
- Try with different test images

**Permission Errors**
- On Windows, run terminal as administrator if needed
- Check file permissions in your project directory

**API Key Issues**
- Verify your `.env` file is properly configured
- Check that API keys are valid and have sufficient credits

## Development Setup

For development and contributing:

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest scripts/testing/

# Format code (if you have black installed)
black .
```

## Next Steps

- Read the main [README.md](../README.md) for usage instructions
- Check out [usage_examples.py](../examples/usage_examples.py) for code examples
- Explore the [scripts/](../scripts/) directory for advanced functionality
