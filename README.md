# Brain Image Analysis


This package provides tools for analyzing brain images to detect perineuronal nets (PNNs) and circular structures, specifically designed for identifying PV protein development in mice brain images using OpenCV and advanced ring detection.

## Features

- **Image Reading**: Load images from local directories with support for common formats (JPG, PNG, TIFF, BMP)
- **PNN Detection**: Specialized detection of perineuronal nets (PNNs) using ring template matching and quality scoring
- **Circle Detection**: Use Hough Circle Transform for general circular structures
- **PV Protein Analysis**: Tuned for detecting protein clusters in brain tissue
- **Batch Processing**: Process multiple images from a directory
- **Visualization**: Generate annotated images showing detected circles
- **Statistics**: Calculate comprehensive statistics across multiple images

## Installation


```bash
# Install with uv (recommended)
uv sync

# Or install dependencies manually
pip install opencv-python numpy pillow scikit-image matplotlib
```

## Quick Start


## Quick Start: PNN Detection

The main runner script is `runner_optimal.py`. This script detects PNNs in all `.tif` images in the `images/` folder and saves annotated results to `output/`.

```bash
uv run python runner_optimal.py
```

You can adjust the input/output folders by editing the script or passing arguments if supported.

The results will be saved as annotated images and a summary in the `output/` directory.

## Module Overview


### `src/pnn_analyzer.PNNAnalyzer`
Performs PNN (perineuronal net) detection using ring template matching, Hough circles, and quality scoring.

**Key Methods:**
- `analyze_image()`: Complete PNN analysis of a single image
- `detect_candidate_circles()`: Finds candidate PNNs using template matching and Hough transform
- `draw_pnn_detections()`: Visualize detected PNNs with quality coloring

**Key Parameters:**
- `min_pnn_radius`/`max_pnn_radius`: Size range for PNNs
- `contrast_threshold`: Minimum ring/center intensity ratio
- `uniformity_threshold`: Minimum ring uniformity


## Parameter Tuning for PNN Detection

You can adjust the detection sensitivity by editing the parameters in `runner_optimal.py` or in the `PNNAnalyzer` constructor:

```python
analyzer = PNNAnalyzer(
    min_pnn_radius=5,
    max_pnn_radius=60,
    contrast_threshold=1.025,   # Lower = more sensitive
    uniformity_threshold=0.025  # Lower = more sensitive
)
```


## Example Output

The runner will save annotated images and a summary text file in the `output/` directory. Each image will have detected PNNs drawn as colored circles (color indicates quality).


## File Structure

```
vva_image_analyzer/
├── src/
│   ├── __init__.py
│   ├── image_reader.py      # Image loading utilities
│   ├── brain_analyzer.py    # (legacy) Circle detection analysis
│   └── pnn_analyzer.py      # PNN detection analysis
├── runner_optimal.py        # Main runner for PNN detection
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_images.py
│   ├── quick_test.py
│   ├── test_runner.py
│   ├── test_images/
│   └── test_results/
├── pyproject.toml
└── README.md
```


## Running Tests

```bash
# Run all pytest tests
uv run pytest tests/ -v
```


## Dependencies

- **OpenCV**: Image processing and Hough Circle Transform
- **NumPy**: Numerical operations and array handling
- **Pillow**: Additional image format support and metadata
- **scikit-image**: Advanced image processing (optional)
- **matplotlib**: Visualization (optional)
- **Python 3.13**: Required Python version
