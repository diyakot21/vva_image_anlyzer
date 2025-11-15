# Brain Image Analysis

Tools for analyzing brain images to detect perineuronal nets (PNNs) and general circular structures (e.g. PV protein–related features). Implements OpenCV‑based preprocessing, Hough circle candidates, lightweight ring template matching, and multi‑metric quality scoring.

---
## ✨ Features

Core
* Image loading (`ImageReader`) with JPG / PNG / TIF(F) / BMP support
* PNN detection (`PNNAnalyzer`) with millimeter-based parameters:
    * Preprocessing: bilateral denoise, optional background subtraction, optional CLAHE
    * Hough gradient circle candidates + ring template matching
    * Quality metrics: contrast, ring uniformity, center darkness, signal/background, size
    * Overlap suppression
    * Results in both pixel and millimeter coordinates
* Visualization utilities for drawing detections

Developer / Analysis
* Batch processing of all images in `images/`
* Quality score breakdown and color legend in console output
* Simple image property helpers (`image_utils`)
* Pytest test suite with synthetic and sample images

---
## 🗂 Project Structure

```
.
├── runner.py                # Main PNN batch runner
├── src/
│   ├── pnn_analyzer.py      # PNN detection with mm parameters
│   ├── image_reader.py      # Directory image loader
│   └── image_utils.py       # Basic image helpers
├── images/                  # (User) input images (.tif, .png, ...)
├── output/                  # Generated annotated outputs
├── tests/                   # Test suite + sample images
│   └── test_images/         # Example test images
├── pyproject.toml           # Project + dependency metadata
└── uv.lock                  # Locked dependency versions
```

---
## 🚀 Installation

Install using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv sync
```

Python 3.13 is targeted (see `pyproject.toml`). Use `uv run` to automatically use the project environment (no manual activation needed).

---
## ▶️ Quick Start: Running the PNN Detector

### 1. Prepare your images
Place your brain section images (`.tif`, `.png`, or other supported formats) in the `images/` directory:
```bash
mkdir -p images
# Copy your images to the images/ folder
```

### 2. Run the analyzer
```bash
uv run python runner.py
```

The runner will:
- Process all images in `images/`
- Detect PNNs using optimized parameters (5-65 µm radius range)
- Display progress and detection statistics in the console
- Save annotated images to `output/`

### 3. View results
**Annotated images**: `output/pnn_<original_name>.tif|png`
- Each detected PNN is drawn as a colored circle
- Circle color indicates quality score

**Console output**:
- Total PNNs detected per image
- Average, min, and max PNN sizes in micrometers
- Quality score breakdown

**Quality color legend**:
- 🟢 Green (>0.8): Excellent detection
- 🟡 Yellow (0.7–0.8): Good detection
- 🔵 Cyan/Blue (0.6–0.7): Moderate detection
- 🔴 Red (≤0.6): Poor detection

---
## 📏 Usage with Millimeter Parameters

The analyzer accepts PNN sizes in millimeters and automatically converts to pixels:

```python
from src.pnn_analyzer import PNNAnalyzer
import cv2

# Initialize with mm-based parameters
analyzer = PNNAnalyzer(
    min_pnn_radius_mm=0.005,   # 5 micrometers
    max_pnn_radius_mm=0.065,   # 65 micrometers
    pixel_size_mm=0.001,       # 1 micrometer per pixel
    contrast_threshold=1.05,
    uniformity_threshold=0.18
)

# Analyze image
image = cv2.imread("images/sample.tif")
result = analyzer.analyze_image(image, "sample.tif")

print(f"Detected {result.pnn_count} PNNs")
print(f"Pixel coordinates: {result.pnn_circles}")       # (x, y, r) in pixels
print(f"MM coordinates: {result.pnn_circles_mm}")        # (x, y, r) in millimeters

# Get sizes in micrometers
for x_mm, y_mm, r_mm in result.pnn_circles_mm:
    print(f"PNN at ({x_mm*1000:.1f}, {y_mm*1000:.1f}) µm, radius: {r_mm*1000:.1f} µm")
```

### Calibrating Pixel Size

Adjust `pixel_size_mm` based on your microscope:

| Magnification | Typical µm/pixel | pixel_size_mm |
|---------------|------------------|---------------|
| 10x           | 0.5 - 1.0        | 0.0005 - 0.001|
| 20x           | 0.25 - 0.5       | 0.00025 - 0.0005|
| 40x           | 0.1 - 0.25       | 0.0001 - 0.00025|
| 63x           | ~0.1             | 0.0001        |

---
## ⚙️ Tuning Key Parameters

```python
analyzer = PNNAnalyzer(
    min_pnn_radius_mm=0.005,          # 5 µm minimum radius
    max_pnn_radius_mm=0.065,          # 65 µm maximum radius
    pixel_size_mm=0.001,              # 1 µm per pixel (adjust for your microscope)
    contrast_threshold=1.05,          # Ring must be brighter than center
    uniformity_threshold=0.18,        # Ring brightness consistency
    template_threshold=0.27,          # Template matching sensitivity
    center_darkness_threshold=0.70,   # Center must be darker than ring
    use_clahe=True,                   # Adaptive histogram equalization
    clahe_clip_limit=2.5,
    clahe_tile_grid=(8, 8),
    apply_background_subtraction=True,
    background_blur_radius=55,        # In pixels
)
```

Adjustment guidance:
* Adjust `min/max_pnn_radius_mm` for your expected PNN sizes
* Lower `contrast_threshold` → more sensitive (may increase false positives)
* Lower `uniformity_threshold` → accept less uniform rings
* Lower `template_threshold` → more template hits (filter later by quality)
* Increase `background_blur_radius` if illumination gradients remain
* Disable `use_clahe` for already high‑contrast fluorescence images

---
## 🧪 Testing

Run all tests:

```bash
uv run pytest -v
```

Focused run (single test file):

```bash
uv run pytest tests/test_images.py::TestImageAnalysis::test_simple_circles_detection -v
```

The test suite validates circle detection, parameter updates, drawing, and statistics.

---
## 🛠 Developer Tips

* Use `uv run python -c "import cv2; print(cv2.__version__)"` to verify OpenCV.
* Use `ImageReader` for batch loops: `for path, img in ImageReader("./images").load_all_images(): ...`.
* Add new dependencies via `pyproject.toml` then `uv sync`.
* Prefer `uv run <cmd>` instead of activating `.venv`.

---
## 📦 Export / Packaging (Optional)

This project is defined as a standard Python package (`[project]` in `pyproject.toml`). You can build a wheel:

```bash
uv build
```

---
## 🧾 License

Add your license statement here (MIT, Apache-2.0, etc.).

---
## ✉️ Contact

For questions or suggestions open an issue or PR.



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
│   ├── image_utils.py       # Image helper functions
│   └── pnn_analyzer.py      # PNN detection with mm parameters
├── runner.py                # Main runner for PNN detection
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_images.py
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
