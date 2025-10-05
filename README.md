# Brain Image Analysis

Tools for analyzing brain images to detect perineuronal nets (PNNs) and general circular structures (e.g. PV protein–related features). Implements OpenCV‑based preprocessing, Hough circle candidates, lightweight ring template matching, and multi‑metric quality scoring.

---
## ✨ Features

Core
* Image loading (`ImageReader`) with JPG / PNG / TIF(F) / BMP support
* PNN detection (`PNNAnalyzer`) combining:
    * Preprocessing: bilateral denoise, optional background subtraction, optional CLAHE
    * Hough gradient circle candidates + single ring template matching
    * Quality metrics: contrast, ring uniformity, center darkness, signal/background, size
    * Overlap suppression
* Generic circular structure detection (`BrainImageAnalyzer`) via Hough Circle Transform
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
│   ├── pnn_analyzer.py      # PNN detection logic
│   ├── brain_analyzer.py    # Generic circle detection
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

Or (fallback) install key runtime dependencies manually:

```bash
pip install opencv-python numpy pillow scikit-image matplotlib
```

Python 3.13 is targeted (see `pyproject.toml`). Use `uv run` to automatically use the project environment (no manual activation needed).

---
## ▶️ Quick Start: PNN Detection

Place one or more `.tif` (or other supported) images in `images/` then run:

```bash
uv run python runner.py
```

Output:
* Annotated images: `output/pnn_<original_name>.tif|png`
* Console summary: counts + quality score distribution

Color legend (default console):
* >0.8 quality = Green
* 0.7–0.8 = Yellow
* 0.6–0.7 = Cyan / Blue
* ≤0.6 = Red

---
## ⚙️ Tuning Key Parameters (`PNNAnalyzer`)

Constructor arguments (shown with runner overrides):

```python
PNNAnalyzer(
        min_pnn_radius=5,
        max_pnn_radius=65,
        contrast_threshold=1.05,
        uniformity_threshold=0.18,
        template_threshold=0.27,
        center_darkness_threshold=0.70,
        use_clahe=True,
        clahe_clip_limit=2.5,
        clahe_tile_grid=(8, 8),
        apply_background_subtraction=True,
        background_blur_radius=55,
)
```

Adjustment guidance:
* Lower `contrast_threshold` → more sensitive (may increase false positives)
* Lower `uniformity_threshold` → accept less uniform rings
* Lower `template_threshold` → more template hits (filter later by quality)
* Increase `background_blur_radius` if illumination gradients remain
* Disable `use_clahe` for already high‑contrast fluorescence images

---
## 🔍 Generic Circle Detection (`BrainImageAnalyzer`)

For simpler circular structure detection (no ring quality scoring):

```python
from src.brain_analyzer import BrainImageAnalyzer
analyzer = BrainImageAnalyzer(dp=1.0, min_dist=20, param1=60, param2=25,
                                                            min_radius=5, max_radius=50)
result = analyzer.analyze_image(image, "sample.png")
print(result.circle_count, result.circles)
```

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
