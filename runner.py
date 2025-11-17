import cv2
import numpy as np
from pathlib import Path

from src.pnn_analyzer import PNNAnalyzer
from src.image_reader import ImageReader


def main():
    reader = ImageReader("./images")
    output_dir = Path("./output")
    # Ensure output directory exists so cv2.imwrite doesn't fail silently
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path, image in reader.load_all_images(color_mode="color"):
        print(f"\n{'='*60}")
        print(f"PNN DETECTION: {image_path.name}")
        print(f"{'='*60}")

        # Initialize analyzer with PV interneuron PNN parameters
        # PV somata: 10-20 µm diameter, PNN rings: 12-28 µm across (radii 6-14 µm)
        # Expanded to detect smaller bright PNNs
        analyzer = PNNAnalyzer(
            min_pnn_radius_mm=0.004,  # 4 microns - detect smaller bright PNNs
            max_pnn_radius_mm=0.014,  # 14 microns - based on PV interneuron PNN data
            pixel_size_mm=0.001,  # 1 micron per pixel (calibrate from microscope settings)
            contrast_threshold=1.30,  # Ring must be 30% brighter - balanced sensitivity
            uniformity_threshold=0.16,  # Stricter - rejects irregular rings
            template_threshold=0.12,  # Very lenient - accepts irregular/oval shapes
            center_darkness_threshold=0.70,  # Stricter - requires darker centers
            min_ring_brightness=0.0,  # Disabled - use only relative contrast
        )
        print(f"\n⚙️  PARAMETERS:")
        print(f"   Pixel size: {analyzer.pixel_size_mm*1000:.2f} µm/pixel")
        print(f"   PNN radius range: {analyzer.min_pnn_radius_mm*1000:.1f}-{analyzer.max_pnn_radius_mm*1000:.1f} µm")
        print(f"   Template threshold: {analyzer.template_threshold}")
        print(f"   Contrast threshold: {analyzer.contrast_threshold}")
        print(f"   Uniformity threshold: {analyzer.uniformity_threshold}")
        print(f"   Center darkness max ratio: {analyzer.center_darkness_threshold}")

        result = analyzer.analyze_image(image, str(image_path))
        print(f"\n📊 RESULTS:")
        print(f"   PNNs detected: {result.pnn_count}")
        if result.pnn_circles_mm:
            radii_mm = [r for _, _, r in result.pnn_circles_mm]
            avg_radius_um = np.mean(radii_mm) * 1000
            min_radius_um = min(radii_mm) * 1000
            max_radius_um = max(radii_mm) * 1000
            print(f"   PNN sizes (µm): avg={avg_radius_um:.1f}, range={min_radius_um:.1f}-{max_radius_um:.1f}")
        if result.quality_scores:
            avg_quality = float(np.mean(result.quality_scores))
            min_quality = float(min(result.quality_scores))
            max_quality = float(max(result.quality_scores))
            print(
                f"   Quality scores: avg={avg_quality:.3f}, range={min_quality:.3f}-{max_quality:.3f}"
            )
            excellent = sum(1 for q in result.quality_scores if q > 0.8)
            good = sum(1 for q in result.quality_scores if 0.7 < q <= 0.8)
            moderate = sum(1 for q in result.quality_scores if 0.6 < q <= 0.7)
            poor = sum(1 for q in result.quality_scores if q <= 0.6)
            print(f"\n📈 QUALITY BREAKDOWN:")
            print(f"   Excellent (>0.8): {excellent}")
            print(f"   Good (0.7-0.8): {good}")
            print(f"   Moderate (0.6-0.7): {moderate}")
            print(f"   Poor (≤0.6): {poor}")

        annotated = analyzer.draw_pnn_detections(
            image, result.pnn_circles, result.quality_scores
        )
        for i, (x, y, r) in enumerate(result.pnn_circles):
            if i < len(result.quality_scores):
                q = result.quality_scores[i]
                if q > 0.8:
                    color = (0, 255, 0)
                elif q > 0.7:
                    color = (0, 255, 255)
                elif q > 0.6:
                    color = (255, 255, 0)
                else:
                    color = (255, 0, 0)
            else:
                color = (128, 128, 128)
            cv2.circle(annotated, (x, y), r, color, 2)
            cv2.circle(annotated, (x, y), 2, (0, 0, 255), -1)

        output_path = output_dir / f"pnn_{image_path.name}"
        cv2.imwrite(str(output_path), annotated)
        print(f"\n💾 PNN detection saved: {output_path}")
        print(f"\n🎨 Color Legend: 🟢 >0.8  🟡 0.7-0.8  🔵 0.6-0.7  🔴 ≤0.6")


if __name__ == "__main__":
    main()
