"""Perineuronal net (PNN) detection using Hough circles and quality assessment."""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class PNNDetectionResult:
    image_path: str
    pnn_circles: List[Tuple[int, int, int]]  # (x, y, r) in pixels
    pnn_circles_mm: List[Tuple[float, float, float]]  # (x, y, r) in millimeters
    pnn_count: int
    quality_scores: List[float]
    detection_params: dict


class PNNAnalyzer:
    def __init__(
        self,
        min_pnn_radius_mm: float = 0.005,  # Default ~8 pixels at 1 micron/pixel
        max_pnn_radius_mm: float = 0.040,  # Default ~35 pixels at 1 micron/pixel
        pixel_size_mm: float = 0.001,  # Default: 1 micron per pixel
        contrast_threshold: float = 0.95,
        uniformity_threshold: float = 0.05,
        template_threshold: float = 0.15,
        center_darkness_threshold: float = 0.80,
        use_clahe: bool = True,
        clahe_clip_limit: float = 3.5,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        apply_background_subtraction: bool = True,
        background_blur_radius: int = 45,
        use_binary_threshold: bool = False,  # Apply statistical thresholding (research method)
        threshold_sd_multiplier: float = 2.5,  # SD multiplier for thresholding
        min_ring_brightness: float = 80.0,  # Minimum absolute brightness for ring (0-255)
    ) -> None:
        """Initialize PNN analyzer with millimeter-based size parameters.
        
        Args:
            min_pnn_radius_mm: Minimum PNN radius in millimeters (typical: 0.005-0.010 = 5-10 µm)
            max_pnn_radius_mm: Maximum PNN radius in millimeters (typical: 0.030-0.065 = 30-65 µm)
            pixel_size_mm: Physical size of one pixel in millimeters (calibrate from microscope magnification)
            contrast_threshold: Ring must be this many times brighter than center (lower = more sensitive)
            uniformity_threshold: Minimum ring brightness consistency 0-1 (lower = accepts patchier rings)
            template_threshold: How well shape must match ring template 0-1 (lower = more lenient matching)
            center_darkness_threshold: Maximum center/ring brightness ratio (lower = requires darker centers)
            use_clahe: Apply contrast-limited adaptive histogram equalization preprocessing
            clahe_clip_limit: CLAHE contrast limit (higher = more aggressive enhancement)
            clahe_tile_grid: CLAHE grid size for local enhancement regions
            apply_background_subtraction: Subtract blurred background to improve foreground contrast
            background_blur_radius: Blur kernel radius for background estimation (larger = smoother)
            use_binary_threshold: Apply statistical thresholding and binarization
            threshold_sd_multiplier: Standard deviation multiplier for thresholding
        """
        if pixel_size_mm <= 0:
            raise ValueError("pixel_size_mm must be > 0")
        
        self.pixel_size_mm = pixel_size_mm
        self.min_pnn_radius_mm = min_pnn_radius_mm
        self.max_pnn_radius_mm = max_pnn_radius_mm
        
        # Convert mm to pixels for internal processing
        self.min_pnn_radius = max(1, int(round(min_pnn_radius_mm / pixel_size_mm)))
        self.max_pnn_radius = max(1, int(round(max_pnn_radius_mm / pixel_size_mm)))
        self.contrast_threshold = contrast_threshold
        self.uniformity_threshold = uniformity_threshold
        self.template_threshold = template_threshold
        self.center_darkness_threshold = center_darkness_threshold
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.apply_background_subtraction = apply_background_subtraction
        self.background_blur_radius = background_blur_radius
        self.use_binary_threshold = use_binary_threshold
        self.threshold_sd_multiplier = threshold_sd_multiplier
        self.min_ring_brightness = min_ring_brightness

    def preprocess_for_pnn(self, image: np.ndarray) -> np.ndarray:
        """Preprocess using research-based approach: CLAHE, threshold at 2-2.5 SD, binarize, median filter."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if gray.dtype == np.uint16:
            max_val = float(gray.max()) if gray.max() > 0 else 1.0
            gray = cv2.convertScaleAbs(gray, alpha=255.0 / max_val)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        if self.apply_background_subtraction:
            blur_base = max(5, self.background_blur_radius)
            k = blur_base if blur_base % 2 == 1 else blur_base + 1
            background = cv2.GaussianBlur(denoised, (k, k), 0)
            subtracted = cv2.subtract(denoised, background)
            norm = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
        else:
            norm = denoised

        if self.use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid
            )
            enhanced = clahe.apply(norm)
        else:
            enhanced = norm
        
        # Optional: Research-based thresholding and binarization
        if self.use_binary_threshold:
            mean_val = enhanced.mean()
            std_val = enhanced.std()
            threshold_val = mean_val + self.threshold_sd_multiplier * std_val
            _, binary = cv2.threshold(enhanced, threshold_val, 255, cv2.THRESH_BINARY)
            filtered = cv2.medianBlur(binary, 3)
            return filtered
        
        return enhanced

    def _create_ring_template(
        self, outer_radius: int = 15, inner_radius: int = 8
    ) -> np.ndarray:
        """Create a ring-shaped template for template matching."""
        size = outer_radius * 2 + 1
        tmpl = np.zeros((size, size), dtype=np.float32)
        cv2.circle(tmpl, (outer_radius, outer_radius), outer_radius, 1.0, -1)
        cv2.circle(tmpl, (outer_radius, outer_radius), inner_radius, 0.0, -1)
        tmpl = cv2.GaussianBlur(tmpl, (3, 3), 1)
        return cv2.normalize(tmpl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def detect_candidate_circles(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find candidate circles using Hough transform and template matching."""
        candidates: List[Tuple[int, int, int]] = []

        # Hough circle detection - very high sensitivity for small bright PNNs
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=6,  # Lower - allows detection of closer small PNNs
            param1=28,
            param2=6,  # Very low - finds small bright circles
            minRadius=self.min_pnn_radius,
            maxRadius=self.max_pnn_radius,
        )
        if circles is not None:
            for x, y, r in np.round(circles[0, :]).astype(int):
                candidates.append((x, y, r))
        
        # Template matching
        base_tmpl = self._create_ring_template()
        base_resp = cv2.matchTemplate(image, base_tmpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(base_resp >= self.template_threshold)
        br = base_tmpl.shape[0] // 2
        for x0, y0 in zip(xs, ys):
            candidates.append((x0 + br, y0 + br, br))

        # Blob-based candidates for non-perfectly-round nets
        _, bin_img = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph)
        for i in range(1, num_labels):
            x_center, y_center = centroids[i]
            area = stats[i, cv2.CC_STAT_AREA]
            if area < np.pi * (self.min_pnn_radius ** 2) * 0.4:
                continue
            if area > np.pi * (self.max_pnn_radius ** 2) * 1.5:
                continue
            approx_r = int(np.sqrt(area / np.pi))
            candidates.append((int(x_center), int(y_center), approx_r))

        # De-duplication
        uniq = {}
        for cx, cy, r in candidates:
            key = (int(cx / 4), int(cy / 4))
            if key not in uniq or r < uniq[key][2]:
                uniq[key] = (cx, cy, r)
        return list(uniq.values())

    def analyze_pnn_quality(
        self, image: np.ndarray, x: int, y: int, r: int
    ) -> Tuple[bool, float, dict]:
        """Score a candidate circle based on contrast, uniformity, and center darkness."""
        h, w = image.shape[:2]
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return False, 0.0, {}

        outer_mask = np.zeros_like(image, dtype=np.uint8)
        inner_mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(outer_mask, (x, y), int(r), 255, -1)
        inner_r = max(1, int(r * 0.4))
        cv2.circle(inner_mask, (x, y), inner_r, 255, -1)
        ring_mask = outer_mask - inner_mask
        ring_pixels = image[ring_mask > 0]
        inner_pixels = image[inner_mask > 0]
        if ring_pixels.size == 0 or inner_pixels.size == 0:
            return False, 0.0, {}

        ring_mean = float(ring_pixels.mean())
        inner_mean = float(inner_pixels.mean())
        ring_std = float(ring_pixels.std())
        background_mean = float(image.mean())
        contrast_ratio = ring_mean / (inner_mean + 1e-6)
        ring_uniformity = 1 - (ring_std / (ring_mean + 1e-6))
        signal_to_background = ring_mean / (background_mean + 1e-6)
        center_darkness_ratio = inner_mean / (ring_mean + 1e-6)
        size_score = 1.0 if self.min_pnn_radius <= r <= self.max_pnn_radius else 0.5

        # Local patchiness / halo measure for non-uniform PNNs
        local_radius = int(r * 1.2)
        y_min = max(0, y - local_radius)
        y_max = min(h, y + local_radius)
        x_min = max(0, x - local_radius)
        x_max = min(w, x + local_radius)
        patch = image[y_min:y_max, x_min:x_max]

        patchy_score = 0.0
        if patch.size > 0:
            patch_mean = float(patch.mean())
            patch_std = float(patch.std())
            patchy_score = (ring_mean - patch_mean) / (patch_std + 1e-6)

        stats = dict(
            ring_mean=ring_mean,
            inner_mean=inner_mean,
            contrast_ratio=contrast_ratio,
            ring_uniformity=ring_uniformity,
            signal_to_background=signal_to_background,
            center_darkness_ratio=center_darkness_ratio,
            size_score=size_score,
            radius=r,
            patchy_score=patchy_score,
        )

        # Standard validation for uniform PNNs
        is_valid_basic = (
            contrast_ratio > self.contrast_threshold
            and ring_uniformity > self.uniformity_threshold
            and signal_to_background > 1.25  # Increased further - only bright PNNs
            and center_darkness_ratio < self.center_darkness_threshold
            and size_score > 0.5
            and ring_mean > self.min_ring_brightness
        )

        # Fallback for patchy / non-circular halos with strong local contrast
        is_valid_patchy = (
            contrast_ratio > (self.contrast_threshold * 0.75)
            and signal_to_background > 1.08
            and patchy_score > 0.6
            and size_score > 0.5
            and ring_mean > (self.min_ring_brightness - 15)
        )

        is_valid = is_valid_basic or is_valid_patchy

        # Rebalanced quality scoring: prioritize contrast over uniformity
        quality = (
            min(contrast_ratio / 2.0, 1.0) * 0.5
            + ring_uniformity * 0.15
            + min(signal_to_background / 2.0, 1.0) * 0.15
            + (1 - center_darkness_ratio) * 0.15
            + size_score * 0.05
        )
        return is_valid, float(quality), stats

    def remove_overlapping_detections(
        self, candidates: List[Tuple[int, int, int]], scores: List[float]
    ) -> List[Tuple[int, int, int]]:
        """Remove overlapping detections, keeping higher-scoring circles."""
        if len(candidates) <= 1:
            return candidates
        order = np.argsort(scores)[::-1]
        kept: List[Tuple[int, int, int]] = []
        for idx in order:
            x1, y1, r1 = candidates[idx]
            keep = True
            for x2, y2, r2 in kept:
                d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if d < min(r1, r2) * 0.8:  # Increased to prevent duplicate detections
                    keep = False
                    break
            if keep:
                kept.append((x1, y1, r1))
        return kept

    def analyze_image(
        self, image: np.ndarray, image_path: str = "unknown"
    ) -> PNNDetectionResult:
        """Detect PNNs in an image and return results in both pixels and millimeters."""
        processed = self.preprocess_for_pnn(image)
        candidates = self.detect_candidate_circles(processed)
        if len(candidates) > 4000:
            candidates = candidates[:4000]

        valid: List[Tuple[int, int, int]] = []
        scores: List[float] = []
        for x, y, r in candidates:
            ok, q, _ = self.analyze_pnn_quality(processed, x, y, r)
            if ok:
                valid.append((x, y, r))
                scores.append(q)
        final = self.remove_overlapping_detections(valid, scores)
        final_scores: List[float] = []
        for x, y, r in final:
            _, q, _ = self.analyze_pnn_quality(processed, x, y, r)
            final_scores.append(q)

        # Convert pixel coordinates to mm
        pnn_circles_mm = [
            (float(x) * self.pixel_size_mm, 
             float(y) * self.pixel_size_mm, 
             float(r) * self.pixel_size_mm)
            for x, y, r in final
        ]
        
        params = dict(
            min_pnn_radius_mm=self.min_pnn_radius_mm,
            max_pnn_radius_mm=self.max_pnn_radius_mm,
            pixel_size_mm=self.pixel_size_mm,
            min_pnn_radius_px=self.min_pnn_radius,
            max_pnn_radius_px=self.max_pnn_radius,
            contrast_threshold=self.contrast_threshold,
            uniformity_threshold=self.uniformity_threshold,
            template_threshold=self.template_threshold,
            use_clahe=self.use_clahe,
            apply_background_subtraction=self.apply_background_subtraction,
            center_darkness_threshold=self.center_darkness_threshold,
        )
        return PNNDetectionResult(
            image_path=image_path,
            pnn_circles=final,
            pnn_circles_mm=pnn_circles_mm,
            pnn_count=len(final),
            quality_scores=final_scores,
            detection_params=params,
        )

    def draw_pnn_detections(
        self,
        image: np.ndarray,
        pnn_circles: List[Tuple[int, int, int]],
        quality_scores: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Draw detected circles on the image with quality-based coloring."""
        out = image.copy()
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        for i, (x, y, r) in enumerate(pnn_circles):
            color = (0, 255, 0)
            cv2.circle(out, (x, y), int(r), color, 2)
            cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
        return out
