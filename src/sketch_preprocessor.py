import cv2
import numpy as np
from PIL import Image

class SketchPreprocessor:
    def __init__(self, target_size=(224,224)):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1) Grayscale
        gray = np.array(img.convert('L'))

        # 2) Binarize: strokes -> white (255), background -> black (0)
        _, mask = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 3) Morphological close to seal gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4) Flood-fill the background to get interior fill
        h_img, w_img = closed.shape
        outer = closed.copy()
        ff_mask = np.zeros((h_img + 2, w_img + 2), np.uint8)
        cv2.floodFill(outer, ff_mask, (0, 0), 255)
        interior = cv2.bitwise_not(outer)
        filled   = cv2.bitwise_or(closed, interior)

        # 5) Find content coords; fallback if none
        coords = cv2.findNonZero(filled)
        if coords is None:
            # no strokes: resize full grayscale
            resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
            rgb     = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(rgb)

        # 6) Crop to bounding box of strokes
        x, y, w_box, h_box = cv2.boundingRect(coords)
        crop = filled[y:y + h_box, x:x + w_box]
        if crop.size == 0:
            # empty crop: fallback
            resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
            rgb     = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(rgb)

        # 7) Pad to square canvas
        side   = max(w_box, h_box)
        canvas = np.zeros((side, side), dtype=np.uint8)
        canvas[:h_box, :w_box] = crop

        # 8) Resize and convert to RGB
        canvas = cv2.resize(canvas, self.target_size, interpolation=cv2.INTER_AREA)
        rgb    = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)