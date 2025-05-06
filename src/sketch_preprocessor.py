import cv2
import numpy as np
from PIL import Image

class SketchPreprocessor:
    def __init__(self, target_size=(224,224)):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"))
        _, mask = cv2.threshold(arr, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(mask)
        if coords is None:
            resized = cv2.resize(arr, self.target_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(rgb)
        
        x, y, w, h = cv2.boundingRect(coords)
        crop = mask[y:y+h, x:x+w]

        if crop.size == 0:
            resized = cv2.resize(arr, self.target_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(rgb)
        
        side = max(w, h)
        canvas = np.zeros((side, side), dtype=np.uint8)
        canvas[:h, :w] = crop
        
        canvas = cv2.resize(canvas, self.target_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)