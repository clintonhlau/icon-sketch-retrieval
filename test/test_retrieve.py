import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from src.retrieve_icons import IconRetriever

sketch = Image.open("test/sketch_arrow.png")

ir = IconRetriever(device="cpu")

matches = ir.retrieve(sketch, k=5)
print("Top matches:", matches)