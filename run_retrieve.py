import os, sys
from PIL import Image
import clip
import torch
import numpy as np

project_root = os.path.dirname(__file__)
src_dir      = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from retrieve_icons import IconRetriever

def self_retrieval(ir, k=5):
    """Test retrieving a specific arrow icon against itself."""
    name = "baseline_arrow_right_alt_black_48dp.png"
    path = os.path.join(project_root, "data", "icons", name)
    img  = Image.open(path).convert("RGB")

    # Encode with CLIP and normalize
    tensor = ir.preprocess(img).unsqueeze(0).to(ir.device)
    with torch.no_grad():
        emb = ir.model.encode_image(tensor).cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    # k-NN search
    dists, idxs = ir.knn.kneighbors(emb, n_neighbors=k)
    results = [(ir.names[i], float(dists[0][j])) for j, i in enumerate(idxs[0])]

    print(f"\nSelf-retrieval for '{name}':")
    for nm, dist in results:
        print(f"  {nm:40s} cos_dist={dist:.4f}")


def sketch_retrieval(ir, sketch_fname="tests/sketch_arrow.png", k=5):
    """Test retrieving icons for a user sketch."""
    path = os.path.join(project_root, sketch_fname)
    sketch = Image.open(path)

    print(f"\nSketch-retrieval for '{sketch_fname}':")
    # Use the retriever's scoring method
    results = ir.retrieve_with_scores(sketch, k=k)
    for nm, dist in results:
        print(f"  {nm:40s} cos_dist={dist:.4f}")

def main():
    print("Loading retriever and index..")
    ir = IconRetriever(device="cpu")

    self_retrieval(ir, k=5)

    sketch_retrieval(ir, sketch_fname="tests/sketch_arrow.png", k=5)

if __name__ == "__main__":
    main()