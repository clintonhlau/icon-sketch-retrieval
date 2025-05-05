import os, pickle, numpy as np, clip, torch
from sklearn.neighbors import NearestNeighbors
from preprocess import SketchPreprocessor

BASE = os.path.dirname(os.path.dirname(__file__))
EMBED = os.path.join(BASE, "data", "icons_embeddings.npy")
NAMES = os.path.join(BASE, "data", "icons_names.pkl")
KNN_IDX = os.path.join(BASE, "data", "knn_index.pkl")

class IconRetriever:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.embs = np.load(EMBED)
        with open(NAMES, "rb") as f: self.names = pickle.load(f)
        with open(KNN_IDX, "rb") as f: self.knn = pickle.load(f)
        self.sketch_pp = SketchPreprocessor(target_size=self.preprocess.transform[-1].size)

    def retrieve(self, sketch, k=5):
        proc = self.sketch_pp(sketch)
        x = self.preprocess(proc).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model.encode_image(x).cpu().numpy()
        _, idxs = self.knn.kneighbors(q, n_neighbors=k)
        return [self.names[i] for i in idxs[0]]