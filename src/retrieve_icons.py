import os, pickle, numpy as np, clip, torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from src.sketch_preprocessor import SketchPreprocessor

BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
EMBED_PATH  = os.path.join(BASE_DIR, 'data', 'icons_embeddings.npy')
NAMES_PATH  = os.path.join(BASE_DIR, 'data', 'icons_names.pkl')
KNN_PATH    = os.path.join(BASE_DIR, 'data', 'knn_index.pkl')

class IconRetriever:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.embs = np.load(EMBED_PATH)
        with open(NAMES_PATH, "rb") as f: 
            self.names = pickle.load(f)

        self.embs = self.embs / np.linalg.norm(self.embs, axis=1, keepdims=True)

        with open(KNN_PATH, "rb") as f: 
            self.knn = pickle.load(f)
        
        self.sketch_pp = SketchPreprocessor(target_size=(224, 224))


    def retrieve(self, sketch: Image.Image, k: int = 5):
        proc = self.sketch_pp(sketch)
        x    = self.preprocess(proc).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model.encode_image(x).cpu().numpy()
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        dists, idxs = self.knn.kneighbors(q, n_neighbors=k)
        return [self.names[i] for i in idxs[0]]

    def retrieve_with_scores(self, sketch: Image.Image, k: int = 5):
        proc = self.sketch_pp(sketch)
        x    = self.preprocess(proc).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.model.encode_image(x).cpu().numpy()
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        dists, idxs = self.knn.kneighbors(q, n_neighbors=k)
        return [(self.names[i], float(dists[0][j])) for j, i in enumerate(idxs[0])]
