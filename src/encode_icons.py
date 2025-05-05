import os
import numpy as np
import pickle
import clip
import torch 
from torchvision import transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors

ICON_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'icons')
EMBED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'icons_embeddings.npy')
NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'icons_names.pkl')
KNN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'knn_index.pkl')

def load_icons():
    files = sorted([f for f in os.listdir(ICON_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    imgs = [Image.open(os.path.join(ICON_DIR, f)).convert('RGB') for f in files]
    return files, imgs

def encode_images(model, preprocess, images, device):
    model.to(device)
    features = []
    with torch.no_grad():
        for img in images:
            x = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(x)
            features.append(feat.cpu().numpy().flatten())
    return np.vstack(features)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)

    names, icons = load_icons()

    embeddings = encode_images(model, preprocess, icons, device)
    np.save(EMBED_PATH, embeddings)
    with open(NAMES_PATH, 'wb') as f:
        pickle.dump(names, f)

    nn = NearestNeighbors(n_neighbors = 10, metric = 'cosine')
    nn.fit(embeddings)
    with open(KNN_PATH, 'wb') as f:
        pickle.dump(nn, f)

    print(f"Encoded {len(names)} icons and built k-NN index.")

if __name__ == '__main__':
    main()