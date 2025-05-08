import os
import numpy as np
import pickle
import clip
import torch
import cv2
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# Directory paths
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
ICON_DIR   = os.path.join(BASE_DIR, 'data', 'icons')
EMBED_PATH = os.path.join(BASE_DIR, 'data', 'icons_embeddings.npy')
NAMES_PATH = os.path.join(BASE_DIR, 'data', 'icons_names.pkl')
KNN_PATH   = os.path.join(BASE_DIR, 'data', 'knn_index.pkl')


def load_icons(style_prefixes=('baseline_',), extensions=('.png','.jpg','.jpeg')):
    """
    Load only those icons whose filenames start with one of the given prefixes.
    e.g. style_prefixes=('baseline_', 'twotone_')
    """
    files = sorted(
        f for f in os.listdir(ICON_DIR)
        if any(f.startswith(pref) for pref in style_prefixes)
           and f.lower().endswith(extensions)
    )
    print(f"Loaded {len(files)} icons using prefixes {style_prefixes}")
    imgs = [Image.open(os.path.join(ICON_DIR, f)).convert('RGBA') for f in files]
    return files, imgs


def encode_images(model, preprocess, images, device, names=None):
    """Encode a list of PIL images into CLIP feature vectors, printing progress."""
    model.to(device)
    features = []
    with torch.no_grad():
        for i, img in enumerate(images, start=1):
            if names:
                print(f"Encoding {i}/{len(images)}: {names[i-1]}")
            x = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(x)
            features.append(feat.cpu().numpy().flatten())
    return np.vstack(features)


def main():
    # 1) Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2) Load CLIP model
    model, preprocess = clip.load('ViT-B/32', device=device)

    # 3) Load icons
    names, icons = load_icons(style_prefixes=['baseline_'])
    print(f"Converting {len(icons)} icons to binary silhouettes...")

    # 4) Set CLIP input size
    target_size = (224, 224)

    # 5) Generate binary silhouette images from alpha channel
    sil_imgs = []
    for i, img in enumerate(icons, start=1):
        # Extract alpha channel as mask
        arr_alpha = np.array(img.split()[-1])  # shape HxW
        # Resize mask to target size
        mask_up   = cv2.resize(arr_alpha, target_size, interpolation=cv2.INTER_NEAREST)
        # Binarize: any non-zero alpha → 255
        _, mask_bin = cv2.threshold(mask_up, 0, 255, cv2.THRESH_BINARY)
        # Optional dilate to thicken strokes
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)
        # Convert to 3-channel RGB for CLIP
        sil_pil  = Image.fromarray(mask_bin).convert('RGB')
        sil_imgs.append(sil_pil)

        # Save debug for first few
        if i <= 5:
            sil_pil.save(os.path.join(BASE_DIR, 'data', f'debug_edge_{names[i-1]}'))
        print(f"Prepared silhouette for {names[i-1]} ({i}/{len(icons)})")

    # 6) Encode silhouette images
    embeddings = encode_images(model, preprocess, sil_imgs, device, names)

    # 7) Normalize embeddings to unit length
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # 8) Save embeddings and names
    np.save(EMBED_PATH, embeddings)
    with open(NAMES_PATH, 'wb') as f:
        pickle.dump(names, f)

    # 9) Build & save k-NN index
    nn = NearestNeighbors(n_neighbors=10, metric='cosine')
    nn.fit(embeddings)
    with open(KNN_PATH, 'wb') as f:
        pickle.dump(nn, f)

    print("✔️ Binary-silhouette encode complete; saved new index.")

if __name__ == '__main__':
    main()