import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle
import os

# Initialize face detector and embedding model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(frame):
    face = mtcnn(frame)
    if face is None:
        return None
    face = face.to(device)
    embedding = model(face.unsqueeze(0)).detach().cpu().numpy()[0]
    return embedding / np.linalg.norm(embedding)  # Normalize

def load_embeddings():
    if os.path.exists("embeddings/embeddings.pkl") and os.path.exists("embeddings/labels.pkl"):
        with open("embeddings/embeddings.pkl", "rb") as f1, open("embeddings/labels.pkl", "rb") as f2:
            embeddings = pickle.load(f1)
            labels = pickle.load(f2)
    else:
        embeddings = []
        labels = []
    return embeddings, labels

def save_embeddings(embeddings, labels):
    os.makedirs("embeddings", exist_ok=True)
    with open("embeddings/embeddings.pkl", "wb") as f1, open("embeddings/labels.pkl", "wb") as f2:
        pickle.dump(embeddings, f1)
        pickle.dump(labels, f2)
