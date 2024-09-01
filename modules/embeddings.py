from sentence_transformers import SentenceTransformer
import torch
import config

class EmbeddingModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(config.HUGGINGFACE_MODEL_NAME).to(self.device)
        print(f"Using device: {self.device}")

    def embed(self, texts):
        return self.model.encode(texts, device=self.device).tolist()