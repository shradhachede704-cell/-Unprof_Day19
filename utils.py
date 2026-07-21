from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class RAGPipeline:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []

    def load_documents(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.documents = [line.strip() for line in f if line.strip()]

    def create_embeddings(self):
        embeddings = self.model.encode(self.documents)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=3):

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distance, index = self.index.search(query_embedding, top_k)

        results = []

        for i in index[0]:
            results.append(self.documents[i])

        return results
