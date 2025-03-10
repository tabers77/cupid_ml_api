import numpy as np
import pickle
import json
import math
from collections import Counter
from scipy.sparse import csr_matrix, save_npz, load_npz


class CustomTfidfVectorizer:
    def __init__(self):
        self.vocab = {}  # Word index mapping
        self.idf = {}  # IDF values

    def fit_transform(self, corpus):
        """Fits vectorizer and transforms corpus into sparse TF-IDF matrix."""
        self.vocab = {word: idx for idx, word in enumerate(set(" ".join(corpus).split()))}

        # Compute TF (Sparse representation)
        row, col, data = [], [], []
        doc_count = len(corpus)

        for doc_idx, doc in enumerate(corpus):
            word_counts = Counter(doc.split())
            for word, count in word_counts.items():
                if word in self.vocab:
                    row.append(doc_idx)
                    col.append(self.vocab[word])
                    data.append(count)

        tf_matrix = csr_matrix((data, (row, col)), shape=(doc_count, len(self.vocab)))

        # Compute IDF
        for word in self.vocab:
            containing_docs = sum(1 for doc in corpus if word in doc)
            self.idf[word] = math.log((doc_count + 1) / (containing_docs + 1)) + 1  # Smooth IDF

        # Apply IDF
        for i, word in enumerate(self.vocab):
            tf_matrix[:, i] *= self.idf[word]

        return tf_matrix

    def transform(self, texts):
        """Transforms new texts into sparse TF-IDF vectors."""
        row, col, data = [], [], []
        for doc_idx, doc in enumerate(texts):
            word_counts = Counter(doc.split())
            for word, count in word_counts.items():
                if word in self.vocab:
                    row.append(doc_idx)
                    col.append(self.vocab[word])
                    data.append(count * self.idf.get(word, 0))

        return csr_matrix((data, (row, col)), shape=(len(texts), len(self.vocab)))

    def save(self, filepath):
        """Save vectorizer to a file."""
        with open(filepath, "wb") as f:
            pickle.dump({"vocab": self.vocab, "idf": self.idf}, f)

    def load(self, filepath):
        """Load vectorizer from a file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.vocab = data["vocab"]
            self.idf = data["idf"]
