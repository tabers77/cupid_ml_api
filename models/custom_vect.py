# import re
# import math
# import numpy as np
# from collections import Counter
# from scipy.sparse import lil_matrix, csr_matrix
#
#
# class TfidfVectorizerCustom:
#     def __init__(self, *,
#                  input="content",
#                  encoding="utf-8",
#                  decode_error="strict",
#                  strip_accents=None,
#                  lowercase=True,
#                  preprocessor=None,
#                  tokenizer=None,
#                  analyzer="word",
#                  stop_words=None,
#                  token_pattern=r"(?u)\b\w\w+\b",
#                  ngram_range=(1, 1),
#                  max_df=1.0,
#                  min_df=1,
#                  max_features=None,
#                  vocabulary=None,
#                  binary=False,
#                  dtype=np.float64,
#                  norm="l2",
#                  use_idf=True,
#                  smooth_idf=True,
#                  sublinear_tf=False):
#         # Most parameters are stored for compatibility;
#         # this simple implementation supports only a subset of options.
#         self.input = input
#         self.encoding = encoding
#         self.decode_error = decode_error
#         self.strip_accents = strip_accents
#         self.lowercase = lowercase
#         self.preprocessor = preprocessor
#         self.tokenizer = tokenizer
#         self.analyzer = analyzer
#         self.stop_words = stop_words
#         self.token_pattern = token_pattern
#         self.ngram_range = ngram_range
#         self.max_df = max_df
#         self.min_df = min_df
#         self.max_features = max_features
#         self.vocabulary = vocabulary  # If provided, a dict mapping term to index.
#         self.binary = binary
#         self.dtype = dtype
#         self.norm = norm
#         self.use_idf = use_idf
#         self.smooth_idf = smooth_idf
#         self.sublinear_tf = sublinear_tf
#
#         # These attributes are set during fitting
#         self.vocabulary_ = {}  # Maps token -> index
#         self.idf_ = None  # Array of IDF weights
#
#         # If no tokenizer is provided, compile a regex for token extraction.
#         if self.tokenizer is None:
#             self.token_pattern_re = re.compile(token_pattern)
#
#     def _preprocess(self, doc):
#         # Apply preprocessor if provided, then lowercase if desired.
#         if self.preprocessor is not None:
#             doc = self.preprocessor(doc)
#         if self.lowercase:
#             doc = doc.lower()
#         return doc
#
#     def _tokenize(self, doc):
#         # Use custom tokenizer or regex-based token extraction.
#         if self.tokenizer is not None:
#             tokens = self.tokenizer(doc)
#         else:
#             tokens = self.token_pattern_re.findall(doc)
#         # Remove stop words if a list is provided.
#         if self.stop_words is not None:
#             tokens = [t for t in tokens if t not in self.stop_words]
#         return tokens
#
#     def _generate_ngrams(self, tokens):
#         min_n, max_n = self.ngram_range
#         ngrams = []
#         for n in range(min_n, max_n + 1):
#             if len(tokens) < n:
#                 continue
#             for i in range(len(tokens) - n + 1):
#                 ngram = " ".join(tokens[i:i + n])
#                 ngrams.append(ngram)
#         return ngrams
#
#     def build_vocabulary(self, raw_documents):
#         """
#         Build vocabulary mapping and collect tokens per document.
#         Returns:
#             vocabulary: dict mapping token -> index
#             documents_tokens: list of token lists for each document.
#         """
#         doc_freq = Counter()
#         documents_tokens = []
#         for doc in raw_documents:
#             processed = self._preprocess(doc)
#             tokens = self._tokenize(processed)
#             if self.ngram_range != (1, 1):
#                 tokens = self._generate_ngrams(tokens)
#             documents_tokens.append(tokens)
#             # Update document frequency (each token counted once per doc)
#             unique_tokens = set(tokens)
#             doc_freq.update(unique_tokens)
#
#         n_docs = len(raw_documents)
#         vocabulary = {}
#         idx = 0
#         for token, freq in doc_freq.items():
#             # Filter tokens based on min_df and max_df.
#             if isinstance(self.min_df, int) and freq < self.min_df:
#                 continue
#             if isinstance(self.max_df, float) and (freq / n_docs) > self.max_df:
#                 continue
#             if isinstance(self.max_df, int) and freq > self.max_df:
#                 continue
#             vocabulary[token] = idx
#             idx += 1
#         return vocabulary, documents_tokens
#
#     def fit(self, raw_documents, y=None):
#         """
#         Learn vocabulary and idf from training documents.
#         """
#         if self.vocabulary is not None:
#             self.vocabulary_ = self.vocabulary
#             # Tokenize documents to compute document frequencies.
#             documents_tokens = []
#             for doc in raw_documents:
#                 processed = self._preprocess(doc)
#                 tokens = self._tokenize(processed)
#                 if self.ngram_range != (1, 1):
#                     tokens = self._generate_ngrams(tokens)
#                 documents_tokens.append(tokens)
#         else:
#             self.vocabulary_, documents_tokens = self.build_vocabulary(raw_documents)
#
#         # Count document frequency for each term.
#         doc_freq = np.zeros(len(self.vocabulary_), dtype=self.dtype)
#         for tokens in documents_tokens:
#             seen = set()
#             for token in tokens:
#                 if token in self.vocabulary_:
#                     idx = self.vocabulary_[token]
#                     if idx not in seen:
#                         doc_freq[idx] += 1
#                         seen.add(idx)
#
#         # Compute the IDF vector.
#         if self.use_idf:
#             n_docs = len(raw_documents)
#             self.idf_ = np.zeros(len(self.vocabulary_), dtype=self.dtype)
#             for token, idx in self.vocabulary_.items():
#                 df = doc_freq[idx]
#                 if self.smooth_idf:
#                     self.idf_[idx] = math.log((n_docs + 1) / (df + 1)) + 1
#                 else:
#                     self.idf_[idx] = math.log(n_docs / df) if df != 0 else 0.0
#         else:
#             self.idf_ = np.ones(len(self.vocabulary_), dtype=self.dtype)
#         return self
#
#     def transform(self, raw_documents):
#         n_docs = len(raw_documents)
#         n_features = len(self.vocabulary_)
#
#         # Use a sparse matrix instead of a dense NumPy array
#         X = lil_matrix((n_docs, n_features), dtype=self.dtype)  # lil_matrix is good for incremental updates
#
#         for doc_idx, doc in enumerate(raw_documents):
#             for term in doc.split():  # Assuming simple whitespace tokenization
#                 if term in self.vocabulary_:
#                     term_idx = self.vocabulary_[term]
#                     X[doc_idx, term_idx] += 1
#
#         return csr_matrix(X)  # Convert to efficient sparse format (CSR)

    # def transform(self, raw_documents):
    #     """
    #     Transform documents to TF-IDF-weighted document-term matrix.
    #     Returns:
    #         X: a NumPy array of shape (n_samples, n_features)
    #     """
    #     n_docs = len(raw_documents)
    #     n_features = len(self.vocabulary_)
    #     X = np.zeros((n_docs, n_features), dtype=self.dtype)
    #     for i, doc in enumerate(raw_documents):
    #         processed = self._preprocess(doc)
    #         tokens = self._tokenize(processed)
    #         if self.ngram_range != (1, 1):
    #             tokens = self._generate_ngrams(tokens)
    #         token_counts = Counter(tokens)
    #         for token, count in token_counts.items():
    #             if token in self.vocabulary_:
    #                 idx = self.vocabulary_[token]
    #                 if self.binary:
    #                     X[i, idx] = 1
    #                 else:
    #                     if self.sublinear_tf:
    #                         X[i, idx] = 1 + math.log(count)
    #                     else:
    #                         X[i, idx] = count
    #     # Apply IDF weighting if enabled.
    #     if self.use_idf:
    #         X = X * self.idf_
    #     # Normalize each document vector.
    #     if self.norm == "l2":
    #         norms = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    #         norms[norms == 0] = 1  # avoid division by zero
    #         X = X / norms
    #     elif self.norm == "l1":
    #         norms = np.linalg.norm(X, ord=1, axis=1, keepdims=True)
    #         norms[norms == 0] = 1
    #         X = X / norms
    #     return X
    #
    # def fit_transform(self, raw_documents, y=None):
    #     """
    #     Learn vocabulary and idf, then return the TF-IDF-weighted document-term matrix.
    #     """
    #     self.fit(raw_documents, y)
    #     return self.transform(raw_documents)
    #
    # def get_feature_names_out(self):
    #     """
    #     Return an array of feature names (tokens) sorted by their index in the vocabulary.
    #     """
    #     feature_names = [None] * len(self.vocabulary_)
    #     for token, idx in self.vocabulary_.items():
    #         feature_names[idx] = token
    #     return np.array(feature_names)

# import numpy as np
# import pickle
# import json
# import math
# from collections import Counter
# from scipy.sparse import csr_matrix, save_npz, load_npz
#
#
# class CustomTfidfVectorizer:
#     def __init__(self):
#         self.vocab = {}  # Word index mapping
#         self.idf = {}  # IDF values
#
#     def fit_transform(self, corpus):
#         """Fits vectorizer and transforms corpus into sparse TF-IDF matrix."""
#         self.vocab = {word: idx for idx, word in enumerate(set(" ".join(corpus).split()))}
#
#         # Compute TF (Sparse representation)
#         row, col, data = [], [], []
#         doc_count = len(corpus)
#
#         for doc_idx, doc in enumerate(corpus):
#             word_counts = Counter(doc.split())
#             for word, count in word_counts.items():
#                 if word in self.vocab:
#                     row.append(doc_idx)
#                     col.append(self.vocab[word])
#                     data.append(count)
#
#         tf_matrix = csr_matrix((data, (row, col)), shape=(doc_count, len(self.vocab)))
#
#         # Compute IDF
#         for word in self.vocab:
#             containing_docs = sum(1 for doc in corpus if word in doc)
#             self.idf[word] = math.log((doc_count + 1) / (containing_docs + 1)) + 1  # Smooth IDF
#
#         # Apply IDF
#         for i, word in enumerate(self.vocab):
#             tf_matrix[:, i] *= self.idf[word]
#
#         return tf_matrix
#
#     def transform(self, texts):
#         """Transforms new texts into sparse TF-IDF vectors."""
#         row, col, data = [], [], []
#         for doc_idx, doc in enumerate(texts):
#             word_counts = Counter(doc.split())
#             for word, count in word_counts.items():
#                 if word in self.vocab:
#                     row.append(doc_idx)
#                     col.append(self.vocab[word])
#                     data.append(count * self.idf.get(word, 0))
#
#         return csr_matrix((data, (row, col)), shape=(len(texts), len(self.vocab)))
#
#     def save(self, filepath):
#         """Save vectorizer to a file."""
#         with open(filepath, "wb") as f:
#             pickle.dump({"vocab": self.vocab, "idf": self.idf}, f)
#
#     def load(self, filepath):
#         """Load vectorizer from a file."""
#         with open(filepath, "rb") as f:
#             data = pickle.load(f)
#             self.vocab = data["vocab"]
#             self.idf = data["idf"]
