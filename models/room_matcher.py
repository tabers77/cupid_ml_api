import os
import pickle
import re
from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from flask import jsonify


class RoomMatcher:
    """
    # # API EXAMPLE OF USAGE
# matcher = RoomMatcher()
# hotel_rooms = pd.read_csv("datasets/referance_rooms.csv")  # Load from local or cloud
# supplier_rooms = pd.read_csv("datasets/updated_core_rooms.csv")
#
# hotel_rooms_grouped, supplier_rooms_grouped = matcher.preprocess_data(hotel_rooms, supplier_rooms)
# hotel_vectors, supplier_vectors = matcher.vectorize_data(hotel_rooms_grouped, supplier_rooms_grouped)

# # Train and save the model
# matcher.knn.fit(supplier_vectors)  # Train the kNN model
# matcher.save_model()  # Save the trained models for later API usage



    """

    def __init__(self, top_k: int = 10, threshold: float = 0.75, vectorizer=TfidfVectorizer()):
        self.supplier_vectors = None
        self.hotel_vectors = None
        self.supplier_rooms_grouped = None
        self.hotel_rooms_grouped = None
        self.top_k = top_k
        self.threshold = threshold
        self.vectorizer = vectorizer
        self.knn = NearestNeighbors(n_neighbors=self.top_k, metric="cosine", algorithm="auto")
        self.model_path = "/dbfs/FileStore/cupid"

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Lowercase, remove special characters and extra spaces."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)  # Keep only alphanumeric
        return re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    def preprocess_data(self, hotel_rooms: pd.DataFrame, supplier_rooms: pd.DataFrame) -> Tuple:
        """Cleans and processes room names for vectorization and matching."""
        hotel_rooms = hotel_rooms.copy()
        supplier_rooms = supplier_rooms.dropna().copy()

        # Apply text preprocessing
        hotel_rooms["clean_room_name"] = hotel_rooms["room_name"].apply(self.preprocess_text)
        supplier_rooms["clean_supplier_room_name"] = supplier_rooms["supplier_room_name"].apply(self.preprocess_text)

        # Group by unique cleaned names
        hotel_rooms_grouped = hotel_rooms.groupby("clean_room_name")["room_id"].apply(list).reset_index()
        supplier_rooms_grouped = supplier_rooms.groupby("clean_supplier_room_name")["supplier_room_id"].apply(
            list).reset_index()

        return hotel_rooms_grouped, supplier_rooms_grouped

    def vectorize_data(self, hotel_rooms_grouped: pd.DataFrame, supplier_rooms_grouped: pd.DataFrame) -> Tuple:
        """Vectorizes room names using TF-IDF."""
        unique_hotel_names = hotel_rooms_grouped["clean_room_name"].tolist()
        unique_supplier_names = supplier_rooms_grouped["clean_supplier_room_name"].tolist()

        all_unique_room_names = unique_hotel_names + unique_supplier_names
        tfidf_matrix = self.vectorizer.fit_transform(all_unique_room_names)

        # Split matrices for hotel and supplier rooms
        hotel_vectors = tfidf_matrix[:len(unique_hotel_names)]
        supplier_vectors = tfidf_matrix[len(unique_hotel_names):]

        return hotel_vectors, supplier_vectors

    def find_best_matches(self, supplier_vectors, hotel_vectors, hotel_rooms_grouped,
                          supplier_rooms_grouped) -> pd.DataFrame:
        """Finds the best matches for hotel rooms using kNN and cosine similarity."""

        self.knn.fit(supplier_vectors)  # Fit on supplier room vectors

        distances, indices = self.knn.kneighbors(hotel_vectors, return_distance=True)

        matches = []
        hotel_room_ids = hotel_rooms_grouped.room_id.values
        supplier_room_ids = supplier_rooms_grouped.supplier_room_id.values

        for i, hotel_room_id in enumerate(hotel_room_ids):
            for j in range(self.top_k):
                supplier_index = indices[i][j]
                similarity_score = 1 - distances[i][j]  # Convert cosine distance to similarity
                if similarity_score > self.threshold:
                    matches.append((hotel_room_id, supplier_room_ids[supplier_index], similarity_score))

        return pd.DataFrame(matches, columns=["hotel_room_id", "supplier_room_id", "similarity_score"])

    def match_rooms(self, hotel_rooms: pd.DataFrame, supplier_rooms: pd.DataFrame) -> pd.DataFrame:
        """End-to-end process to match rooms."""
        hotel_rooms_grouped, supplier_rooms_grouped = self.preprocess_data(hotel_rooms, supplier_rooms)
        hotel_vectors, supplier_vectors = self.vectorize_data(hotel_rooms_grouped, supplier_rooms_grouped)
        return self.find_best_matches(supplier_vectors, hotel_vectors, hotel_rooms_grouped, supplier_rooms_grouped)

    # def evaluate(self, supplier_vectors, hotel_vectors, hotel_rooms_grouped, supplier_rooms_grouped):
    #     self.knn.fit(supplier_vectors)  # Fit on supplier room vectors
    #
    #     distances, indices = self.knn.kneighbors(hotel_vectors, return_distance=True)
    #
    #     matches = []
    #     hotel_room_ids = hotel_rooms_grouped.room_id.values
    #
    #     # Loop through each hotel room and its nearest neighbors
    #     for i, hotel_room_id in tqdm(enumerate(hotel_room_ids), total=len(hotel_room_ids), desc="Evaluating Matches"):
    #         for j in range(3):  # top 3 matches (change this as per your requirement)
    #             supplier_index = indices[i][j]
    #             similarity_score = 1 - distances[i][j]  # Convert cosine distance to similarity
    #
    #             # Access hotel and supplier rooms by their position using .iloc[]
    #             hotel_room_name = hotel_rooms_grouped.iloc[i]["clean_room_name"]
    #             supplier_room_name = supplier_rooms_grouped.iloc[supplier_index]["clean_supplier_room_name"]
    #
    #             # Check if the similarity score meets the threshold
    #             if similarity_score > self.threshold:
    #                 matches.append({
    #                     "hotel_room_name": hotel_room_name,
    #                     "supplier_room_name": supplier_room_name,
    #                     "similarity_score": similarity_score,
    #                     "match": 1  # Mark as match if similarity score > threshold
    #                 })
    #             else:
    #                 matches.append({
    #                     "hotel_room_name": hotel_room_name,
    #                     "supplier_room_name": supplier_room_name,
    #                     "similarity_score": similarity_score,
    #                     "match": 0  # Mark as no match if similarity score < threshold
    #                 })
    #
    #     return pd.DataFrame(matches), matches

    def save_model(self):
        """Save the vectorizer, kNN model, and precomputed vectors."""

        os.makedirs(self.model_path, exist_ok=True)
        pickle.dump(self.vectorizer, open(os.path.join(self.model_path, "vectorizer.pkl"), "wb"))
        pickle.dump(self.knn, open(os.path.join(self.model_path, "knn_model.pkl"), "wb"))

    def load_model(self):
        """Load the vectorizer, kNN model, and precomputed vectors."""

        self.vectorizer = pickle.load(open(os.path.join(self.model_path, "vectorizer.pkl"), "rb"))
        self.knn = pickle.load(open(os.path.join(self.model_path, "knn_model.pkl"), "rb"))


# UTIL FUNCTION TO TEST API RESULTS

def match_rooms_test(vectorizer, knn_model, supplier_rooms_grouped, room_name):
    if room_name not in vectorizer.vocabulary_:
        print("WARNING: 'apartment' not found in the vectorizer vocabulary!")

    if not room_name:
        return jsonify({"error": "room_name is required"}), 400

    # Vectorize the input room name
    room_vector = vectorizer.transform([room_name])

    # Find top-k matches
    distances, indices = knn_model.kneighbors(room_vector, return_distance=True)

    # Initialize matched and unmatched rooms
    results = []
    unmapped_rooms = []

    # Iterate over the indices and distances
    for idx, dist in zip(indices[0], distances[0]):
        similarity_score = 1 - dist  # Convert cosine distance to similarity
        if similarity_score > 0.75:  # Ensure threshold condition

            try:
                # Get matched room details
                matched_room = {
                    "supplierRoomName": supplier_rooms_grouped.iloc[idx]["clean_supplier_room_name"],
                    "mappedRooms": [
                        {
                            "score": similarity_score,
                            "supplierRoomId": supplier_rooms_grouped.iloc[idx]["supplier_room_id"],
                            "supplierRoomName": supplier_rooms_grouped.iloc[idx]["clean_supplier_room_name"]
                        }
                    ],

                }

                results.append(matched_room)

            except IndexError:
                print(f"Warning: Index {idx} is out of bounds for supplier_rooms_grouped!")

        else:
            # Add to unmapped rooms if similarity score is below threshold
            unmapped_room = {
                "supplierRoomName": supplier_rooms_grouped.iloc[idx]["clean_supplier_room_name"],
                "supplierRoomId": supplier_rooms_grouped.iloc[idx]["supplier_room_id"],
            }
            unmapped_rooms.append(unmapped_room)

    return results, unmapped_rooms
