import pickle

from app import app
from models.room_matcher import RoomMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os

import pytest
import pandas as pd

base_dir = r"C:\Users\delacruzribadenc\Documents\Repos\cupid_ml_api"


def get_base_dir(path):
    return os.path.join(base_dir, path)


@pytest.fixture
def client():
    """Creates a Flask test client for API testing"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_vectorizer():
    """Mock a pre-trained vectorizer"""
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["room deluxe", "standard suite", "penthouse"])
    return vectorizer


@pytest.fixture
def mock_knn_model(mock_vectorizer):
    """Mock a trained kNN model"""
    knn_model = NearestNeighbors(n_neighbors=3, metric="cosine", algorithm="auto")
    tfidf_matrix = mock_vectorizer.transform(["room deluxe", "standard suite", "penthouse"])
    knn_model.fit(tfidf_matrix)
    return knn_model


@pytest.fixture
def mock_supplier_rooms():
    """Mock supplier room dataset"""
    data = {
        "clean_supplier_room_name": ["room deluxe", "standard suite", "penthouse"],
        "supplier_room_id": [101, 102, 103],
    }
    return pd.DataFrame(data)


# 📌 **FLASK APP TESTS**
def test_match_rooms_success(client, mock_vectorizer, mock_knn_model, mock_supplier_rooms):
    """Test successful room matching"""
    response = client.post(
        "/match_rooms",
        json={"room_name": "deluxe"}
    )
    data = response.get_json()

    assert response.status_code == 200
    assert "Results" in data
    assert isinstance(data["Results"], list)


def test_match_rooms_missing_param(client):
    """Test API error when 'room_name' is missing"""
    response = client.post("/match_rooms", json={})
    data = response.get_json()

    assert response.status_code == 400
    assert "error" in data
    assert data["error"] == "room_name is required"


def test_match_rooms_not_in_vocab(client, mock_vectorizer):
    """Test behavior when room name is not in vectorizer vocabulary"""
    response = client.post(
        "/match_rooms",
        json={"room_name": "nonexistent room"}
    )
    data = response.get_json()

    assert response.status_code == 200
    assert "Results" in data
    assert isinstance(data["Results"], list)


def test_model_loading():
    """Test model loading from disk"""
    vectorizer_path = get_base_dir('models/vectorizer.pkl')
    knn_model_path = get_base_dir('models/knn_model.pkl')

    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    knn_model = pickle.load(open(knn_model_path, "rb"))

    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(knn_model, NearestNeighbors)


# 📌 **ROOM MATCHER CLASS TESTS**
def test_preprocess_text():
    """Test text preprocessing"""
    matcher = RoomMatcher()

    assert matcher.preprocess_text(" Deluxe Room!! ") == "deluxe room"
    assert matcher.preprocess_text("Suite@penthouse") == "suitepenthouse"
    assert matcher.preprocess_text("Standard   Suite  ") == "standard suite"


def test_vectorization(mock_vectorizer):
    """Test vectorization output shape"""
    texts = ["deluxe room", "standard suite"]
    vectors = mock_vectorizer.transform(texts)

    assert vectors.shape[0] == 2  # Two input texts
    assert vectors.shape[1] > 0  # Should have TF-IDF features


def test_match_rooms_knn(mock_knn_model, mock_vectorizer, mock_supplier_rooms):
    """Test kNN matching process"""
    matcher = RoomMatcher(vectorizer=mock_vectorizer)
    matcher.knn = mock_knn_model
    matcher.supplier_rooms_grouped = mock_supplier_rooms

    # Mocking a match request
    room_vector = matcher.vectorizer.transform(["deluxe"])
    distances, indices = matcher.knn.kneighbors(room_vector, return_distance=True)

    assert len(indices[0]) == 3  # Ensure kNN returned top-3 matches
    assert all(0 <= (1 - distances[0][i]) <= 1 for i in range(3))  # Similarity scores should be between 0 and 1
