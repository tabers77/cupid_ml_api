from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load saved models
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
knn_model = pickle.load(open("models/knn_model.pkl", "rb"))

# Load room data
# hotel_rooms_grouped = pd.read_csv("datasets/preprocessed_hotel_rooms.csv")  # Save this file earlier
supplier_rooms_grouped = pd.read_csv("datasets/preprocessed_supplier_rooms.csv")  # Save this file earlier


@app.route("/match_rooms", methods=["POST"])
def match_rooms():
    data = request.get_json()
    room_name = data.get("room_name", "")

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

    return jsonify({
        "Results": results,
        "UnmappedRooms": unmapped_rooms
    })


if __name__ == "__main__":
    # Use dynamic port for deployment
    port = 80  # int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
