import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from io import BytesIO

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained Keras model
model_path = 'tomato_leaf_disease_model12.keras'  # Path to your saved model
model = load_model(model_path)

# Define class labels
class_indices = {
    0: "Bacterial Spot",
    1: "Early Blight",
    2: "Late Blight",
    3: "Leaf Mold",
    4: "Septoria Leaf Spot",
    5: "Spider Mites",
    6: "Target Spot",
    7: "Yellow Leaf Curl Virus",
    8: "Mosaic Virus",
    9: "Healthy",
    10: "Other Disease"
}

# Define image dimensions
img_height, img_width = 150, 150  # Same as used in training

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Convert the file to a BytesIO object, which load_img can accept
    try:
        img_bytes = file.read()
        img = load_img(BytesIO(img_bytes), target_size=(img_height, img_width))  # Use BytesIO
        img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(predicted_class_idx, "Unknown")

        # Return prediction result
        return jsonify({
            "predicted_class": predicted_class_name,
            "confidence": float(predictions[0][predicted_class_idx])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define a route for health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



