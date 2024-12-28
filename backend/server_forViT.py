import os
from PIL import Image, ImageOps
import torch
from transformers import AutoProcessor, ViTForImageClassification

from flask import Flask, request, jsonify, send_file, render_template
from rembg import remove
import io
import base64
import json

# Initialize Flask app
app = Flask(__name__)

# Load saved model and processor
model_path = "model/ViT-plant-classifier"
model = ViTForImageClassification.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

# Load prediction descriptions from external JSON file
description_file_path = "descriptions.json"
with open(description_file_path, "r", encoding="utf-8") as f:
    prediction_descriptions = json.load(f)

# 測試圖片輸入至模型後回傳預測結果
def predict_image(image):
    """Predict the class of the given image."""
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()

        # Get the highest confidence
        confidence = probabilities[0, predicted_class_idx].item()

    return predicted_class_idx, confidence


@app.route('/')
def index():
    """Render the upload page."""
    return render_template('upload.html')

# 上傳到模型進行預測
@app.route('/upload', methods=['POST'])
def upload_image():
    """Endpoint for image upload and processing."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file found."}), 400

    image_file = request.files['image']
    try:
        # Save uploaded image to an in-memory buffer
        buffer = io.BytesIO()
        image_file.save(buffer)
        buffer.seek(0)

        # Open image
        image = Image.open(buffer).convert("RGB")

        # Predict class
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()

        # Get description from JSON
        prediction_data = prediction_descriptions.get(str(predicted_class_idx), None)
        if prediction_data:
            prediction_details = prediction_data[0]
            prediction = prediction_details.get("No.", "Unknown") + prediction_details.get("Name", "Unknown")
            description = prediction_details.get("description", "No description available.")
            scientific_name = prediction_details.get("Scientific Name", "Unknown")
            family = prediction_details.get("Family", "Unknown")
            family_tw = prediction_details.get("Family_TW", "Unknown")
            image_path = prediction_details.get("image", "./static/images/placeholder.png")  # Default placeholder
        else:
            prediction = "Unknown"
            description = "No description available."
            scientific_name = "Unknown"
            family = "Unknown"
            family_tw = "Unknown"
            image_path = "./static/images/placeholder.png"  # Default placeholder

        # Convert processed image to Base64
        processed_buffer = io.BytesIO()
        image.save(processed_buffer, format="PNG")
        processed_buffer.seek(0)
        image_base64 = base64.b64encode(processed_buffer.getvalue()).decode('utf-8')

        # Render the result page
        return render_template(
            'result.html',
            prediction=prediction,
            name=prediction_details.get("Name", "Unknown"),
            description=description,
            confidence=confidence,
            scientific_name=scientific_name,
            family=family,
            family_tw=family_tw,
            image_path=image_path,
            image_data=image_base64
        )

    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500



if __name__ == "__main__":
    app.run(debug=True)
