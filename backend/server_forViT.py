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

@app.route('/upload', methods=['POST'])
def upload_image():
    """Endpoint for image upload and processing."""
    if 'image' not in request.files:
        return jsonify({"error": "No image file found."}), 400

    image_file = request.files['image']

    try:
        # Open the image file
        image = Image.open(image_file).convert("RGB")

        # Remove background
        input_bytes = io.BytesIO()
        image.save(input_bytes, format="PNG")
        input_bytes.seek(0)

        # Pass the PNG data to rembg
        output_bytes = remove(input_bytes.getvalue())
        image_no_bg = Image.open(io.BytesIO(output_bytes))

        # Ensure rembg output is valid
        if image_no_bg.mode != "RGBA":
            image_no_bg = image_no_bg.convert("RGBA")

        # Add white background
        white_bg = Image.new("RGBA", image_no_bg.size, (255, 255, 255, 255))
        image_with_bg = Image.alpha_composite(white_bg, image_no_bg).convert("RGB")

        # Save to in-memory file
        buffer = io.BytesIO()
        image_with_bg.save(buffer, format="PNG")
        buffer.seek(0)

        # Predict class
        predicted_class_idx, confidence = predict_image(image_with_bg)

        # Get the description from JSON
        prediction = prediction_descriptions.get(str(predicted_class_idx), "Unknown class")
        description = f"The predicted plant is {prediction}."

        # Encode image for rendering
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return render_template('result.html', prediction=prediction, description=description, confidence=confidence, image_data=image_base64)

    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500



if __name__ == "__main__":
    app.run(debug=True)
