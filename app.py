from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

# -----------------------------
# Initialize Flask
# -----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# Create static folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("crop_disease_model.h5")

# -----------------------------
# Define class names (exactly matching dataset folders)
# -----------------------------
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_YellowLeaf_Curl_Virus",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two_spotted"
]

# -----------------------------
# Helper function to preprocess image
# -----------------------------
def prepare_image(image_path):
    img = load_img(image_path, target_size=(128,128))  # Resize to match training
    img_array = img_to_array(img) / 255.0              # Normalize
    img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension
    return img_array

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess and predict
            img_array = prepare_image(filepath)
            preds = model.predict(img_array)
            class_idx = np.argmax(preds[0])
            confidence = preds[0][class_idx]
            disease = class_names[class_idx]

            return render_template("result.html",
                                   image_file=filename,
                                   prediction=disease,
                                   confidence=f"{confidence*100:.2f}%")
    return render_template("index.html")

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    print("\n🚀 Flask app running at http://127.0.0.1:5000/\n")
    app.run(debug=True)
