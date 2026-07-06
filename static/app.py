from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename



app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



model = load_model("crop_disease_model.h5")


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



disease_info = {

    "Pepper__bell___Bacterial_spot": {
        "description": "A bacterial disease causing dark water-soaked spots on pepper leaves and fruits.",
        "treatment": "Apply copper-based bactericides and remove infected leaves.",
        "prevention": "Use disease-free seeds and avoid overhead watering."
    },

    "Pepper__bell___healthy": {
        "description": "The pepper plant is healthy with no signs of disease.",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper irrigation and balanced fertilization."
    },

    "Potato___Early_blight": {
        "description": "A fungal disease causing concentric brown spots on potato leaves.",
        "treatment": "Apply fungicides like Chlorothalonil or Mancozeb.",
        "prevention": "Practice crop rotation and remove infected plant debris."
    },

    "Potato___healthy": {
        "description": "The potato plant is healthy.",
        "treatment": "No treatment required.",
        "prevention": "Continue proper crop management."
    },

    "Potato___Late_blight": {
        "description": "A severe fungal disease causing dark lesions on leaves and tubers.",
        "treatment": "Use fungicides containing Metalaxyl or Mancozeb.",
        "prevention": "Avoid excessive moisture and use resistant varieties."
    },

    "Tomato___Target_Spot": {
        "description": "A fungal disease producing circular brown spots on leaves.",
        "treatment": "Apply appropriate fungicides.",
        "prevention": "Improve air circulation and avoid leaf wetness."
    },

    "Tomato___Tomato_mosaic_virus": {
        "description": "A viral disease causing mosaic patterns and distorted leaves.",
        "treatment": "No chemical cure available. Remove infected plants.",
        "prevention": "Disinfect tools and use virus-free seeds."
    },

    "Tomato___Tomato_YellowLeaf_Curl_Virus": {
        "description": "A viral disease causing yellow curled leaves and stunted growth.",
        "treatment": "Control whiteflies and remove infected plants.",
        "prevention": "Use resistant varieties and insect-proof nets."
    },

    "Tomato___Bacterial_spot": {
        "description": "A bacterial disease producing dark lesions on leaves and fruits.",
        "treatment": "Apply copper-based bactericides.",
        "prevention": "Use certified seeds and avoid overhead irrigation."
    },

    "Tomato___Early_blight": {
        "description": "A fungal disease causing brown concentric spots on leaves.",
        "treatment": "Use fungicides like Mancozeb or Chlorothalonil.",
        "prevention": "Rotate crops and remove infected debris."
    },

    "Tomato___healthy": {
        "description": "The tomato plant is healthy.",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper nutrition and watering."
    },

    "Tomato___Late_blight": {
        "description": "A destructive fungal disease affecting leaves, stems and fruits.",
        "treatment": "Apply Metalaxyl or Copper-based fungicides.",
        "prevention": "Avoid excessive humidity and remove infected plants."
    },

    "Tomato___Leaf_Mold": {
        "description": "A fungal disease causing yellow spots and mold on leaf undersides.",
        "treatment": "Apply fungicides and improve ventilation.",
        "prevention": "Reduce humidity inside the crop canopy."
    },

    "Tomato___Septoria_leaf_spot": {
        "description": "A fungal disease producing many small circular spots on leaves.",
        "treatment": "Use recommended fungicides.",
        "prevention": "Remove infected leaves and avoid overhead watering."
    },

    "Tomato___Spider_mites_Two_spotted": {
        "description": "A pest infestation causing yellow speckles and webbing on leaves.",
        "treatment": "Apply miticides or neem oil.",
        "prevention": "Maintain adequate humidity and inspect plants regularly."
    }

}


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def format_disease_name(name):
    return (
        name.replace("___", " ")
            .replace("__", " ")
            .replace("_", " ")
            .strip()
    )


def prepare_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "file" not in request.files:
            return "No file uploaded."

        file = request.files["file"]

        if file.filename == "":
            return "Please select an image."

        if not allowed_file(file.filename):
            return "Only JPG, JPEG and PNG images are allowed."

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(filepath)

        try:

            img_array = prepare_image(filepath)

            predictions = model.predict(img_array, verbose=0)

            class_index = np.argmax(predictions[0])

            confidence = float(predictions[0][class_index])

            disease = class_names[class_index]

            display_name = format_disease_name(disease)

            info = disease_info.get(
                disease,
                {
                    "description": "Information not available.",
                    "treatment": "Information not available.",
                    "prevention": "Information not available."
                }
            )

            return render_template(
                "result.html",
                image_file=filename,
                prediction=display_name,
                confidence=round(confidence * 100, 2),
                description=info["description"],
                treatment=info["treatment"],
                prevention=info["prevention"]
            )

        except Exception as e:

            return f"Prediction Error: {e}"

    return render_template("index.html")



if __name__ == "__main__":

    print("\n🚀 Crop Disease Detection running at:")
    print("http://127.0.0.1:5000\n")

    app.run(debug=False)