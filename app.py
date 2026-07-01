from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


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


def prepare_image(image_path):
    img = load_img(image_path, target_size=(128,128))  
    img_array = img_to_array(img) / 255.0              
    img_array = np.expand_dims(img_array, axis=0)      
    return img_array


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


if __name__ == "__main__":
    print("\n🚀 Flask app running at http://127.0.0.1:5000/\n")
    app.run(debug=False)
