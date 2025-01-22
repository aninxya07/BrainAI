from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
from xai import limework
import os
import requests
import gdown
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['OUTPUT_FOLDER'] = 'static/output'

# Google Drive model download configuration
MODEL_DRIVE_ID = "1vLZsYv9gdR_dQEtrtPujPNJaYcS10n1n"  # Replace with your Google Drive file ID
MODEL_PATH = 'prediction/brain_tumor_classifier.h5'


def download_model_from_drive(drive_id, destination):
    """Download the model from Google Drive using gdown."""
    url = f"https://drive.google.com/uc?id={drive_id}"
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        gdown.download(url, destination, quiet=False)
        print("Model downloaded successfully.")
    except Exception as e:
        raise Exception(f"Failed to download the model: {e}")


# Check if model exists, if not, download it from Google Drive
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    download_model_from_drive(MODEL_DRIVE_ID, MODEL_PATH)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Confidence threshold for detecting unknown tumors
CONFIDENCE_THRESHOLD = 0.50


# Define allowed file extensions
def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/resnet')
def resnet():
    return render_template('resnet.html')


@app.route('/vgg')
def vgg():
    return render_template('vgg.html')


@app.route('/gans')
def gans():
    return render_template('gans.html')


@app.route('/detection')
def detection():
    return render_template('detection.html')


@app.route('/gan_dataset')
def gan_dataset():
    return render_template('gan_dataset.html')


@app.route('/xai')
def xai():
    return render_template('xai.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']

    # Read the image file and convert it to a format suitable for keras
    img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_probabilities = predictions[0]
    max_probability = np.max(predicted_probabilities)
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class_label = class_labels[predicted_class_index]

    # Check for 'notumor' class or low confidence
    if predicted_class_label == 'notumor':
        result = f'No tumor detected with confidence: {max_probability * 100:.2f}%'
    elif max_probability < CONFIDENCE_THRESHOLD:
        result = f'Tumor detected, but type is not in trained classes. Confidence: {max_probability * 100:.2f}%'
    else:
        result = f'Tumor detected: {predicted_class_label} with confidence: {max_probability * 100:.2f}%'

    return render_template('detection_output.html', result=result)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image_files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('image_files[]')
    uploaded_files = []

    if not files:
        flash('No selected file')
        return redirect(request.url)

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(file_path)
        else:
            flash('Invalid file type')
            return redirect(request.url)

    if uploaded_files:
        try:
            explanations = []
            for file_path in uploaded_files:
                output_files = limework.apply_gradcam_and_lime(file_path, limework.model)

                # Add URLs to static output folder
                output_files = {
                    "original": url_for('static', filename='output/original_image.png'),
                    "heatmap": url_for('static', filename='output/heatmap.png'),
                    "superimposed": url_for('static', filename='output/superimposed_image.png'),
                    "lime": url_for('static', filename='output/lime_explanation_image.png')
                }
                explanations.append(output_files)

            return render_template('xai.html', explanations=explanations)
        except Exception as e:
            flash(f"Error processing the image(s): {str(e)}")
            return redirect(request.url)

    flash('No valid images processed')
    return redirect(request.url)


if __name__ == '__main__':
    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    port = int(os.environ.get('PORT', 10000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)  # Bind to the dynamic port
