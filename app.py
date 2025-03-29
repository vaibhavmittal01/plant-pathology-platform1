from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model.h5')
class_names = ['Apple scab', 'Apple Black rot', 'Apple Cedar rust', 'Healthy Apple',
               'Cherry Powdery mildew', 'Healthy Cherry',
               'Corn Cercospora leaf spot Gray leaf spot', 'Corn(maize) Common rust',
               'Corn(maize) Northern Leaf Blight', 'Corn(maize) Healthy', 'Grape Black rot',
               'Grape Esca(Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)', 'Grape Healthy',
               'Peach Bacterial spot', 'Peach Healthy', 'Pepper bell Bacterial spot', 'Pepper bell Healthy',
               'Potato Early blight', 'Potato Late blight', 'Potato Healthy', 'Strawberry Leaf scorch',
               'Strawberry Healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
               'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two-spotted spider mite',
               'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato  mosaic virus',
               'Tomato Healthy']

# Disease descriptions
disease_data = {
    "Apple scab": {
        "description": "A fungal disease that causes dark, scabby lesions on apple leaves, fruit, and twigs.",
        "stage": "Early to Mid Growth Stage"
    },
    "Apple Black rot": {
        "description": "A fungal infection leading to black, circular lesions on apples and leaves.",
        "stage": "Mid to Late Growth Stage"
    },
    "Apple Cedar rust": {
        "description": "A fungal disease forming orange spore masses, affecting apple trees and junipers.",
        "stage": "Early Growth Stage"
    },
    "Healthy Apple": {
        "description": "No disease detected. The apple plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Cherry Powdery mildew": {
        "description": "A fungal infection that appears as a white powdery coating on cherry leaves and fruit.",
        "stage": "Flowering and Fruit Development Stage"
    },
    "Healthy Cherry": {
        "description": "No disease detected. The cherry plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Corn Cercospora leaf spot Gray leaf spot": {
        "description": "A fungal disease causing grayish leaf spots, leading to reduced photosynthesis.",
        "stage": "Vegetative Stage"
    },
    "Corn(maize) Common rust": {
        "description": "A fungal disease producing reddish-brown pustules on corn leaves.",
        "stage": "Vegetative to Reproductive Stage"
    },
    "Corn(maize) Northern Leaf Blight": {
        "description": "A fungal infection causing cigar-shaped lesions, leading to yield loss.",
        "stage": "Mid to Late Growth Stage"
    },
    "Corn(maize) Healthy": {
        "description": "No disease detected. The corn plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Grape Black rot": {
        "description": "A fungal disease causing dark spots on leaves and shriveling fruit.",
        "stage": "Fruit Development Stage"
    },
    "Grape Esca(Black Measles)": {
        "description": "A disease that affects grapevines, leading to black streaks and wilting.",
        "stage": "Mid to Late Growth Stage"
    },
    "Grape Leaf blight (Isariopsis Leaf Spot)": {
        "description": "A fungal disease causing irregular leaf spots and defoliation.",
        "stage": "Early to Mid Growth Stage"
    },
    "Grape Healthy": {
        "description": "No disease detected. The grape plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Peach Bacterial spot": {
        "description": "A bacterial infection causing sunken, dark lesions on peach fruits and leaves.",
        "stage": "Early to Mid Growth Stage"
    },
    "Peach Healthy": {
        "description": "No disease detected. The peach plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Pepper bell Bacterial spot": {
        "description": "A bacterial disease causing water-soaked lesions on leaves and fruits.",
        "stage": "Vegetative to Fruit Development Stage"
    },
    "Pepper bell Healthy": {
        "description": "No disease detected. The pepper plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Potato Early blight": {
        "description": "A fungal disease causing dark concentric rings on potato leaves.",
        "stage": "Early to Mid Growth Stage"
    },
    "Potato Late blight": {
        "description": "A severe fungal disease causing large, water-soaked lesions leading to crop loss.",
        "stage": "Late Growth Stage"
    },
    "Potato Healthy": {
        "description": "No disease detected. The potato plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Strawberry Leaf scorch": {
        "description": "A fungal disease causing brown, dried leaf edges, reducing fruit yield.",
        "stage": "Mid to Late Growth Stage"
    },
    "Strawberry Healthy": {
        "description": "No disease detected. The strawberry plant is in a healthy condition.",
        "stage": "All Growth Stages"
    },
    "Tomato Bacterial spot": {
        "description": "A bacterial infection causing water-soaked spots on tomato leaves and fruit.",
        "stage": "Early to Mid Growth Stage"
    },
    "Tomato Early blight": {
        "description": "A fungal disease causing dark, target-like spots on lower tomato leaves.",
        "stage": "Mid to Late Growth Stage"
    },
    "Tomato Late blight": {
        "description": "A severe fungal disease causing large, dark lesions on leaves and stems.",
        "stage": "Late Growth Stage"
    },
    "Tomato Leaf Mold": {
        "description": "A fungal disease causing yellow spots on leaves, leading to mold growth.",
        "stage": "Mid to Late Growth Stage"
    },
    "Tomato Septoria leaf spot": {
        "description": "A fungal infection causing small, circular, brown spots on tomato leaves.",
        "stage": "Early to Mid Growth Stage"
    },
    "Tomato Spider mites Two-spotted spider mite": {
        "description": "An infestation of tiny spider mites causing leaf bronzing and defoliation.",
        "stage": "All Growth Stages"
    },
    "Tomato Target Spot": {
        "description": "A fungal disease causing circular leaf lesions with a dark center.",
        "stage": "Early to Mid Growth Stage"
    },
    "Tomato Tomato Yellow Leaf Curl Virus": {
        "description": "A viral disease causing yellow, curled leaves and stunted growth.",
        "stage": "Early Growth Stage"
    },
    "Tomato mosaic virus": {
        "description": "A viral infection leading to mottled, yellowed tomato leaves.",
        "stage": "Seedling to Vegetative Stage"
    },
    "Tomato Healthy": {
        "description": "No disease detected. The tomato plant is in a healthy condition.",
        "stage": "All Growth Stages"
    }
}


IMAGE_SIZE = 255

# Function to preprocess and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            # Read and preprocess the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))

            # Get prediction
            predicted_class, confidence = predict(img)

            # Retrieve disease details (description + stage)
            disease_info = disease_data.get(predicted_class, {"description": "No description available", "stage": "Unknown"})
            description = disease_info["description"]
            stage = disease_info["stage"]

            return render_template(
                'index.html',
                image_path=filepath,
                predicted_label=predicted_class,
                confidence=confidence,
                description=description,
                stage=stage
            )

    return render_template('index.html', message='Upload an image')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
