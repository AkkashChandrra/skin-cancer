from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import datetime
import os

app = Flask(__name__)

# Load the model
model = load_model('weights.h5')

# Ensure the directory for static files exists
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

def preprocess_image(image):
    img = image.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    return img

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return render_template('index.html', pred="No image provided")

    # Get form data
    name = request.form['name']
    mobile = request.form['mobile']
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Process image
    image = request.files['image']
    image_path = os.path.join('static/uploads', image.filename)
    image.save(image_path)
    
    img = preprocess_image(Image.open(image_path))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    # Diagnosis result
    diagnosis = "Malignant" if prediction[0][0] > 0.5 else "Benign"

    # Return the results to the template
    return render_template(
        'index.html',
        name=name,
        mobile=mobile,
        datetime=current_time,
        image_path=image.filename,
        pred=diagnosis
    )

# Intro page route
@app.route('/')
def intro():
    return render_template('intro.html')

# Index page route
@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
