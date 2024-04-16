from flask import Flask, request, render_template
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

loaded_model = tf.keras.models.load_model("/home/amitabh/PycharmProjects/MNIST-DIGIT-RECOGNITION/Results/Models/ANN-Model.h5")

UPLOAD_FOLDER = '/home/amitabh/PycharmProjects/MNIST-DIGIT-RECOGNITION/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_digit(image_file):
    try:

        img = image.load_img(image_file, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = img_array.reshape((1, 784))
        img_array = img_array.astype('float32') / 255.0
        predictions = loaded_model.predict(img_array)
        predicted_label = np.argmax(predictions[0])

        return predicted_label

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', prediction='No selected file')

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_digit(filepath)
            print(prediction)
            return render_template('index.html', prediction=f'Digit: {prediction}', image_file=filename)

    return render_template('index.html', prediction='')


# Run the application
if __name__ == '__main__':
    app.run(debug=True)
