import os
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from flask_ngrok import run_with_ngrok
import time
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np

timestr = time.strftime("%Y%m%d%H%M%S")
UPLOAD_FOLDER = '/content/Image_Detection/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
run_with_ngrok(app) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = ResNet50(weights='imagenet')
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename=filename.replace('.','_'+timestr+'.')
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html')

@app.route('/show/<filename>')
def uploaded_file(filename):
    value1 = model_predict(filename)
    return render_template('template.html', filename=filename,value=value1)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def model_predict(filename):
  img = image.load_img(os.path.join(UPLOAD_FOLDER, filename), target_size=(224, 224))
  x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
  predictions = decode_predictions(model.predict(x), top=1)[0][0] # Get Top-3 Accuracy
  print(predictions)
  _,label,accuracy = predictions
  return str(label)

if __name__ == '__main__':
    app.run()