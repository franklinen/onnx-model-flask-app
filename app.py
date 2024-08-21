from flask import Flask, render_template, request, send_from_directory
import os
from PIL import Image
import numpy as np
import onnxruntime as rt

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def styletransferProcess(image_path, model_path='./mosaic-9.onnx'):
    # Open and preprocess the image
    image = Image.open(image_path)
    image = image.resize((224, 224), Image.LANCZOS)
    
    x = np.array(image).astype('float32')
    x = np.transpose(x, [2, 0, 1])
    x = np.expand_dims(x, axis=0)
    
    # Model inference
    session = rt.InferenceSession(model_path)
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    result = session.run([output_name], {input_name: x})[0][0]
    
    # Postprocess the result
    result = np.clip(result, 0, 255)
    result = result.transpose(1, 2, 0).astype("uint8")
    imag = Image.fromarray(result)
    output_path = os.path.join(APP_ROOT, 'images', 'styled_' + os.path.basename(image_path))
    imag.save(output_path)
    
    return output_path

@app.route('/')
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    if not os.path.isdir(target):
        os.mkdir(target)
    
    file = request.files['file']
    filename = file.filename
    destination = os.path.join(target, filename)
    file.save(destination)
    
    # Apply style transfer
    processed_image_path = styletransferProcess(destination)
    
    return render_template("complete.html", image_name=os.path.basename(processed_image_path))

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=True)
