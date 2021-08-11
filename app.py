from flask import Flask, render_template, request, send_file

from PIL import Image
import numpy as np
import onnxruntime as rt


#mosaic-9 = wget https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx 

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/mosaic', methods=['POST'])
def mosaic():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = Image.open(image_path)
    size_reduction_factor = 1 
    image = image.resize((224,224), Image.ANTIALIAS)

    # Preprocess image
    x = np.array(image).astype('float32')
    x = np.transpose(x, [2, 0, 1])
    x = np.expand_dims(x, axis=0)

    session = rt.InferenceSession("./mosaic-9.onnx")
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    result = session.run([output_name], {input_name: x})[0][0]

    # postprocess
    result = np.clip(result, 0, 255)
    result = result.transpose(1,2,0).astype("uint8")
    imag = Image.fromarray(result)
    print(imag)


    return render_template('index.html', tr_image=imag)



@app.route('/get_image')
def get_image():
    if request.args.get('type') == '1':
       filename = 'ok.gif'
    else:
       filename = 'error.gif'
    return send_file(filename, mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True)