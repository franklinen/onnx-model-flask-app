from PIL import Image
import numpy as np
import onnxruntime as rt
from flask import request, Flask

app = Flask(__name__)

@app.route('/styletransfer', methods=['POST'])
def styletransferProcess():
    # check if the post request has the file part
    if 'imagefile' not in request.files:
        return 'No file part'
        
    imagefile= request.files['imagefile']
    
    if imagefile.filename == '':
        return 'No selected file'
        
    if imagefile:
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        image = Image.open(image_path)
        size_reduction_factor = 1 
        image = image.resize((224,224), Image.LANCZOS)

        # Preprocess image
        x = np.array(image).astype('float32')
        x = np.transpose(x, [2, 0, 1])
        x = np.expand_dims(x, axis=0)

        # Model inference
        session = rt.InferenceSession("./mosaic-9.onnx")
        output_name = session.get_outputs()[0].name
        input_name = session.get_inputs()[0].name
        result = session.run([output_name], {input_name: x})[0][0]

        # postprocess
        result = np.clip(result, 0, 255)
        result = result.transpose(1,2,0).astype("uint8")
        imag = Image.fromarray(result)
        output_path = "./images/styled_" + imagefile.filename
        imag.save(output_path)
    
        return f"Image saved at {output_path}"
        
    return 'Error processing the image'


if __name__ == '__main__':
    app.run(debug=True)
    #styletransferProcess('/directory', 'image', 'model')




