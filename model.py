from PIL import Image
import numpy as np
import onnxruntime as rt


image = Image.open("./too.jpg")
size_reduction_factor = 1 
image = image.resize((224, 224), Image.ANTIALIAS)


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
img = Image.fromarray(result)
print(img)