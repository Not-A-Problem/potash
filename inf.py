import numpy as np
import tensorflow as tf


def predict(image):

    clases = ['Early_blight', 'Late_blight', 'Healthy']
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    print(input_shape)
    print("*"*50, input_details)
    interpreter.set_tensor(input_details[0]['index'], image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    confidence = round((output_data[0][np.argmax(output_data)]) * 100, 3)
    return clases[np.argmax(output_data)], confidence

# print(predict(img))



