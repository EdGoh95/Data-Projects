#!/usr/bin/env python3
"""
Machine Learning Model Serving Patterns and Best Practices (Packt Publishing)
Chapter 8: Two-Phase Model Serving
"""
import numpy as np
import tensorflow as tf
import pathlib

(MNIST_images_train, MNIST_labels_train), (MNIST_images_test, MNIST_labels_test) = tf.keras.datasets.mnist.load_data()
MNIST_images_train = MNIST_images_train.astype(np.float32)/255.0
MNIST_images_test = MNIST_images_test.astype(np.float32)/255.0

CNN_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = (28, 28)), tf.keras.layers.Reshape(target_shape = (28, 28, 1)),
    tf.keras.layers.Conv2D(filters = 12, kernel_size = (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(10)])

CNN_model.compile(optimizer = 'adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
CNN_model.fit(MNIST_images_train, MNIST_labels_train, epochs = 5,
              validation_data = (MNIST_images_test, MNIST_labels_test))
CNN_model.save('MNIST Model')

def representative_data_generator():
    for input_value in tf.data.Dataset.from_tensor_slices(MNIST_images_train).batch(1).take(100):
        yield [input_value]

model_converter = tf.lite.TFLiteConverter.from_keras_model(CNN_model)
model_converter.representative_dataset = representative_data_generator
model_converter.optimizations = [tf.lite.Optimize.DEFAULT]
model_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
model_converter.inference_input_type = tf.uint8
model_converter.inference_output_type = tf.uint8

tflite_model_quantized = model_converter.convert()
tflite_model_quantized_file = pathlib.Path("MNIST Model/quantized_model.tflite")
tflite_model_quantized_file.write_bytes(tflite_model_quantized)

interpreter = tf.lite.Interpreter(model_path = str(tflite_model_quantized_file))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

MNIST_images_test_indices = range(MNIST_images_test.shape[0])
predictions = np.zeros((len(MNIST_images_test_indices),), dtype = int)
for i, image_test_index in enumerate(MNIST_images_test_indices):
    image_test = MNIST_images_test[image_test_index]
    label_test = MNIST_labels_test[image_test_index]
    # Check if the input type is quantized. If yes, rescale the input data to uint8
    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details['quantization']
        image_test = (image_test/input_scale) + input_zero_point
    image_test = np.expand_dims(image_test, axis = 0).astype(input_details['dtype'])
    interpreter.set_tensor(input_details['index'], image_test)
    # Execute a prediction
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])[0]
    predictions[i] = output.argmax()

accuracy = (np.sum(MNIST_labels_test == predictions)/len(MNIST_images_test)) * 100
print('Accuracy of quantized model: {:.4f} (for {} samples)'.format(accuracy, len(MNIST_images_test)))