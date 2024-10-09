import tensorflow as tf
import os

keras_model_path = 'model.keras'
if not os.path.exists(keras_model_path):
    print(f"Error: {keras_model_path} does not exist. Please train and save the model first.")
    exit(1)

print("Loading the Keras model...")
model = tf.keras.models.load_model(keras_model_path)


print("Converting the Keras model to TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_file_path = 'model.tflite'
with open(tflite_file_path, 'wb') as f:
    f.write(tflite_model)

print(f"TensorFlow Lite model has been saved to: {os.path.abspath(tflite_file_path)}")


labels_file_path = 'labels.txt'
if os.path.exists(labels_file_path):
    print(f"Labels file has been found at: {os.path.abspath(labels_file_path)}")
else:
    print(f"Warning: {labels_file_path} does not exist. Please ensure the labels are generated.")
