import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import subprocess
import os

# Set paths
base_dir = 'dataset'  # Ensure your dataset folder is structured properly

# Data preprocessing with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Save class labels to labels.txt
labels_file_path = 'labels.txt'
with open(labels_file_path, 'w') as f:
    for label, index in train_generator.class_indices.items():
        f.write(f"{label}\n")
print(f"Labels saved to {labels_file_path}")

# Build the model
model = Sequential()
model.add(Input(shape=(224, 224, 3)))  # Correctly add the Input layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the model in Keras format
model.save('my_model.keras')

# Run the convert_to_tflite.py script to convert the model to TensorFlow Lite
print("Converting the Keras model to TensorFlow Lite...")
subprocess.run(['python', 'convert_to_tflite.py'], check=True)

# Show where the .tflite file is saved
tflite_file_path = os.path.join(os.getcwd(), 'my_model.tflite')
if os.path.exists(tflite_file_path):
    print(f"TensorFlow Lite model has been generated at: {tflite_file_path}")
else:
    print("Failed to generate TensorFlow Lite model.")
