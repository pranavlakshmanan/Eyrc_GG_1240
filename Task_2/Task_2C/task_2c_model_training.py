import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(50, 50, 3))
batch_size = 12  # (80*5)//32 = 12
num_epochs = 15  # From hit and trial

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(5, activation="softmax")(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Data preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)
train_generator = train_datagen.flow_from_directory(
    "/mnt/Storage Drive/Projects/E-YRC/EYRC_2023/Task 2/Task 2C/Task2_dataset/training/",
    target_size=(50, 50),
    batch_size=batch_size,
    class_mode="categorical",
)

# Train the model
model.fit(train_generator, epochs=num_epochs, steps_per_epoch=400 // 32)

# Save the trained model
model.save("vgg16_transfer_learning_model.keras")
