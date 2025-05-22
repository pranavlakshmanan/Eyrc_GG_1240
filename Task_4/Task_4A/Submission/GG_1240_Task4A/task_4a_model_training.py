import torch
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import tensorflow as tf
from keras.models import Model
import keras.optimizers
import matplotlib.pyplot as plt
from datetime import datetime
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class_labels = [
    "combat",
    "destroyedbuilding",
    "fire",
    "humanitarianaid",
    "militaryvehicles",
]
IMG_SIZE = (90, 90)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess_and_display(image_path):
    img = Image.open(image_path)
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)

    if display:
        numpy_img = input_tensor.permute(1, 2, 0).numpy()
        numpy_img = cv.cvtColor(numpy_img, cv.COLOR_BGR2RGB)
        cv.imshow("Preprocessed Image", numpy_img)
        cv.waitKey(200)
        cv.destroyAllWindows()

    return input_batch


train_data_dir = "/mnt/Storage Drive/Dataset/Training"
test_data_dir = "/mnt/Storage Drive/Dataset/Testing"

train_datagen = ImageDataGenerator(
    rotation_range=3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.02,
    zoom_range=0.01,
    horizontal_flip=1,
    brightness_range=(0.1, 1.9),
    fill_mode="constant",
    cval=255,
    preprocessing_function=preprocess_input,
)
test_datagen = ImageDataGenerator(
    rotation_range=3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.02,
    zoom_range=0.01,
    horizontal_flip=1,
    brightness_range=(0.1, 1.9),
    fill_mode="constant",
    cval=255,
    preprocessing_function=preprocess_input,
)

# Define constants
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 1
LR = 0.01

TRAIN_DIR = train_data_dir
TEST_DIR = test_data_dir

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
)

for layer in base_model.layers:
    layer.trainable = False
model = Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(Dense(NUM_CLASSES, activation="softmax"))
optim = keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=["accuracy"])

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)


metric = "accuracy"
checkpoint = ModelCheckpoint(
    filepath="/mnt/Storage Drive/Dataset/test_model.h5",
    monitor=metric,
    verbose=2,
    save_best_only=True,
    mode="max",
)
callbacks = [checkpoint]

start = datetime.now()

model_history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    steps_per_epoch=1756 // BATCH_SIZE,
    validation_steps=258 // BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)
duration = datetime.now() - start
print("Training completed in time: ", duration)

plt.plot(model_history.history["accuracy"])
plt.plot(model_history.history["val_accuracy"])
plt.title("CNN Model accuracy values")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()