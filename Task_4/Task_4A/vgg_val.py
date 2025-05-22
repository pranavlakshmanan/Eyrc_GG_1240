
import numpy as np
import os
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

class_labels = [
    "combat",
    "destroyedbuilding",
    "fire",
    "humanitarianaid",
    "militaryvehicles",
]
SIZE = (50, 50)

class_labels = [
    "combat",
    "destroyedbuilding",
    "fire",
    "humanitarianaid",
    "militaryvehicles",
]
SIZE = (50, 50)

#loaded_model = load_model("/mnt/Storage Drive/Projects/E-YRC/EYRC_2023/Task_2/Task_2B/vgg16_transfer_learning_model.keras")
#loaded_model = keras.models.load_model("/mnt/Storage Drive/Downloads/vgg16_custom_80.keras")
loaded_model = load_model("/mnt/Storage Drive/Downloads/vgg16_custom.keras")
print(loaded_model.summary())
val_dir = "/mnt/Storage Drive/Dataset/Validate"
# dir_class = ["cbt","db","f","har","mww"]
result_list =[]
for classes in os.listdir(val_dir):
    total_images = 0
    correct = 0
    for images in os.listdir(val_dir + "/" + classes):
        if images.endswith(".jpg"):
            img_path = val_dir + "/" + classes + "/" + images
            # img = cv.resize(cv.imread(image),SIZE)
            # img = np.reshape(img, [1, SIZE[0], SIZE[1], 3])
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            probabilities = loaded_model.predict(img_array)
            predicted_class_index = np.argmax(probabilities)
            pred = class_labels[predicted_class_index]
            if (classes) == pred:
                correct += 1
            # if(classes =="cbt"):
            #     print(pred)
        total_images += 1
    result_list.append(str(str(correct) + "/" + str(total_images)+"    " + str(round((correct / total_images) * 100)) + "%   --> "+classes))
for i in result_list:
    print(i)


