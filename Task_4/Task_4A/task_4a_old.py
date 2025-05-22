"""
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
"""

# Team ID:			[ 1240 ]
# Author List:		[ Bharani, Pranav, Srikar ]
# Filename:			task_4a.py


####################### IMPORT MODULES #######################
from keras.models import load_model
import os
import numpy as np
import cv2 as cv
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input

##############################################################


################# ADD UTILITY FUNCTIONS HERE #################

"""
You are allowed to add any number of functions to this code.
"""
class_labels = [
    "combat",
    "destroyed_buildings",
    "fire",
    "human_aid_rehabilitation",
    "military_vehicles",
]
IMG_SIZE = (90, 90)
IMAGE_ORDER = ["C", "B", "E", "D", "A"]
predicted_class_list = {}


def predict_class(img_path):
    loaded_model = load_model("/mnt/Storage Drive/Dataset/model (1).h5")
    # print(loaded_model.summary())
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    probabilities = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(probabilities)
    pred = class_labels[predicted_class_index]
    return pred


def add_rectangle(frame, event_list, image_cnts, frame_count):
    for i in range(0, image_cnts):
        pts = np.reshape(event_list[i], (-1, 1, 2))
        width = 150
        height = 150
        dstPts = [[0, 0], [width, 0], [width, height], [0, height]]
        matrix = cv.getPerspectiveTransform(np.float32(pts), np.float32(dstPts))
        out = cv.warpPerspective(frame, matrix, (int(width), int(height)))
        out = cv.rotate(out, cv.ROTATE_90_COUNTERCLOCKWISE)
        image = cv.polylines(frame, [pts], 1, (0, 255, 0), 3)
        predicted_class = ""
        if frame_count < 1:
            image_path = (
                "/mnt/Storage Drive/Projects/E-YRC/EYRC_2023/Task_4/Task_4A/Live_Images/Img_"
                + str(i)
                + ".jpg"
            )
            cv.imwrite(image_path, out)
            predicted_class = predict_class(image_path)
            predicted_class_list[IMAGE_ORDER[i]] = predicted_class

    return image, predicted_class_list


def add_event_name(image, predicted_class_list, event_list, image_cnts):
    image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
    image = image[500:1750, 0:1080]
    for i in range(0, image_cnts):
        image = cv.putText(
            image,
            predicted_class_list[IMAGE_ORDER[i]],
            (
                event_list[i][0][1] + 100,
                abs(image.shape[0] - event_list[i][0][0]) + 130,
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=3,
        )
    image = cv.resize(image, (900, 1000))
    return image


def event_square_identification(arena):
    event_list = []
    image_cnts = 0
    arena_gray = cv.cvtColor(arena, cv.COLOR_BGR2GRAY)
    arena_gray = cv.GaussianBlur(arena_gray, (3, 3), 1)
    # ret, thresh = cv.threshold(arena_gray, 150, 256, cv.THRESH_BINARY)
    thresh = cv.adaptiveThreshold(
        arena_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 9, -2
    )
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.1 * cv.arcLength(cnt, True), True)
        if len(approx) == 4:
            x1, y1, w, h = cv.boundingRect(cnt)
            ratio = float(w) / h
            area = w * h
            if ratio >= 0.9 and ratio <= 1.2 and area > 7000 and area < 9800:
                image_cnts += 1
                list = []
                for i in range(0, 4):
                    list.append(approx[i][0])
                event_list.append(list)
    return event_list, image_cnts


##############################################################


def task_4a_return():
    """
    Purpose:
    ---
    Only for returning the final dictionary variable

    Arguments:
    ---
    You are not allowed to define any input arguments for this function. You can
    return the dictionary from a user-defined function and just call the
    function here

    Returns:
    ---
    `identified_labels` : { dictionary }
        dictionary containing the labels of the events detected
    """
    identified_labels = {}

    ##############	ADD YOUR CODE HERE	##############
    try:
        cap = cv.VideoCapture("/dev/video2")
        if not cap.isOpened():
            raise Exception("Could not open /dev/video2")
    except:
        cap = cv.VideoCapture("/dev/video3")
        if not cap.isOpened():
            print("Could not open /dev/video3 either. Please check your device.")
            try:
                cap = cv.VideoCapture("/dev/video0")
                if not cap.isOpened():
                    raise Exception("Could not open /dev/video0")
            except:
                cap = cv.VideoCapture("/dev/video1")
                if not cap.isOpened():
                    print(
                        "Could not open /dev/video1 either. Please check your device."
                    )
    desired_width = 1920
    desired_height = 1080
    cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height)
    frame_count = 0
    ret, frame = cap.read()
    image_cnts = 0
    while image_cnts < 5:
        event_list, image_cnts = event_square_identification(frame)
        print(image_cnts)
    image, predicted_class_list = add_rectangle(
        frame, event_list, image_cnts, frame_count
    )
    image = add_event_name(image, predicted_class_list, event_list, image_cnts)
    identified_labels = {k: v for k, v in sorted(predicted_class_list.items())}
    print("predicted_class_list", predicted_class_list)
    cv.imshow("img", image)
    # identified_labels = predicted_class_list
    # if  & 0xFF == ord("q"):
    #     break
    cv.waitKey(20000)
    cap.release()
    cv.destroyAllWindows()
    ##################################################
    return identified_labels


###############	Main Function	#################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)
