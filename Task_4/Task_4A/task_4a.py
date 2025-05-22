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
from torchvision import transforms
import numpy as np
import cv2 as cv
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import math
from time import sleep

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
IMAGE_ORDER = ["A", "B", "C", "D", "E"]
predicted_class_list = []


def preprocess_arena(arena):
    arena = cv.rotate(arena, cv.ROTATE_90_COUNTERCLOCKWISE)
    arena = arena[450:1750, 0:1080]
    return arena


def detect_ArUco_details(image):
    ArUco_details_dict = {}
    ArUco_corners = {}

    ##############	ADD YOUR CODE HERE	##############
    # image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # image_gray = cv.GaussianBlur(image_gray, (3, 3), 1)
    # dictionary = aruco.Dictionary_get(cv.aruco.DICT_4X4_1000)
    # parameters = aruco.DetectorParameters_create()
    # # detector = aruco.detectMarkers(image_gray, dictionary, parameters)
    # bboxs, ids, _ = aruco.detectMarkers(image_gray, dictionary, parameters=parameters)

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # image_gray = cv.GaussianBlur(image_gray, (5, 5), 3)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    bboxs, ids, _ = detector.detectMarkers(image_gray)

    # image = cv.flip(image, 1)

    if ids is not None:
        # bboxs[0][0][3],bboxs[0][0][2] = bboxs[0][0][2],bboxs[0][0][3]
        # for i in range(0, 3):
        #     bboxs[0][0][i][0] = abs(bboxs[0][0][i][0] - frame_width)
        ArUco_corners2 = (np.int0(bboxs)).ravel()
        # print(bboxs[0][0][0][0])
        ArUco_corners = {}
        ArUco_corners_list = []
        count1 = 0
        count2 = 0
        for i in range(0, len(ArUco_corners2), +2):
            temp = []
            temp.append(ArUco_corners2[i])
            temp.append(ArUco_corners2[i + 1])
            count1 += 1
            ArUco_corners_list.append(temp)
            if count1 % 4 == 0 and count1 != 0:
                ArUco_corners[count2] = ArUco_corners_list
                count2 += 1
                ArUco_corners_list = []
        ArUco_ID = ids.ravel()
        centre = []
        for i in range(0, len(ArUco_corners)):
            sumx = 0
            sumy = 0
            for j in range(0, 4):
                sumx += ArUco_corners[i][j][0]
                sumy += ArUco_corners[i][j][1]
            lst = []
            lst.append(int(sumx / 4.0))
            lst.append(int(sumy / 4.0))
            centre.append(lst)

        slope = []
        for i in range(0, len(ArUco_corners)):
            vr = 0
            x1 = int(round((ArUco_corners[i][0][0] + ArUco_corners[i][1][0]) / 2))
            y1 = int(round((ArUco_corners[i][0][1] + ArUco_corners[i][1][1]) / 2))
            x2 = centre[i][0]
            y2 = centre[i][1]
            c = 0
            if x2 == x1 and y2 <= y1:  # 2nd Quadrant
                vr = -180
                c += 1
            elif x2 <= x1 and y2 == y1:  # 1st Quadrant
                vr = -90
                c += 1
            elif x2 >= x1 and y2 == y1:  # 3rd Quadrant
                vr = 90
                c += 1
            elif x2 == x1 and y2 >= y1:  # 4th Quadrant
                vr = 0
                c += 1
            if c == 0:
                x1 = ArUco_corners[i][0][0]
                x2 = ArUco_corners[i][1][0]
                y1 = ArUco_corners[i][0][1]
                y2 = ArUco_corners[i][1][1]
                if x1 != x2:
                    vr = int(round(math.degrees(math.atan2((y1 - y2), (x2 - x1)))))
                vr = int(vr)
                x1 = int(round((ArUco_corners[i][0][0] + ArUco_corners[i][1][0]) / 2))
                y1 = int(round((ArUco_corners[i][0][1] + ArUco_corners[i][1][1]) / 2))
                x2 = centre[i][0]
                y2 = centre[i][1]
            slope.append(vr)
        lst = []
        for i in range(0, len(ArUco_ID)):
            lst.append(centre[i])
            lst.append(int(slope[i]))
            ArUco_details_dict[int(ArUco_ID[i])] = lst
            lst = []
        ArUco_corners_copy = ArUco_corners.copy()
        ArUco_corners = {}
        for i in range(0, len(ArUco_ID)):
            ArUco_corners[int(abs(ArUco_ID[i]))] = ArUco_corners_copy[i]

    return ArUco_details_dict, ArUco_corners


def get_ROI_Images(arena):
    roi_images_array = []
    event_arcs = [21, 29, 30, 34, 48]
    for i in event_arcs:
        ArUco_details_dict, _ = detect_ArUco_details(arena)
        print("ArUco_details_dict", ArUco_details_dict)
        point1 = (ArUco_details_dict[i][0][0], ArUco_details_dict[i][0][1] - 50)
        point2 = (
            ArUco_details_dict[i][0][0] - 200,
            ArUco_details_dict[i][0][1] + 120,
        )
        top_left = (min(point1[0], point2[0]), min(point1[1], point2[1]))
        bottom_right = (max(point1[0], point2[0]), max(point1[1], point2[1]))
        roi = arena[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]

        roi_images = np.zeros_like(arena)
        roi_images[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = roi
        roi_images_array.append(roi_images)
    return roi_images_array


def threshold_sweep(roi_image):
    flag = 0
    block_size = 11
    roi_image_gray = cv.GaussianBlur(roi_image, (3, 3), 1)
    roi_image_gray = cv.cvtColor(roi_image_gray, cv.COLOR_BGR2GRAY)
    # ret, thresh = cv.threshold(roi_image_gray, l_thresh, 256, cv.THRESH_BINARY)
    thresh = cv.adaptiveThreshold(
        roi_image_gray,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY_INV,
        block_size,
        -2,
    )
    event_corners = []
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.1 * cv.arcLength(cnt, True), True)
        # print("approx", approx[0])
        if len(approx) == 4:
            x1, y1, w, h = cv.boundingRect(cnt)
            ratio = float(w) / h
            area = w * h
            if ratio >= 0.9 and ratio <= 1.2 and area > 7000 and area < 9800:
                for i in range(0, 4):
                    event_corners.append(approx[i][0])
                # flag = 1
                # thresh = cv.rectangle(
                #     thresh, approx[1][0], approx[3][0], (0, 255, 0), 3
                # )
    #             break
    # if flag == 1:
    #     break

    return thresh, event_corners


def put_bounding_square(arena, event_corners):
    pts = np.reshape(event_corners, (-1, 1, 2))
    width = 200
    height = 200
    dstPts = [[0, 0], [width, 0], [width, height], [0, height]]
    matrix = cv.getPerspectiveTransform(np.float32(pts), np.float32(dstPts))
    cropped_event = cv.warpPerspective(arena, matrix, (int(width), int(height)))
    arena = cv.polylines(arena, [pts], 1, (0, 255, 0), 3)

    return arena, cropped_event


def predict_class(img_path):
    # loaded_model = load_model("/mnt/Storage/Dataset/Task-4A_Weights.h5")
    loaded_model = load_model("/mnt/Storage/Downloads/model (2).h5")
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    probabilities = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(probabilities)
    pred = class_labels[predicted_class_index]
    return pred


def add_event_name(arena, event, event_corners):
    arena = cv.putText(
        arena,
        event,
        (
            event_corners[0] + 10,
            event_corners[1] + 50,
        ),
        cv.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),
        thickness=3,
    )
    return arena


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
            print("Could not open /dev/video2")
            raise Exception("Could not open /dev/video2")
    except:
        cap = cv.VideoCapture("/dev/video3")
        if not cap.isOpened():
            print("Could not open /dev/video3 either. Please check your device.")
            try:
                cap = cv.VideoCapture("/dev/video0")
                if not cap.isOpened():
                    print("Could not open /dev/video0")
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
    ret, arena = cap.read()
    if detect_ArUco_details(arena) == {}:
        try:
            cap = cv.VideoCapture("/dev/video0")
            if not cap.isOpened():
                print("Could not open /dev/video0")
                raise Exception("Could not open /dev/video0")
        except:
            cap = cv.VideoCapture("/dev/video1")
            if not cap.isOpened():
                print("Could not open /dev/video1 either. Please check your device.")
        desired_width = 1920
        desired_height = 1080
        cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height)
        
    cv.imshow("arena_in",arena)
    arena = preprocess_arena(arena)

    roi_images = get_ROI_Images(arena)
    for i in range(0, 5):
        thresh, event_corners = threshold_sweep(roi_images[i])
        if len(event_corners) != 0:
            arena, cropped_event = put_bounding_square(arena, event_corners)
            if i != 4:
                cropped_event = cv.rotate(cropped_event, cv.ROTATE_90_CLOCKWISE)
            cropped_event = cv.flip(cropped_event, 1)
            cropped_image_path = (
                "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_4/Task_4A/Live_Images/Img_"
                + str(i)
                + ".jpg"
            )
            cv.imwrite(cropped_image_path, cropped_event)
            event = predict_class(cropped_image_path)
            if i != 4:
                arena = add_event_name(arena, event, event_corners[3])
            else:
                arena = add_event_name(arena, event, event_corners[0])
            identified_labels[IMAGE_ORDER[i]] = event

    while 1:
        arena = cv.resize(arena, (900, 1000))
        cv.imshow("Arena", arena)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
    ##################################################
    return identified_labels


###############	Main Function	#################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)
