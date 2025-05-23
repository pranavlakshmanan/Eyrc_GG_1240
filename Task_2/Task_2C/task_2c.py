"""
*****************************************************************************************
*
*        		===============================================*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2C of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
"""
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ 1240 ]
# Author List:		[ Srikar, Barani, Pranav ]
# Filename:			task_2c.py
# Functions:	    [`classify_event(image)` ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
from sys import platform
import numpy as np
import subprocess
import cv2 as cv  # OpenCV Library
import shutil
import ast
import sys
import os

# Additional Imports
"""
You can import your required libraries here
"""
from keras.models import load_model

# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
arena_path = "arena.png"  # Path of generated arena image
event_list = []
detected_list = []

# Declaring Variables
"""
You can delare the necessary variables here
"""

# EVENT NAMES
"""
We have already specified the event names that you should train your model with.
DO NOT CHANGE THE BELOW EVENT NAMES IN ANY CASE
If you have accidently created a different name for the event, you can create another 
function to use the below shared event names wherever your event names are used.
(Remember, the 'classify_event()' should always return the predefined event names)  
"""
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"


# Extracting Events from Arena
def arena_image(
    arena_path,
):  # NOTE: This function has already been done for you, don't make any changes in it.
    """
        Purpose:
        ---
        This function will take the path of the generated image as input and
    read the image specified by the path.

        Input Arguments:
        ---
        `arena_path`: Generated image path i.e. arena_path (declared above)

        Returns:
        ---
        `arena` : [ Numpy Array ]

        Example call:
        ---
        arena = arena_image(arena_path)
    """
    """
    ADD YOUR CODE HERE
    """
    frame = cv.imread(arena_path)
    arena = cv.resize(frame, (700, 700))
    return arena


def event_identification(
    arena,
):  # NOTE: You can tweak this function in case you need to give more inputs
    """
        Purpose:
        ---
        This function will select the events on arena image and extract them as
    separate images.

        Input Arguments:
        ---
        `arena`: Image of arena detected by arena_image()

        Returns:
        ---
        `event_list`,  : [ List ]
                            event_list will store the extracted event images

        Example call:
        ---
        event_list = event_identification(arena)
    """
    """
    ADD YOUR CODE HERE
    """
    arena_org = arena.copy()
    image_cnts = 0
    border_width = 5
    arena_gray = cv.cvtColor(arena, cv.COLOR_BGR2GRAY)
    arena_gray = cv.GaussianBlur(arena_gray, (7, 7), 1)
    ret, thresh = cv.threshold(arena_gray, 250, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        if len(approx) == 4:
            x1, y1, w, h = cv.boundingRect(cnt)
            ratio = float(w) / h
            area = w * h
            if ratio >= 0.9 and ratio <= 1.1 and area > 20:
                cropped_image = arena_org[
                    y1 + border_width : y1 + 50 - border_width,
                    x1 + border_width : x1 + 50 - border_width,
                ]
                event_list.append([cropped_image])
                image_cnts += 1
    return event_list


# Event Detection
def classify_event(image):
    """
        Purpose:
        ---
        This function will load your trained model and classify the event from an image which is
    sent as an input.

        Input Arguments:
        ---
        `image`: Image path sent by input file

        Returns:
        ---
        `event` : [ String ]
                                                  Detected event is returned in the form of a string

        Example call:
        ---
        event = classify_event(image_path)
    """
    """
    ADD YOUR CODE HERE
    """
    event = ""
    model = load_model("/mnt/Storage Drive/Projects/E-YRC/EYRC_2023/Task 2/Task 2B/vgg16_transfer_learning_model.keras")
    IMAGE_SIZE = [50,50]
    img = cv.resize(image[0], (IMAGE_SIZE))
    img = np.reshape(img, [1, IMAGE_SIZE[0],IMAGE_SIZE[1], 3])
    probabilities = model(img)
    predicted_class_index = np.argmax(probabilities)
    class_labels = ['combat', 'destroyedbuilding', 'fire','humanitarianaid', 'militaryvehicles']
    event = class_labels[predicted_class_index]
    return event


# ADDITIONAL FUNCTIONS
"""
Although not required but if there are any additonal functions that you're using, you shall add them here. 
"""


###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################
def classification(event_list):
    for img_index in range(0, 5):
        img = event_list[img_index]
        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    os.remove("arena.png")
    return detected_list


def detected_list_processing(detected_list):
    try:
        detected_events = open("detected_events.txt", "w")
        detected_events.writelines(str(detected_list))
    except Exception as e:
        print("Error: ", e)


def input_function():
    if platform == "win32":
        try:
            subprocess.run("input.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./input")
        except Exception as e:
            print("Error: ", e)


def output_function():
    if platform == "win32":
        try:
            subprocess.run("output.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./output")
        except Exception as e:
            print("Error: ", e)


###################################################################################################
def main():
    ##### Input #####
    input_function()
    #################

    ##### Process #####
    arena = arena_image(arena_path)
    event_list = event_identification(arena)
    detected_list = classification(event_list)
    detected_list_processing(detected_list)
    ###################

    ##### Output #####
    output_function()
    ##################


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        if os.path.exists("arena.png"):
            os.remove("arena.png")
        if os.path.exists("detected_events.txt"):
            os.remove("detected_events.txt")
        sys.exit()
