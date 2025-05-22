"""
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
"""

# Team ID:			[ 1240 ]
# Author List:		[ Bharani, Pranav, Srikar]
# Filename:			task_2a.py
# Functions:		detect_ArUco_details
# 					[ Comma separated list of functions in this file ]


####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
## You have to implement this task with the five available  ##
## modules for this task                                    ##
##############################################################
import numpy as np
import cv2
from cv2 import aruco
import math

##############################################################

################# ADD UTILITY FUNCTIONS HERE #################


##############################################################


def detect_ArUco_details(image):
    """
    Purpose:
    ---
    This function takes the image as an argument and returns two dictionaries where one
    contains details regarding the center coordinates and orientation of the marker
    and the second dictionary contains values of the 4 corner coordinates of the marker.

    First output: The dictionary `ArUco_details_dict` should should have the id of the marker
    as the key and the value corresponding to that id should be a list containing the following details
    in this order: [[center_x, center_y], angle from the vertical]
    This order should be strictly maintained in the output
    Datatypes:
    1. id - int
    2. center coordinates - int
    3. angle - int, x and y coordinates should be combined as a list for each corner

    Second output: The dictionary `ArUco_corners` should contain the id of the marker as key and the
    corresponding value should be an array of the coordinates of 4 corner points of the markers
    Datatypes:
    1. id - int
    2. corner coordinates - each coordinate value should be float, x and y coordinates should
    be combined as a list for each corner

    Input Arguments:
    ---
    `image` :	[ numpy array ]
            numpy array of image returned by cv2 library
    Returns:
    ---
    `ArUco_details_dict` : { dictionary }
            dictionary containing the details regarding the ArUco marker

    `ArUco_corners` : { dictionary }
            dictionary containing the details regarding the corner coordinates of the ArUco marker

    Example call:
    ---
    ArUco_details_dict, ArUco_corners = detect_ArUco_details(image)

    Example output for 2 markers in an image:
    ---
    * ArUco_details_dict = {9: [[311, 490], 0], 3: [[158, 175], -22]}
    * ArUco_corners =
       {9: array([[211., 389.],
       [412., 389.],
       [412., 592.],
       [211., 592.]], dtype=float32),
       3: array([[109.,  46.],
       [284., 118.],
       [207., 304.],
       [ 33., 232.]], dtype=float32)}
    """
    ArUco_details_dict = {}
    ArUco_corners = {}

    ##############	ADD YOUR CODE HERE	##############
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    bboxs, ids, _ = detector.detectMarkers(image_gray)
    ArUco_corners2 = (np.int0(bboxs)).ravel()
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
    ##################################################

    return ArUco_details_dict, ArUco_corners


######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THE CODE BELOW #########


def mark_ArUco_image(image, ArUco_details_dict, ArUco_corners):
    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.circle(image, center, 5, (0, 0, 255), -1)

        corner = ArUco_corners[int(ids)]
        cv2.circle(image, (int(corner[0][0]), int(corner[0][1])), 5, (50, 50, 50), -1)
        cv2.circle(image, (int(corner[1][0]), int(corner[1][1])), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(corner[2][0]), int(corner[2][1])), 5, (128, 0, 255), -1)
        cv2.circle(image, (int(corner[3][0]), int(corner[3][1])), 5, (25, 255, 255), -1)

        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2)

        cv2.line(image, center, (tl_tr_center_x, tl_tr_center_y), (255, 0, 0), 5)
        display_offset = int(
            math.sqrt(
                (tl_tr_center_x - center[0]) ** 2 + (tl_tr_center_y - center[1]) ** 2
            )
        )
        cv2.putText(
            image,
            str(ids),
            (center[0] + int(display_offset / 2), center[1]),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        angle = details[1]
        # cv2.putText(
        #     image,
        #     str(angle),
        #     (center[0] - display_offset, center[1]),
        #     cv2.FONT_HERSHEY_COMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2,
        # )
    return image


if __name__ == "__main__":
    # path directory of images in test_images folder
    img_dir_path = "public_test_cases/"

    marker = "aruco"

    for file_num in range(0, 2):
        # img_file_path = img_dir_path + marker + '_' + str(file_num) + '.png'
        img_file_path = "/mnt/Storage Drive/Pictures/picture_2023-12-14_13-17-41.jpg"
        # read image using opencv
        img = cv2.imread(img_file_path)
        #img = cv2.resize(img, (800, 800))

        print("\n============================================")
        print("\nFor " + marker + str(file_num) + ".png")

        ArUco_details_dict, ArUco_corners = detect_ArUco_details(img)
        print("Detected details of ArUco: ", ArUco_details_dict)
        print("Size", len(ArUco_details_dict))

        # displaying the marked image
        img = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners)
        # img = cv2.resize(img, (900, 900))
        cv2.imshow("Marked Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
