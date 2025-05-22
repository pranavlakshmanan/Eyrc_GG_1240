# Team ID:			[ 1240 ]
# Author List:		[ Bharani, Pranav, Srikar]
# Filename:			task_2a.py
# Functions:		detect_ArUco_details
# 					[ Comma separated list of functions in this file ]

import numpy as np
import cv2 as cv
from cv2 import aruco
import math
import time

# frame_width  = 1920
# frame_height = 1080
frame_width = 1920
frame_height = 1080


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


def mark_ArUco_image(image, ArUco_details_dict, ArUco_corners):
    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv.circle(image, center, 5, (0, 0, 255), -1)

        corner = ArUco_corners[int(ids)]
        cv.circle(image, (int(corner[0][0]), int(corner[0][1])), 5, (50, 50, 50), -1)
        cv.circle(image, (int(corner[1][0]), int(corner[1][1])), 5, (0, 255, 0), -1)
        cv.circle(image, (int(corner[2][0]), int(corner[2][1])), 5, (128, 0, 255), -1)
        cv.circle(image, (int(corner[3][0]), int(corner[3][1])), 5, (25, 255, 255), -1)

        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2)

        cv.line(image, center, (tl_tr_center_x, tl_tr_center_y), (255, 0, 0), 5)
        display_offset = int(
            math.sqrt(
                (tl_tr_center_x - center[0]) ** 2 + (tl_tr_center_y - center[1]) ** 2
            )
        )
        cv.putText(
            image,
            str(ids),
            (center[0] + int(display_offset / 2), center[1]),
            cv.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        angle = details[1]
        # cv.putText(
        #     image,
        #     str(angle),
        #     (center[0] - display_offset, center[1]),
        #     cv.FONT_HERSHEY_COMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2,
        # )
    return image


def undistort_image(img, npz_file):
    data = np.load(npz_file)
    mtx = data["camMatrix"]
    dist = data["distCoef"]
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]

    return dst


if __name__ == "__main__":
    # path directory of images in test_images folder
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
        # desired_width = 1920
        # desired_height = 1080
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height)

    # cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        ret, img = cap.read()
        start_time = time.time()
        img = undistort_image(
            img,
            "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_5/Task_5A/MultiMatrix.npz",
        )
        # kernel = np.ones((3, 3), np.uint8)
        # img = cv.fastNlMeansDenoising(
        #     src=img,
        #     dst=None,
        #     h=10,
        #     templateWindowSize=7,
        #     searchWindowSize=3,
        # )

        # img = cv.resize(img, (800, 800))
        ArUco_details_dict, ArUco_corners = detect_ArUco_details(img)
        # print("Detected details of ArUco: ", ArUco_details_dict)
        print("Size", len(ArUco_details_dict))

        # displaying the marked image
        # img = cv.flip(img, 1)
        img = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners)
        time_taken = time.time() - start_time

        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        img = img[450:1600, 0:1080]
        img = cv.resize(img, (900, 1000))
        # img = cv.resize(img, (1920,1000))
        fps = 1 / time_taken
        print(f"FPS: {fps}")
        # img = cv.flip(img, 1)
        cv.imshow("Marked Image", img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            # After the loop release the cap object
            cap.release()
            # Destroy all the windows
            cv.destroyAllWindows()
            break
