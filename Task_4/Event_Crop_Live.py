import cv2 as cv
import tkinter as tk
from tkinter import ttk
import numpy as np
import math


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
    image_gray = cv.GaussianBlur(image_gray, (5, 5), 7)
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


def event_identification(arena):
    event_list = []
    arena_org = arena.copy()
    image_cnts = 0
    border_width = 15
    arena_gray = cv.cvtColor(arena, cv.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    arena_gray = cv.dilate(arena_gray, kernel, iterations=2)
    arena_gray = cv.fastNlMeansDenoising(
        src=arena_gray, dst=None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    # kernel = np.array([[-1, -1, -1], [-1, 7, -1], [-1, -1, -1]])
    # arena_gray = cv.filter2D(arena_gray, -1, kernel)
    cv.imshow("gray", arena_gray)
    # arena_gray = cv.GaussianBlur(arena_gray, (3, 3), 1)
    # ret, thresh = cv.threshold(arena_gray, 150, 255, cv.THRESH_BINARY)
    thresh = cv.adaptiveThreshold(
        arena_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, -2
    )
    # thresh = cv.Canny(arena_gray, 140, 240)
    cv.imshow("thresh", thresh)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        if len(approx) == 4:
            x1, y1, w, h = cv.boundingRect(cnt)
            ratio = float(w) / h
            area = w * h
            # print("approx", approx[0])
            if ratio >= 0.9 and ratio <= 1.2 and area > 9600 and area < 12000:
                # if ratio >= 0.9 and ratio <= 1.1 and area > 3000 and area < 15000:
                # cropped_image = arena_org[
                #     y1 + border_width : y1 + 120 - border_width,
                #     x1 + border_width : x1 + 120 - border_width,
                # ]
                # cv.imshow("CI",cropped_image)
                print("Area at ", image_cnts, " is ", area)
                image_cnts += 1
                list = []
                for i in range(0, 4):
                    list.append(approx[i][0])
                event_list.append(list)
                # print("event_list",event_list[0][0])
    return event_list, image_cnts

    # img =cv.imread("/mnt/Storage Drive/Projects/E-YRC/EYRC_2023/Task_4/Arena.png")
    # event_list , image_cnts = event_identification(img)
    # print("event_list",event_list)
    # print("image_cnts",image_cnts)
    # for i in range(0,image_cnts):
    #     image = cv.rectangle(img, (event_list[i][0],event_list[i][1]), (event_list[i][0]+event_list[i][2],event_list[i][1]+event_list[i][3]), (0,255,0), 3)
    # cv.imshow("window_name", image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


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
                print("Could not open /dev/video1 either. Please check your device.")
desired_width = 1920
desired_height = 1080
cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height)
_, f = cap.read()
print("detect_ArUco_details", detect_ArUco_details(f))
if detect_ArUco_details(f) == ({}, {}):
    cap.release()
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


while 1:
    # Capture the video frame
    # by frame
    # frame = cv.imread(
    #     "/mnt/Storage Drive/Pictures/picture_2023-12-23_23-59-36 (copy).jpg"
    # )
    _, frame = cap.read()
    frame = undistort_image(
        frame,
        "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_5/Task_5A/MultiMatrix.npz",
    )
    # frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    cv.imshow("Inp", frame)
    event_list, image_cnts = event_identification(frame)
    # print("event_list", event_list)
    print("image_cnts", image_cnts)
    for i in range(0, image_cnts):
        # frame = cv.rectangle(frame, (event_list[i][0][0],event_list[i][0][1]), (event_list[i][2][0],event_list[i][2][1]), (0,255,0), 3)
        # frame = cv.rectangle(frame, (event_list[i][0][0],event_list[i][0][1]), (event_list[i][0][0]+100,event_list[i][0][1]+100), (0,255,0), 3)

        pts = np.reshape(event_list[i], (-1, 1, 2))
        # print("pts", event_list[i])
        width = 90
        height = 90
        # for i in pts:
        #     frame = cv.circle(
        #         frame, (i[0][0], i[0][1]), radius=0, color=(0, 255, 0), thickness=10
        #     )
        # intersect_pts = list()
        # for j in pts:
        #     lst = list()
        #     lst.append(i[0][0])
        #     lst.append(i[0][1])
        #     intersect_pts.append(i)
        dstPts = [[0, 0], [width, 0], [width, height], [0, height]]
        # print("intersect_pts", pts)
        # Get the transform
        matrix = cv.getPerspectiveTransform(np.float32(pts), np.float32(dstPts))
        # Transform the image
        out = cv.warpPerspective(frame, matrix, (int(width), int(height)))
        # out = cv.flip(out, 1)
        image_path = (
            "/mnt/Storage Drive/Projects/E-YRC/EYRC_2023/Task_4/Task_4A/Saved_Images/Img_"
            + str(i)
            + ".jpg"
        )
        cv.imwrite(image_path, out)
        image = cv.polylines(frame, [pts], 1, (0, 255, 0), 3)

        image = cv.resize(image, (1800, 900))
        cv.imshow("Image", image)
        print("Out", out.shape)

    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # Display the resulting frame

    # cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
