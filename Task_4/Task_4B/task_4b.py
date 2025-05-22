import os
import numpy as np
import cv2 as cv
import math
from time import sleep
import math
import pandas as pd
import csv


def detect_ArUco_details(image):
    ArUco_details_dict = {}
    ArUco_corners = {}
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # image_gray = cv.GaussianBlur(image_gray, (5, 5), 3)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    bboxs, ids, _ = detector.detectMarkers(image_gray)

    if ids is not None:
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

    return ArUco_details_dict, ArUco_corners


def get_bot_Aruco_centre(arena):
    ArUco_details_dict, ArUco_corners = detect_ArUco_details(arena)
    for i in ArUco_corners:
        # print("i",i)
        # print(ArUco_corners[i])
        cntr_x = 0
        cntr_y = 0
        for j in ArUco_corners[i]:
            cntr_x += j[0]
            cntr_y += j[1]

        if i == BOT_ARUCO_ID:
            return [cntr_x // 4, cntr_y // 4]
        else:
            return [0, 0]


def read_csv(min_dist_Aruco_id):
    lat_lon = []
    live_loc_csv = (
        "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_4/Task_4B/Live_location.csv"
    )
    df = pd.read_csv(
        "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_4/Task_4B/lat_long (1).csv"
    )
    for __, row in df.iterrows():
        if row["id"] == min_dist_Aruco_id:
            lat_lon = [float(row["lat"]), float(row["lon"])]
            with open(live_loc_csv, "w+", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["lat", "lon"])

            with open(live_loc_csv, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(lat_lon)
                csvfile.close()
    return lat_lon


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

cnt = 0
ArUco_details_dict1 = {}
ArUco_corners1 = {}

while 1:
    ret, arena = cap.read()
    # arena = cv.imread("/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_4/frame_screenshot_09.01.2024.png")
    arena = cv.rotate(arena, cv.ROTATE_90_COUNTERCLOCKWISE)
    arena = arena[450:1550, 0:1080]
    arena = cv.resize(arena, (900, 900))
    Aruco_centres = []
    Aruco_ids = []
    ArUco_details_dict, ArUco_corners = detect_ArUco_details(arena)
    if (
        ArUco_details_dict1.keys() == ArUco_details_dict.keys()
        and len(ArUco_details_dict) > 46
    ):
        cnt += 1
    else:
        cnt = 0
        ArUco_details_dict1 = ArUco_details_dict.copy()
        ArUco_corners1 = ArUco_corners.copy()
        print("ArUco_details_dict", ArUco_details_dict1)
        print("ArUco_corners", ArUco_corners1)
        print("Len", len(ArUco_details_dict1))

        print("")

    if cnt >= 1:
        Aruco_ids = list(ArUco_details_dict.keys())
        break

bot_aruco_centre = []
BOT_ARUCO_ID = 100
while 1:
    ret, arena = cap.read()
    # arena = cv.imread("/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_4/frame_screenshot_09.01.2024.png")
    arena = cv.rotate(arena, cv.ROTATE_90_COUNTERCLOCKWISE)
    arena = arena[450:1550, 0:1080]
    arena = cv.resize(arena, (900, 900))

    for i in ArUco_corners:
        # print("i",i)
        # print(ArUco_corners[i])
        cntr_x = 0
        cntr_y = 0
        if i == BOT_ARUCO_ID:
            _, ArUco_corners2 = detect_ArUco_details(arena)
            if i in ArUco_corners2.keys():
                for j in ArUco_corners2[i]:
                    cntr_x += j[0]
                    cntr_y += j[1]
                bot_aruco_centre = [cntr_x // 4, cntr_y // 4]
                Aruco_centres.append(bot_aruco_centre)
        else:
            for j in ArUco_corners[i]:
                cntr_x += j[0]
                cntr_y += j[1]
            Aruco_centres.append([cntr_x // 4, cntr_y // 4])

    print("Aruco_centres", Aruco_centres)
    print("Aruco_ids", Aruco_ids)

    for i in range(0, len(Aruco_ids)):
        if Aruco_ids[i] == BOT_ARUCO_ID:
            arena = cv.circle(
                arena.copy(),
                (Aruco_centres[i][0], Aruco_centres[i][1]),
                10,
                (255, 0, 0),
                -1,
            )
        else:
            arena = cv.circle(
                arena, (Aruco_centres[i][0], Aruco_centres[i][1]), 5, (0, 255, 0), -1
            )
    distance_min = 10000
    min_dist_Aruco = []
    min_dist_Aruco_id = 0
    for i in range(0, len(Aruco_ids)):
        distance = math.sqrt(
            (bot_aruco_centre[0] - Aruco_centres[i][0]) ** 2
            + (bot_aruco_centre[1] - Aruco_centres[i][1]) ** 2
        )
        if distance_min > distance and distance > 2 and Aruco_ids[i] != BOT_ARUCO_ID:
            min_dist_Aruco = Aruco_centres[i]
            distance_min = distance
            min_dist_Aruco_id = Aruco_ids[i]

    print("min_dist_Aruco", distance_min)
    print("min_dist_Aruco_id", min_dist_Aruco_id)
    print("Lat lon", read_csv(min_dist_Aruco_id))
    arena = cv.circle(
        arena, (min_dist_Aruco[0], min_dist_Aruco[1]), 15, (0, 0, 255), -1
    )

    cv.imshow("Arena", arena)

    # sleep(0.5)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
