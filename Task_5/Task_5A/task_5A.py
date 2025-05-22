import cv2 as cv
import numpy as np
import math
from time import sleep
from keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import pandas as pd
import csv
import socket
import errno

esp32_ip = "192.168.234.184"  # Replace with the actual IP address of your ESP32

# Efficient Aruco Detect


def detect_aruco(frame):
    frame = cv.GaussianBlur(frame, (3, 3), 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
    parameters = cv.aruco.DetectorParameters()
    corners, ids, _ = cv.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    centers = []
    if len(corners):
        for corner in corners:
            center = np.mean(corner, axis=1, dtype=int)
            centers.append(center)
    ids = (np.ravel(ids)).tolist()
    return ids, centers


# Arena Orientation Detect

ANGLE = -1


def correct_Orientation(frame):
    global ANGLE
    if ANGLE == -1:
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)

        def get_ANGLE(marker_corners):
            p1 = marker_corners[0][0]
            p2 = marker_corners[0][1]
            ANGLE = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            return ANGLE

        marker_corners, marker_ids, _ = cv.aruco.detectMarkers(frame, dictionary)

        if marker_ids is not None:
            for i, marker_id in enumerate(marker_ids):
                ANGLE = get_ANGLE(marker_corners[i])
                break
    if ANGLE < 95 and ANGLE > 85:
        frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    elif ANGLE < -85 and ANGLE > -95:
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    elif ANGLE < 185 and ANGLE > 175:
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

    return frame


# Camera Matrix and Distortion Coefficients


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


# Open the Camera


def open_camera(cam_num):
    if cam_num == 0:
        try:
            cap = cv.VideoCapture("/dev/video0")
            if not cap.isOpened():
                raise Exception("Could not open /dev/video0")
        except:
            cap = cv.VideoCapture("/dev/video1")
            if not cap.isOpened():
                print("Could not open /dev/video1 either. Please check your device.")
    elif cam_num == 1:
        try:
            cap = cv.VideoCapture("/dev/video2")
            if not cap.isOpened():
                raise Exception("Could not open /dev/video2")
        except:
            cap = cv.VideoCapture("/dev/video3")
            if not cap.isOpened():
                print("Could not open /dev/video3 either. Please check your device.")

    desired_width = 1920
    desired_height = 1080
    cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height)
    return cap


def arrange_clockwise(pts):
    pts = np.array(pts).reshape((4, 2))
    centroid = np.mean(pts, axis=0)
    ANGLEs = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    pts = pts[np.argsort(ANGLEs)]
    pts = pts.tolist()
    for i in range(len(pts)):
        pts[i] = [pts[i]]
    return pts


# Put Bounding Boxes and Events Name


def put_box_and_name(arena, event_boxes, events):
    arena = cv.rectangle(
        arena,
        event_boxes[0],
        event_boxes[1],
        (0, 255, 0),
        3,
    )
    if events != "empty":
        arena = cv.putText(
            arena,
            events,
            (event_boxes[0][0] + 100, event_boxes[0][1] + 50),
            cv.FONT_HERSHEY_COMPLEX,
            0.9,
            (0, 255, 0),
            3,
        )
    return arena


#   Load the Model


def predict_class(img):
    class_labels = [
        "combat",
        "destroyed_buildings",
        "empty",
        "fire",
        "human_aid_rehabilitation",
        "military_vehicles",
    ]
    IMG_SIZE = (160, 160)
    IMAGE_ORDER = ["A", "B", "C", "D", "E"]
    predicted_class_list = []
    loaded_model = load_model("/mnt/Storage/Dataset/Yolo-cls/model_HC3.h5")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)
    probabilities = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(probabilities)
    pred = class_labels[predicted_class_index]

    return pred


# Moves Generator

map = np.array(
    [
        [0, None, None, None, None, None],
        [None, 1, 5, 6, 11, None],  #
        [None, 21, None, 24, None, 25],  #
        [None, 2, 4, 7, 10, None],  #
        [None, None, 22, 23, None, None],  #
        [None, None, 3, 8, 9, None],
    ]
)


def map_rotate(dir):
    global map
    if dir == 1:  # Clockwise
        map = np.flip(map.transpose(), 0)
    elif dir == -1:  # AntiClockwise
        map = np.flip(map.transpose(), 1)
    # print(map)


def mov_gen(path_str):
    global map
    orientation_vector = 0
    moves = ""
    path_counter = 0  # pointer for shortest_path string
    nodes_list = []  # list for contating all the nodes for each event visit
    nodes = []
    for i in range(0, len(path_str)):
        if path_str[i] == "P":
            node = int(path_str[path_counter + 1])
            if path_str[path_counter + 2] == "0" or path_str[path_counter + 2] == "1":
                node += int(path_str[path_counter + 2]) + 9
            nodes.append(node)
        elif path_str[i] == "A":
            nodes.append(21)
        elif path_str[i] == "B":
            nodes.append(22)
        elif path_str[i] == "C":
            nodes.append(23)
        elif path_str[i] == "D":
            nodes.append(24)
        elif path_str[i] == "E":
            nodes.append(25)
        elif path_str[i] == "s":
            nodes_list = nodes
            nodes = []

        path_counter += 1

    line_follow_time = 0
    forward_time = 0
    backward_time = 0
    right_time = 0
    left_time = 0

    for i in range(0, len(nodes_list) - 1):
        node1 = nodes_list[i]
        node2 = nodes_list[i + 1]
        node0 = 0
        if i > 0:
            node0 = nodes_list[i - 1]
        pos_num1 = np.argwhere(map == node1)
        pos_num2 = np.argwhere(map == node2)
        if node1 in [21, 22, 23, 24, 25]:
            if node0 in [1, 4, 7, 6, 11] and [node0, node1] != [7, 24]:
                moves += str(chr(64 + (node1 - 20))) + "z"
            else:
                moves += str(chr(96 + (node1 - 20))) + "z"
        else:
            moves += "l" + str(line_follow_time)

        if (
            abs(pos_num1[0][0] - pos_num2[0][0]) <= 2
            and abs(pos_num1[0][1] - pos_num2[0][1]) <= 2
            and (pos_num1[0][0] == pos_num2[0][0] or pos_num1[0][1] == pos_num2[0][1])
            and [node1, node2] != [0, 1]
        ):
            if (
                pos_num1[0][0] + 1 == pos_num2[0][0]
                or pos_num1[0][0] + 2 == pos_num2[0][0]
            ):
                # print("Right")
                moves += "r" + str(right_time)
                map_rotate(1)
            elif (
                pos_num1[0][0] == pos_num2[0][0] + 1
                or pos_num1[0][0] == pos_num2[0][0] + 2
            ):
                # print("Left")
                moves += "a" + str(left_time)
                map_rotate(-1)
            elif (
                pos_num1[0][1] + 1 == pos_num2[0][1]
                or pos_num1[0][1] + 2 == pos_num2[0][1]
            ):
                # print("forward")
                moves += "f" + str(forward_time)
            elif (
                pos_num1[0][1] == pos_num2[0][1] + 1
                or pos_num1[0][1] == pos_num2[0][1] + 2
            ):
                # print("U-Turn")
                moves += "u" + str(backward_time)
                map_rotate(1)
                map_rotate(1)
            else:
                print("path not generated")

        elif [node0, node1, node2] == [6, 11, 25]:
            # print("Forward")
            moves += "f" + str(forward_time)
            map_rotate(1)
        elif [node0, node1, node2] == [10, 11, 25]:
            # print("Right")
            moves += "c" + str(forward_time)
            map_rotate(1)
            map_rotate(1)
        elif [node1, node2] == [25, 11]:
            # print("U-turn")
            moves += "u" + str(forward_time)
            map_rotate(1)
        elif [node0, node1, node2] == [2, 1, 0] or [node0, node1, node2] == [21, 1, 0]:
            # print("Left")
            moves += "a" + str(forward_time)
            map_rotate(-1)
        elif [node0, node1, node2] == [5, 1, 0]:
            # print("Forward")
            moves += "f" + str(forward_time)
        elif [node1, node2] == [0, 1]:
            # print("Forward")
            moves = moves[:-2]
        else:
            print("path not generated")
    if "d" in moves:
        # moves = moves[: i] + moves[i + 2 :]
        moves = moves.replace("dzf0", "dz")
    if "E" in moves:
        moves = moves.replace("Ezu0", "E")
    if moves[2] == "f":
        moves = moves[:2] + "a1" + moves[4:]
    moves = moves[:3] + "1" + moves[4:]
    moves += "ss"
    return moves


# Path Planning

import heapq


def path_plan(priority_string):
    graph = {
        "P0": {"P1": 27},
        "P1": {"P0": 27, "P2": 97, "P5": 50, "A": 35},
        "P2": {"P4": 52, "P1": 97, "A": 62},
        "P3": {"P4": 103, "P2": 160, "B": 45},
        "P4": {"P2": 52, "P7": 51, "P5": 100},
        "P5": {"P6": 57, "P1": 50, "P4": 100},
        "P6": {"P7": 100, "P11": 40, "P5": 57, "D": 38},
        "P7": {"P6": 100, "P4": 51, "D": 62, "P8": 103, "P10": 38, "C": 65},
        "P8": {"P7": 103, "C": 41, "P3": 50, "P9": 38},
        "P9": {"P8": 38, "P10": 102},
        "P10": {"P9": 102, "P11": 101},
        "P11": {"P10": 101, "E": 71, "P6": 40},
        "A": {"P1": 35, "P2": 62},
        "B": {"P3": 45, "P4": 60},
        "C": {"P7": 65, "P8": 41},
        "D": {"P7": 62, "P6": 38},
        "E": {"P11": 71},
    }

    start_node = "P0"

    def dijkstras(start, target):
        distances = {node: float("inf") for node in graph}
        distances[start] = 0
        priority_queue = [(0, start)]
        previous_nodes = {}

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            if current_node == target:
                path = []
                while current_node in previous_nodes:
                    path.insert(0, current_node)
                    current_node = previous_nodes[current_node]
                path.insert(0, start)
                return path

            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))
        return []

    node_priority = priority_string
    path = [start_node]
    for i in range(len(node_priority)):
        current_node = path[-1]
        next_node = node_priority[i]
        shortest_path = dijkstras(current_node, next_node)
        path.extend(shortest_path[1:])
    shortest_path_to_P0 = dijkstras(path[-1], start_node)
    path.extend(shortest_path_to_P0[1:])
    path_str = "".join(path) + "s"

    return path_str


# Sort According to Priority


def sort_priority(events_list):
    priority_order = [
        "fire",
        "destroyed_buildings",
        "human_aid_rehabilitation",
        "military_vehicles",
        "combat",
        "empty",
    ]
    events_dict = {}
    events_dict2 = {}

    priority_string = ""
    for i in range(0, len(events_list)):
        if events_list[i] != "empty":
            events_dict[chr(65 + i)] = events_list[i]
            if events_list[i] == "fire":
                events_dict2[chr(65 + i)] = "Fire"
            elif events_list[i] == "destroyed_buildings":
                events_dict2[chr(65 + i)] = "Destroyed buildings"
            elif events_list[i] == "human_aid_rehabilitation":
                events_dict2[chr(65 + i)] = "Humanitarian Aid and rehabilitation"
            elif events_list[i] == "military_vehicles":
                events_dict2[chr(65 + i)] = "Military Vehicles"
            elif events_list[i] == "combat":
                events_dict2[chr(65 + i)] = "Combat"

    for i in priority_order:
        for j in events_dict:
            if i == events_dict[j]:
                priority_string += list(events_dict.keys())[
                    list(events_dict.values()).index(i)
                ]
                break

    return events_dict2, priority_string


# Extract the events


def event_extract(arena):
    extracted_events = []
    ids, centers = detect_aruco(arena)
    roi_images_array = []
    event_corners = []
    event_boxes = []
    events = []
    marked_arena = arena.copy()
    event_arcs = [21, 29, 30, 34, 48]
    for i in event_arcs:
        for j in range(0, len(ids)):
            if ids[j] == i:
                point1 = (centers[j][0][0] - 10, centers[j][0][1] - 40)
                point2 = (
                    centers[j][0][0] - 160,
                    centers[j][0][1] + 110,
                )

                top_left = (min(point1[0], point2[0]), min(point1[1], point2[1]))
                bottom_right = (max(point1[0], point2[0]), max(point1[1], point2[1]))
                roi = arena[
                    top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
                ]
                roi_images_array.append(roi)
                roi_gray = cv.cvtColor(roi.copy(), cv.COLOR_BGR2GRAY)
                kernel = np.ones((5, 5), np.uint8)
                roi_denoise = cv.fastNlMeansDenoising(
                    src=roi_gray,
                    dst=None,
                    h=10,
                    templateWindowSize=7,
                    searchWindowSize=21,
                )
                roi_dialate = cv.dilate(
                    roi_denoise, kernel, iterations=1
                )  # decrease iterations if image is too bright
                thresh = cv.adaptiveThreshold(
                    roi_dialate,
                    255,
                    cv.ADAPTIVE_THRESH_MEAN_C,
                    cv.THRESH_BINARY_INV,
                    11,
                    -2,
                )
                contours, _ = cv.findContours(
                    thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE
                )
                for cnt in contours:
                    approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
                    if len(approx) == 4:
                        x1, y1, w, h = cv.boundingRect(cnt)
                        x2 = x1 + top_left[0]
                        y2 = y1 + top_left[1]
                        ratio = float(w) / h
                        area = w * h
                        if (
                            ratio >= 0.8
                            and ratio <= 1.2
                            and area > 8000
                            and area < 11000
                        ):
                            event_boxes.append([[x2, y2], [x2 + w, y2 + h]])
                            event_corners = []
                            for k in range(0, 4):
                                event_corners.append(approx[k][0])
                            pts = np.reshape(event_corners, (-1, 1, 2))
                            if len(pts) != 0:
                                pts = arrange_clockwise(pts)
                                width = 160
                                height = 160
                                dstPts = [
                                    [0, 0],
                                    [width, 0],
                                    [width, height],
                                    [0, height],
                                ]
                                matrix = cv.getPerspectiveTransform(
                                    np.float32(pts), np.float32(dstPts)
                                )
                                cropped_event = cv.warpPerspective(
                                    roi, matrix, (int(width), int(height))
                                )
                                extracted_events.append(cropped_event)
                                # cv.imshow(str(i), cropped_event)
                                cv.imwrite("Event_" + str(i) + ".jpg", cropped_event)
                                predected_event = predict_class(cropped_event)
                                events.append(predected_event)
                                marked_arena = put_box_and_name(
                                    marked_arena,
                                    [[x2, y2], [x2 + w, y2 + h]],
                                    predected_event,
                                )
                                print(str(i) + " : ", predected_event)
                                break

    return extracted_events, event_boxes, events, marked_arena


# Recording and processing

cap = open_camera(1)

extracted_events = []

while len(extracted_events) < 5:
    print("Scanning the event Boundaries")
    _, frame = cap.read()
    frame = undistort_image(
        frame,
        "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_5/Task_5A/MultiMatrix.npz",
    )
    frame = correct_Orientation(frame)[450:1600, 0:1080]
    # cv.imshow("frame_in", frame)
    extracted_events, event_boxes, events, marked_arena = event_extract(frame)
    print(len(extracted_events))
events_dict, priority_string = sort_priority(events)
print("identified_labels = ", events_dict)
path_str = path_plan(priority_string)
moves_str = mov_gen(path_str)
print("moves_str", moves_str)
marked_arena = cv.resize(marked_arena, (900, 1000))

while 1:
    cv.imshow("marked_arena", marked_arena)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cv.destroyAllWindows()

#  Send the Move Instructions to the bot

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (esp32_ip, 80)
client_socket.connect(server_address)
message = moves_str
while 1:
    try:
        client_socket.sendall(message.encode())
        data = client_socket.recv(1024).decode()
        if data == "Instructions Recieved":
            print("Received:", data)
            break
    except IOError as e:
        if e.errno == errno.EPIPE:
            print("Broken pipe error occurred.")
            break


client_socket.close()

# QGIS Mapping

BOT_ARUCO_ID = 100


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


Const_ids_Check = []
Const_corner_Check = []
Const_centres = []
Const_ids = []
cnt = 0
while 1:
    _, frame = cap.read()
    frame = undistort_image(
        frame,
        "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_5/Task_5A/MultiMatrix.npz",
    )
    frame = correct_Orientation(frame)[450:1600, 0:1080]

    Const_ids, Const_centres = detect_aruco(frame)
    if len(Const_centres) > 48 and (BOT_ARUCO_ID in Const_ids):
        # print("Const_centres", Const_centres)
        # print("Const_ids", Const_ids)
        # print("len(Const_ids)", len(Const_ids))
        break

bot_aruco_centre = []
for i in range(0, len(Const_ids)):
    if Const_ids[i] == BOT_ARUCO_ID:
        bot_aruco_centre = np.ravel(Const_centres[i]).tolist()
        break

while 1:
    _, arena = cap.read()
    arena = undistort_image(
        arena,
        "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_5/Task_5A/MultiMatrix.npz",
    )
    arena = correct_Orientation(arena)[450:1600, 0:1080]

    ids, centres = detect_aruco(arena)

    for i in range(0, len(ids)):
        if ids[i] == BOT_ARUCO_ID:
            bot_aruco_centre = np.ravel(centres[i]).tolist()
            break
    # print("bot_aruco_centre", bot_aruco_centre)
    distance_min = 10000
    min_dist_Aruco = []
    min_dist_Aruco_id = 0
    for i in range(0, len(Const_ids)):
        distance = math.sqrt(
            (bot_aruco_centre[0] - Const_centres[i][0][0]) ** 2
            + (bot_aruco_centre[1] - Const_centres[i][0][1]) ** 2
        )
        if (
            distance_min + 10 > distance
            and distance > 2
            and Const_ids[i] != BOT_ARUCO_ID
        ):
            min_dist_Aruco = Const_centres[i][0]
            distance_min = distance
            min_dist_Aruco_id = Const_ids[i]

    lat_lon = read_csv(min_dist_Aruco_id)
    # arena = cv.circle(
    #     arena, (min_dist_Aruco[0], min_dist_Aruco[1]), 15, (0, 0, 255), -1
    # )
    arena = cv.resize(arena, (900, 1000))
    cv.imshow("Arena", arena)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
