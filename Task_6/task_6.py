"""
* Team Id : GG_1240
* Author List : Srikar Bharadwaj, Pranav Lakshmanan
* Filename: task_6.py
* Theme: GeoGuide
* Functions: detect_aruco , correct_Orientation, undistort_image, open_camera, arrange_clockwise, put_box_and_name,
             predict_class, map_rotate, mov_gen, path_plan, sort_priority, event_extract, get_nearest_lat_lon
* Global Variables: ANGLE, BOT_ARUCO_ID, map
"""

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
import heapq


esp32_ip = "192.168.59.184"  # Replace with the actual IP address of ESP32


# Aruco Detect

"""
Function Name: detect_aruco
Input: frame (numpy array): The input image frame on which Aruco markers are to be detected.
Output: ids (list): List of ids of the detected Aruco markers.
        centers (list): List of centers of the detected Aruco markers.
Logic: This function detects Aruco markers in a given image frame. It first applies Gaussian blur to the frame and then converts it to grayscale. 
       It then uses the predefined dictionary for Aruco markers to detect them in the grayscale image.
       If any markers are detected, it calculates their centers and returns the ids and centers.
Example Call: ids, centers = detect_aruco(frame)
"""


def detect_aruco(frame):
    # Apply Gaussian blur to the frame
    frame = cv.GaussianBlur(frame, (3, 3), 1)

    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Get the predefined dictionary for Aruco markers
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)

    # Get the default parameters for Aruco marker detection
    parameters = cv.aruco.DetectorParameters()

    # Detect Aruco markers in the grayscale image
    corners, ids, _ = cv.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    # Initialize an empty list to store the centers of the detected markers
    centers = []

    # If any markers are detected
    if len(corners):
        # For each detected marker
        for corner in corners:
            # Calculate the center of the marker
            center = np.mean(corner, axis=1, dtype=int)

            # Add the center to the list of centers
            centers.append(center)

    # Flatten the list of ids
    ids = (np.ravel(ids)).tolist()

    # Return the ids and centers of the detected markers
    return ids, centers


# Arena Orientation Detect

"""
Function Name: correct_Orientation
Input: frame (numpy array): The input image frame on which the orientation is to be corrected.
Output: frame (numpy array): The corrected image frame.
Logic: This function corrects the orientation of a given image frame. It first detects Aruco markers in the frame.
If any markers are detected, it calculates the angle of rotation required to correct the orientation.
Depending on the calculated angle, it rotates the frame accordingly.
Example Call: corrected_frame = correct_Orientation(frame)
"""

# Global variable to store the angle of rotation
ANGLE = -1


def correct_Orientation(frame):
    # Access the global variable ANGLE
    global ANGLE

    # If ANGLE is not yet calculated
    if ANGLE == -1:
        # Get the predefined dictionary for Aruco markers
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)

        # Function to calculate the angle of rotation
        def get_ANGLE(marker_corners):
            # Get the first two points of the marker
            p1 = marker_corners[0][0]
            p2 = marker_corners[0][1]

            # Calculate the angle of rotation
            ANGLE = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

            # Return the calculated angle
            return ANGLE

        # Detect Aruco markers in the frame
        marker_corners, marker_ids, _ = cv.aruco.detectMarkers(frame, dictionary)

        # If any markers are detected
        if marker_ids is not None:
            # For each detected marker
            for i, marker_id in enumerate(marker_ids):
                # Calculate the angle of rotation
                ANGLE = get_ANGLE(marker_corners[i])

                # Break after calculating the angle for the first marker
                break

    # If the calculated angle is between 85 and 95 degrees
    if ANGLE < 95 and ANGLE > 85:
        # Rotate the frame 90 degrees counterclockwise
        frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    # If the calculated angle is between -85 and -95 degrees
    elif ANGLE < -85 and ANGLE > -95:
        # Rotate the frame 90 degrees clockwise
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    # If the calculated angle is between 175 and 185 degrees
    elif ANGLE < 185 and ANGLE > 175:
        # Rotate the frame 180 degrees
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

    # Return the corrected frame
    return frame


# Camera Matrix and Distortion Coefficients

"""
Function Name: undistort_image
Input: img (numpy array): The input image to be undistorted.
       npz_file (str): The file path of the .npz file containing the camera matrix and distortion coefficients.
Output: dst (numpy array): The undistorted image.
Logic: This function undistorts a given image using the camera matrix and distortion coefficients stored in a .npz file.
        It first loads the camera matrix and distortion coefficients from the .npz file.
        It then calculates the optimal new camera matrix and the region of interest (roi) for the given image size.
        It then undistorts the image using the original camera matrix, distortion coefficients, and the optimal new camera matrix. 
        Finally, it crops the undistorted image to the region of interest and returns it.
Example Call: undistorted_img = undistort_image(img, 'calibration.npz')
"""


def undistort_image(img, npz_file):
    # Load the camera matrix and distortion coefficients from the .npz file
    data = np.load(npz_file)
    mtx = data["camMatrix"]
    dist = data["distCoef"]

    # Get the shape of the image
    h, w = img.shape[:2]

    # Calculate the optimal new camera matrix and the region of interest
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort the image
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # Get the region of interest
    x, y, w, h = roi

    # Crop the undistorted image to the region of interest
    dst = dst[y : y + h, x : x + w]

    # Return the undistorted image
    return dst


# Open the Camera

"""
Function Name: open_camera
Input: cam_num (int): The camera number to be opened. It can be either 0 or 1.
Output: cap (cv.VideoCapture): The opened camera capture object.
Logic: This function opens a camera based on the given camera number.
       If the camera number is 0, it tries to open /dev/video0. 
       If it fails, it tries to open /dev/video1. If the camera number is 1, it tries to open /dev/video2. 
       If it fails, it tries to open /dev/video3. After opening the camera, it sets the frame width and height to 1920 and 1080 respectively.
Example Call: cap = open_camera(0)
"""


def open_camera(cam_num):
    # If the camera number is 0
    if cam_num == 0:
        try:
            # Try to open /dev/video0
            cap = cv.VideoCapture("/dev/video0")

            # If the camera is not opened, raise an exception
            if not cap.isOpened():
                raise Exception("Could not open /dev/video0")
        except:
            # If an exception is raised, try to open /dev/video1
            cap = cv.VideoCapture("/dev/video1")

            # If the camera is not opened, print an error message
            if not cap.isOpened():
                print("Could not open /dev/video1 either. Please check your device.")
    # If the camera number is 1
    elif cam_num == 1:
        try:
            # Try to open /dev/video2
            cap = cv.VideoCapture("/dev/video2")

            # If the camera is not opened, raise an exception
            if not cap.isOpened():
                raise Exception("Could not open /dev/video2")
        except:
            # If an exception is raised, try to open /dev/video3
            cap = cv.VideoCapture("/dev/video3")

            # If the camera is not opened, print an error message
            if not cap.isOpened():
                print("Could not open /dev/video3 either. Please check your device.")

    # Set the desired frame width and height
    desired_width = 1920
    desired_height = 1080
    cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height)

    # Return the opened camera capture object
    return cap


"""
Function Name: arrange_clockwise
Input: pts - A list of points where each point is a list of two coordinates [x, y]
Output: Returns a list of points arranged in clockwise order. Each point is a list of two coordinates [x, y]
Logic: This function takes a list of points and arranges them in clockwise order.
       It first calculates the centroid of the points. Then it calculates the angles of each point with respect to the centroid.
        The points are then sorted based on these angles. It is done to make sure that the cropped event images are in the correct orientation,
        and are not flipped.
Example Call: arrange_clockwise([[1, 2], [2, 3], [3, 4], [4, 5]])
"""


def arrange_clockwise(pts):
    # Convert the list of points to a numpy array and reshape it to a 4x2 matrix
    pts = np.array(pts).reshape((4, 2))

    # Calculate the centroid of the points
    centroid = np.mean(pts, axis=0)

    # Calculate the angles of each point with respect to the centroid
    ANGLEs = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])

    # Sort the points based on the calculated angles
    pts = pts[np.argsort(ANGLEs)]

    # Convert the numpy array back to a list
    pts = pts.tolist()

    # Wrap each point in a list
    for i in range(len(pts)):
        pts[i] = [pts[i]]

    # Return the list of points arranged in clockwise order
    return pts


# Put Bounding Boxes and Events Name

"""
Function Name: put_box_and_name
Input: 
    - arena: The image where the box and name will be put.
    - event_boxes: A list of two points representing the top left and bottom right corners of the box.
    - events: The name to be put inside the box. If it is "empty", no name will be put.
Output: Returns the image with the box and name put on it.
Logic: This function first draws a rectangle on the image using the points in event_boxes.
       Then, if events is not "empty", it puts the name next to the box.
Example Call: put_box_and_name(image, [[10, 10], [100, 100]], "fire")
"""


def put_box_and_name(arena, event_boxes, events):
    # Draw a rectangle on the image using the points in event_boxes
    arena = cv.rectangle(
        arena,
        event_boxes[0],
        event_boxes[1],
        (0, 255, 0),  # The color of the rectangle is green
        3,  # The thickness of the rectangle is 3
    )

    # If events is not "empty", put the name inside the box
    if events != "empty":
        if events == "human_aid_rehabilitation":
            events = "humanitarian_aid"  # since model is trained with the class name "human_aid_rehabilitation"
        arena = cv.putText(
            arena,
            events,  # The text to be put
            (
                event_boxes[0][0] + 100,
                event_boxes[0][1] + 50,
            ),  # The position of the text
            cv.FONT_HERSHEY_COMPLEX,  # The font of the text
            0.9,  # The scale of the text
            (0, 255, 0),  # The color of the text is green
            2,  # The thickness of the text is 2
        )

    # Return the image with the box and name put on it
    return arena


#   Load the Model

"""
Function Name: predict_class
Input: img - An image in the form of a numpy array.
Output: Returns the predicted class of the image as a string.
Logic: This function takes an image as input and predicts its class using a pre-trained model.
       The image is first converted to RGB format. Then, it is preprocessed and fed into the model for prediction. 
       The function returns the class label corresponding to the highest probability.
Example Call: predict_class(image)
"""


def predict_class(img):
    # Define the class labels
    class_labels = [
        "combat",
        "destroyed_buildings",
        "empty",
        "fire",
        "human_aid_rehabilitation",
        "military_vehicles",
    ]

    # Load the pre-trained model
    loaded_model = load_model(
        "/mnt/Storage/Dataset/Final_Dataset/modelv10.h5"
    )  # https://drive.google.com/file/d/1IXRfSyjA_E-Xl64oUktysgN0OW4Srfu7/view?usp=sharing
    # Convert the image to RGB format
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Preprocess the image
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the class of the image
    probabilities = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(probabilities)

    # Get the class label corresponding to the highest probability
    pred = class_labels[predicted_class_index]

    # Return the predicted class
    return pred


# Moves Generator

# The map is a 2D numpy array representing the map structure, refer EYRC-Nodes.pdf
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

"""
Function Name: map_rotate
Input: dir - An integer representing the direction of rotation. 1 for clockwise and -1 for anti-clockwise.
Output: None. The function modifies the global variable 'map' in place.
Logic: This function rotates the global variable 'map' in the specified direction.
       If the direction is 1, it rotates the map clockwise. If the direction is -1, it rotates the map anti-clockwise.
        The rotation is achieved by transposing the map and then flipping it along the appropriate axis.
Example Call: map_rotate(1)  # Rotates the map clockwise
"""


def map_rotate(dir):
    global map  # The map to be rotated
    if dir == 1:  # Clockwise
        # Transpose the map and then flip it along the 0th axis
        map = np.flip(map.transpose(), 0)
    elif dir == -1:  # AntiClockwise
        # Transpose the map and then flip it along the 1st axis
        map = np.flip(map.transpose(), 1)


"""
 Function Name: mov_gen
* Input: path_str (a string representing the path)
* Output: None. The function modifies the global variable 'moves'.
* Logic: This function generates a sequence of moves based on the input path string. 
         It iterates over the path string and for each character, it determines the corresponding node and appends it to the nodes list.
         It then generates the moves based on the relative positions of the nodes.
* Example Call: mov_gen("P0A")
"""


def mov_gen(path_str):
    global map  # global map variable
    orientation_vector = 0  # orientation vector
    moves = ""  # string to store the moves
    path_counter = 0  # pointer for shortest_path string
    nodes_list = []  # list for containing all the nodes for each event visit
    nodes = []  # list to store the nodes

    # iterate over the path string
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

    # iterate over the nodes list
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
                moves += (
                    str(chr(64 + (node1 - 20))) + "z"
                )  # z for buzzer ,  capital event name means it is approaching the event from left
            else:
                moves += (
                    str(chr(96 + (node1 - 20))) + "z"
                )  # bot should approach the event from right.
        else:
            moves += "l" + str(line_follow_time)

        # generate moves based on the relative positions of the nodes
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
                moves += "r" + str(right_time)  # rotate right
                map_rotate(1)
            elif (
                pos_num1[0][0] == pos_num2[0][0] + 1
                or pos_num1[0][0] == pos_num2[0][0] + 2
            ):
                moves += "a" + str(left_time)  # rotate left
                map_rotate(-1)
            elif (
                pos_num1[0][1] + 1 == pos_num2[0][1]
                or pos_num1[0][1] + 2 == pos_num2[0][1]
            ):
                moves += "f" + str(forward_time)  # move forward
            elif (
                pos_num1[0][1] == pos_num2[0][1] + 1
                or pos_num1[0][1] == pos_num2[0][1] + 2
            ):
                if moves[-2:-1] in ["A", "B", "C", "D", "E"]:
                    moves += "u" + str(1)  # take U-Turn
                else:
                    moves += "u" + str(0)
                map_rotate(1)
                map_rotate(1)
            else:
                print("path not generated")

        elif [node0, node1, node2] == [6, 11, 25]:
            moves += "f" + str(forward_time)
            map_rotate(1)
        elif [node0, node1, node2] == [10, 11, 25]:
            moves += "c" + str(forward_time)
            map_rotate(1)
            map_rotate(1)
        elif [node1, node2] == [25, 11]:
            moves += "u" + str(forward_time)
            map_rotate(1)
        elif [node0, node1, node2] == [2, 1, 0] or [node0, node1, node2] == [21, 1, 0]:
            moves += "a" + str(forward_time)
            map_rotate(-1)
        elif [node0, node1, node2] == [5, 1, 0]:
            moves += "f" + str(forward_time)
        elif [node1, node2] == [0, 1]:
            moves = moves[:-2]
        else:
            print("path not generated")
    if "d" in moves:
        moves = moves.replace("dzf0", "dz")
    if "E" in moves:
        moves = moves.replace("Ezu0", "E")
    if "Czu1" in moves:
        moves = moves.replace("Czu1", "Czu3")
    if "Az" in moves:
        moves = moves.replace("Az", "Aza1")
    if moves[2] == "f":
        moves = moves[:2] + "a1" + moves[4:]
    moves = moves[:3] + "1" + moves[4:]
    moves += "ss"
    if "a0ss" in moves:
        moves = moves.replace("a0ss", "a2ss")
    return moves


# Path Planning

"""

# Function Name: path_plan
# Input: priority_string (str): A string representing the priority of nodes to visit
# Output: path_str (str): A string representing the shortest path to visit all nodes in the priority_string and return to the start node
# Logic: This function uses Dijkstra's algorithm to find the shortest path between nodes in a graph. 
It iterates over the priority_string to visit each node in order and then returns to the start node.
# Example Call: path_plan("ABCDE")
"""


def path_plan(priority_string):
    # The graph represents the nodes and their connections with weights
    graph = {
        "P0": {"P1": 27},
        "P1": {"P0": 27, "P2": 97, "P5": 50, "A": 35},
        "P2": {"P4": 52, "P1": 97, "A": 62, "P3": 160},
        "P3": {"P4": 103, "P2": 160, "B": 45},
        "P4": {"P2": 52, "P7": 51, "P5": 100, "B": 60},
        "P5": {"P6": 57, "P1": 50, "P4": 100},
        "P6": {"P7": 100, "P11": 40, "P5": 57, "D": 38},
        "P7": {"P6": 100, "P4": 51, "D": 62, "P8": 103, "P10": 38, "C": 65},
        "P8": {"P7": 103, "C": 41, "P3": 50, "P9": 38},
        "P9": {"P8": 38, "P10": 102},
        "P10": {"P9": 102},
        "P11": {"P10": 101, "E": 71, "P6": 40},
        "A": {"P1": 35, "P2": 62},
        "B": {"P3": 45, "P4": 60},
        "C": {"P7": 65, "P8": 41},
        "D": {"P7": 62, "P6": 38},
        "E": {"P11": 71},
    }

    # The start node is always "P0"
    start_node = "P0"

    """   
    # Function Name: dijkstras
    # Input: start (str): The start node, target (str): The target node
    # Output: path (list): The shortest path from start to target
    # Logic: This function implements Dijkstra's algorithm to find the shortest path between two nodes in a graph.
    """

    def dijkstras(start, target):
        # Initialize the distances to all nodes as infinity
        distances = {node: float("inf") for node in graph}
        # The distance to the start node is 0
        distances[start] = 0
        # Initialize the priority queue with the start node
        priority_queue = [(0, start)]
        # This dictionary will store the previous node for each node
        previous_nodes = {}

        # While there are nodes in the priority queue
        while priority_queue:
            # Pop the node with the smallest distance
            current_distance, current_node = heapq.heappop(priority_queue)
            # If the current node is the target, we have found the shortest path
            if current_node == target:
                path = []
                # Build the path by following the previous nodes
                while current_node in previous_nodes:
                    path.insert(0, current_node)
                    current_node = previous_nodes[current_node]
                path.insert(0, start)
                return path

            # For each neighbor of the current node
            for neighbor, weight in graph[current_node].items():
                # Calculate the distance to the neighbor through the current node
                distance = current_distance + weight
                # If this distance is smaller than the current distance to the neighbor
                if distance < distances[neighbor]:
                    # Update the distance to the neighbor
                    distances[neighbor] = distance
                    # Update the previous node of the neighbor
                    previous_nodes[neighbor] = current_node
                    # Add the neighbor to the priority queue
                    heapq.heappush(priority_queue, (distance, neighbor))
        # If there is no path to the target, return an empty list
        return []

    # The node_priority is the priority_string
    node_priority = priority_string
    # Initialize the path with the start_node
    path = [start_node]
    # For each node in the node_priority
    for i in range(len(node_priority)):
        # The current node is the last node in the path
        current_node = path[-1]
        # The next node is the next node in the node_priority
        next_node = node_priority[i]
        # Find the shortest path from the current node to the next node
        shortest_path = dijkstras(current_node, next_node)
        # Extend the path with the shortest path (excluding the current node)
        path.extend(shortest_path[1:])
    # Find the shortest path from the last node in the path to the start node
    shortest_path_to_P0 = dijkstras(path[-1], start_node)
    # Extend the path with the shortest path to the start node (excluding the last node in the path)
    path.extend(shortest_path_to_P0[1:])
    # Convert the path to a string and append "s" to stop the bot
    path_str = "".join(path) + "s"

    # Return the path string
    return path_str


# Sort According to Priority

"""
Function Name: sort_priority
* Input: events_list (a list of events)
* Output: Returns two dictionaries - events_dict2 (a dictionary mapping event codes to event names) and priority_string (a string representing the priority order of events)
* Logic: This function sorts a list of events according to a predefined priority order. It iterates over the events_list and for each event, 
         it assigns a code and maps it to the event name in events_dict2.
         It then generates the priority_string by iterating over the priority_order and appending the codes of the events in the order they appear in the priority_order.
* Example Call: sort_priority(["fire", "destroyed_buildings", "empty"])
"""


def sort_priority(events_list):
    priority_order = [  # predefined priority order
        "fire",
        "destroyed_buildings",
        "human_aid_rehabilitation",
        "military_vehicles",
        "combat",
        "empty",
    ]
    events_dict = {}  # dictionary to store event codes
    events_dict2 = {}  # dictionary to map event codes to event names

    priority_string = ""  # string to store the priority order of events
    for i in range(0, len(events_list)):  # iterate over the events_list
        if events_list[i] != "empty":  # if the event is not 'empty'
            events_dict[chr(65 + i)] = events_list[i]  # assign a code to the event
            # map the event code to the event name
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

    for i in priority_order:  # iterate over the priority_order
        for j in events_dict:  # iterate over the events_dict
            if i == events_dict[j]:  # if the event matches the current priority
                priority_string += str(
                    j
                )  # append the event code to the priority_string

    return events_dict2, priority_string  # return the dictionaries


# Extract the events

"""
* Function Name: event_extract
* Input: arena (numpy array): The image from which events are to be extracted.
* Output: None. This function modifies the 'arena' in-place.
* Logic: This function detects ArUco markers in the given image, identifies those that correspond to events,
         and extracts these events. It applies several image processing techniques such as denoising, dilation,
         and adaptive thresholding to identify the contours of the events. It then uses perspective transformation
         to crop the events from the image.
* Example Call: event_extract(arena_image)
"""


def event_extract(arena):
    # List to store the extracted events
    extracted_events = []
    # Detect ArUco markers in the image
    ids, centers = detect_aruco(arena)
    # Lists to store ROI images, event corners, event bounding boxes, and events
    roi_images_array = []
    event_corners = []
    event_boxes = []
    events = []
    # Copy of the original arena image
    marked_arena = arena.copy()
    # List of ArUco IDs corresponding to events
    event_arcs = [21, 29, 30, 34, 48]
    # Loop over each event ArUco ID
    for i in event_arcs:
        # Loop over each detected ArUco marker
        for j in range(0, len(ids)):
            # If the ID of the current marker matches the current event ID
            if ids[j] == i:
                # Define two points for extracting the ROI around the event
                point1 = (centers[j][0][0] - 10, centers[j][0][1] - 40)
                point2 = (
                    centers[j][0][0] - 160,
                    centers[j][0][1] + 110,
                )
                # Determine the top-left and bottom-right points of the ROI
                top_left = (min(point1[0], point2[0]), min(point1[1], point2[1]))
                bottom_right = (max(point1[0], point2[0]), max(point1[1], point2[1]))
                # Extract the ROI from the arena image
                roi = arena[
                    top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
                ]
                # Append the ROI image to the list
                roi_images_array.append(roi)
                # Convert the ROI image to grayscale
                roi_gray = cv.cvtColor(roi.copy(), cv.COLOR_BGR2GRAY)
                # Define a kernel for dilation
                kernel = np.ones((5, 5), np.uint8)
                # Denoise the grayscale ROI image
                roi_denoise = cv.fastNlMeansDenoising(
                    src=roi_gray,
                    dst=None,
                    h=10,
                    templateWindowSize=7,
                    searchWindowSize=21,
                )
                # Dilate the denoised image
                roi_dialate = cv.dilate(
                    roi_denoise, kernel, iterations=1
                )  # decrease iterations if image is too bright

                # Apply adaptive thresholding to the dilated image
                thresh = cv.adaptiveThreshold(
                    roi_dialate,
                    255,
                    cv.ADAPTIVE_THRESH_MEAN_C,
                    cv.THRESH_BINARY_INV,
                    11,
                    -2,
                )
                # Find contours in the thresholded image
                contours, _ = cv.findContours(
                    thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE
                )
                # Loop over each contour
                for cnt in contours:
                    # Approximate the contour
                    approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
                    # If the contour has 4 points (i.e., it is a rectangle)
                    if len(approx) == 4:
                        # Get the bounding rectangle of the contour
                        x1, y1, w, h = cv.boundingRect(cnt)
                        # Adjust the rectangle coordinates relative to the original image
                        x2 = x1 + top_left[0]
                        y2 = y1 + top_left[1]
                        # Calculate the aspect ratio and area of the rectangle
                        ratio = float(w) / h
                        area = w * h
                        # If the rectangle meets the criteria for an event
                        if (
                            ratio >= 0.8
                            and ratio <= 1.2
                            and area > 8000
                            and area < 11000
                        ):
                            # Append the rectangle to the list of event bounding boxes
                            event_boxes.append([[x2, y2], [x2 + w, y2 + h]])
                            # Initialize a list to store the corners of the event
                            event_corners = []
                            # Loop over each point in the approximated contour
                            for k in range(0, 4):
                                # Append the point to the list of event corners
                                event_corners.append(approx[k][0])
                            # Reshape the list of event corners
                            pts = np.reshape(event_corners, (-1, 1, 2))
                            # If the list of points is not empty
                            if len(pts) != 0:
                                # Arrange the points in clockwise order
                                pts = arrange_clockwise(pts)
                                # Define the dimensions of the cropped event image
                                width = 160
                                height = 160
                                # Define the destination points for the perspective transformation
                                dstPts = [
                                    [0, 0],
                                    [width, 0],
                                    [width, height],
                                    [0, height],
                                ]
                                # Calculate the perspective transformation matrix
                                matrix = cv.getPerspectiveTransform(
                                    np.float32(pts), np.float32(dstPts)
                                )
                                # Apply the perspective transformation to the ROI image
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
                                # print(str(i) + " : ", predected_event)
                                break

    return extracted_events, event_boxes, events, marked_arena


# Recording and processing

cap = open_camera(1)

extracted_events = []

while len(extracted_events) < 5:
    # print("Scanning the event Boundaries")
    _, frame = cap.read()
    frame = undistort_image(
        frame,
        "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_5/Task_5A/MultiMatrix.npz",
    )
    frame = correct_Orientation(frame)[450:1500, 0:1080]
    extracted_events, event_boxes, events, marked_arena = event_extract(frame)
events_dict, priority_string = sort_priority(events)
print("identified_labels = ", events_dict)
path_str = path_plan(priority_string)
moves_str = mov_gen(path_str)
# print("priority_string", priority_string)
# print("path_str", path_str)
# print("moves_str", moves_str)
marked_arena = cv.resize(marked_arena, (900, 900))

while 1:
    cv.imshow("marked_arena", marked_arena)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cv.destroyAllWindows()

#  Send the Move Instructions to the bot

"""
* Logic: It creates a client socket and connects to the server at the given IP address. 
         It then enters a loop where it continuously tries to send the instructions to the server. 
         If the server acknowledges the receipt of the instructions, the function breaks out of the loop. 
         If a broken pipe error occurs, the function prints an error message and breaks out of the loop. 
         Finally, the client socket is closed.
"""


# Create a client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Define the server address
server_address = (esp32_ip, 80)
# Connect to the server
client_socket.connect(server_address)
# Define the message to be sent
message = moves_str
# Enter a loop
while 1:
    try:
        # Try to send the message to the server
        client_socket.sendall(message.encode())
        # Receive data from the server
        data = client_socket.recv(1024).decode()
        # If the server acknowledges the receipt of the instructions
        if data == "Instructions Recieved":
            # Break out of the loop
            break
    except IOError as e:
        # If a broken pipe error occurs
        if e.errno == errno.EPIPE:
            # Print an error message
            print("Broken pipe error occurred.")
            # Break out of the loop
            break
# Close the client socket
client_socket.close()


# QGIS Mapping

BOT_ARUCO_ID = 100

"""
* Function Name: get_nearest_lat_lon
* Input: bot_aruco_centre (a list of two float values representing the x and y coordinates of the bot's ArUco marker)
* Output: None. The function writes the nearest latitude and longitude to a CSV file.
* Logic: This function reads a CSV file containing pixel coordinates and their corresponding latitudes and longitudes.
         It calculates the Euclidean distance between the bot's ArUco centre and each pixel coordinate in the CSV file. 
         The function then writes the latitude and longitude corresponding to the minimum distance to a CSV file.
* Example Call: get_nearest_lat_lon([100, 200])
"""


def get_nearest_lat_lon(bot_aruco_centre):
    # Open the CSV file containing pixel coordinates and their corresponding latitudes and longitudes
    with open(
        "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_6/PixelstoLatLon.csv", "r"
    ) as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header

        # Initialize the minimum distance to infinity and the nearest latitude and longitude to None
        min_distance = float("inf")
        nearest_lat_lon = None

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Convert the pixel coordinates and latitudes and longitudes to float
            px, py, lat, lon = [float(val) for val in row]

            # Calculate the Euclidean distance between the bot's ArUco centre and the current pixel coordinate
            distance = math.sqrt(
                (px - bot_aruco_centre[0]) ** 2 + (py - bot_aruco_centre[1]) ** 2
            )

            # If the calculated distance is less than the current minimum distance, update the minimum distance and the nearest latitude and longitude
            if distance < min_distance:
                min_distance = distance
                nearest_lat_lon = (lat, lon)

    # Store the nearest latitude and longitude
    lat_lon = nearest_lat_lon
    live_loc_csv = (
        "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_4/Task_4B/Live_location.csv"
    )

    # Open the CSV file to write the nearest latitude and longitude
    with open(live_loc_csv, "w+", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["lat", "lon"])

    # Append the nearest latitude and longitude to the CSV file
    with open(live_loc_csv, "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(lat_lon)
        csvfile.close()


# Initialize the lists to store the ArUco IDs and centres
Const_ids_Check = []
Const_corner_Check = []
Const_centres = []
Const_ids = []
cnt = 0

# Start the loop to capture frames from the camera
while 1:
    # Capture a frame from the camera
    _, frame = cap.read()
    # Undistort the captured frame
    frame = undistort_image(
        frame, "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_5/Task_5A/MultiMatrix.npz"
    )
    # Correct the orientation of the frame
    frame = correct_Orientation(frame)[450:1500, 0:1080]

    # Detect the ArUco markers in the frame
    Const_ids, Const_centres = detect_aruco(frame)
    # If the number of detected ArUco centres is greater than 48 and the bot's ArUco ID is in the list of detected ArUco IDs, break the loop
    if len(Const_centres) > 48 and (BOT_ARUCO_ID in Const_ids):
        break

# Initialize the lists to store the centres of the bot's ArUco marker and the offset ArUco marker
bot_aruco_centre = []
offset_aruco_centre = []

# Iterate over the list of detected ArUco IDs
for i in range(0, len(Const_ids)):
    # If the current ArUco ID is the bot's ArUco ID, store the centre of the bot's ArUco marker
    if Const_ids[i] == BOT_ARUCO_ID:
        bot_aruco_centre = np.ravel(Const_centres[i]).tolist()
    # If the current ArUco ID is 32, store the centre of the offset ArUco marker
    if Const_ids[i] == 32:
        offset_aruco_centre = np.ravel(Const_centres[i]).tolist()

# Calculate the centre of the offset ArUco marker
offset_centre = [offset_aruco_centre[0] - 574, offset_aruco_centre[1] - 542]

# Start the loop to capture frames from the camera
while 1:
    # Capture a frame from the camera
    _, arena = cap.read()
    # Undistort the captured frame
    arena = undistort_image(
        arena, "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_5/Task_5A/MultiMatrix.npz"
    )
    # Correct the orientation of the frame
    arena = correct_Orientation(arena)[450:1500, 0:1080]

    # Detect the ArUco markers in the frame
    ids, centres = detect_aruco(arena)

    # Iterate over the list of detected ArUco IDs
    for i in range(0, len(ids)):
        # If the current ArUco ID is the bot's ArUco ID, store the centre of the bot's ArUco marker and break the loop
        if ids[i] == BOT_ARUCO_ID:
            bot_aruco_centre = np.ravel(centres[i]).tolist()
            break

    # Call the get_nearest_lat_lon function with the centre of the bot's ArUco marker as the argument
    get_nearest_lat_lon(
        [bot_aruco_centre[0] - offset_centre[0], bot_aruco_centre[1] - offset_centre[1]]
    )

    # Resize the frame
    arena = cv.resize(arena, (900, 900))
    # Display the frame
    cv.imshow("Arena", arena)
    # If the 'q' key is pressed, break the loop
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera
cap.release()
# Destroy all windows
cv.destroyAllWindows()
