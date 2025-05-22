# import cv2


# # define a video capture object
# # vid = cv2.VideoCapture("/dev/video2")

# vid = cv2.VideoCapture("/dev/video2",cv2.CAP_DSHOW)
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# ret, frame = vid.read()


# while(True):

#     # Capture the video frame
#     # by frame
#     ret, frame = vid.read()

#     # Display the resulting frame

#     cv2.imshow('frame', frame)

#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # After the loop release the cap object
# vid.release()
# # Destroy all the windows
# cv2.destroyAllWindows()


import cv2 as cv
import numpy as np

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


# Get the default resolution
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

print(f"Default resolution: {width}x{height}")

# Set the desired resolution
desired_width = 1920
desired_height = 1080
cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height)

# cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
# Get the resolution after setting
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)


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


while True:
    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    frame = undistort_image(
        frame,
        "/mnt/Storage/Projects/E-YRC/EYRC_2023/Task_5/Task_5A/MultiMatrix.npz",
    )
    # Display the resulting frame
    frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)[450:1600, 0:1080]
    frame = cv.resize(frame, (900, 900))
    cv.imshow("frame", frame)
    cv.imwrite("Arena_Image.jpg", frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv.destroyAllWindows()


print(f"Resolution set to: {width}x{height}")
