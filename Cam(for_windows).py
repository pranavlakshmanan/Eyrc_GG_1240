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

cap = cv.VideoCapture(0)

# try:
#     cap = cv.VideoCapture(0)
#     if not cap.isOpened():
#         raise Exception("Could not open 0")
# except:
#     cap = cv.VideoCapturue(1)
#     if not cap.isOpened():
#         print("Could not open 1 either. Please check your device.")
#         try:
#             cap = cv.VideoCapture(2)
#             if not cap.isOpened():
#                 raise Exception("Could not open 2")
#         except:
#             cap = cv.VideoCapture(3)
#             if not cap.isOpened():
#                 print("Could not open 3 either. Please check your device.")

                

# Get the default resolution
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

print(f"Default resolution: {width}x{height}")

# Set the desired resolution
desired_width = 1920
desired_height = 1080
cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height)
# cap.set(cv.CAP_PROP_EXPOSURE, -1)

# Get the resolution after setting
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = cap.read() 
  
    # Display the resulting frame
    frame= cv.resize(frame, (1800,900))
    cv.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv.destroyAllWindows() 


print(f"Resolution set to: {width}x{height}")
