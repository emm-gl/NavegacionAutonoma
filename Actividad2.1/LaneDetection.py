#Libraries:
import cv2
import numpy as np


# Open the video file
cap = cv2.VideoCapture('./Videos/LaneDetection.mp4')

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was successfully read 
    if ret:
        # Process the frame
        img = frame

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection to the grayscale image
        edges = cv2.Canny(gray, 100, 150, apertureSize=3)


        #ROI
        vertices = np.array ([[(100,700), (600, 550), (700, 550), (1000, 700)]], dtype = np.int32)
        img_roi = np.zeros_like(edges)
        cv2.fillPoly (img_roi, vertices, 255)
        img_mask = cv2.bitwise_and(edges, img_roi)



        #Hough Transform Parameters:
        rho = 2
        theta = np.pi/180
        threshold = 40
        min_line_len = 10
        max_line_len = 10
        lines = cv2.HoughLinesP(img_mask, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap=max_line_len)

        #print(lines)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display the orginal image with the lane detaction and the Canny Transform:
        cv2.imshow('Canny Transform', img_mask)
        cv2.imshow('Original', frame)



        # Wait for a key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
