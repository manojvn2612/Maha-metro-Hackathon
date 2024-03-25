'''import cv2
import numpy as np

img = cv2.imread("Hackthon\MahaMetro\image.jpeg")

#using canny edge detection, Hough transform will be applied on the output #of cv2.Canny()
cannyedges = cv2.Canny(img, 75, 150)

#applying cv2.HoughlinesP(), the coordinates of the endpoints of the #detected lines are stored 
detectedlines = cv2.HoughLinesP(cannyedges, 1, np.pi/180, 60,maxLineGap=30)

#iterating over the points and drawing lines on the image by using the #coordinates that we got from HoughLinesP()
for line in detectedlines:
  x0, y0, x1, y1 = line[0]
  cv2.line(img, (x0, y0), (x1, y1), (0, 0, 250), 1)

#getting the output
cv2.imshow("linesDetectedusingHoughTransform", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
import cv2
import numpy as np

img = cv2.imread("Hackthon\MahaMetro\image.png")

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using Canny
cannyedges = cv2.Canny(blurred, 50, 150)

# Apply a region of interest mask to focus on the area where the railway tracks are expected
mask = np.zeros_like(cannyedges)
height, width = cannyedges.shape
polygon = np.array([[
    (0, height * 0.6),
    (width, height * 0.6),
    (width, height),
    (0, height),
]], np.int32)
cv2.fillPoly(mask, polygon, 255)
masked_edges = cv2.bitwise_and(cannyedges, mask)

# Applying cv2.HoughLinesP(), the coordinates of the endpoints of the detected lines are stored
detectedlines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 60, maxLineGap=30)

# Iterating over the points and drawing lines on the image using the coordinates obtained from HoughLinesP()
for line in detectedlines:
    x0, y0, x1, y1 = line[0]
    cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 2)  # Green color for railway tracks

# Display the output
cv2.imshow("Railway Tracks Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
