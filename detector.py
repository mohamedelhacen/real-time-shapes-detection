import cv2
import numpy as np
from stackImages import stackImages


def empty(a):
    pass


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("threshold1", "Parameters", 23, 255, empty)
cv2.createTrackbar("threshold2", "Parameters", 20, 255, empty)
cv2.createTrackbar("Area", "Parameters", 5000, 30000, empty)


def getContours(image, imageContour):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imageContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imageContour, (x, y), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(imageContour, "Points: " + str(len(approx)), (x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX, .7, (0,255, 0), 2)
            cv2.putText(imageContour, "Area: " + str(int(area)), (x+w+20, y+45), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)


image = cv2.imread("shapes.png")
imageBlur = cv2.GaussianBlur(image, (7, 7), 1)
imageGray = cv2.cvtColor(imageBlur, cv2.COLOR_BGR2GRAY)
while True:
    imageContour = image.copy()
    threshold1 = cv2.getTrackbarPos("threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("threshold2", "Parameters")
    imageCanny = cv2.Canny(imageGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imageDilated = cv2.dilate(imageCanny, kernel, iterations=1)
    getContours(imageDilated, imageContour)
    stackedImage = stackImages(0.8, ([image, imageCanny],
                                     [imageDilated, imageContour]))
    cv2.imshow("Output", stackedImage)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
