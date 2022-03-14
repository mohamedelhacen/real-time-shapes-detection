import cv2
import numpy as np


def stackImages(scale, imageArray):
    rows = len(imageArray)
    cols = len(imageArray[0])
    rowsAvailable = isinstance(imageArray[0], list)
    width = imageArray[0][0].shape[1]
    height = imageArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imageArray[x][y].shape[:2] == imageArray[0][0].shape[:2]:
                    imageArray[x][y] = cv2.resize(imageArray[x][y], (0, 0), None, scale, scale)
                else:
                    imageArray[x][y] = cv2.resize(imageArray[x][y], (imageArray[0][0].shape[1], imageArray[0][0].shape[0]), None, scale, scale)
                if len(imageArray[x][y].shape) == 2:
                    imageArray[x][y]  = cv2.cvtColor(imageArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imageArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imageArray[x].shape[:2] == imageArray[0].shape[:2]:
                imageArray[x] = cv2.resize(imageArray[x], (0, 0), None, scale, scale)
            else:
                imageArray[x] = cv2.resize(imageArray[x], (imageArray[0].shape[1], imageArray[0].shape[0]), None, scale, scale)
            if len(imageArray[x].shape) == 2:
                imageArray[x] = cv2.cvtColor(imageArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imageArray)
        ver = hor
    return ver
