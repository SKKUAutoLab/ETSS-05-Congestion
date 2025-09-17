import os.path
import cv2
import numpy as np

def areaCal(contour):
    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour)
    return area

def findmaxcontours(distance_map, fname):
    threshold = min(255 * (np.mean(distance_map) * 4 ) / np.max(distance_map),150)
    distance_map = 255 * distance_map / np.max(distance_map)
    distance_map = distance_map[0][0]
    distance_map[distance_map < 0] = 0
    img = distance_map.astype(np.uint8)
    ret, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    if not os.path.exists('middle_process'):
        os.makedirs('middle_process')
    cv2.imwrite("middle_process/binary2.jpg", binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list_index = []
    for i in range(len(contours)):
        list_index.append(areaCal(contours[i]))
    list_index.sort(reverse=True)
    first = list_index[0]
    first_index = 0
    img = cv2.applyColorMap(img, 2)
    for i in range(len(contours)):
        if areaCal(contours[i]) == first:
            first_index = i
    cv2.drawContours(img, contours[first_index], -1, (0, 0, 255), 2)
    x, y, w, h = cv2.boundingRect(contours[first_index])
    coordinate_first = [x, y, w, h]
    if not os.path.exists('middle_process/contours_result_mean'):
        os.makedirs('middle_process/contours_result_mean')
    save_path = "middle_process/contours_result_mean/" + fname[0]
    save_path = save_path.replace('.h5','.jpg')
    cv2.imwrite(save_path, img)
    return coordinate_first