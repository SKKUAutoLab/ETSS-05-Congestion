import os
import cv2
import numpy as np
import random

def areaCal(contour):
    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour)
    return area

def findmaxcontours(distance_map, find_max, fname): # [1, 1, 704, 1024]
    distance_map = 255 * distance_map / np.max(distance_map)
    distance_map = distance_map[0][0]
    img = distance_map.astype(np.uint8)
    Img = img
    gray = img
    Thresh = 8.0/11.0 * 255.0
    ret, binary = cv2.threshold(gray, Thresh, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list_index = []
    for i in range(len(contours)):
        list_index.append(areaCal(contours[i]))
    list_index.sort(reverse = True)
    if len(list_index) == 0:
        return [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]
    if len(list_index)<3:
        list_index.append(list_index[0])
        list_index.append(list_index[0])
    if len(list_index) >= 5:
        list_index = list_index[0:5]
        index_new = random.sample(list_index, 2)
        index_new.sort(reverse = True)
        first = index_new[0]
        sceond = index_new[1]
    else:
        first = list_index[0]
        sceond = list_index[1]
    if find_max == True:
        first = list_index[0]
        sceond = list_index[1]
    first_index = 0
    sceond_index = 0
    for i in range(len(contours)):
        if areaCal(contours[i]) == first:
            first_index = i
        if areaCal(contours[i]) == sceond:
            sceond_index = i
    cor_array = []
    cv2.drawContours(img, contours[first_index], -1, (0, 0, 255), 3)
    x, y, w, h = cv2.boundingRect(contours[first_index])
    cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    coordinate_first = [x, y, w, h]
    cor_array.append([x, y])
    x, y, w, h = cv2.boundingRect(contours[sceond_index])
    cor_array.append([x, y])
    if not os.path.exists('save_file/contours_result'):
        os.makedirs('save_file/contours_result')
    save_path = os.path.join("save_file/contours_result/", fname[0])
    cv2.imwrite(save_path.replace('h5','jpg'), Img)
    return coordinate_first