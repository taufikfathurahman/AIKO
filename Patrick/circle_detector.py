import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from imagesearch import config

def readandconv_image(image_path):
    img = cv2.imread(image_path, 1)
    img_orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img, img_orig

def image_sharpening(my_img):
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    return cv2.filter2D(my_img, -1, kernel_sharpening)

def median_blur(my_img):
    for _ in range(12):
        my_img = cv2.medianBlur(my_img, 21)
    
    return my_img

def hough_transform(my_img, M):
    all_circle = cv2.HoughCircles(my_img, cv2.HOUGH_GRADIENT, 1, round((90/3024)*M), 
                              param1=50,param2=30, 
                              minRadius=round((6/3024)*M), maxRadius=round((150/3024)*M))
    return all_circle

def show_circle(my_img, all_circle):
    detected_circle = 0 
    all_circle_rounded = 0
    if all_circle is not None:
        all_circle_rounded = np.uint16(np.around(all_circle))
        for i in all_circle_rounded[0,:]:
            cv2.circle(my_img,(i[0],i[1]),i[2],(0,255,0),15)
            cv2.circle(my_img,(i[0],i[1]),2,(0,0,255),5)

        all_circle_rounded = all_circle_rounded[0].tolist()
        detected_circle = len(all_circle_rounded)
    
    return my_img, detected_circle, all_circle_rounded

def circle_radius_avg(all_circle_rounded):
    radius_avg = 0
    result = 0
    if all_circle_rounded != 0:
        for i in all_circle_rounded:
            radius_avg += i[2]
        result = radius_avg/len(all_circle_rounded)
    return result

def circle_dens(detected_circle, radius_avg, M, N):
    result = 0
    if detected_circle != 0:
        result = (detected_circle * radius_avg) / (M * N)
    return result

def execute():
    i = 1
    img_name = []
    circles = []
    circle_rad = []
    circle_den = []
    dir_path = os.listdir(config.ORIG_DATASET_DIR)
    for img_folder in dir_path:
        img_path = os.path.sep.join([config.ORIG_DATASET_DIR, img_folder])
        img_selected = random.choices(os.listdir(img_path), k=10)
        img_name += img_selected
        for my_img in img_selected:
            # Read image from listed dir
            img, img_orig = readandconv_image(os.path.sep.join([img_path, my_img]))
            # Get image dimension
            M, N = img.shape
            # Image sharpening 2x
            img = image_sharpening(img)
            img = image_sharpening(img)
            # Image blurring using median blur
            img = median_blur(img)
            # Apply hough transform
            all_circle = hough_transform(img, M)
            img_orig, detected_circle, all_circle_rounded = show_circle(img_orig, all_circle)
        
            # save image
            cv2.imwrite(config.CIRCLE_ROUNDED + '/' + my_img, img_orig)
            
            # Save to list for cvs file
            circles.append(detected_circle)

            radius_avg = circle_radius_avg(all_circle_rounded)
            circle_rad.append(radius_avg)

            circle_den.append(circle_dens(detected_circle, radius_avg, M, N))
            print(i, '-', my_img)
            i += 1
        
    csv_dict = {
        'Data Kayu' : img_name,
        'Detected Circle' : circles,
        'Radius Avg' : circle_rad,
        'Circle Density' : circle_den
    }
    df = pd.DataFrame(csv_dict)
    df.to_csv(config.CIRCLE_DETECTOR, index = False)

execute()                                                                          