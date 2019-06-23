#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import cv2
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read image & Convert Image to Gray-Scale

# In[2]:


def readandconv_image(image_name):
    img = cv2.imread('Asset/'+image_name, 1)
    img_orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img, img_orig


# ## Sharpening Image

# In[3]:


def image_sharpening(my_img):
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    return cv2.filter2D(my_img, -1, kernel_sharpening)


# ## Blur Image to Reduce Noise

# In[4]:


def median_blur(my_img):
    for _ in range(5):
        my_img = cv2.medianBlur(my_img, 21)
    
    return my_img


# In[5]:


def gaussian_blur(my_img):
    for _ in range(5):
        my_img = cv2.GaussianBlur(my_img, (21, 21), cv2.BORDER_DEFAULT)
    
    return my_img


# ## Image Thresholding
# 
# #### 1. Image Thresholding (Otsu's Binarization)

# In[6]:


def otsu_thresholding(my_img):
    ret, th = cv2.threshold(my_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return th


# #### 2. Image Thresholding (Adaptive Thresholding)

# In[7]:


def adaptive_thresholding(my_img):
    th = cv2.adaptiveThreshold(my_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,                               cv2.THRESH_BINARY, 31, 20)
    return th


# ## Hough Circle Transform

# In[8]:


def hough_transform(my_img):
    all_circle = cv2.HoughCircles(my_img, cv2.HOUGH_GRADIENT, 1, 100, 
                              param1=50,param2=30, 
                              minRadius=10, maxRadius=150)
    return all_circle


# ## Show Detected Circles

# In[9]:


def show_circle(my_img, all_circle):
    all_circle_rounded = np.uint16(np.around(all_circle))
    for i in all_circle_rounded[0,:]:
        # draw the outer circle
        cv2.circle(my_img,(i[0],i[1]),i[2],(0,255,0),15)
        # draw the center of the circle
        cv2.circle(my_img,(i[0],i[1]),2,(0,0,255),5)
        
    all_circle_rounded = all_circle_rounded[0].tolist()
    detected_circle = len(all_circle_rounded)
    
    return my_img, detected_circle, all_circle_rounded


# ##### Get Average Radius

# In[10]:


def circle_radius_avg(all_circle_rounded):
    radius_avg = 0
    for i in all_circle_rounded:
        radius_avg += i[2]
    
    return radius_avg/len(all_circle_rounded)


# ##### Get Circle Density

# In[11]:


def circle_dens1(detected_circle, radius_avg, M, N):
    return (detected_circle * radius_avg) / (M * N)


# # Main Program

# In[12]:


def main():
    # Directory listing
    entries = os.listdir('Asset')
    circles = []
    circle_rad = []
    circle_den = []

    for my_img in entries:
        # Read image from listed dir
        img, img_orig = readandconv_image(my_img)
        # Get image dimension
        M, N = img.shape
        # Image sharpening
        img = image_sharpening(img)
        # Image blurring uring using median blur
        img = median_blur(img)
        # Apply hough transform
        all_circle = hough_transform(img)
        img_orig, detected_circle, all_circle_rounded = show_circle(img_orig, all_circle)
        
        # save image
        cv2.imwrite('Result/' + my_img, img_orig)
        
        # Save to list for cvs file
        circles.append(detected_circle)

        radius_avg = circle_radius_avg(all_circle_rounded)
        circle_rad.append(radius_avg)

        circle_den.append(circle_dens1(detected_circle, radius_avg, M, N))
        
    csv_dict = {
        'Data Kayu' : entries,
        'Detected Circle' : circles,
        'Radius Avg' : circle_rad,
        'Circle Density' : circle_den
    }
    df = pd.DataFrame(csv_dict)
    df.to_excel('Result/Result.xlsx', index = False)

