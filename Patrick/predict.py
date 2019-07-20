# from tensorflow.keras.models import load_model
import numpy as np 
import argparse
import os
import pandas as pd
from tensorflow.keras.models import load_model
import PIL
import matplotlib.pyplot as plt
import cv2
import imutils

from imagesearch import config
import hough_transform as ht
import kmean_2 as km

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", 
    "--image", 
    type=str, 
    required=True, 
    help="path  to our input image"
    )
args = vars(ap.parse_args())

#################################################################################################
image = [args["image"]]
# ht.execute(image)

# #################################################################################################
# selected_file_path = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht'+str(1)+'.csv'])
# selected_files = pd.read_csv(selected_file_path)

# input_file_path = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht'+str(3)+'.csv'])
# input_files = pd.read_csv(input_file_path)

# new_file = {
#     'Data Kayu' : list(selected_files['Data Kayu']),
#     'Detected Circle' : list(selected_files['Detected Circle']),
#     'Radius Avg' : list(selected_files['Radius Avg']),
#     'Circle Density' : list(selected_files['Circle Density'])
# }
# new_file['Data Kayu'] += list(input_files['Data Kayu'])
# new_file['Detected Circle'] += list(input_files['Detected Circle'])
# new_file['Radius Avg'] += list(input_files['Radius Avg'])
# new_file['Circle Density'] += list(input_files['Circle Density'])

# new_file_path = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht'+str(3)+'.csv'])
# df = pd.DataFrame(new_file)
# df.to_csv(new_file_path, index = False)

# ##################################################################################################
# # Call kmean_2.py
# j = 3
# try:
#     km.execute(j)
# except:
#     print('ERROR : Program failed executed.....')

# new_file_cluster = pd.read_excel(config.KMEAN_XLSX[j])
# input_img_cluster = list(new_file_cluster['Wood Class'])[-1]
# print('Image Cluster => Cluster ', input_img_cluster)

##################################################################################################
# Load Model
model = load_model(os.path.sep.join([config.MODEL_PATH, config.CLUSTER[0] + '.h5']))
label = pd.read_excel(os.path.sep.join([config.MODEL_PATH, config.CLUSTER[0]+'.xlsx']))

input_shape = (280, 280)
img = PIL.Image.open(image[0])
img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

img_array = np.expand_dims(np.array(img_resized), axis=0)
pred = model.predict(img_array)[0]
i = np.argmax(pred)
image_label = list(label[i])

print(label)
print(pred)

img = cv2.imread(image[0])
img = imutils.resize(img, width=400)
text = "{}: {:.2f}%".format(image_label, pred[i] * 100)
cv2.putText(img, 
            text, 
            (3, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            2)
cv2.imshow('Output', img)
cv2.waitKey(0)
