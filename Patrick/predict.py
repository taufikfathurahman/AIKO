# from tensorflow.keras.models import load_model
import numpy as np 
import argparse
import os
import pandas as pd

from imagesearch import config
import hough_transform as ht

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
ht.execute(image)

#################################################################################################
selected_file_path = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht'+str(1)+'.csv'])
selected_files = pd.read_csv(selected_file_path)

input_file_path = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht'+str(3)+'.csv'])
input_files = pd.read_csv(input_file_path)

new_file = {
    'Data Kayu' : list(selected_files['Data Kayu']),
    'Detected Circle' : list(selected_files['Detected Circle']),
    'Radius Avg' : list(selected_files['Radius Avg']),
    'Circle Density' : list(selected_files['Circle Density'])
}
new_file['Data Kayu'] += list(input_files['Data Kayu'])
new_file['Detected Circle'] += list(input_files['Detected Circle'])
new_file['Radius Avg'] += list(input_files['Radius Avg'])
new_file['Circle Density'] += list(input_files['Circle Density'])

new_file_path = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht'+str(3)+'.csv'])
df = pd.DataFrame(new_file)
df.to_csv(new_file_path, index = False)
##################################################################################################

new_file_cluster = pd.read_excel(config.KMEAN_XLSX[3])
input_img_cluster = list(new_file_cluster['Wood Class'])[-1]