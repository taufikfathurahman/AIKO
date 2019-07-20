import numpy as np
import argparse
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt

from imagesearch import config
import hough_transform as ht
import kmean as km

image_size = 280

# ap = argparse.ArgumentParser()
# ap.add_argument(
#     '-i',
#     '--image',
#     type=str,
#     required=True,
#     help='path to your input image'
# )
# args = vars(ap.parse_args())
#
# #################################################################################################
# image = [args["image"]]

model = load_model(os.path.sep.join([config.MODEL_PATH, config.CLUSTER[3] + '.h5']))
label = pd.read_excel(os.path.sep.join([config.MODEL_PATH, config.CLUSTER[3]+'.xlsx']))

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'test_me',
    target_size=(image_size, image_size))
predictions = model.predict_generator(validation_generator)
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
