from tensorflow.keras.models import load_model
import numpy as np 
import argparse
import imutils
import cv2
import os

from imagesearch import config

model = load_model(os.path.sep.join([config.MODEL_PATH, config.CLUSTER[0] + '.h5']))

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", 
    "--image", 
    type=str, 
    required=True, 
    help="path  to our input image"
    )

args = vars(ap.parse_args())

image = cv2.imread(args["image"])
output = image.copy()
output = imutils.resize(output, width=400)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

preds = model.predict(np.expand_dims(image, axis=0))[0]
i = np.argmax(preds)

print(preds)
print('Predicted => ', i)