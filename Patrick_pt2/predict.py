import numpy as np
import argparse
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import shutil as sh
import cv2 as cv
import json

from imagesearch import config
import hough_transform as ht
import kmean as km


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-i',
        '--image',
        type=str,
        required=True,
        help='path to your input image'
    )
    args = vars(ap.parse_args())

    return args['image']


def get_cluster():
    selected_file_path = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht4.csv'])
    selected_files = pd.read_csv(selected_file_path)

    input_file_path = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht3.csv'])
    input_files = pd.read_csv(input_file_path)

    new_file = {
        'Data Kayu': list(selected_files['Data Kayu']) + list(input_files['Data Kayu']),
        'Detected Circle': list(selected_files['Detected Circle']) + list(input_files['Detected Circle']),
        'Radius Avg': list(selected_files['Radius Avg']) + list(input_files['Radius Avg']),
        'Circle Density': list(selected_files['Circle Density']) + list(input_files['Circle Density'])
    }

    new_file_path = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht3.csv'])
    df = pd.DataFrame(new_file)
    df.to_csv(new_file_path, index=False)

    km.execute()
    my_cluster = km.get_neighbor()

    return int(my_cluster)


def predict_image(my_cluster):
    model = load_model(os.path.sep.join([config.MODEL_PATH, config.CLUSTER[my_cluster] + '.h5']))
    with open(os.path.sep.join([config.MODEL_PATH, config.CLUSTER[my_cluster] + '.txt']), 'r') as f:
        label = json.loads(f.read())

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(
        'test_me',
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))
    prediction = model.predict_generator(validation_generator)
    predicted_classes = int(np.argmax(prediction, axis=1)[0])

    return label[str(predicted_classes)], prediction


def show_result(image, image_label):
    img = cv.imread(image)
    img = cv.resize(img, (400, 400))
    text = "{}".format(image_label)
    cv.putText(img,
               text,
               (3, 20),
               cv.FONT_HERSHEY_SIMPLEX,
               0.5,
               (150, 0, 50),
               2)
    cv.imshow('Output', img)
    cv.waitKey(0)


def execute():
    image = parser()
    print(image)
    sh.copy2(image, config.TEST_ME)
    ht.execute([image])
    my_cluster = get_cluster()
    image_label, prediction = predict_image(my_cluster)
    show_result(image, image_label)
    image = image.split('/')
    os.remove(os.path.sep.join([config.TEST_ME, image[-1]]))


execute()
