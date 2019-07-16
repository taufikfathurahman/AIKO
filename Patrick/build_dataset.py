import pandas as pd
from imutils import paths
import shutil
import os
import numpy as np
import random

from imagesearch import config

cluster_data = pd.read_excel(config.KMEAN2_XLSX)
my_cluster = []
my_cluster.append(cluster_data['Data Kayu'])
my_cluster.append(cluster_data['Wood Class'])

clusters = {}
for cluster_num in range(config.K):
    wood_id = []
    for idx in range(len(my_cluster[0])):
        if my_cluster[1][idx] == cluster_num:
            wood_id.append(my_cluster[0][idx].split('_')[0])
    clusters[cluster_num] = np.unique(wood_id)

for key in clusters:
    for img_sp in clusters[key]:
        validation_path = os.path.sep.join([config.DATASET_DIR, 'cluster_'+str(key), config.VAL, img_sp])
        test_path = os.path.sep.join([config.DATASET_DIR,  'cluster_'+str(key), config.TEST, img_sp])
        train_path = os.path.sep.join([config.DATASET_DIR,  'cluster_'+str(key), config.TRAIN, img_sp])

        if not os.path.exists(validation_path):
            os.makedirs(validation_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        p = os.path.sep.join([config.ORIG_DATASET_DIR, img_sp])

        imagePaths = list(paths.list_images(p))
        imagePaths = set(imagePaths)
        for_val = random.sample(imagePaths, 40)
        for imagePath in for_val:
            shutil.move(imagePath, validation_path)

        imagePaths = list(paths.list_images(p))
        imagePaths = set(imagePaths)
        for_test = random.sample(imagePaths, k=10)
        for imagePath in for_test:
            shutil.move(imagePath, test_path)

        imagePaths = list(paths.list_images(p))
        imagePaths = set(imagePaths)
        for imagePath in imagePaths:
            shutil.move(imagePath, train_path)