import pandas as pd
from imutils import paths
import shutil
import os
import random
from statistics import mode
import json

from imagesearch import config


def get_cluster(cluster_xlsx=config.KMEAN_XLSX[4]):
    cluster_data = pd.read_excel(cluster_xlsx)
    err_sps = pd.read_excel(os.path.sep.join([config.FILTERED_KMEAN, 'delete_this_sps.xlsx']))
    err_sps = list(err_sps['Sp will be deleted'])
    my_cluster = [cluster_data['Data Kayu'], cluster_data['Wood Class']]

    list_sp = os.listdir(config.ORIG_DATASET_DIR)

    clusters = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
    }

    for sp in list_sp:
        if sp not in err_sps:
            cluster_id = []
            for wid in range(len(my_cluster[0])):
                if my_cluster[0][wid].split('_')[0] == sp:
                    cluster_id.append(my_cluster[1][wid])
            mode_cluster = mode(cluster_id)
            clusters[mode_cluster].append(sp)

    return clusters


def move_image():
    clusters = get_cluster()

    with open(config.MY_CLUSTER, 'w') as f:
        json.dump(clusters, f)

    for key in clusters:
        for img_sp in clusters[key]:
            validation_path = os.path.sep.join([config.DATASET_DIR, 'cluster_' + str(key), config.VAL, img_sp])
            test_path = os.path.sep.join([config.DATASET_DIR, 'cluster_' + str(key), config.TEST, img_sp])
            train_path = os.path.sep.join([config.DATASET_DIR, 'cluster_' + str(key), config.TRAIN, img_sp])

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
