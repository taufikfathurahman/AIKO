import pandas as pd
import time as tm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
import os
import json
from statistics import mode

from imagesearch import config


def get_csv(j):
    csv_file = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht' + str(j) + '.csv'])
    ht_result = np.genfromtxt(csv_file, delimiter=",", skip_header=1)

    return ht_result[:, 1:], csv_file


def find_clusters(X, n_clusters, rseed=300):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


def get_neighbor():
    input_img_cluster = pd.read_excel(config.KMEAN_XLSX[3])
    img_cluster = list(input_img_cluster['Wood Class'])[-1]
    list_cluster = [input_img_cluster['Data Kayu'], input_img_cluster['Wood Class']]
    cp_list_cluster = []
    with open(config.MY_CLUSTER, 'r') as f:
        my_cluster = json.loads(f.read())
    clusters = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: []
    }

    for img in range(len(list_cluster[0])):
        cp_list_cluster.append(list_cluster[0][img].split('_')[0])

    for sp in np.unique(cp_list_cluster):
        cluster_id = []
        for wid in range(len(list_cluster[0])):
            if list_cluster[0][wid].split('_')[0] == sp:
                cluster_id.append(list_cluster[1][wid])
        mode_cluster = mode(cluster_id)
        clusters[mode_cluster].append(sp.split('_')[0])

    find_me = clusters[img_cluster][0]
    real_cluster = 0

    for key in my_cluster.keys():
        for val in my_cluster[key]:
            if val == find_me:
                real_cluster = key

    print('Image Cluster : ', real_cluster)

    return real_cluster


def plot_cluster(centers, labels, X, j):
    plt.scatter(X[:, 0], X[:, 1], c=labels,
                s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1],
                c='black', s=200, alpha=0.5)
    plt.savefig(config.KMEAN_IMG[j])


def save_to_xlsx(labels, csv_file, j):
    df = pd.read_csv(csv_file)
    df['Wood Class'] = labels
    df.to_excel(config.KMEAN_XLSX[j], index=False)


def execute(j=3, K=config.K):
    start = tm.time()

    X, csv_file = get_csv(j)
    centers, labels = find_clusters(X, K)
    plot_cluster(centers, labels, X, j)
    save_to_xlsx(labels, csv_file, j)

    print('kmean clustering done.....')
    end = tm.time()
    menit = (end - start) / 60
    print('Time spent => ', menit, ' minutes')

