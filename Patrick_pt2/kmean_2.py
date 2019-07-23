import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os

from imagesearch import config


def get_csv(j):
    csv_file = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht' + str(j) + '.csv'])
    ht_result = np.genfromtxt(csv_file, delimiter=",", skip_header=1)

    return ht_result[:, 1:], csv_file


def find_cluster(X, k):
    kmeans = KMeans(n_clusters=k, max_iter=2000, algorithm='auto')
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_

    return y_kmeans, centers


def plot_cluster(X, y_kmeans, centers, j):
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.savefig(config.KMEAN_IMG[j])


def save_to_xlsx(csv_file, y_kmeans, j):
    df = pd.read_csv(csv_file)
    df['Wood Class'] = y_kmeans
    df.to_excel(config.KMEAN_XLSX[j], index=False)


def execute(j=3, K=config.K):
    X, csv_file = get_csv(j)
    y_kmeans, centers = find_cluster(X, K)
    plot_cluster(X, y_kmeans, centers, j)
    save_to_xlsx(csv_file, y_kmeans, j)


execute(1)
