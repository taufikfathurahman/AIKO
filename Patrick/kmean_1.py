import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os

from imagesearch import config

csv_file = os.path.sep.join([config.CIRCLE_DETECTOR, 'ht0.csv'])
ht_result = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
    
X = ht_result[:,1:]

def find_cluster(k):
    kmeans = KMeans(n_clusters=k, max_iter=1000, algorithm='auto')
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_

    return y_kmeans, centers

def plot_cluster(y_kmeans, centers):
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.savefig(config.KMEAN1_IMG)

def save_to_xlsx(y_kmeans):
    df = pd.read_csv(csv_file)
    df['Wood Class'] = y_kmeans
    df.to_excel(config.KMEAN1_XLSX, index = False)

def execute(K = config.K):
    y_kmeans, centers = find_cluster(K)
    plot_cluster(y_kmeans, centers)
    save_to_xlsx(y_kmeans)    

execute()