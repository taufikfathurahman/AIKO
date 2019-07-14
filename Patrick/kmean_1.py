import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

ht_result = np.genfromtxt("Result/circle_detector/circle_detected.csv", delimiter=",", skip_header=1)
    
X = ht_result[:,1:]

def execute():
    kmeans = KMeans(n_clusters=6, max_iter=1000, algorithm='auto')
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.savefig('Result/clustering/kmean1.png')

    df = pd.read_csv('Result/circle_detector/circle_detected.csv')
    df['Wood Class'] = y_kmeans
    df.to_excel('Result/clustering/Kmean1.xlsx', index = False)

execute()