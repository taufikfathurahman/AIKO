import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import MinMaxScaler

from imagesearch import config

ht_result = np.genfromtxt(config.CIRCLE_DETECTOR, delimiter=",", skip_header=1)
    
X = ht_result[:,1:]

def find_clusters(X, n_clusters, rseed=500):
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

def plot_cluster(centers, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels,
                s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], 
                c='black', s=200, alpha=0.5)
    plt.savefig(config.KMEAN2_IMG)

def save_to_xlsx():
    df = pd.read_csv(config.CIRCLE_DETECTOR)
    df['Wood Class'] = labels
    df.to_excel('Result/clustering/Kmean2.xlsx', index = False)

def execute():
    centers, labels = find_clusters(X, 6)
    plot_cluster(centers, labels)
    
execute()