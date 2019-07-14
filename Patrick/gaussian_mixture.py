import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture as GMM

def draw_ellipse(n_class, position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance[:, :2])
    
    # Draw the Ellipse
    for nsig in range(1, n_class):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs)) 

def plot_gmm(n_class, gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(n_class, pos, covar, alpha=w * w_factor)
        
    return labels

def save_result(labels):
    df = pd.read_csv('Result/circle_detected.csv')
    df['Wood Class'] = labels
    df.to_excel('Result/GMM_result.xlsx', index = False)

def execute_gmm(n_class):
    gmm = GMM(n_components=n_class, covariance_type='full', random_state=500)
    labels = plot_gmm(n_class, gmm, X)
    save_result(labels)

ht_result = np.genfromtxt("Result/circle_detected.csv", delimiter=",", skip_header=1)
X = ht_result[:,1:3]
execute_gmm(6)