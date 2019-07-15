import os

ORIG_DATASET_DIR = os.path.sep.join(['..', '..', 'dataset'])
DATASET_DIR = os.path.sep.join([['..', '..', 'patrick_dataset'])

################################################ CNN ################################################
TRAIN = "training"
TEST = "test"
VAL = "validation"

CLUSTER = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'cluster6']

BATCH_SIZE = 32

COUNTED_CLASSES = len(os.listdir(ORIG_DATASET_DIR))

MODEL1_PATH = os.path.sep.join(['..', 'model', 'patrick.h5'])

############################################### Circle ##############################################
CIRCLE_ROUNDED = os.path.sep.join(['result', 'circle_rounded'])
CIRCLE_DETECTOR = os.path.sep.join(['result', 'circle_detector', 'circle_detected.csv'])

############################################## Cluster ##############################################
K = 6
KMEAN1_IMG = os.path.sep.join(['result', 'clustering', 'kmean1.png'])
KMEAN2_IMG = os.path.sep.join(['result', 'clustering', 'kmean2.png'])
GMM_IMG = os.path.sep.join(['result', 'clustering', 'gmm.png'])

KMEAN1_XLSX = os.path.sep.join(['result', 'clustering', 'kmean1.xlsx'])
KMEAN2_XLSX = os.path.sep.join(['result', 'clustering', 'kmean2.xlsx'])
GMM_XLSX = os.path.sep.join(['result', 'clustering', 'gmm.xlsx'])