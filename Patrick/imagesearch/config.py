import os

ORIG_DATASET_DIR = os.path.sep.join(['..', '..', 'dataset'])
DATASET_DIR = os.path.sep.join(['..', '..', 'patrick_dataset'])

################################################ CNN ################################################
TRAIN = "training"
TEST = "test"
VAL = "validation"

CLUSTER = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5']

BATCH_SIZE = 32
EPOCH = 20

COUNTED_CLASSES = len(os.listdir(ORIG_DATASET_DIR))
MODEL_PATH = os.path.sep.join(['result', 'model'])

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