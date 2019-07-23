import os

ORIG_DATASET_DIR = os.path.sep.join(['..', '..', 'dataset'])
DATASET_DIR = os.path.sep.join(['..', '..', 'patrick_dataset'])
TEST_ME = os.path.sep.join(['test_me', 'img'])

################################################ CNN ################################################
TRAIN = "training"
TEST = "test"
VAL = "validation"

CLUSTER = ['cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5']

BATCH_SIZE = 8
EPOCH = 20
IMAGE_SIZE = 280

COUNTED_CLASSES = len(os.listdir(ORIG_DATASET_DIR))
MODEL_PATH = os.path.sep.join(['result', 'model'])

PREDICTED_IMG = os.path.sep.join(['result, predicted_img'])

############################################### Circle ##############################################
CIRCLE_ROUNDED = os.path.sep.join(['result', 'circle_rounded'])
CIRCLE_DETECTOR = os.path.sep.join(['result', 'circle_detector'])

############################################## Cluster ##############################################
K = 6

KMEAN_IMG = [os.path.sep.join(['result', 'clustering', 'kmean_0.png']),
             os.path.sep.join(['result', 'clustering', 'kmean_1.png']),
             os.path.sep.join(['result', 'clustering', 'kmean_2.png']),
             os.path.sep.join(['result', 'clustering', 'kmean_new.png']),
             os.path.sep.join(['result', 'clustering', 'kmean_fixed.png'])
             ]
GMM_IMG = os.path.sep.join(['result', 'clustering', 'gmm.png'])

KMEAN_XLSX = [os.path.sep.join(['result', 'clustering', 'kmean_0.xlsx']),
              os.path.sep.join(['result', 'clustering', 'kmean_1.xlsx']),
              os.path.sep.join(['result', 'clustering', 'kmean_2.xlsx']),
              os.path.sep.join(['result', 'clustering', 'kmean_new.xlsx']),
              os.path.sep.join(['result', 'clustering', 'kmean_fixed.xlsx'])
              ]
GMM_XLSX = os.path.sep.join(['result', 'clustering', 'gmm.xlsx'])

FILTERED_KMEAN = os.path.sep.join(['result', 'filtered_kmean'])

MY_CLUSTER = os.path.sep.join(['result', 'clustering', 'my_cluster.txt'])
