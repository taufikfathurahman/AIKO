import os

ORIG_DATASET_DIR = os.path.sep.join(['..', 'dataset'])

################################################ CNN ################################################
TRAIN = "training"
TEST = "test"
VAL = "validation"

BATCH_SIZE = 32

COUNTED_CLASSES = len(os.listdir(ORIG_DATASET_DIR))

MODEL_PATH = os.path.sep.join(['..', 'model', 'patrick.h5'])

############################################### Output ###############################################
CIRCLE_ROUNDED = os.path.sep.join(['Result', 'circle_rounded'])
CIRCLE_DETECTOR = os.path.sep.join(['Result', 'circle_detector', 'circle_detected.csv'])

KMEAN1_IMG = os.path.sep.join(['Result', 'clustering', 'kmean1.png'])
KMEAN2_IMG = os.path.sep.join(['Result', 'clustering', 'kmean2.png'])