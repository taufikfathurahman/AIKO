from imagesearch import config

import os
from imutils import paths
import random as rd

sp_names = os.listdir(config.ORIG_DATASET_DIR)

for sp in sp_names:
    path = os.listdir(os.path.sep.join([config.ORIG_DATASET_DIR, sp]))
    print(sp,' = ', len(path))