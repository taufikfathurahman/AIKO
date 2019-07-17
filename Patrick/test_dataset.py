import os

from imagesearch import config

orig_dir = os.listdir(config.ORIG_DATASET_DIR)

for i in orig_dir:
    x = os.listdir(os.path.sep.join([config.ORIG_DATASET_DIR, i]))
    print(i, ' - ', len(x))