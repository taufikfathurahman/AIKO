from imagesearch import config
import hough_transform as ht

import random
import os
from imutils import paths
import random as rd
import time as tm

start = tm.time()

sp_names = os.listdir(config.ORIG_DATASET_DIR)

def execute():
    for i in range(3):
        selected_imgpaths = []
        for img_sp in sp_names:
            p = os.path.sep.join([config.ORIG_DATASET_DIR, img_sp])
            imagePaths = list(paths.list_images(p))
            selected_imgpaths += random.sample(imagePaths, int(len(imagePaths)*0.4))

        print('Selected images => ', len(selected_imgpaths))
        ht.execute(selected_imgpaths, i)

try:
    execute()
except:
    print('ERROR : Program failed executed.....')

end = tm.time()
menit = (end-start)/60
print('Time spent => ', menit)