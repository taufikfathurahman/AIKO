import os
from imagesearch import config

orig_dir = os.listdir(config.DATASET_DIR)

sum_sp = 0
for i in orig_dir:
    x = os.listdir(os.path.sep.join([config.ORIG_DATASET_DIR, i]))
    print(i, ' - ', len(x))
    sum_sp += len(x)

print(sum_sp)
