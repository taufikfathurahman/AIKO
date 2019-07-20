import pandas as pd
import os
import numpy as np
import time as tm

from imagesearch import config

start = tm.time()


def filter_ht(j):
    cluster_data = pd.read_excel(config.KMEAN_XLSX[j])
    my_cluster = [cluster_data['Data Kayu'], cluster_data['Wood Class']]

    clusters = {}
    for cluster_num in range(config.K):
        wood_id = []
        for idx in range(len(my_cluster[0])):
            if my_cluster[1][idx] == cluster_num:
                wood_id.append(my_cluster[0][idx].split('_')[0])
        clusters[cluster_num] = np.unique(wood_id)

    sp_id = os.listdir(config.ORIG_DATASET_DIR)
    sp_dict = {}
    for ids in sp_id:
        counted_sp = 0
        for key in clusters:
            if ids in clusters[key]:
                counted_sp += 1
        sp_dict[ids] = counted_sp

    d1 = []
    d2 = []
    count_err = 0

    for key in sp_dict:
        if sp_dict[key] < 3:
            d1.append(key)
            d2.append('-')
        else:
            d2.append(key)
            d1.append('-')
            count_err += 1

    csv_dict = {
        'Duplikasi < 3': d1,
        'Duplikasi => 3': d2
    }

    df = pd.DataFrame(csv_dict)
    df.to_excel(os.path.sep.join([config.FILTERED_KMEAN, 'filtered_ht' + str(j) + '.xlsx']), index=False)

    return count_err


def evaluate_filtered_data():
    fht0 = os.path.sep.join([config.FILTERED_KMEAN,
                             'filtered_ht' + str(0) + '.xlsx'])
    fht0 = list(pd.read_excel(fht0)['Duplikasi => 3'])

    fht1 = os.path.sep.join([config.FILTERED_KMEAN,
                             'filtered_ht' + str(1) + '.xlsx'])
    fht1 = list(pd.read_excel(fht1)['Duplikasi => 3'])

    fht2 = os.path.sep.join([config.FILTERED_KMEAN,
                             'filtered_ht' + str(2) + '.xlsx'])
    fht2 = list(pd.read_excel(fht2)['Duplikasi => 3'])

    selected_sp = []
    for i in fht0:
        if ((i in fht1) or (i in fht2)) and i != '-':
            selected_sp.append(i)

    for i in fht1:
        if ((i in fht0) or (i in fht2)) and i != '-':
            selected_sp.append(i)

    for i in fht2:
        if ((i in fht0) or (i in fht1)) and i != '-':
            selected_sp.append(i)

    selected_sp = np.unique(selected_sp)
    csv_dict = {
        'Sp will be deleted': selected_sp
    }
    df = pd.DataFrame(csv_dict)
    df.to_excel(os.path.sep.join([config.FILTERED_KMEAN, 'delete_this_sps' + '.xlsx']), index=False)


def execute(j=3):
    count_err = []
    for i in range(j):
        count_err.append(filter_ht(i))

    print('Err list => ', count_err)

    evaluate_filtered_data()


execute()

print('done.....')
end = tm.time()
minute = (end - start) / 60
print('Time spent => ', minute, ' minutes')
