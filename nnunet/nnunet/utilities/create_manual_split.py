import pandas as pd
import glob
import os
from batchgenerators.utilities.file_and_folder_operations import save_pickle
import numpy as np
from collections import OrderedDict
import pdb

splits = []
#tuning = pd.read_csv('tuning.csv', header=None)
#training = pd.read_csv('training.csv', header=None)
#training = training[~training[0].isin(tuning[0])]

#train_keys = np.array(training[0])
#test_keys = np.array(tuning[0])

labelsTr = glob.glob('./labelsTr/*.nii.gz')
test_keys = [fname for fname in labelsTr if 'tuning' in fname]
test_keys = np.array([str(os.path.basename(k).replace('.nii.gz', '')) for k in test_keys])
train_keys = [fname for fname in labelsTr if 'tuning' not in fname]
train_keys = np.array([str(os.path.basename(k).replace('.nii.gz', '')) for k in train_keys])

splits.append(OrderedDict())
splits[-1]['train'] = train_keys
splits[-1]['val'] = test_keys
save_pickle(splits, "splits_final.pkl")

print(splits[-1]['train'])
print(splits[-1]['val'])
print(f"numTrain: {len(splits[-1]['train'])} || numTuning: {len(splits[-1]['val'])}")
# pdb.set_trace()
