import glob
import sys
import os
import pandas as pd
import numpy as np
import pdb

folder_name = sys.argv[1]

epoch_folders = glob.glob(os.path.join(folder_name, 'inference*/'))
summary = []

for epoch in epoch_folders:
    print(epoch)
    csv = pd.read_csv(os.path.join(epoch, f'{epoch.split(os.sep)[-2]}.csv'))
    metrics_df = csv[['avg_dc',  'avg_sens',  'avg_prec']]
    mean_list = metrics_df.mean().tolist()
    mean_list.insert(0, epoch.split(os.path.sep)[-2])
    summary.append(mean_list)

columns = metrics_df.columns.tolist()
columns.insert(0, 'epoch')

df = pd.DataFrame(summary, columns=columns)
df.to_csv(os.path.join(folder_name, "summary.csv"))
