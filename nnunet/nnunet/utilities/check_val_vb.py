import os
import glob
import argparse
import numpy as np
import SimpleITK as sitk
from medpy.metric.binary import dc, sensitivity, precision
import matplotlib.pyplot as plt
import pandas as pd
import pdb


def main():
    parser = argparse.ArgumentParser(description='Iterative Fully Convolutional Network')
    parser.add_argument('--label_dir', type=str, default='./crop_isotropic_dataset/test/seg',
                        help='folder of test label')
    parser.add_argument('--pred_dir', type=str, default='./pred',
                        help='folder of pred masks')
    parser.add_argument('--label_known', action='store_true',
                        help='use this when label for predicted result is known (nnunet)')
    args = parser.parse_args()

    labels = glob.glob(os.path.join(args.label_dir, '*.nii.gz'))
    # preds = glob.glob(os.path.join(args.pred_dir, '*.nii.gz'))

    summary = pd.DataFrame()
    for n_, label in enumerate(labels):
        #label = '/home/user/Documents/data/vertebral/trainall20210923/valnofilter/iso/train/segniigz/subverse030.nii.gz'
        predict = os.path.join(args.pred_dir, os.path.basename(label))
        print(f"Process {n_}/{len(labels)} {os.path.basename(predict)} and {os.path.basename(label)}")
        
        label_arr = sitk.GetArrayFromImage(sitk.ReadImage(label))
        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(predict))
        
        sum_dict = {f'{i}_{metric}': [np.nan] for i in range(1, 21) for metric in ['dice', 'sensitivity', 'precision', 'pred_gt']}
        sum_dict['id'] = os.path.basename(label)

        for i in np.unique(label_arr)[1:]:
            li = np.where(label_arr == i, 1, 0)
            
            if args.label_known:
                val = i
                pi = np.where(pred_arr == i, 1, 0)
            else: 
                pi = np.where(li, pred_arr, 0)

                pi_unique = np.unique(pi, return_counts=True)
                
                if len(pi_unique[0]) > 1:
                    val = pi_unique[0][pi_unique[1][1:].argmax() + 1]    
                else:
                    val = -1
                pi = np.where(pred_arr == val, 1, 0)

            dice = dc(pi, li)
            sens = sensitivity(pi, li)
            prec = precision(pi, li)
            sum_dict[f'{i}_dice'] = [dice]
            sum_dict[f'{i}_sensitivity'] = [sens]
            sum_dict[f'{i}_precision'] = [prec]
            sum_dict[f'{i}_pred_gt'] = [val]
        
        sum_df = pd.DataFrame.from_dict(sum_dict)
        summary = pd.concat([summary, sum_df])
    
    dice_col = [col for col in summary.columns if 'dice' in col]
    sens_col = [col for col in summary.columns if 'sensitivity' in col]
    prec_col = [col for col in summary.columns if 'precision' in col]
    summary['avg_dc'] = summary[dice_col].mean(axis=1, skipna=True)
    summary['avg_sens'] = summary[sens_col].mean(axis=1, skipna=True)
    summary['avg_prec'] = summary[prec_col].mean(axis=1, skipna=True)
    # shift column 'id' to first position
    first_column = summary.pop('id')
    summary.insert(0, 'id', first_column)
    summary.to_csv(os.path.join(args.pred_dir, os.path.split(args.pred_dir)[-1] + '.csv'))
    #pdb.set_trace()


if __name__ == '__main__':
    main()
