import os
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import pdb

# command example
# python "D:\\medicalip\\trunk\\execution\\val\\val_manual_niigz.py"
# --gt_dir "Y:\\DeepInsthink\\data\\vertebral\\train_nnunet\\Task517_vb2indvb\\labelsTntemp"
# --predict_dir "X:\\weights\\vertebral\\vb2indvb_nnunet3d\\Task517_vb2indvb\\inference_modeltn_899"


def check_val_manual(opt):
    target_filenames = sorted(glob.glob(os.path.join(opt.gt_dir) + "/*.nii.gz*"))

    tp, tn, fp, fn, hausdorff, voxel_count = {}, {}, {}, {}, {}, {}

    for i, target_filename in enumerate(target_filenames):
        nii = nib.load(target_filename)
        target_allcls = np.transpose(np.array(nii.dataobj), axes=[2, 1, 0])

        nii = nib.load(target_filename.replace(opt.gt_dir, opt.predict_dir))
        predict_allcls = np.transpose(np.array(nii.dataobj), axes=[2, 1, 0])

        uniques = np.unique(target_allcls)
        uniques = uniques[uniques != 0]

        for cls in uniques:
            target_ = np.where(target_allcls == cls, 1, 0)
            predict_ = np.where(predict_allcls == cls, 1, 0)
        
            print("Target: " + target_filename + " | Class: " + str(cls))

            predict_eq_0 = predict_ == 0
            predict_eq_1 = predict_ == 1

            target_eq_0 = target_ == 0
            target_eq_1 = target_ == 1

            index_filename = os.path.basename(target_filename) + "_gt" + str(cls)

            tp[index_filename] = np.sum(np.logical_and(predict_eq_1, target_eq_1))
            tn[index_filename] = np.sum(np.logical_and(predict_eq_0, target_eq_0))
            fp[index_filename] = np.sum(np.logical_and(predict_eq_1, target_eq_0))
            fn[index_filename] = np.sum(np.logical_and(predict_eq_0, target_eq_1))

            hausdorff[index_filename] = 0
            voxel_count[index_filename] = np.sum(predict_)
        

    dices = calculate_dice(tp, fp, fn)
    sensitivity = calculate_sensitivity(tp, fn)
    precision = calculate_precision(tp, fp)
    ious = calculate_iou(tp, fp, fn)

    dices_df = pd.DataFrame.from_dict(dices, orient='index', columns=['Dice'])
    sensitivity_df = pd.DataFrame.from_dict(sensitivity, orient='index', columns=['Sensitivity'])
    precision_df = pd.DataFrame.from_dict(precision, orient='index', columns=['Precision'])
    ious_df = pd.DataFrame.from_dict(ious, orient='index', columns=['IoU'])
    hausdorff_df = pd.DataFrame.from_dict(hausdorff, orient='index', columns=['Hausdorff'])
    voxel_df = pd.DataFrame.from_dict(voxel_count, orient='index', columns=['Voxel Count'])
    # dices_df['tp'] = dices_df.index.map(tp)
    df_all_ = pd.concat([dices_df, sensitivity_df, precision_df, ious_df, hausdorff_df, voxel_df], axis=1)
    
    # pivot table for easier manual modification
    df_all_ = df_all_.reset_index()
    df_all_[['filename', 'gt']] = df_all_['index'].str.rsplit('_', 1, expand=True)
    df_all_ = df_all_[['filename', 'gt', 'Dice', 'Sensitivity', 'Precision', 'IoU', 'Voxel Count']]
    df_all_ = df_all_.pivot_table(index='filename', columns='gt')

    df_all_.to_csv(os.path.join(opt.predict_dir, 'dice_sensitivity_precision_hausdorff.csv'))
    print('dice_sensitivity_precision_hausdorff.csv saved successfully at ' + opt.predict_dir)


def calculate_dice(tp, fp, fn):
    dices = {}
    print('calculating dice...')
    for key in tp.keys():
        intersection = 2 * tp[key]
        union = (2 * tp[key] + fp[key] + fn[key])

        if union == 0:
            dice = 0
        else:
            dice = intersection / union

        dices[key] = dice

    return dices


def calculate_sensitivity(tp, fn):
    sensitivity = {}
    print('calculating sensitivity...')
    for key in tp.keys():
        intersection = tp[key]
        union = tp[key] + fn[key]

        if union == 0:
            sensitivity_ = 0
        else:
            sensitivity_ = intersection / union

        sensitivity[key] = sensitivity_

    return sensitivity


def calculate_precision(tp, fp):
    precision = {}
    print('calculating precision...')
    for key in tp.keys():
        intersection = tp[key]
        union = tp[key] + fp[key]

        if union == 0:
            precision_ = 0
        else:
            precision_ = intersection / union

        precision[key] = precision_

    return precision


def calculate_iou(tp, fp, fn):
    ious = {}
    print('calculating dice...')
    for key in tp.keys():
        intersection = tp[key]
        union = (tp[key] + fp[key] + fn[key] + 1e-10)

        if union == 0:
            dice = 0
        else:
            iou = intersection / union

        ious[key] = iou

    return ious


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True, help="gt directory")
    parser.add_argument("--predict_dir", type=str, required=True, help="predict directory")

    opt = parser.parse_args()

    check_val_manual(opt)
