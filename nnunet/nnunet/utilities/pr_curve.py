from batchgenerators.utilities.file_and_folder_operations import load_pickle
import numpy as np
# from skimage.measure import label
from sklearn.metrics import average_precision_score, auc, precision_recall_curve
import nibabel as nib
import pandas as pd
import glob
import os
from matplotlib import pyplot
from pathlib import Path

import pdb

# def precision_recall_curves(y_true, pred_scores, thresholds):
    # import sklearn.metrics

    # precisions = []
    # recalls = []

    # for threshold in thresholds:
    #     #y_pred = [1 if score >= threshold else 0 for score in pred_scores]
    #     y_pred = np.where(pred_scores >= threshold, 1, 0)

    #     precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred)
    #     recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred)

    #     precisions.append(precision)
    #     recalls.append(recall)

    # return np.array(precisions), np.array(recalls)

target_dir = './inference_modelgmm_best/'
save_dir = './pr_curve/'
gt_dir = "/home/user/Documents/y/DeepInsthink/data/meningioma/train20211019/val3d/gt/"
Path(save_dir).mkdir(parents=True, exist_ok=True)

npzs = sorted(glob.glob(os.path.join(target_dir, '*.npz')))
pkls = sorted(glob.glob(os.path.join(target_dir, '*.pkl')))
niigzs = sorted(glob.glob(os.path.join(target_dir, '*.nii.gz')))

f_list = []

for npz, pkl, niigz in zip(npzs, pkls, niigzs):
    patient_id = os.path.basename(npz).replace('.npz', '')
    softmax = np.load(npz)['softmax'][1]
    # nii = nib.Nifti1Image(np.transpose(softmax.astype(np.float32), axes=[2, 1, 0]), affine=None)
    # nib.save(nii, npz + '.nii.gz')
    # pdb.set_trace()

    properties_dict = load_pickle(pkl)
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    bbox = properties_dict.get('crop_bbox')
    
    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping).astype(np.float16)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + softmax.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = softmax
    else:
        print('no bbox')
        seg_old_size = softmax
    
    gtpath = os.path.join(gt_dir, os.path.basename(npz).replace('.npz', '_gt1.raw'))
    gt = np.fromfile(gtpath, dtype='uint8', sep="")
    # thresholds = np.linspace(0.01, 0.99, num=10)
    # gt = gt.reshape(seg_old_size.shape)
    precision, recall, thresholds = precision_recall_curve(gt, seg_old_size.flatten())
    # precision, recall = precision_recall_curves(gt, seg_old_size.flatten(), thresholds)
    avg_prec = average_precision_score(gt, seg_old_size.flatten())
    pr_auc = auc(recall, precision)
    fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    ix = np.argmax(fscore)
    print(f'{patient_id} Best Threshold={thresholds[ix]}, F-Score={fscore[ix]}')

    no_skill = len(gt[gt==1]) / len(gt)
    pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='Ratio of Positive Ex.')
    pyplot.plot(recall, precision, marker='.', label='Prediction')
    pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.legend()
    # show the plot
    # pyplot.show()
    pyplot.savefig(os.path.join(save_dir, f'{patient_id}.png'))
    pyplot.clf()
    #pdb.set_trace()
    stat = np.array([precision[:-1], recall[:-1], thresholds])
    # stat = np.array([precision, recall, thresholds])
    stat = np.transpose(stat, axes=[1,0])
    pr_df = pd.DataFrame(stat, columns=['precision', 'recall', 'threshold'])
    pr_df.to_csv(os.path.join(save_dir, f'{patient_id}.csv'))

    f_list.append([patient_id, precision[ix], recall[ix], fscore[ix], thresholds[ix], avg_prec, pr_auc, np.min(thresholds), np.max(thresholds)])
    f_df = pd.DataFrame(f_list, columns=['id', 'precision', 'recall', 'f_score', 'threshold', 'avg_prec', 'pr_auc', 'min_th', 'max_th'])
    f_df.to_csv(os.path.join(save_dir, 'f_score.csv'))

    #output = seg_old_size >= thresholds[ix]

    # raw_filename = npz.replace(target_dir, save_dir).replace('.npz', '_gt1.raw')
    # fileobj = open(raw_filename, mode='wb')
    # off = np.array(output, dtype=np.uint8)
    # off.tofile(fileobj)
    # fileobj.close()
    # pdb.set_trace()