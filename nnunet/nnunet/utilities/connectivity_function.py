import os
import glob
import argparse
import numpy as np
from skimage.measure import label
import nibabel as nib
import matplotlib.pyplot as plt
import pdb

def get_biggest_connected_region(mask, n_region=1):
    """ return n_biggest connected region -> similar to region growing in Medip """
    labels = label(mask)  # label each connected region with index from 0 - n of connected region found
    n_connected_region = np.bincount(labels.flat)  # number of pixel for each connected region
    if n_connected_region[0] != np.max(n_connected_region):  # if number of background's pixel is not the biggest
        n_connected_region[0] = np.max(n_connected_region) + 1  # make it the biggest
    biggest_regions_index = (-n_connected_region).argsort()[1:n_region + 1]  # get n biggest regions index without BG

    biggest_regions = np.array([])
    for ind in biggest_regions_index:
        if biggest_regions.size == 0:
            biggest_regions = labels == ind
        else:
            biggest_regions += labels == ind
    return biggest_regions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", type=str, required=True, help="mask directory")
    opt = parser.parse_args()

    output_path = opt.mask_dir + '_conn'
    os.makedirs(output_path, exist_ok=True)
    filenames = glob.glob(opt.mask_dir + '/*.nii.gz')

    for filename in filenames:
        nii = nib.load(filename)
        header = nii.header
        arr = np.array(nii.dataobj)
        new_arr = np.zeros_like(arr)
        noise = np.zeros_like(arr)
        final_arr = np.zeros_like(arr)

        n_cls = np.unique(arr)

        # get connected region per class and accumulate noises
        for c in n_cls:
            if c != 0:
                cls_arr = np.where(arr == c, 1, 0)
                conn_arr = get_biggest_connected_region(cls_arr)
                noise += cls_arr.astype(np.uint8) - conn_arr.astype(np.uint8)
                new_arr += np.where(conn_arr, c, 0).astype(np.uint8)

        noise_vxcount = get_biggest_connected_region(noise).sum()

        # add noises to each classes then connected region
        for c in n_cls:
            if c != 0:
                cls_arr = np.where(new_arr == c, 1, 0)

                # make sure no noise that is bigger than class prediction
                if noise_vxcount < cls_arr.sum():
                    cls_arr += noise
                    cls_arr = get_biggest_connected_region(cls_arr)
                conn_cls = np.where(cls_arr, c, 0)
                final_arr = np.where(conn_cls == c, conn_cls, final_arr)
                #final_arr += np.where(conn_arr, c, 0).astype(np.uint8)

        nii = nib.Nifti1Image(final_arr.astype(np.uint8), affine=None, header=header)
        nib.save(nii, os.path.join(output_path, os.path.basename(filename)))

        print(filename, 'saved')
        # pdb.set_trace()
