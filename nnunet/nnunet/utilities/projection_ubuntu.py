import nibabel as nib
from PIL import Image
import math
import numpy as np
import glob
import os
import pandas as pd
# import matplotlib.pyplot as plt
import argparse
import pdb

def resize_keep_ratio(img, target_size):
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = img.resize(new_size, Image.LANCZOS)
    return im


def resize_pad(img, target_size=2048, bg_val=-1024, save_mask=False):
    im_c_resize = resize_keep_ratio(Image.fromarray(img), 512)
    img_c_resize = np.array(im_c_resize)
    height_c, width_c = img_c_resize.shape
    start_y = math.ceil((512 - height_c) / 2)
    start_x = math.ceil((512 - width_c) / 2)
    pad = np.full((512, 512), bg_val)
    pad[start_y:start_y + height_c, start_x:start_x + width_c] = img_c_resize
    if save_mask:
        pad = pad.astype(np.uint8)
    pad2048 = resize_keep_ratio(Image.fromarray(pad), target_size)
    return pad2048


def get_volume_area(mask3d, vesselcoronal, voxel_sizes):
    volume = np.sum(mask3d) * voxel_sizes[0] * voxel_sizes[0] * voxel_sizes[0] / 1000
    
    coronal2dmask = np.where(vesselcoronal > -1024, 1, 0)
    area = np.sum(coronal2dmask) * voxel_sizes[0] * voxel_sizes[2] / 100
    return volume, area


parser = argparse.ArgumentParser(description='Process args')
parser.add_argument('--data_folder', type=str, help='input data path')
parser.add_argument('--mask3d_folder', type=str, help='gt path')
parser.add_argument('--projection_folder', type=str, help='outfolder path')
parser.add_argument('--projection_name', type=str, help='projection_name')
args = parser.parse_args()

data_folder = args.data_folder # "/home/user/Documents/x/Link to /TisepX/nii/20220124_2"
mask3d_folder = args.mask3d_folder
# projection_folder = args.projection_folder
projection_name = args.projection_name
filenames = glob.glob(f'{data_folder}/*.nii')

os.makedirs(os.path.join(mask3d_folder, f'gt_{projection_name}'), exist_ok=True)
os.makedirs(os.path.join(mask3d_folder, f'gt_{projection_name}2048'), exist_ok=True)
os.makedirs(os.path.join(mask3d_folder, f'gt_{projection_name}mask2048'), exist_ok=True)

statistic = []

for filename in filenames:
    savefilename = os.path.basename(filename.replace('_0000', '')).split('.')[0]
    
    nii_ = nib.load(filename)
    data = np.transpose(np.array(nii_.dataobj), axes=[2, 1, 0])
    data = np.clip(data, -1024, None)
    depth = data.shape[0]
    header = nii_.header
    voxel_sizes = header.get_zooms()
    #datacoronal = np.nanmean(np.transpose(data, axes=[0, 2, 1]), axis=2)

    # gtvessel = np.fromfile(filename.replace(data_folder, 'gt_merge').replace('.nii', '_gt1.raw'), dtype='uint8', sep="")
    # gtvessel = gtvessel.reshape([depth, 512, 512])
    gtvessel = nib.load(filename.replace(data_folder, mask3d_folder) + '.gz')
    gtvessel = np.transpose(np.array(gtvessel.dataobj), axes=[2, 1, 0])
    
    vesselcoronal = np.nanmean(np.transpose(np.where(gtvessel, data, -1024), axes=[0, 2, 1]), axis=2)
    # vesselcoronal = np.where(np.isnan(vesselcoronal), -1024, vesselcoronal)
    #pdb.set_trace()
    
    nii = nib.Nifti1Image(np.transpose(vesselcoronal.astype(np.int16), axes=[1, 0]), affine=None)
    nib.save(nii, os.path.join(mask3d_folder, f'gt_{projection_name}', savefilename + '_gt1.nii'))
    
    min_val, max_val = vesselcoronal.min(), vesselcoronal.max()
    normalize_ = (vesselcoronal - min_val) / (max_val - min_val)
    resize_ = np.array(resize_pad(normalize_, bg_val=normalize_.min()))
    denormalize_ = resize_ * (max_val - min_val) + min_val
    denormalize_ = np.clip(denormalize_, -1024, None)
    nii = nib.Nifti1Image(np.transpose(denormalize_.astype(np.int16), axes=[1, 0]), affine=None)
    nib.save(nii, os.path.join(mask3d_folder, f'gt_{projection_name}2048', savefilename + '_gt1.nii'))

    #coronal_projection_mask3d = np.transpose(gtvessel, axes=[0, 2, 1])
    #coronal_projection_mask2d = np.fmin(np.sum(coronal_projection_mask3d, axis=2), 1)
    # vesselcoronal = np.where(vesselcoronal <= -1022, -1024, vesselcoronal)
    coronal_projection_mask2d = np.where(vesselcoronal > -1024, 1, 0)
    fileobj = open(os.path.join(mask3d_folder, f'gt_{projection_name}mask2048', savefilename + '_gt1.raw'), mode='wb')
    off = np.array(resize_pad(coronal_projection_mask2d.astype(np.uint8), bg_val=0., save_mask=True), dtype=np.uint8)
    off.tofile(fileobj)
    fileobj.close()
    
    # save mask as 16bit nii
    #nii = nib.Nifti1Image(np.transpose(np.array(resize_pad(coronal_projection_mask2d.astype(np.uint8), bg_val=0., save_mask=True), dtype=np.int16), axes=[1, 0]), affine=None)
    #nib.save(nii, os.path.join(projection_folder, 'pulmonarymask2048', savefilename + '_gt1.nii'))
    
    pulmonary_volume, pulmonary_area = get_volume_area(gtvessel, vesselcoronal, voxel_sizes)
    statistic.append([savefilename, pulmonary_volume, pulmonary_area, voxel_sizes, data.shape])
    
    print(filename)
    #pdb.set_trace()

statistic_df = pd.DataFrame(statistic, columns=['id', f'{projection_name}_volume', f'{projection_name}_area', 'voxel_sizes', 'shape'])
statistic_df.to_csv(os.path.join(mask3d_folder, f'statistic_{projection_name}.csv'))
