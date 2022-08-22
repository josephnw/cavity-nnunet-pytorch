import glob
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pdb

parser = argparse.ArgumentParser(description='Process args')
parser.add_argument('--path', type=str, help='entry path')
parser.add_argument('--datafolder', type=str, help='input data path')
parser.add_argument('--filterfolder', type=str, help='filter path')
parser.add_argument('--gtfolder', type=str, default='', help='gt path')
parser.add_argument('--outfolder', type=str, help='outfolder path')
parser.add_argument('--filtergt', type=str, default="1", help='filter gt number')
parser.add_argument('--mergegt', default=False, action='store_true')
args = parser.parse_args()

path = args.path
datafolder = args.datafolder
filterfolder = args.filterfolder + '/'
gtfolder = args.gtfolder + '/'
outfolder = args.outfolder
filtergt = args.filtergt
merge_filter_with_gt = args.mergegt

os.makedirs(os.path.join(path, outfolder), exist_ok=True)
nii_files = glob.glob(os.path.join(path, f'{datafolder}/*.nii'))
nii_files.sort()

for nii_file in nii_files:
	print(nii_file)
	nii_input = nib.load(nii_file)
	header = nii_input.header
	# voxel_sizes = header.get_zooms()
	id_ = os.path.basename(nii_file)[:-4]
	raw_files = glob.glob(path + filterfolder + id_ + f'_gt{filtergt}.raw')  # multiple class gt per id
	if len(raw_files) == 0:
		pdb.set_trace()
	raw_files.sort()

	mask_out = np.zeros((nii_input.shape[2], nii_input.shape[1], nii_input.shape[0]))
	arr = np.transpose(np.array(nii_input.dataobj), axes=[2, 1, 0])
	
	#for i, raw_file in enumerate(raw_files, 1):
	#cls = int(raw_file[raw_file.rfind('_gt')+3:raw_file.rfind('.')])
	mask = np.fromfile(raw_files[0], dtype='uint8', sep="")

	if merge_filter_with_gt:
		gt_files = glob.glob(path + gtfolder + id_ + '_gt*.raw')  # merge filter with gt
		gt_files.sort()
		for gt_file in gt_files:
			gt = np.fromfile(gt_file, dtype='uint8', sep="")
			mask = np.logical_or(mask, gt)
	
	try:	
		mask = mask.reshape(mask_out.shape)
	except:
		pdb.set_trace()
	arr = np.where(mask, arr, -1024)
	# arr = np.transpose(arr, axes=[2, 0, 1])

	nii_np = np.transpose(arr, axes=[2, 1, 0])
	
	nii = nib.Nifti1Image(nii_np.astype(np.int16), affine=None, header=header)
	#idx = [0,1,3,2,4,5,6,7]
	#nii.header['pixdim'] = header['pixdim'][idx]
	nib.save(nii, os.path.join(outfolder.join(nii_file.rsplit(datafolder, 1)).replace('.nii', '_0000.nii.gz')))

	print(nii_file[:-4] + '_0000.nii.gz')
	#pdb.set_trace()


