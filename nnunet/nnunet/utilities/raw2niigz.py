import glob
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb

path = './vnc/train3d/data/'
nii_files = glob.glob(os.path.join(path, '*.nii'))
os.makedirs(path.replace('data', 'gt_niigz'), exist_ok=True)

for nii_file in nii_files:
	nii_input = nib.load(nii_file)
	header = nii_input.header
	# voxel_sizes = header.get_zooms()
	id_ = os.path.basename(nii_file)[:-4]
	raw_files = []
	raw_files = glob.glob(path.replace('data', 'gt') + id_ + '_gt*.raw') # multiple class gt per id
	raw_files.sort()
	mask_out = np.zeros((nii_input.shape[2], nii_input.shape[1], nii_input.shape[0]))
	
	for i, raw_file in enumerate(raw_files, 1):
		cls = int(raw_file[raw_file.rfind('_gt')+3:raw_file.rfind('.')])
		mask = np.fromfile(raw_file, dtype='uint8', sep="")
		mask = mask.reshape(mask_out.shape)
		mask_out = np.where(mask, cls, mask_out)
		print(raw_file, cls)
	# mask_out = np.transpose(mask_out, axes=[2, 0, 1])
	nii = nib.Nifti1Image(np.transpose(mask_out.astype(np.int16), axes=[2, 1, 0]), affine=None, header=header)
	# idx = [0,1,3,2,4,5,6,7]
	# nii.header['pixdim'] = header['pixdim'][idx]
	# nib.save(nii, os.path.join(raw_file[:-8] + '.nii.gz'))
	nib.save(nii, os.path.join(nii_file.replace('data', 'gt_niigz').replace('.nii', '.nii.gz')))
	# print(raw_file[:-8] + '.nii.gz', 'saved')
	print(nii_file[:-4] + '.nii.gz', 'saved')
