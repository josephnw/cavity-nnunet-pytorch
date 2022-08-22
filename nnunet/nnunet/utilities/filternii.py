import glob
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb

path = './vnc/train3d/'

nii_files = glob.glob(os.path.join(path, 'data/*.nii'))
nii_files.sort()

for nii_file in nii_files:
	print(nii_file)
	nii_input = nib.load(nii_file)
	header = nii_input.header
	# voxel_sizes = header.get_zooms()
	id_ = os.path.basename(nii_file)[:-4]
	raw_files = glob.glob(path + 'filter/' + id_ + '_gt7.raw') # multiple class gt per id
	raw_files.sort()
	mask_out = np.zeros((nii_input.shape[2], nii_input.shape[1], nii_input.shape[0]))
	arr = np.transpose(np.array(nii_input.dataobj), axes=[2, 1, 0])
	
	#for i, raw_file in enumerate(raw_files, 1):
	#cls = int(raw_file[raw_file.rfind('_gt')+3:raw_file.rfind('.')])
	mask = np.fromfile(raw_files[0], dtype='uint8', sep="")
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
	nib.save(nii, nii_file[:-4] + '_0000.nii.gz')

	print(nii_file[:-4] + '_0000.nii.gz')
	#pdb.set_trace()


