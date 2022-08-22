import nibabel as nib
import glob
import numpy as np
import SimpleITK as sitk
import sys
import os
import pdb

# convert 2D nii to 3D niigz for nnunet prediction

# niigzs = glob.glob('./labelsTr5/*.nii.gz')
# niigzs = glob.glob('./xray - xray2lung - HN/*.nii')
path = sys.argv[1]
niigzs = glob.glob(os.path.join(path, '*.nii'))
os.makedirs(path.replace('nii', 'niigz'), exist_ok=True)

# is_seg = True
is_seg = False

# spacing=(999, 1, 1)
for count, nii in enumerate(niigzs):
	# output_filename_truncated = nii.replace('.nii.gz', '')
	output_filename_truncated = nii.replace('/nii', '/niigz').replace('_0000.nii.gz', '')
	nii_ = nib.load(nii)
	arr = np.array(nii_.dataobj)
	header = nii_.header
	spacing = (999, float(header['pixdim'][1]), float(header['pixdim'][2]))
	# pdb.set_trace()
	if len(arr.shape) == 4:
		arr = arr[:, :, :, 0]
	if len(arr.shape) == 2:
		arr = arr[:,:, None]
	if len(arr.shape) == 3 :
		arr = np.transpose(arr, axes=[2, 1, 0])[None]
		for j, i in enumerate(arr):

			if is_seg:
				i = i.astype(np.uint32)

			itk_img = sitk.GetImageFromArray(i)
			itk_img.SetSpacing(list(spacing)[::-1])
			if not is_seg:
				sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
			else:
				sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")

	else:
		pdb.set_trace()
	if count % 100 == 0:
		print(count, nii)
