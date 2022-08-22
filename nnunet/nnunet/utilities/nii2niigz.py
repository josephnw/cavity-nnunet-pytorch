import glob
import nibabel as nib
import pdb

nii_files = glob.glob('./train3d/*.nii')

for nii_file in nii_files:
	nii = nib.load(nii_file)
	nib.save(nii, nii_file[:-4] + '_0000.nii.gz')

	print(nii_file[:-4] + '.nii.gz')

