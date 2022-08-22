import json
import os
from datetime import date
import glob
import nibabel as nib
import numpy as np
import pdb


name = os.path.split(os.getcwd())[-1]
license = "CC-BY-SA 4.0"
release_date = date.today().strftime("%d/%m/%Y")
release = f"1.0 {release_date}"
tensorImageSize= "3D"


#### Check dataset modality!! ####
modality = {'0': "CT"}
# modality = {'0': "MRI"}
print('modality', modality['0'])
#### Check dataset modality!! ####


labelsTr = glob.glob(os.path.join('.', 'labelsTr', '*.nii.gz'))
labelsTs = glob.glob(os.path.join('.', 'labelsTs', '*.nii.gz'))


#### Get the number of class labels ####
arr = np.array(nib.load(labelsTr[0]).dataobj)
cls = np.unique(arr)
labels = {str(k):f'gt_{str(v)}' for k, v in enumerate(cls)}
print('Detected class: ', cls)
#### Get the number of class labels ####


numTraining = len(labelsTr)
numTest = len(labelsTs)
print(f'numTraining: {numTraining} || numTest: {numTest}')


#### List training and test data ####
training = [{"image": label.replace('labelsTr', 'imagesTr'), "label":label} for label in labelsTr]
test = [image.replace('labelsTs', 'imagesTs') for image in labelsTs]
#### List training and test data ####

#### Export to json ####
data = {'name': name, 'description': name, 'reference': name,
    'license': license, 'release': release, 'modality': modality, 
    'labels': labels, 'numTraining': numTraining, 'numTest': numTest,
    'training': training, 'test': test
    }

filename = 'dataset.json'
with open(filename, 'w') as outfile:
    json.dump(data, outfile)

print(f'{filename} saved!')
