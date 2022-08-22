from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle
import numpy as np

planner = load_pickle('nnUNetPlansv2.1_plans_3D.pkl')

planner['plans_per_stage'][0]['patch_size'] = np.array([64,256,256])
planner['plans_per_stage'][1]['patch_size'] = np.array([64,256,256])

save_pickle(planner, "patch64x256x256_plans_3D.pkl")