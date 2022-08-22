import random
import numpy as np
from threading import Timer

import pdb
#from PIL import Image > for resizing but we dont use
import re
import os
import glob
import csv
import sys
import math
import torch
from PIL import Image
import torchvision.transforms as transforms
from math import ceil, floor

import nibabel as nib

from utils.preprocessing import add_gaussian_noise, add_uniform_noise
from scipy.ndimage.interpolation import rotate as imrotate
from scipy.ndimage.interpolation import zoom as imzoom
from scipy.ndimage.interpolation import shift as imshift
from skimage.measure import label

from utils.shared_memory import *
import datetime
import traceback

from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt

def force_shutdown(shm, message="", no_server=True):
    print("shutdown")
    f = open("error.log", "a")
    try:
        current = datetime.datetime.now()
        f.write(str(current) + " " + str(message) + "\n")
    finally:
        f.close()
    if no_server is False:
        global status
        UpdateMemory(shm, -3)
        status = -3
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def ensure_terminator(path, terminator="/"):
    if path == "":
        return path
    if path.endswith(terminator):
        return path
    return path + terminator

def lreplace(pattern, sub, string):
    """
    Replaces 'pattern' in 'string' with 'sub' if 'pattern' starts 'string'.
    """
    return re.sub('^%s' % pattern, sub, string)



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def reshape_np2tensorinput(img, flag_3d=False):
    if flag_3d:
        img = img.reshape([1,1,img.shape[0], img.shape[1], img.shape[2]]).astype(np.float32)
    else:
        img = img.reshape([img.shape[0], 1, img.shape[1], img.shape[2]]).astype(np.float32)
        #img = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]]).astype(np.float32)

    return img

def reshape_tensor2npoutput(img, flag_3d=False,n_class=1):
    if flag_3d:
        img = np.zeros((n_class, img.shape[2], img.shape[3],img.shape[4]),dtype=np.int8)
    else:
        img = np.zeros((n_class, img.shape[0], img.shape[2],img.shape[3]), dtype=np.int8)
    return img


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    pdb.set_trace()
    w = pilimg.shape[0]
    h = pilimg.shape[1]
    newW = int(w * scale)
    newH = int(h * scale)
    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    if n > 0:        
        random.shuffle(dataset)
        return {'train': dataset[:-n], 'val': dataset[-n:]}
    else:
        random.shuffle(dataset)
        return {'train': dataset[:], 'val': dataset[:-0]}

def split_train_val_withdir(dataset_val):
    dataset_val = list(dataset_val)
    return {'val': dataset_val[:]}

def normalize(x,normalization):
    ww = normalization["ww"]
    wl = normalization["wl"]
    new_norm = normalization["new_norm"]
    norm_values = normalization["norm_values"]
    standardization = normalization["standardization"]
    if(standardization):
        return standardize_norm(x)
    if(len(norm_values) > 0 and len(norm_values) % 2 == 0):
        x = x.astype(np.float32)
        nc = int(len(norm_values) / 2)
        cumulative = np.zeros(x.shape)
        for n in range(nc):
            cond = np.logical_and(x <= norm_values[n], np.logical_not(cumulative))
            if n == 0:
                x[cond] = norm_values[nc + n]
            else:
                x[cond] = norm_values[nc + n - 1] + (x[cond] - norm_values[n-1]) / (norm_values[n] - norm_values[n-1]) \
                    * (norm_values[nc + n] - norm_values[nc + n - 1] )
            cumulative = np.logical_or(cumulative, cond)
        cond = np.logical_not(cumulative)
        x[cond] = norm_values[nc*2-1]
        return x
    realMinHU = np.amin(x)
    #new wl
    #if realMinHU is 0 instead of -1024,
    #adjust wl by +1024
    if realMinHU > 100:
        wl = (realMinHU + 1024) + wl
    
    # ww wl selection
    minHU = wl - (ww / 2)
    maxHU = wl + (ww / 2)

    x = np.clip(x, minHU, maxHU)
    if new_norm:
        return ((x - minHU) * 2 / (maxHU - minHU)) -1
    return (x - minHU) / (maxHU - minHU)

    condlist = np.logical_and(x >= minHU, x <= maxHU)
    x = x*condlist
    
    minX = np.amin(x)
    if(minX < 0):
        minX = -minX
    x = x + minX

    # normalized 0~1
    minX = np.amin(x)
    maxX = np.amax(x)
    if minX == maxX:
        return x
    else:
        return (x - minX) / (maxX - minX)


def find_mean_std(directory):
    print('Finding Mean and STD of dataset')
    gathered_value = np.zeros(0)
    filenames = [f for f in glob.glob(directory + '\\*.nii')]

    for i, file in enumerate(filenames):
        nii = nib.load(file)
        a_arr = np.array(nii.dataobj).flatten()
        gathered_value = np.concatenate([gathered_value, a_arr])
        print("({}/{}) files read.".format(i+1, len(filenames)), end='\r')
        if i == len(filenames) - 1:
            print("({}/{}) files read.".format(i+1, len(filenames)))

    mean = np.mean(gathered_value)
    std = np.std(gathered_value)
    return mean, std


def standardize_norm(input_np):
    # return (input_np - mean) / std
    return (input_np - input_np.mean()) / (input_np.std() + 0.00000000001 )

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]
    pdb.set_trace()

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

def mask_to_image_binary(mask):
    return np.array(mask * 1* 1).astype(np.int8)


'''
def generate_excel43d(slice, n_class, dir_data, dir_gt, s_w_dir, Hdimen, Wdimen):

    ids = sorted(os.listdir(dir_data))
    ids.sort(key=natural_keys)

    idTrain = []

    gt_single = []

    for i in range(0,len(ids)):
        #count = 0
        fileroot = ids[i]
        sep = fileroot.split('.')
        filename=sep[0]

        positive = False

        all_gt = []
        for k in range(0,n_class):
            gtname = filename + '_gt' + str(k+1) + '.raw'
            load_gt = np.fromfile(dir_gt + gtname, dtype='uint8', sep="")
            depth = int(load_gt.shape[0] / Hdimen / Wdimen)
            load_gt = load_gt.reshape([depth, Hdimen, Wdimen])
            all_gt.append(load_gt)

        minSlice = 100000
        maxSlice = 1

        for n in range(0,depth):
            for k in range(0,n_class):
                if np.sum(all_gt[k][n]) > 0:
                    if n <= minSlice:
                        minSlice = n
                    elif n>= maxSlice:
                        maxSlice = n
                    break
        gt_single.append([filename, minSlice, maxSlice])

    for i in range(0,len(gt_single)):
        start = gt_single[i][1]
        to = gt_single[i][2]

        for j in range(start,to-slice+1):
            pdb.set_trace()
            idTrain.append([gt_single[i][0], j, j+slice-1])
    pdb.set_trace()
'''


def load_crop_data(args):
    ids = sorted(os.listdir(args.train_dir))
    ids.sort(key=natural_keys)

    train_data = []
    for id in ids:
        loadImage = np.fromfile(args.train_dir + id, dtype='int16', sep="")
        train_data.append([id, 0, loadImage.shape[0] // args.Hdimen // args.Wdimen])

    return train_data


def crop_3d_input_from_mask(img, gt, mask, padding=0):
    non_zero_indices = np.nonzero(mask)
    z_start, z_end = non_zero_indices[0].min()-padding, non_zero_indices[0].max()+padding
    x_start, x_end = non_zero_indices[1].min()-padding, non_zero_indices[1].max()+padding
    y_start, y_end = non_zero_indices[2].min()-padding, non_zero_indices[2].max()+padding

    # make x and y to have same size
    x_size = x_end - x_start
    y_size = y_end - y_start
    if x_size > y_size:
        # pdb.set_trace()
        if y_start - floor((x_size-y_size)/2) < 0:
            y_start = 0
            remainder = y_start - floor((x_size-y_size)/2) * -1
            y_end += ceil((x_size - y_size) / 2) + remainder
        elif y_end + ceil((x_size - y_size) / 2) > img.shape[-1]:
            y_end = img.shape[-1]
            remainder = y_end + ceil((x_size - y_size) / 2) - img.shape[-1]
            y_start -= floor((x_size-y_size)/2) - remainder
        else:
            y_start -= floor((x_size-y_size)/2)
            y_end += ceil((x_size - y_size) / 2)
    elif y_size > x_size:
        if x_start - floor((y_size-x_size) / 2) < 0:
            x_start = 0
            remainder = x_start - floor((y_size-x_size)/2) * -1
            x_end += ceil((y_size - x_size) / 2) + remainder
        elif x_end + ceil((y_size - x_size) / 2) > img.shape[-2]:
            x_end = img.shape[-2]
            remainder = x_end + ceil((y_size - x_size) / 2) - img.shape[-2]
            x_start -= floor((y_size-x_size)/2) - remainder
        else:
            x_start -= floor((y_size-x_size) / 2)
            x_end += ceil((y_size-x_size) / 2)

    crop_img = img[z_start:z_end, x_start:x_end, y_start:y_end]

    # crop_img_ = nib.Nifti1Image(np.transpose(crop_img.astype(np.int16), axes=[2, 1, 0]), affine=None)
    # nib.save(crop_img_, '_crop_img.nii')
    crop_gt = np.zeros(1)
    if gt is not None:
        crop_gt = np.zeros((gt.shape[0], z_end-z_start, x_end-x_start, y_end-y_start))
        for i in range(gt.shape[0]):
            crop_gt_ = gt[i, z_start:z_end, x_start:x_end, y_start:y_end]
            # print(crop_gt.shape, crop_gt_.shape)
            crop_gt[i] = crop_gt_
            # crop_gt_ = nib.Nifti1Image(np.transpose(crop_gt_.astype(np.uint8), axes=[2, 1, 0]), affine=None)
            # nib.save(crop_gt_, str(i) + '_crop_gt.nii')

    # pdb.set_trace()
    crop_coordinate = (z_start, z_end, x_start, x_end, y_start, y_end)
    return crop_img, crop_gt, crop_coordinate


def resize3d(img, gt, args):
    img_resize_xy = np.zeros((img.shape[0], args.volume[1], args.volume[2]))
    for depth in range(img.shape[0]):
        temp = Image.fromarray((img[depth]))
        img_resize_xy[depth] = temp.resize((args.volume[1], args.volume[2]), Image.LANCZOS)

    # img_resize_ = nib.Nifti1Image(np.transpose(img_resize_xy.astype(np.int16), axes=[2, 1, 0]), affine=None)
    # nib.save(img_resize_, '_resize_img.nii')

    gt_resize_xy = np.zeros((gt.shape[0], gt.shape[1], args.volume[1], args.volume[2]))
    for n_class in range(gt.shape[0]):
        for depth in range(gt.shape[1]):
            temp = Image.fromarray(gt[n_class, depth])
            gt_resize_ = temp.resize((args.volume[1], args.volume[2]), Image.LANCZOS)
            gt_resize_xy[n_class, depth] = gt_resize_
        # gt_resize_ = nib.Nifti1Image(np.transpose(gt_resize_xy[n_class], axes=[2, 1, 0]), affine=None)
        # nib.save(gt_resize_, str(n_class) + '_resize_gt.nii')

    pdb.set_trace()
    return img_resize_xy, gt_resize_xy


def resize3dzoom(img, gt, args, threshold=0.2):
    x_ratio = min(args.volume[2], img.shape[2]) / max(args.volume[2], img.shape[2])
    y_ratio = min(args.volume[1], img.shape[1]) / max(args.volume[1], img.shape[1])
    z_ratio = min(args.volume[3], img.shape[0]) / max(args.volume[3], img.shape[0])
    img_resize = imzoom(img, zoom=(z_ratio, y_ratio, x_ratio))

    # img_resize_ = nib.Nifti1Image(np.transpose(img_resize.astype(np.int16), axes=[2, 1, 0]), affine=None)
    # nib.save(img_resize_, '_zoom_img.nii')
    gt_resize = np.zeros(1)
    if gt is not None:
        gt_resize = np.zeros((gt.shape[0], args.volume[3], args.volume[1], args.volume[2]))
        for n_class in range(gt.shape[0]):
            # temp = Image.fromarray(gt[n_class, depth])
            gt_resize_temp = imzoom(gt[n_class], zoom=(z_ratio, y_ratio, x_ratio))
            gt_resize[n_class] = np.where(gt_resize_temp > threshold, 1, 0)
            # gt_resize_ = nib.Nifti1Image(np.transpose(gt_resize[n_class], axes=[2, 1, 0]), affine=None)
            # nib.save(gt_resize_, str(n_class) + '_zoom_gt.nii')

    # pdb.set_trace()
    return img_resize, gt_resize


def generate_excel42d_perslice(args,state, shm):#padding
    #slice = state['slice']
    slice = args.cube[2]
    n_class = state['n_class']
    boxing = state['boxing']

    dir_data=args.train_dir
    dir_gt=args.groundtruth_dir
    height=args.Hdimen
    width=args.Wdimen
    check_dataset=args.check_dataset
    no_server=args.no_server
    nii_input=args.nii_input
    '''
    pseudococode of generating excel:
    1. list all of training data
    2. input the list of all training data into dictionary with format:
    dict --> |patient id | number of slice |
    dictMin --> |patient id | starting minimum slice |
    3. if check is true:
        a. generate idPos and idNeg from beginning
            a.1 use dict and dictMin to generate id pos dictionary with format
            |patient id | start slice | end slice
        b. load train_idPos.csv and train_idNeg.csv
    4. save
    5. return idPos,idNeg
    '''

    #check all data in train dir
    ids = sorted(os.listdir(dir_data))
    ids.sort(key=natural_keys)

    idPos = []
    idNeg = []

    gt_single = []

    #store id and number of slice
    dict = {}

    #sotre id and the starting minimum slice
    dictMin = {}

    # store file ext
    dictExt = {}

    #get the patientid and the number of slices per patient
    for i in range(0,len(ids)):
        sep = ids[i][:-4].split('_')
        separator = '_'

        filename = separator.join(sep[:-2])

        if filename in dict:
            dict[filename] = dict.get(filename,0) + 1
        else:
            dict[filename] = 0
            dictMin[filename] = int(sep[-1])
            dictExt[filename] = ids[i][-4:]
            # print(filename)
            # print(sep[-1])

    dict_trainDist = {}

    #request to load fom csv as a data multiplier for each dataset per patient)
    #ex: patient1 --> 2 times, patient 1--> 3 times larger than normal
    if os.path.isfile(dir_data + "../traindist.csv"):
        print("Will use data multiplier from csv")
        with open(dir_data + "../traindist.csv", 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[0] in dict_trainDist:
                    pass
                else:
                    dict_trainDist[row[0]] = row[1]

    else:
        print("Do not use data multiplier from csv")
        dict_trainDist = {}

    # if check argument is active, then generate train csv.
    posFilename = dir_data + "../train_idPos_" + str(slice) + ".csv"
    negFilename = dir_data + "../train_idNeg_" + str(slice) + ".csv"
    posFile = Path(posFilename)
    negFile = Path(negFilename)
    
    if check_dataset == False and posFile.is_file() and negFile.is_file():  
        #if no check argument and csv already exists
        print('Skip generating data sample - Loading data from csv file')
        #print("To do: check whether the slice from arguments match with the csv. If not, need to regenerate csv")
        with open(posFilename, "r") as f:
            reader = csv.reader(f)
            idPos = list(reader)
        with open(negFilename, "r") as f:
            reader = csv.reader(f)
            idNeg = list(reader)
    else:
        if check_dataset == False:
            print('Csv for slice ' + str(slice) + ' does not exist.')
        print('Generating dataset')
        for i in range(0, len(dict)):  # Loop through patient id
            patient_id = list(dict)[i]
            end_patient = list(dict.values())[i]

            start_patient = list(dictMin.values())[i]
            total_slice = start_patient + end_patient + 1
            
            #Joseph's todo
            #A. loop within all, keeping track of:
            #- n number of data
            #- how many positive slices
            #if positive slices > 1, add to idPos, k number of times
            #k = int(dict_trainDist.get(patient_id,1))
            #else, add to idNeg
            #B. try to add bounding box if needed for box segmentation
            #if not needed, fill bounding box with all values to be zero or null
            MIN_VALUE = -1
            MAX_VALUE = 10000
            yxmin = np.zeros([2,slice]) #ymin, xmin
            yxmax = np.zeros([2,slice]) #ymax, xmax
            posvalue = np.zeros([slice])
            k = int(dict_trainDist.get(patient_id,1))

            #circular for example start_patient = 10, slice = 3
            #then, posvalue table
            #slice 10 -> overwrite -> 13
            #slice 11 -> overwrite -> 14
            #slice 12 -> overwrite -> 15
            nii_input = False
            if dictExt[patient_id] == '.nii':
                nii_input = True
            for j in range(start_patient, total_slice):
                if nii_input:
                    data_name = patient_id + '_slice_' + str(j) + '.nii'
                    nii_ = nib.load(dir_data + data_name)
                    load_data = np.array(nii_.dataobj)
                    # nii_height = load_data.shape[0]
                    # nii_width = load_data.shape[1]
                    load_data = load_data.flatten()
                else:
                    data_name = patient_id + '_slice_' + str(j) + '.raw'
                    load_data = np.fromfile(dir_data + data_name, dtype='int16', sep="")
                # pdb.set_trace()

                if load_data.shape[0] != (height * width):#currently require exact same size
                    # print(data_name + ' must have size of ' + str(height) + 'x' + str(width))
                    print(data_name + ' is not ' + str(height) + 'x' + str(width))
                    # force_shutdown(shm, traceback.format_exc(), no_server)
                index = j - start_patient
                
                #important
                posvalue[index % slice] = 0
                yxmin[0, index % slice], yxmin[1, index % slice] = [MAX_VALUE, MAX_VALUE]
                yxmax[0, index % slice], yxmax[1, index % slice] = [MIN_VALUE, MIN_VALUE]
                #START POSITIVE CHECK
                for cls in range(0, n_class):  # Loop through all gt
                    #for cls in range(2, 4):  # Loop through specific gt
                    
                    #skip 0 if there are multiple classes
                    '''
                    if cls == 0 and n_class > 1:
                        print("skip")
                        pass
                    '''
                    gtname = patient_id + '_slice_' + str(j) + '_gt' + str(cls+1) + '.raw'
                    load_gt = np.fromfile(dir_gt + gtname, dtype='uint8', sep="")

                    if load_data.shape[0] != load_gt.shape[0]:  # check if data and gt have same shape
                        print('shape mismatch ' + data_name + ' - ' + gtname)
                        force_shutdown(shm, traceback.format_exc(), no_server)

                    if np.sum(load_gt) > 0:  # mark positive sample
                        posvalue[index % slice] = 1
                        
                        if boxing:
                            load_gt = load_gt.reshape([height, width])
                            withmask = np.array(np.where(load_gt))
                            
                            yxmin[0, index % slice], yxmin[1, index % slice] = np.amin(withmask, axis=1)
                            yxmax[0, index % slice], yxmax[1, index % slice] = np.amax(withmask, axis=1)
                            #ymin, xmin = np.amin(withmask, axis=1)
                            #ymax, xmax = np.amax(withmask, axis=1)
                
                if index >= slice - 1:#already finishes 1 iteration
                    if np.sum(posvalue > 0):
                        for n in range(k):
                            if boxing:
                                idPos.append([patient_id, j - slice + 1, j,
                                    int(yxmin[0, index % slice]), int(yxmin[1, index % slice]),
                                    int(yxmax[0, index % slice]), int(yxmax[1, index % slice]),
                                    int(yxmax[0, index % slice] - yxmin[0, index % slice] + 1),
                                    int(yxmax[1, index % slice] - yxmin[1, index % slice] + 1)])
                                # idPos is patient_id, first slice, end slice, y min, x min, y max, x max, height, width
                            else:
                                idPos.append([patient_id, j - slice + 1, j, dictExt[patient_id]])
                    else:
                        idNeg.append([patient_id, j - slice + 1, j, dictExt[patient_id]])
                        #double proportion of negative sample in lower half
                        '''
                        if (j - slice) > (total_slice/2) :
                            idNeg.append([patient_id, j - slice + 1, j])
                        '''
                    #pdb.set_trace()
            
            # TODO check dataset
            # Check if data has correct shape base on height and weight
            # Additionally, mark positive sample
            '''
            idPat = []
            positive = np.zeros(total_slice)
            for j in range(start_patient, total_slice):  # Loop through patient slices
                data_name = patient_id + '_slice_' + str(j) + '.raw'
                load_data = np.fromfile(dir_data + data_name, dtype='int16', sep="")

                #if load_data.shape[0] % (height) != 0 or load_data.shape[0] % (width) != 0:
                if load_data.shape[0] != (height * width):#currently require exact same size
                    print(data_name + ' must have size of ' + str(height) + 'x' + str(width))
                    force_shutdown(shm, traceback.format_exc(), no_server)

                for cls in range(0, n_class):  # Loop through gt
                    #if cls == 0:
                    gtname = patient_id + '_slice_' + str(j) + '_gt' + str(cls+1) + '.raw'
                    load_gt = np.fromfile(dir_gt + gtname, dtype='uint8', sep="")

                    if load_data.shape[0] != load_gt.shape[0]:  # check if data and gt have same shape
                        print('shape mismatch ' + data_name + ' - ' + gtname)
                        force_shutdown(shm, traceback.format_exc(), no_server)

                    if np.sum(load_gt) > 0:  # mark positive sample
                        positive[j] = 1
                        load_gt = load_gt.reshape([height, width])
                        withmask = np.array(np.where(load_gt))
                        
                        ymin, xmin = np.amin(withmask, axis=1)
                        ymax, xmax = np.amax(withmask, axis=1)
                        
                        # break
            # END TO DO

            rangeLoop = np.argwhere(positive)  # select index that only has groundtruth
            '''
            '''
            if slice%2==1:
                before = int((slice-1)/2)
                after = int((slice-1)/2)
            else:
                before = int((slice)/2)
                after = int((slice)/2) - 1
            '''
            '''
            # if no groundtruth --> all become idNeg
            if len(rangeLoop) == 0:
                for j in range(start_patient,total_slice):
                    idNeg.append([patient_id, j, j + slice - 1])
            else: #if not, generate training id positive
                start = int(rangeLoop[0])

                prev = start-1
                startDict = {}
                startDict[0] = start
                toDict = {}

                for j in range(0,len(rangeLoop)):
                    vls = int(rangeLoop[j])
                    # pdb.set_trace()
                    if prev + 1 != vls:
                        startDict[len(startDict)] = vls
                        toDict[len(toDict)]=int(rangeLoop[j-1])
                    prev = vls
                toDict[len(toDict)] = int(rangeLoop[-1])

                for j in range(0,len(startDict)):
                    firstStart = startDict[j] - padding
                    if firstStart < 0:
                        firstStart = 0


                    # max 100, slice 6, last 95 - 100
                    #gt: 0-5 -> 0,5 1,6 2,7 [5 + 2 - 6 + 1]
                    #gt: 95-100 -> 93,98, 94,99 95,100
                    #pdb.set_trace()
                    lastStart = toDict[j] + padding - slice + 1
                    if lastStart > total_slice - slice + 1:#97 > 95
                        lastStart = (total_slice - slice) + 1

                    #basis 0: misal toDict
                    #basis 0: (lastStart) 95 - 100
                    #multiplier per patient
                    for k in range(firstStart,lastStart+1):
                        idPat.append([patient_id, k, k + slice - 1])
                        #k = 0, slice=6 -> 0, 5

                k = int(dict_trainDist.get(patient_id,1))
                #k = int(len(idPat)*k)

                selected = []

                if k==0:
                    idPat = []
                elif k==1:
                    selected=[]
                    idPat = idPat + selected
                else:
                    for mp in range(1,k+1):
                        selected = selected + idPat
                    idPat = idPat + selected

                idPos = idPos + idPat

                # pdb.set_trace()

                #negative
                negative = positive[start_patient:total_slice]==0
                padding = 0
                rangeLoop = np.argwhere(negative)

                if len(rangeLoop) == 0:
                    pass
                else:
                    start = int(rangeLoop[0])
                    prev = start - 1
                    startDict = {}
                    startDict[0] = start
                    toDict = {}

                    for j in range(0, len(rangeLoop)):
                        vls = int(rangeLoop[j])
                        # pdb.set_trace()
                        if prev + 1 != vls:
                            startDict[len(startDict)] = vls
                            toDict[len(toDict)] = int(rangeLoop[j - 1])
                        prev = vls
                    toDict[len(toDict)] = int(rangeLoop[-1])

                    for j in range(0, len(startDict)):
                        firstStart = startDict[j] - padding + start_patient
                        if firstStart < 0:
                            firstStart = 0

                        # max 100, slice 6, last 95 - 100
                        # gt: 0-5 -> 0,5 1,6 2,7 [5 + 2 - 6 + 1]
                        # gt: 95-100 -> 93,98, 94,99 95,100
                        lastStart = toDict[j] + padding - slice + 1 + start_patient
                        if lastStart > total_slice - slice + 1:  # 97 > 95
                            lastStart = (total_slice - slice) + 1

                        # basis 0: misal toDict
                        # basis 0: (lastStart) 95 - 100
                        for k in range(firstStart, lastStart + 1):
                            idNeg.append([patient_id, k, k + slice - 1])
                            # k = 0, slice=6 -> 0, 5
            '''
            p = float(i+1) / len(dict)
            sys.stdout.write("\rGenerating data ({0:.0f}/{1:.0f}): {2:.0f} %".format(i+1, len(dict), p*100))
            sys.stdout.flush()

        # TODO write 2 files --> idPos, idNeg.csv
        f = open(posFilename, "w")
        for i in range(0, len(idPos)):
            if boxing:
                f.write('{0:},{1:d},{2:d},{3:d},{4:d},{5:d},{6:d},{7:d},{8:d}'.format(
                    idPos[i][0], idPos[i][1], idPos[i][2], idPos[i][3], idPos[i][4],
                    idPos[i][5], idPos[i][6], idPos[i][7], idPos[i][8]) + "\n")
            else:
                f.write('{0:},{1:},{2:},{3:}'.format(idPos[i][0], idPos[i][1], idPos[i][2], idPos[i][3]) + "\n")
        f.close()

        f = open(negFilename, "w")
        for i in range(0, len(idNeg)):
            f.write('{0:},{1:},{2:},{3:}'.format(idNeg[i][0], idNeg[i][1], idNeg[i][2], idNeg[i][3]) + "\n")
        f.close()
        # END TO DO
        

    return idPos, idNeg


def generate_excel42d_patches(args):
    filenames = glob.glob(args.train_dir + "/*.nii")
    idPos, idNeg = [], []
    for filename in filenames:
        nii = nib.load(filename)
        nii_np = np.array(nii.dataobj)
        nii_np = np.transpose(nii_np, axes=[2, 1, 0])  # z, y, x
        z, y, x = nii_np.shape
        patch_size = args.patches_size
        # patch_size = 20

        try:
            raw = np.fromfile(args.groundtruth_dir + os.path.basename(filename)[:-4] + '_gt1.raw', dtype='int8', sep="")
            gt_class = 0
        except:
            raw = np.fromfile(args.groundtruth_dir + os.path.basename(filename)[:-4] + '_gt2.raw', dtype='int8', sep="")
            gt_class = 1

        raw = raw.reshape(nii_np.shape)

        for i in range(z):
            if (args.arch[-2:] == '3d') and (i + patch_size > z):
                break
            for j in range(y):
                if j+patch_size < y:
                    for k in range(x):
                        if k+patch_size < x:
                            if args.arch[-2:] == '3d':
                                patch = raw[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                                pos_sample = patch[patch_size//2, patch_size//2, patch_size//2] == 1
                            else:
                                # print(patch[patch_size//2, patch_size//2])
                                patch = raw[i, j:j+patch_size, k:k+patch_size]
                                pos_sample = patch[patch_size // 2, patch_size // 2] == 1

                            if pos_sample:
                                idPos.append((filename, i, j, k, gt_class))
                                # plt.imshow(raw[i, j:j+patch_size, k:k+patch_size])
                                # plt.show()
                            else:
                                idNeg.append((filename, i, j, k, 2))
        print(filename)
    # idPos_df = pd.DataFrame(idPos, columns=['filename', 'z', 'y', 'x', 'class'])
    # idNeg_df = pd.DataFrame(idNeg, columns=['filename', 'z', 'y', 'x', 'class'])

    # idPos_df.to_csv(os.path.dirname(os.path.dirname(filename)) + '/train_idPos.csv')
    # idNeg_df.to_csv(os.path.dirname(os.path.dirname(filename)) + '/train_idNeg.csv')
    return idPos, idNeg


def generate_excel42d_fullpatch(args):
    idPos = glob.glob(args.train_dir + "/*.nii")
    idNeg = []
    return idPos, idNeg

def crop_center(img, cropx, cropy):
    _, y, x = img.shape
    start_x = x // 2 - (cropx // 2)
    start_y = y // 2 - (cropy // 2)
    return img[:, start_y:start_y + cropy, start_x:start_x + cropx]


def data_augmentation(input, gt, n_class, height, width, background, args):
    '''gaussian_max_std=0, uniform_max_range=0,
                      max_rotation_degree=0, max_zoom_in_percentage=0, flip_lr=0'''
    gaussian_max_std = args.gaussian_max_std
    uniform_max_range = args.uniform_max_range
    flip_lr = args.flip_lr
    flip_ud = args.flip_ud
    max_rotation_degree = args.max_rotation_degree
    max_zoom_in_percentage = args.max_zoom_in_percentage
    max_zoom_out_percentage = args.max_zoom_out_percentage
    max_shift_percentage = args.max_shift_percentage

    augmented_img = np.copy(input)
    augmented_mask = np.copy(gt)

    if input.ndim < 4:  # if dataset 2d
        augmented_img = augmented_img[:, np.newaxis, :, :]
        augmented_mask = augmented_mask[:, np.newaxis, :, :]

    choices = []
    if gaussian_max_std > 0 or uniform_max_range > 0:
        choices.append('noise')
    if max_rotation_degree > 0:
        choices.append('rotate')
    if max_zoom_in_percentage > 0:
        choices.append('zoom')
    if flip_lr > 0:
        choices.append('flip_lr')
    if flip_ud > 0:
        choices.append('flip_ud')
    if max_zoom_out_percentage > 0:
        choices.append('zoom_out')
    if max_shift_percentage > 0:
        choices.append('shift')

    selected_augmentation = np.random.choice(choices, len(choices))
    # selected_augmentation = choices # for debugging
    # print('selected augmentation ', selected_augmentation, sep=': ')

    if 'noise' in selected_augmentation:
        if gaussian_max_std != 0 and uniform_max_range != 0:
            selected_noise = np.random.choice(['gaussian', 'uniform'], 1, p=[0.5, 0.5])
            if selected_noise == 'gaussian':
                uniform_max_range = 0
            else:
                gaussian_max_std = 0

        if gaussian_max_std != 0:
            gaus_sd, gaus_mean = random.randint(0, gaussian_max_std), 0
            augmented_img = add_gaussian_noise(augmented_img, gaus_mean, gaus_sd)
            # print('gaussian noise with: ', gaus_sd, ', ', gaus_mean)

        elif uniform_max_range != 0:
            noise_range = random.randint(0, uniform_max_range)
            augmented_img = add_uniform_noise(augmented_img, -noise_range, noise_range)
            # print("uniform noise with: ", noise_range)

    if 'flip_lr' in selected_augmentation and flip_lr == 1:
        #print('fliplr =', flip_lr)
        fliplr = np.zeros(augmented_img.shape)
        for depth in range(augmented_img.shape[1]):
            fliplr[0, depth, :, :] = np.fliplr(augmented_img[0, depth, :, :])
        augmented_img = fliplr[:, :, :, :]
        # fliplr_ = nib.Nifti1Image(np.transpose(augmented_img[0, :, :, :].astype(np.int16), axes=[2, 1, 0]), affine=None)
        # nib.save(fliplr_, '_fliplr.nii')

        fliplr_gt = np.zeros(augmented_mask.shape)
        # print(fliplr_gt.shape)
        for num_class in range(n_class):
            for depth in range(augmented_mask.shape[1]):
                fliplr_gt[num_class, depth, :, :] = np.fliplr(augmented_mask[num_class, depth, :, :])
        augmented_mask = fliplr_gt[:,:,:,:]
        # fileobj = open('_fliplr_gt0.raw', mode='wb')
        # off = np.array(augmented_mask[0, :, :, :], dtype=np.uint8)
        # off.tofile(fileobj)
        # fileobj.close()

        # print('flip lr')

    if 'flip_ud' in selected_augmentation and flip_ud == 1:
        #print('flipud =', flip_ud)
        flipud = np.zeros(augmented_img.shape)
        for depth in range(augmented_img.shape[1]):
            flipud[0, depth, :, :] = np.flipud(augmented_img[0, depth, :, :])
        augmented_img = flipud[:, :, :, :]
        # flipud_ = nib.Nifti1Image(np.transpose(augmented_img[0, :, :, :].astype(np.int16), axes=[2, 1, 0]), affine=None)
        # nib.save(flipud_, '_flipud.nii')

        flipud_gt = np.zeros(augmented_mask.shape)
        # print(flipud_gt.shape)
        for num_class in range(n_class):
            for depth in range(augmented_mask.shape[1]):
                flipud_gt[num_class, depth, :, :] = np.flipud(augmented_mask[num_class, depth, :, :])
        augmented_mask = flipud_gt[:,:,:,:]
        # fileobj = open('_flipud_gt0.raw', mode='wb')
        # off = np.array(augmented_mask[0, :, :, :], dtype=np.uint8)
        # off.tofile(fileobj)
        # fileobj.close()

        # print('flip ud')

    if 'rotate' in selected_augmentation:
        rotation_degree = random.randint(-max_rotation_degree, max_rotation_degree)
        augmented_img[0, :, :, :] = imrotate(augmented_img[0, :, :, :], rotation_degree, axes=(1, 2), reshape=False,
                                             cval=0)

        for num_class in range(n_class):
            augmented_mask[num_class, :, :, :] = imrotate(augmented_mask[num_class, :, :, :], rotation_degree,
                                                          axes=(1, 2),
                                                          reshape=False, cval=0)
        # print('rotation with degree: ', rotation_degree)

    if 'zoom' in selected_augmentation:
        random_zoom_in_percentage = random.randint(0, max_zoom_in_percentage)
        center_area_to_keep = 100 - random_zoom_in_percentage  # how many percent of center area to keep
        crop_center_img = crop_center(augmented_img[0, :, :, :], int(width * center_area_to_keep / 100),
                                      int(height * center_area_to_keep / 100))
        augmented_img[0, :, :, :] = imzoom(crop_center_img,
                                           zoom=(1, width / crop_center_img.shape[1],
                                                 height / crop_center_img.shape[2]))
        for num_class in range(n_class):
            crop_center_mask = crop_center(augmented_mask[num_class, :, :, :],
                                           int(width * center_area_to_keep / 100),
                                           int(height * center_area_to_keep / 100))
            augmented_mask[num_class, :, :, :] = imzoom(crop_center_mask,
                                                        zoom=(1, width / crop_center_mask.shape[1],
                                                              height / crop_center_mask.shape[2]))
        # print('zoom with percentage: ', random_zoom_in_percentage)

    if 'zoom_out' in selected_augmentation:
        random_zoom_out_percentage = random.randint(0, max_zoom_out_percentage)
        zoom_out = imzoom(augmented_img[0, :, :, :],
                                           zoom=(1, (100-random_zoom_out_percentage)/100,
                                                 (100 - random_zoom_out_percentage)/100))
        for channel in range(zoom_out.shape[0]):
            zoom_out = Image.fromarray(zoom_out[channel])
            zoom_out_pad = pad_image(zoom_out, width, pad_value=int(augmented_img.min()))
            augmented_img[0, channel, :, :] = zoom_out_pad(zoom_out)

        for num_class in range(n_class):
            zoom_out_mask = imzoom(augmented_mask[num_class, :, :, :],
                                   zoom=(1,  (100-random_zoom_out_percentage)/100,
                                         (100 - random_zoom_out_percentage)/100))
            for channel in range(zoom_out_mask.shape[0]):
                zoom_out_mask = Image.fromarray(zoom_out_mask[channel])
                zoom_out_pad = pad_image(zoom_out_mask, width, pad_value=int(augmented_mask.min()))
                augmented_mask[num_class, channel, :, :] = zoom_out_pad(zoom_out_mask)

        # print('zoom out with percentage: ', random_zoom_out_percentage)
    if 'shift' in selected_augmentation:
        shift_lr_percentage = random.randint(-max_shift_percentage, max_shift_percentage)
        shift_ud_percentage = random.randint(-max_shift_percentage, max_shift_percentage)

        shift_img = imshift(augmented_img[0], shift=[0, shift_lr_percentage, shift_ud_percentage], cval=int(augmented_mask.min()))
        # plt.subplot(1,2,1)
        # plt.imshow(shift_img[0])
        # plt.subplot(1,2,2)
        # plt.imshow(augmented_img[0,0])
        # plt.show()
        augmented_img[0] = shift_img
        for num_class in range(n_class):
            shift_mask = imshift(augmented_mask[num_class], shift=(0, shift_lr_percentage, shift_ud_percentage), cval=int(augmented_mask.min()))
            augmented_mask[num_class] = shift_mask

        # print('shift image: ', shift_lr_percentage, shift_ud_percentage)

    augmented_mask[augmented_mask < 0.6] = 0
    augmented_mask[augmented_mask > 0.6] = 1

    if background:
        background = np.ones([augmented_mask.shape[1], height, width])
        for num_class in range(0, n_class):
            temp_background = augmented_mask[num_class, :, :, :] == 0
            background = temp_background * background
        augmented_mask[-1, :, :, :] = background

    if input.ndim < 4:  # if dataset2d
        augmented_img = augmented_img[:, 0, :, :]
        augmented_mask = augmented_mask[:, 0, :, :]

    # print(augmented_img.shape, augmented_mask.shape, sep=' - ')
    return augmented_img, augmented_mask


def resize_keep_ratio(img, target_size, use_ceil=False):
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(target_size) / max(old_size)
    if int(max(old_size) * ratio) <= target_size: use_ceil = True
    if use_ceil:
        new_size = tuple([ceil(x * ratio) for x in old_size])
    else:
        new_size = tuple([int(x * ratio) for x in old_size])

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = img.resize(new_size, Image.LANCZOS)
    return im


def pad_image(img, target_size, target_size_w=None, pad_value=0):
    old_size = img.size
    target_size_w = target_size if not target_size_w else target_size_w
    pad_size_w = (target_size_w - old_size[0]) / 2
    pad_size_h = (target_size - old_size[1]) / 2

    if pad_size_w % 2 == 0:
        wl, wr = int(pad_size_w), int(pad_size_w)
    else:
        wl = ceil(pad_size_w)
        wr = floor(pad_size_w)

    if pad_size_h % 2 == 0:
        ht, hb = int(pad_size_h), int(pad_size_h)
    else:
        ht = ceil(pad_size_h)
        hb = floor(pad_size_h)

    return transforms.Compose(
        [
            transforms.Pad((wl, ht, wr, hb), fill=pad_value),
            # transforms.ToTensor(),
        ]
    )


def get_biggest_connected_region(mask3d, n_region=2):
    if np.sum(mask3d) == 0:
        return mask3d
    """ return n_biggest connected region -> similar to region growing in Medip """
    labels = label(mask3d)  # label each connected region with index from 0 - n of connected region found
    n_connected_region = np.bincount(labels.flat)  # number of pixel for each connected region
    if n_connected_region[0] != np.max(n_connected_region):  # if number of background's pixel is not the biggest
        n_connected_region[0] = np.max(n_connected_region) + 1  # make it the biggest
    biggest_regions_index = (-n_connected_region).argsort()[1:n_region+1]  # get n biggest regions index without BG

    biggest_regions = np.array([])
    for ind in biggest_regions_index:
        if biggest_regions.size == 0:
            biggest_regions = labels == ind
        else:
            biggest_regions += labels == ind
    return biggest_regions

def get_highest_sum_probability_region(mask3d, prob3d, n_candidate=10):
    if np.sum(mask3d) == 0:
        return mask3d
    """ return region with highest sum of probability in prob3d"""
    labels = label(mask3d)  # label each connected region with index from 0 - n of connected region found
    n_connected_region = np.bincount(labels.flat)  # number of pixel for each connected region
    biggest_regions_index = (-n_connected_region).argsort()[1:n_candidate+1]  # get n biggest regions index without BG

    #biggest_regions = np.array([])
    max_sum = 0
    for ind in biggest_regions_index:
        current_region = prob3d * (labels == ind)
        # COUNT SUM
        prob_sum = np.sum(current_region)
        if prob_sum > max_sum:
            max_sum = prob_sum
            max_ind = ind
        '''
        if biggest_regions.size == 0:
            biggest_regions = labels == ind
        else:
            pdb.set_trace()
            biggest_regions += labels == ind
        '''
    #pdb.set_trace()
    return labels == max_ind

def lung_lobe_boundary_correction(result_mask):
    # https://link.springer.com/content/pdf/10.1007/s10278-019-00223-1.pdf
    print('lung lobe boundary correction...')

    fileobj = open('result_mask.raw', mode='wb')
    off = np.array(result_mask, dtype=np.int8)
    off.tofile(fileobj)
    fileobj.close()

    combined_mask = np.zeros_like(result_mask[0])
    for i in range(result_mask.shape[0]):
        temp_mask = np.where(result_mask[i] == 1, i+1, 0)
        combined_mask += temp_mask

    # fileobj = open('combined_mask.raw', mode='wb')
    # off = np.array(combined_mask, dtype=np.int8)
    # off.tofile(fileobj)
    # fileobj.close()
    count = 0
    for mask in result_mask:
        count += 1
        print("Class: " + str(count))

        labels = label(mask)
        n_connected_region = np.bincount(labels.flat)

        if len(n_connected_region) > 2:
            connected_regions_index = (-n_connected_region).argsort()[2:]  # get sorted index of connected region
            for index_region in connected_regions_index:
                # print(index_region)
                list_pixel_index = np.where(labels == index_region)

                # Get cube coordinate
                z_min = list_pixel_index[0].min() - 1
                z_max = list_pixel_index[0].max() + 1
                y_min = list_pixel_index[1].min() - 1
                y_max = list_pixel_index[1].max() + 1
                x_min = list_pixel_index[2].min() - 1
                x_max = list_pixel_index[2].max() + 1

                class_region = np.where(labels == index_region, 1, 0)
                class_boundary = np.zeros_like(class_region)
                class_boundary[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = 1  # Cube around ROI
                class_boundary = np.where(class_region, 0, class_boundary)  # Cube around ROI without ROI
                class_boundary = np.where(combined_mask != 0, class_boundary, 0)  # Cube inside Lung only

                # New class voting
                class_boundary_value = np.where(class_boundary, combined_mask, np.nan)
                unique_elements, counts_elements = np.unique(class_boundary_value[~np.isnan(class_boundary_value)], return_counts=True)
                if len(counts_elements) > 0:
                    new_class = unique_elements[counts_elements.argmax()]
                    combined_mask = np.where(class_region, new_class, combined_mask)

                # fileobj = open('class_region.raw', mode='wb')
                # off = np.array(class_region, dtype=np.int8)
                # off.tofile(fileobj)
                # fileobj.close()
                # fileobj = open('class_boundary.raw', mode='wb')
                # off = np.array(class_boundary, dtype=np.int8)
                # off.tofile(fileobj)
                # fileobj.close()
        # print('index region done')

    # fileobj = open('combined_mask_new.raw', mode='wb')
    # off = np.array(combined_mask, dtype=np.int8)
    # off.tofile(fileobj)
    # fileobj.close()

    for i in range(result_mask.shape[0]):
        result_mask[i] = np.where(combined_mask == i+1, 1, 0)

    fileobj = open('result_mask_new.raw', mode='wb')
    off = np.array(result_mask, dtype=np.int8)
    off.tofile(fileobj)
    fileobj.close()

    return result_mask


def read_2d_nii(path):
    nii = nib.load(path)
    header = nii.header
    voxel_sizes = header.get_zooms()

    nii_shape = len(np.array(nii.dataobj).shape)

    if nii_shape == 3:
        np_input = np.transpose(np.array(nii.dataobj), axes=[2, 1, 0])[0, :, :]
    elif nii_shape == 4:
        np_input = np.transpose(np.array(nii.dataobj), axes=[3, 2, 1, 0])[0, 0, :, :]
    else:
        np_input = np.transpose(np.array(nii.dataobj), axes=[1, 0])

    return np_input, voxel_sizes

def read_3d_nii(path):
    nii = nib.load(path)
    header = nii.header
    voxel_sizes = header.get_zooms()

    nii_shape = len(np.array(nii.dataobj).shape)

    if nii_shape == 3:
        np_input = np.transpose(np.array(nii.dataobj), axes=[2, 1, 0])[:]
    elif nii_shape == 4:
        np_input = np.transpose(np.array(nii.dataobj), axes=[3, 2, 1, 0])[0, :, :, :]
    else:
        print('shape is not supported')
        pdb.set_trace()

    return np_input, voxel_sizes


def apply_filter(img, args, ids, filter_value, postintersect, class_output=0):
    depth, inputHeight, inputWidth = img.shape
    filter_val_data_dir = args.filter_val_data_dir
    increment = 0
    while increment < 99:
        fileFilter = str(filter_val_data_dir + ids[:-4])
        if increment > 0:
            fileFilter += "_gt" + str(increment)
        fileFilter += ".raw"
        if os.path.exists(fileFilter):
            print(fileFilter)
            filterDicom = np.fromfile(fileFilter, dtype='uint8', sep="")
            filterDicom = filterDicom.reshape([depth, inputHeight, inputWidth])
            if postintersect:
                img = np.logical_and(filterDicom, img)
            elif args.val_groundtruth_dir != '':
                gt_filename = args.val_groundtruth_dir + str(ids[:-4])
                if args.filter_nogt == False:
                    for class_output in range(0, args.n_class):
                        temp_gt = np.fromfile(gt_filename + '_gt' + str(class_output + 1) + '.raw', dtype='uint8',
                                              sep="")
                        temp_gt = temp_gt.reshape([depth, inputHeight, inputWidth])
                        filterDicom = np.logical_or(filterDicom, temp_gt)
                    if args.volume[0] != 4:  # change value outside filter into FILTER_VALUE
                        img[filterDicom == 0] = filter_value
            break
        increment += 1
    if increment == 99:
        print('Filter file not found. Please check again")')
        pdb.set_trace()
    return img, filterDicom


def apply_threshold(result_mask, filterDicom, args, thresholds):
    if args.loss == 1:
        output = np.zeros_like(result_mask)
        if args.ignore_bg_val or args.lung_lobe_correction:
            result_mask = result_mask[:-1]
        result_mask_tensor = torch.from_numpy(result_mask).float()
        output_ = torch.argmax(result_mask_tensor, dim=0)
        output_ = output_.cpu().numpy()
        for n_cls in range(0, args.n_class):
            output[n_cls, :, :, :] = output_ == n_cls
        output = output[:args.n_class]
        if args.ignore_bg_val or args.lung_lobe_correction:
            output = np.where(filterDicom, output, 0)  # Outside lung mask = 0 (BG)
    elif args.loss == 0 or args.loss == 2 or args.loss == 4 or args.loss == -1 or args.loss == 5:
        if args.lung_lobe_fissure:
            output = np.zeros_like(result_mask)
            lobe = np.argmax(result_mask[:-1], axis=0)
            for n_cls in range(0, args.n_class - 1):
                output[n_cls, :, :, :] = lobe == n_cls
            fissure = result_mask[-1] >= thresholds[0]
            output[-1] = fissure
        else:
            # NORMAL RESULT
            output = result_mask >= thresholds[0]

    return output


def split_train_tuning_val(df_pos, df_neg, df_val_pos, df_val_neg, ratio=None):
    '''
    :param df_pos: dataframe of positive data excluding positive val set
    :param df_neg: dataframe of negative data excluding negative val set
    :param df_val_pos: dataframe of positive val set
    :param df_val_neg: dataframe of negative val set
    :param ratio: train, val, tuning ratio
    :return: df_pos and df_neg with is_tuning columns and has been concatenated with val set
    '''

    if ratio is None:
        ratio = [0.8, 0.1, 0.1]

    # 0:training set 1:tuning set 2:val set
    training_set, tuning_set, val_set = 0, 1, 2

    # Positive
    n_pos_tuning = len(df_val_pos)
    n_pos_train = len(df_pos) - n_pos_tuning
    # n_pos_val = math.ceil(0.10 * len(self.ids_pos))
    val_index_pos = [training_set] * n_pos_train + \
                    [tuning_set] * n_pos_tuning  # + \
    # [0] * n_pos_val
    random.shuffle(val_index_pos)
    df_pos['is_tuning'] = val_index_pos
    df_val_pos['is_tuning'] = val_set
    df_pos = pd.concat([df_pos, df_val_pos])

    # Negative
    n_neg_tuning = len(df_val_neg)  # n_tuning is = n_val neg
    # n_neg_val = math.ceil(neg_samp * n_pos_val)
    n_neg_train = len(df_neg) - n_neg_tuning  # neg_train is all neg data except tuning and val
    val_index_neg = [training_set] * n_neg_train + \
                    [tuning_set] * n_neg_tuning  # + \
    # [0] * n_neg_val
    random.shuffle(val_index_neg)
    df_neg['is_tuning'] = val_index_neg
    df_val_neg['is_tuning'] = val_set
    df_neg = pd.concat([df_neg, df_val_neg])

    return df_pos, df_neg


def fixstrdir(strdir):
    strdir = strdir.replace('//', '/')
    strdir = strdir.replace('\\\\', '/')
    strdir = strdir.replace('\\', '')
    if strdir[-1] != '/':
        strdir += '/'
    return strdir


def precision_recall_curves(y_true, pred_scores, thresholds):
    import sklearn.metrics

    precisions = []
    recalls = []

    for threshold in thresholds:
        #y_pred = [1 if score >= threshold else 0 for score in pred_scores]
        y_pred = np.where(pred_scores >= threshold, 1, 0)

        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred)
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred)

        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)

def generate_pr_curve(pred, gt_file, save_dir):
    from sklearn.metrics import average_precision_score, auc
    patient_id = os.path.basename(gt_file)
    f_list = []

    gt = np.fromfile( gt_file+'_gt1.raw', dtype='uint8', sep="")
    thresholds = np.linspace(0.01, 0.99, num=100)
    # precision, recall, thresholdss = precision_recall_curve(gt, pred.flatten())
    precision, recall = precision_recall_curves(gt, pred.flatten(), thresholds)
    avg_prec = average_precision_score(gt, pred.flatten())
    pr_auc = auc(recall, precision)
    fscore = (2 * precision * recall) / (precision + recall + 1e-10)
    ix = np.argmax(fscore)
    print(f'{patient_id} Best Threshold={thresholds[ix]}, F-Score={fscore[ix]}')

    no_skill = len(gt[gt == 1]) / len(gt)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Ratio of Positive Ex.')
    plt.plot(recall, precision, marker='.', label='Prediction')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'{patient_id}.png'))
    plt.clf()

    # stat = np.array([precision[:-1], recall[:-1], thresholds])
    stat = np.array([precision, recall, thresholds])
    stat = np.transpose(stat, axes=[1, 0])
    pr_df = pd.DataFrame(stat, columns=['precision', 'recall', 'threshold'])
    pr_df.to_csv(os.path.join(save_dir, f'{patient_id}.csv'))

    f_list.append([patient_id, precision[ix], recall[ix], fscore[ix], thresholds[ix], avg_prec, pr_auc, np.min(thresholds), np.max(thresholds)])
    f_df = pd.DataFrame(f_list, columns=['id', 'precision', 'recall', 'f_score', 'threshold', 'avg_prec', 'pr_auc', 'min_th', 'max_th'])
    if os.path.exists(os.path.join(save_dir, 'f_score.csv')):
        f_df.to_csv(os.path.join(save_dir, 'f_score.csv'), mode='a', header=False)
    else:
        f_df.to_csv(os.path.join(save_dir, 'f_score.csv'))

    return thresholds[ix]


if __name__ == "__main__":
    #dir_img, dir_mask, height, weight, ww, wl, n_class
    #dir_img = '//147.46.188.185/media/Share DATA/toydata/segmentation/oneclass/train/data/raw/'
    #dir_mask = '//147.46.188.185/media/Share DATA/toydata/segmentation/oneclass/train/label/raw/'

    dir_data = 'E:/data/wholebody/train/data43d/'
    dir_gt = 'E:/data/wholebody/train/gt43d/'

    dir_data = 'E:/data/ntmcavity/train/data/'
    dir_gt = 'E:/data/ntmcavity/train/gtmerge/'

    n_class = 1
    slice = 6
    #padding = 0

    test = generate_excel42d_perslice(slice,n_class,dir_data, dir_gt)

