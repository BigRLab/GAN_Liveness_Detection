import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import ntpath
import scipy.misc
import pickle

import pdb

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 # the default range of image is (-1, 1)
    return image_numpy.astype(imtype)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
    	os.makedirs(path)

def save_images(visuals, save_path, img_path):

    img_name = ntpath.basename(img_path[0])
    folder_name = img_name[:-4]
    ext = img_name[-4:]
    save_folder_path = os.path.join(save_path, folder_name) # path to the folder saving individual images
    mkdir(save_folder_path)

    # save the individual images
    for key in visuals.keys():
        scipy.misc.imsave(os.path.join(save_folder_path, key+ext), visuals[key])

    # combine the images into single result and save
    comb = np.concatenate((visuals['real_A'],
                           visuals['fake_B'],
                           visuals['real_B']), 1)
    scipy.misc.imsave(os.path.join(save_path, img_name), comb)

    return

def load_pretrained_params(weight_path, bias_path):
    print 'Loading pre-trained VGG_FACE parameters for perceptual loss'

    # load weights
    with open(weight_path, 'rb') as weight_reader:
        weight = pickle.load(weight_reader)
    weight_reader.close()

    # load bias
    with open(bias_path, 'rb') as bias_reader:
        bias = pickle.load(bias_reader)
    bias_reader.close()

    return weight, bias

