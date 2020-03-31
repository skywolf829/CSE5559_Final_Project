import imageio
import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import numpy as np
from skimage import transform, io
import torch

# Must set up options :/
opt = TestOptions().parse()

# Create model and put it into evaluation mode (we're not training)
model = Pix2PixModel(opt)
model.eval()

def create_generated_image(seg_map):
    '''
    Full feedforward through Pix2Pix for generating an
    output from a segmentation map of shape (x, y) where all
    entries are ints representing class labels for the trained network
    you are using. This image will be converted to (256, 256) for 
    feedforward.
    '''
    seg = create_seg_map_tensor(seg_map)
    # Set up data as they do in the original code
    data = {}
    data['label'] = seg
    # image isn't important, but must exist anyway
    data['image'] = seg
    # instance isn't important, but must exist anyway
    data['instance'] = torch.zeros(1)-1
    data['path'] = None
    generated = model(data, mode='inference')
    # Detach the generated output of size (1, 1, 256, 256) and return a tensor of size (1, 256, 256)
    generated = generated.cpu().detach().numpy()[0]
    return generated

def create_seg_map_tensor(seg_map):
    '''
    Converts a seg map of shape (256, 256) into a correct torch tensor for Pix2Pix,
    that will be of shape (1, 1, 256, 256).
    The segmentation map input should be a single channel image, with each pixel
    corresponding to a class for the trained model. Example classes labels can be 
    found in SPADE/ADEChallengeData2016/objectInfo150.txt.
    '''
    # We don't use anti_aliasing because that will introduce unwanted classes
    # Preserve range to keep the range 0-255
    seg_map = transform.resize(seg_map, (256, 256), preserve_range=True, anti_aliasing=False)
    seg_map = torch.from_numpy(seg_map)
    t = torch.zeros([1, 1, 256, 256]) - 1
    t[0,0]=seg_map
    return t

def generated_to_savable_image(gen_image):
    '''
    Converts the generated image into a savable format for imageio library.
    Original shape is (3, 256, 256) as c,x,y, but needs to be in form (y,x,c).
    The image also needs to be denormalized and converted to uint8 from 0-255
    ''' 
    gen_image = gen_image.swapaxes(0,2).swapaxes(0,1)
    gen_image = 255*((gen_image+1)/2)
    gen_image = np.uint8(gen_image)
    return gen_image

def save_image(im, path):   
    '''
    Saves an image im to file path
    ''' 
    imageio.imsave(path, im)


# Just a placeholder variable for the current directory
folder_path = os.path.dirname(os.path.abspath(__file__))

seg = imageio.imread(os.path.join(folder_path, "TestFolder", "SegMaps", "ADE_train_00000002.png"))
gen = create_generated_image(seg)
gen_image = generated_to_savable_image(gen)
save_image(gen_image, os.path.join(folder_path, "TestResults","image2.png"))
