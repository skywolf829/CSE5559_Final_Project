import imageio
import os
from collections import OrderedDict

import data
from options.custom_options import CustomOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import numpy as np
from skimage import transform, io
import torch


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

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

def get_cmap(N):
    '''
    Creates an arbitrary color map for some number of classes using bit shifting
    '''
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i + 1  # let's give 0 a color
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
        #print(str(i+1)+","+str(cmap[i,0])+","+str(cmap[i,1])+","+str(cmap[i,2]))
    return cmap

def create_cmap(file_location, file_location_out, N):
    '''
    Was used as a temporary utility function to combine two csv's
    and create a final cmap text document for future reading
    '''
    cmap = get_cmap(N)
    f = open(file_location, "r")
    f_out = open(file_location_out, "a+")
    if f.mode == "r":
        fl = f.readlines()
        for line in fl:
            if line[0] != "#":
                contents = line.split(",")
                i = int(contents[0].strip())-1
                c = contents[1].strip()
                line = line.strip() + "," + str(cmap[i,0]) + "," + str(cmap[i,1]) + "," + str(cmap[i,2])+"\n"
                f_out.write(line)
            else:                
                f_out.write(line)
    f.close()
    f_out.close()

def load_cmap(file_location):
    '''
    Reades a text document at file_location for color mapping 
    between classes, the class' grayscale segmentation map color,
    and a visualization color that's 3 channel. 
    Returns 2 color maps, one for the rgb colors (3 channel) to black 
    and white (1 channel), and another for black and white to colored
    '''
    rgb2bw = {}
    bw2rgb = {}
    class2rgb = {}
    classes = []
    f = open(file_location, "r")
    seen_first_line = False
    if f.mode == "r":
        fl = f.readlines()
        for line in fl:
            if line[0] != "#":
                if seen_first_line:
                    contents = line.split(",")
                    bw = int(contents[0].strip())
                    c = contents[1].strip()
                    r = int(contents[2].strip())
                    g = int(contents[3].strip())
                    b = int(contents[4].strip())
                    rgb2bw[str(r)+","+str(g)+","+str(b)] = bw
                    bw2rgb[bw] = np.array([r, g, b])
                    class2rgb[c] = str(r)+","+str(g)+","+str(b)
                    classes.append(c)
                else:
                    num_classes = int(line.strip())
                    seen_first_line = True
    f.close()
    return rgb2bw, bw2rgb, classes, class2rgb

def convert_drawing_to_segmentation_map(drawing, rgb2bw):
    '''
    Converts a drawing using colors from a specific color map into the
    segmentation map values for use in Pix2Pix network.
    drawing comes in as (256, 256, 3)
    '''
    seg_map = np.zeros((drawing.shape[0], drawing.shape[1]))
    for i in range(seg_map.shape[0]):
        for j in range(seg_map.shape[1]):
            asStr = str(drawing[i,j,0])+","+str(drawing[i,j,1])+","+str(drawing[i,j,2])
            seg_map[i, j]= rgb2bw[asStr]
    return seg_map

class GAUGAN():

    def __init__(self):
        # Must set up options :/
        self.opt = CustomOptions().parse()

        # Create model and put it into evaluation mode (we're not training)
        self.model = Pix2PixModel(self.opt)
        self.model.eval()

    def generate_from_seg_map(self, seg_map):
        '''
        Full feedforward through Pix2Pix for generating an
        output from a segmentation map of shape (x, y) where all
        entries are ints representing class labels for the trained network
        you are using. This image will be converted to (256, 256) for 
        feedforward.
        '''
        # Set up data as they do in the original code
        data = {}
        data['label'] = seg_map
        # image isn't important, but must exist anyway
        data['image'] = seg_map
        # instance isn't important, but must exist anyway
        data['instance'] = torch.zeros(1)-1
        data['path'] = None
        generated = self.model(data, mode='inference')
        # Detach the generated output of size (1, 1, 256, 256) and return a tensor of size (1, 256, 256)
        generated = generated.cpu().detach().numpy()[0]
        return generated


# Just a placeholder variable for the current directory
#folder_path = os.path.dirname(os.path.abspath(__file__))
'''
seg = imageio.imread(os.path.join(folder_path, "TestFolder", "SegMaps", "ADE_train_00000002.png"))
seg = create_seg_map_tensor(seg)
g = GAUGAN()
gen_image = g.generate_from_seg_map(seg)
gen_image = generated_to_savable_image(gen_image)
save_image(gen_image, os.path.join(folder_path, "TestResults","image2.png"))
'''
#create_cmap(os.path.join(folder_path, "ade20k_classColors.txt"), os.path.join(folder_path,"ade20k_cmap.txt"), 150)