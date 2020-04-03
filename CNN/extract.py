## Basic Python libraries
import os
import yaml
from PIL import Image

## Deep learning and array processing libraries
import numpy as np 
import torch
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms 

## Inner-project imports
from CNN.models import ModelBuilder

##### Code begins #####

# Path to config file
config_path = 'CNN/config.yaml'

# Open the yaml config file
try:
    with open(os.path.abspath(config_path)) as config_file: 
        config = yaml.safe_load(config_file)

        # Location of any saved images
        image_directory = config['Paths']['image_directory']

        # Location of saved network weights
        network_directory = config['Paths']['network_directory']

except:
    raise Exception('Error loading data from config file.')

# Setting up other necessary paths
input_image_directory = f'{image_directory}input/'
output_image_directory = f'{image_directory}output/'
encoder_name = 'resnet18dilated'
decoder_name = 'ppm_deepsup'
weight_path = f'{network_directory}ade20k-{encoder_name}-{decoder_name}.pth'

# Define the compute device (either GPU or CPU)
if torch.cuda.is_available():
    compute_device = torch.device('cuda:0')
else:
    compute_device = torch.device('cpu')
print(f'Using device: {compute_device}')

# Create the data transforms for evaluating
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Configure network
network = ModelBuilder.build_encoder(arch=encoder_name, fc_dim=512, weights=weight_path)
network = network.to(compute_device)
network.eval()

def get_visual_features(img):
    """
    Extracts the visual features from an input image. Converts input
    into PIL Image, normalizes the image, then feeds it through a CNN.
    The features returned from the CNN are then pooled into a 1x512x1x1
    and finally squeezed to produce our [512] array output.

    Input
    img :: 3D NumPy array
        Takes a [x, y, 3] NumPy array to be converted into a PIL Image

    Output
    features :: 1D NumPy array
        Returns a [512] NumPy array of the visual features from the CNN
    """

    # Convert to PIL Image and perform transformation
    img = Image.fromarray(img)
    img = transform(img)

    # Add a 4th dimension and send to compute device (GPU or CPU)
    img = img.unsqueeze(0)
    img = img.to(compute_device)

    # Feed input through CNN
    features = network(img)[0]

    # Take outputted [1, 512, 30, 30] tensor and pool into [1, 512, 1, 1]
    features = F.adaptive_avg_pool2d(features, 1)

    # TODO: Can change this to a different shape as needed for the LSTM
    # Squeeze into a [512] vector
    features = features.squeeze()

    # Convert to NumPy
    features = features.cpu().detach().numpy()
    return features

# Below is only there for testing, commented out for now
"""
if __name__ == '__main__':
    # Inference
    img = Image.open(f'{image_directory}input/1.png')
    img = np.asarray(img)
    features = get_visual_features(img)
    print('End')
"""




