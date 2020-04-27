## Basic Python libraries
import os
from PIL import Image
import pickle

## Deep learning and array processing libraries
import numpy as np 
import torch
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms 

## Inner-project imports
from model import EncoderCNN, DecoderRNN
from build_vocab import build_vocab

## Other
import nltk
nltk.download('punkt')

##### Code begins #####

# Path to config file
image_directory = './LSTM/images/'
network_directory = './LSTM/models/'
data_directory = './LSTM/data/'

# Setting up other necessary paths
decoder_path = f'{network_directory}decoder-5-3000.pkl'
vocab_path = f'{data_directory}vocab.pkl'
annotations_path = f'{data_directory}annotations/captions_train2014.json'

# Define the compute device (either GPU or CPU)
if torch.cuda.is_available():
    compute_device = torch.device('cuda:0')
else:
    compute_device = torch.device('cpu')
print(f'Using device: {compute_device}')

vocab = build_vocab(json=annotations_path, threshold=4)
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)

# Configure network
network = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1)
network = network.eval()
network.load_state_dict(torch.load(decoder_path, map_location='cpu'))
network = network.to(compute_device)

def caption_image(features):
    """
    features :: a [512] NumPy array
    """

    features = torch.from_numpy(features)
    features = features.unsqueeze(0)
    features = features.to(compute_device)

    sampled_ids = network.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    return sentence

# Below is only there for testing, commented out for now
"""
if __name__ == '__main__':
    import sys 
    sys.path.append(os.getcwd())
    import CNN.extract as extract
    # Inference
    img = Image.open(f'{image_directory}input/3.png')
    img = np.asarray(img)
    features = extract.get_visual_features(img)
    sentence = caption_image(vocab, features)
    print('End')
"""
