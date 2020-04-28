# CSE5559_Final_Project
The final project for CSE 5559 by team VisAiR with members Skylar Wurster (wurster.18@osu.edu), Logan Frank (frank.580@osu.edu), and Njoki Mwagni.

# Intro
// Introduce the topic, ex:

With the recent improvements in generative adversarial networks (GANs), many networks have come about that allow for generation of realistic images. They do this using a generator-discriminator pair. The generator's goal is to generate convincing looking images that will "fool" the discriminator, and the discriminator is learning real from fake images.

Some create cars, some create human faces, and some create cats! Here are some examples:

// Insert example images

Some GANs are specifically trained to generate an output image from an input sketch or segmentation map. A  segmentation map is simply an image where each pixel color corresponds to a specific class, like "grass", or "sky" or "table". Here is an example of a segmentation map, and the corresponding actual photo:

// Insert segmentation image example

Some fun tools online allow you draw sketches and see the output, without having to run the network on your own machine! 

// Insert links here

Another GAN that won a few awards at SIGGRAPH 2019 is called GAUGAN, which was trained on an unreleased Flickr landscapes dataset. We find this one particularly interesting because it is lightweight and allows us to see changes to the output image (after drawing) reasonably quickly on new hardware.

// Insert images of GAUGAN generations

// Talk about image captioning here

With these two new advances, we were curious - what changes in our input to a generative image network would create large changes in our output image and caption? To test this, we use NVidia's GAUGAN for the generator network and _________ for the captioning network. We create a interactive browser based HTML page to draw the segmentation map and view outputs, a python backend to run all of our models, and flask to tie the front and backend together.

To allow the users to see how much their changes have an impact on the output, we will also show metrics between any two previous images, such as MSE, SSIM, and feature distance, measured by a feature extracting network.

// Finished with introduction 

// Next, talk about the backend networks


// Next, talk about the frontend and how it uses flask to get info from the backend


// Done talking about the GUI, now we should talk about interesting results 

// Discussion and things to try (future work)

# Introduction
This is a VISxAI project that follows this sequence of events:
1. The user draws onto a canvas by selecting colors from classes. Effectively, they are drawing a segmentation map.
2. The segmentation map is used as input to GAUGAN, which generates a realistic looking image from the segmentation map. The output is displayed next to the drawing on screen.
3. Visual features are extracted from the generated image via DilatedResNet18.
4. The extracted visual features are used as input into an image captioning network.
5. The image caption is displayed on screen.

The goal is to see what changes in the input drawing cause interesting changes in the output caption. We hypothesize that small changes may make large changes in the caption prediction, especially when inserting classes that may not be related to the other surroundings in the scene (ex. inserting a "ceiling" in an outdoor scene).

## Instructions and Installation

This project uses python 3+ and PyTorch 1.0 for the machine learning, and flask for connecting the HTML/web-based front end with the back-end.

### Install dependencies

```
pip install -r requirements.txt
```
Navigate to https://pytorch.org/, select your configuration, and run the command instructed. For example, we used:
```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```
If you're runing on a unix based system, install pycocotools with:
```
pip install pycocotools
```
Otherwise on Windows, you'll have to install it with this command:
```
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

### Install ade20k checkpoint for SPADE
Download the tar of the pretrained models from the [Google Drive Folder](https://drive.google.com/file/d/12gvlTbMvUcJewQlSEaZdeb2CdOB-b8kQ/view?usp=sharing), save it in 'SPADE/checkpoints/', and run
```
cd checkpoints
tar xvf checkpoints.tar.gz
```
### Set up CNN weights
Downlost the encoder, decoder, and vocab pickle files from ADD_LINK_HERE. Then, place the encoder.pkl file into CNN/models, the decoder.pkl into LSTM/models, and vocab.pkl into LSTM/data



## Running the code
After proper installation, navigate to the base folder and run 
```bash
export FLASK_APP=HTML/playground.py
flask run
```

In the terminal, you should see proper initialization for SPADE and that ResNet properly loaded weights for net_encoder.

Then, open a browser and navigate to localhost:5000.

You can interact with the painting by clicking to draw. You can select classes to "draw" on the left side. Scroll through the list and click on the class desired. On the right, the output image will be shown, and below, the caption will be shown.

## Code Structure

### HTML
This folder is responsible for running the application via flask. 

- `HTML/routes.py` is responsible for routing the calls properly. This serves the index page at HTML/app/index.html and also calls the python function to generate an image from a segmentation map input.
- `HTML/app/index.html` is loaded as a flask template, allowing flask to insert python code where necessary. When {{ }} is used, the inside is the python code to be called on load.

### SPADE (GAUGAN)

- `SPADE/generate.py` does most of the backend work. It allows for creation of a GAUGAN object, which will load the ade20k pretrained model. Then, `generate_from_seg_map` will return a generated image (NumPy array) from an input segmentation map.
- `SPADE/ade20k_cmap.txt` holds color mapping information for the segmentation. GAUGAN takes a (256x256x1) input, but a grayscale segmentation map wouldn't be very fun to draw, and would be very difficult to differentiate similar gray colors. Therefore, we color map 0-255 grayscale to RGB. These mappings are saved in this text document, which also has the class names.

For the rest of the files and structure, please see SPADE/README.md.

### CNN

- `CNN/extract.py` is for extracting the visual features from an input image. When this file is imported, a ResNet-18-Dilated model is loaded using the weights provided by the ADE20K authors to be used for extracting the features. The only method in the file is `get_visual_features` which takes a (x, y, 3) RGB NumPy array and returns a (512) NumPy array that is the visual features for the provided input image.
- `CNN/models/models.py` is what `CNN/extract.py` uses to construct the model. Specifically the `build_encoder` method in the `ModelBuilder` class constructs the network and loads the pretrained weights.
- `CNN/networks/ade20k-resnet18dilated-ppm_deepsup.pth` contains the weights used in our encoder network.

