# CSE5559_Final_Project
The final project for CSE 5559 by team VisAiR with members Skylar Wurster (wurster.18@osu.edu), Logan Frank, and Njoki Mwagni.

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

### Install flask:
```bash
pip install flask
```

### Install SPADE (GAUGAN) dependencies
```bash
cd SPADE
pip install -r requirements.txt
```
### Install ade20k checkpoint for SPADE
Download the tar of the pretrained models from the [Google Drive Folder](https://drive.google.com/file/d/12gvlTbMvUcJewQlSEaZdeb2CdOB-b8kQ/view?usp=sharing), save it in 'SPADE/checkpoints/', and run

```
cd checkpoints
tar xvf checkpoints.tar.gz
```

## Running the code
After proper installation, navigate to the base folder and run 
```bash
python -m flask run
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