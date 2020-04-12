## Standard library imports
import base64
import sys
import os

## Append to path
folder_path = os.path.dirname(os.path.abspath(__file__))
SPADE_folder_path = os.path.join(folder_path, "..", "..", "SPADE")
sys.path.append(SPADE_folder_path)

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'HTML'))
sys.path.append(os.path.join(os.getcwd(), 'CNN'))

## Random Imports
import flask
from flask import render_template
import numpy as np
from PIL import Image
import cv2
from skimage.measure import compare_ssim as ssim

## Inner-project Imports
from app import app
from generate import *
import extract

# Comment the below line out and insert wherever needed
# visual_features = extract.get_visual_features(generated_img)

g = GAUGAN()
rgb2bw, bw2rgb, classes, class2rgb = load_cmap(os.path.join(SPADE_folder_path, "ade20k_cmap.txt"))

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', classes=classes, class2rgb=class2rgb)

#background process happening without any refreshing
@app.route('/get_generated_image')
def get_generated_image():
    sent_data = flask.request.args.get('img')
    encoded_data = sent_data.split(',')[1]
    decoded_data = base64.b64decode(encoded_data)
    import io
    img = Image.open(io.BytesIO(decoded_data))
    # Shape (256, 256, 4) in RGBA
    img = np.asarray(img)[:,:,0:3].astype(np.uint8)
    seg_map = convert_drawing_to_segmentation_map(img, rgb2bw)
    seg_map = create_seg_map_tensor(seg_map)
    generated_img = g.generate_from_seg_map(seg_map)
    generated_img = generated_to_savable_image(generated_img)
    #save_image(generated_img, os.path.join(folder_path, "test.png"))
    # Color space issue correction
    generated_img = cv2.cvtColor(generated_img, cv2.COLOR_BGR2RGB)
    success, return_img = cv2.imencode(".png", generated_img)
    return_img = return_img.tobytes()
    return flask.jsonify({"img":str(base64.b64encode(return_img))})

@app.route('/get_image_metrics')
def get_image_metrics():
    sent_data_1 = flask.request.args.get('img1')
    sent_data_2 = flask.request.args.get('img2')
    encoded_data_1 = sent_data_1.split(',')[1]
    encoded_data_2 = sent_data_2.split(',')[1]
    decoded_data_1 = base64.b64decode(encoded_data_1)
    decoded_data_2 = base64.b64decode(encoded_data_2)
    import io
    img1 = Image.open(io.BytesIO(decoded_data_1))
    img1 = np.asarray(img1)[:,:,0:3].astype(np.uint8)
    img2 = Image.open(io.BytesIO(decoded_data_2))
    img2 = np.asarray(img2)[:,:,0:3].astype(np.uint8)

    ssim_result = ssim(img1, img2, multichannel=True)
    mse = np.mean((img1 - img2) ** 2) ** 0.5
    return flask.jsonify({"mse":str(mse), "ssim":str(ssim_result)})