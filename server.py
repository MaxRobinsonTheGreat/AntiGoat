from flask import Flask, flash, request, render_template, redirect, url_for
import torch
from model import GoatDetector, threshold
from torchvision import transforms
from werkzeug.utils import secure_filename
from PIL import Image
import os
import numpy as np

model = GoatDetector()
model.load_state_dict(torch.load("./saved_models/temp"))
model = model.eval()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = 'super secret key'


def rgba_to_rgb(im):
    x = np.array(im)
    if len(x.shape) < 3:
        # convert from grayscale
        new = np.array([[[i, i, i] for i in y] for y in x])
        return Image.fromarray(new)
    if x.shape[2] > 3:
        # remove alpha channel
        return Image.fromarray(x[:,:,0:3])
    return im

def prepare_image(im):
    im = Image.open(im)
    im = rgba_to_rgb(im)

    trans = transforms.Compose(
                            [
                                transforms.Resize((64, 64)),
                                transforms.ToTensor()
                            ])
    return trans(im).unsqueeze(0)

def predict(image_file):
    tensor = prepare_image(image_file)
    return model(tensor).item() >= threshold


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@app.route('/<page>')
def public(page=None):
    return render_template("/"+page)


@app.route('/upload', methods=['GET', 'POST'])
def detect():
    # print(request.files)
    # return "what is up my dude"
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No image')
        return "No image"
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected Image')
        return "No selected image"
    if file and allowed_file(file.filename):
        print("hello")
        
        # filename = secure_filename(file.filename)
        # file.save(file_path)
        
        pred = predict(file)
        return str(pred)

    return "Could not read file"

    