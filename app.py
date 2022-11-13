import json
from flask import Flask
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.utils  import secure_filename
import os

import sklearn
import cv2
import numpy as np
import pickle
import pandas as pd
from sklearn.decomposition import PCA



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload'

rslt = -1
file = ""

@app.route('/')
def index():    
    return render_template('index.html')


@app.route('/predict',methods=['POST']) 
def predict():
    global rslt,file
    if request.method=='POST':
        f = request.files['myfile']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file=filename
        
        pt = os.path.join(os.getcwd(),app.config['UPLOAD_FOLDER'], filename)        
        inp = input(pt)
        
        
        mod = pickle.load(open('model_knn.sav', 'rb'))
        
        res = mod.predict(inp)
        print(res)
        rslt = res[0]
        return render_template('result.html',res=rslt,file=file)        
    else:
        return "else"



@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


def input(pth):
    hog = cv2.HOGDescriptor()

    # DECLARE PARAMETER
    NUM_FEATURES = 25  # this is the number of features to be extracted to a the csv file

    # Declare empty container to hold extracted category
    category = []

    # Declare empty container to hold extracted features
    hogArray = []

    # Use open cv to read the image
    image = pth
    img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)

    # Resize the image to (64, 128)
    # Default for hog
    resized = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)

    # Compute the Hog Features
    h = hog.compute(resized)

    # Transpose the result from a vector to an array
    hogImage = h.T

    hogArray.append(hogImage)

    hogArray_np = np.array(hogArray)

    # Reshaped the Features to the acurrate size ####errror
    reshaped_hog_Array = np.reshape(hogArray_np, (hogArray_np.shape[0], hogArray_np.shape[1]))


    # setup PCA for dimensionality reduction
    # pca = PCA(n_components=1)
    
    pca1 = pickle.load(open('pca.pkl', 'rb'))
    reduced_features = pca1.transform(reshaped_hog_Array)
    features = reduced_features.tolist()
    return features


