from __future__ import print_function
from flask import Flask, request, render_template
from flask import  session
# from feature_extractor import FeatureExtractor
from feature_extractor import FeatureExtractor
from class_yolo import YOLO_MODEL
from datetime import datetime
from PIL import Image
import numpy as np
import pickle
import glob
import os
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing import image

import json
from Classification import test
import cv2
from keras.models import load_model
import tensorflow as tf
global graph
graph = tf.get_default_graph()
model = load_model('Classification/Dogs.h5')
app = Flask(__name__)
from gevent.pywsgi import WSGIServer

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
import tensorflow as tf
# global graph
# graph = tf.get_default_graph()
fe = FeatureExtractor()
pred=YOLO_MODEL()
features = []
img_paths = []
labels =test.labels
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')

print(img_paths)
@app.route('/', methods=['GET', 'POST'])
def index():
    searched = (request.method == 'POST')
    if searched:
        file = request.files['query_img']
        try:
            img = Image.open(file.stream)  # PIL image
        except OSError:
            return render_template('index.html',  file_error=True, searched=False)
        uploaded_img_path = "static/uploaded/" + "_" + file.filename
        img.save(uploaded_img_path)
        print("BEFORE PREDICTION")
        breed_before = image.load_img(uploaded_img_path, target_size=(299, 299))

        query_1,coords,image_paths,unique_class=pred.create_boxes(uploaded_img_path,file.filename) # predicting class


        #Create Dictionary
        paths_dict={}
        score_list=[]
        threshold=0.00
        temp_threshold=0.00
        for i in range(len(image_paths)):
            img_temp=Image.open(image_paths[i])
            # print(image_paths[i])
            # print(type(img_temp))
            query = fe.extract(img_temp)  # extract features
            dists = np.linalg.norm(features - query, axis=1)  # Do search
            ids = np.argsort(dists)  # select top features
            while(1):        
                scores = [(dists[id], img_paths[id]) for id in ids if dists[id] <= temp_threshold]
                if (len(scores)>=6):
                    break
                temp_threshold=temp_threshold+0.05
            temp_threshold=0.00
            # print(scores[0][1])
            # print(scores)
            # print(scores)
            for j in range(len(scores)):
                if(query_1[i] in paths_dict):
                    paths_dict[query_1[i]].append(scores[j][1])
                    
                else:
                    paths_dict[query_1[i]]=[]
                    paths_dict[query_1[i]].append(scores[j][1])
            score_list.append(scores)
            # if(!'query_1[i]' /in paths_dict):

            # img.save()
        app_json = json.dumps(paths_dict, sort_keys=True)
        print(paths_dict)

        detected_img_path="static/detected/"+file.filename
        cv2.imread(detected_img_path)
        breed = test.predict(model,breed_before)
        print(breed.argmax())
        print(labels[breed.argmax()])
        top_values = [breed[i] for i in np.argsort(breed)[-5:]]
        # for top in top_values:
        #     print(labels[top])

        # session.clear()
        # keras.backend.clear_session()
        # with tf.Graph().as_default():
        #
        #     breed = test.predict(breed_before)
        #     print(breed.argmax())
        #     print(labels[breed.argmax()])
        print(paths_dict)
        return render_template('index.html',
                               query_path=detected_img_path,
                               coords_dict=coords,
                               score_list=score_list,
                               searched=searched,
                               predicted_class=query_1,
                               tag_dictionary=paths_dict,
                               json_form=app_json,
                               unique_class=unique_class,
                               file_error=False,breed=labels[breed.argmax()])
    else:
        return render_template('index.html', searched=searched, file_error=False)


if __name__ == "__main__":
    app.run(debug=True)
    # http_server = WSGIServer(('localhost', 5000), app)
    # http_server.serve_forever()