from keras.preprocessing import image
import keras as keras
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model,load_model
import numpy as np
import tensorflow as tf


class FeatureExtractor:
    def __init__(self):
        # tf.reset_default_graph()
        self.graph1 = tf.Graph()
        base_model = MobileNet(weights='imagenet')
        self.model1 = Model(inputs=base_model.input, outputs=base_model.get_layer('conv_preds').output)
        # self.base_model_class = load_model('class_model.h5')
        # self.graph1 = tf.Graph()
        self.graph = tf.get_default_graph()
        # self.graph2 = tf.Graph()

    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
        print("I have got into extract method")
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        # To np.array. Height x Width x Channel. dtype=float32
        x = image.img_to_array(img)
        # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        # feature = self.model1.predict(x)[0][0][0]  # (1, 4096) -> (4096, )
        # return feature / np.linalg.norm(feature)
        with self.graph.as_default():
            # base_model = MobileNet(weights='imagenet')
            print("I have loaded the model")
            # model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv_preds').output)
            feature = self.model1.predict(x)[0][0][0]  # (1, 4096) -> (4096, )
            return feature / np.linalg.norm(feature)  # Normalize



