from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pydl_image_encoders.library.download_utils import reporthook

slim = tf.contrib.slim
from PIL import Image
from pydl_image_encoders.library.tf.inception_resnet_v2 import *
import urllib.request
import numpy as np
import tarfile
import os

INCEPTION_RESNET_MODEL_NAME = 'inception_resnet_v2_2016_08_30.ckpt'


def download_pretrained_model(data_dir_path):
    model_file_path = os.path.join(data_dir_path, INCEPTION_RESNET_MODEL_NAME)
    if os.path.exists(model_file_path):
        return

    zip_file_path = data_dir_path + '/inception_resnet_v2_2016_08_30.tar.gz'

    if not os.path.exists(zip_file_path):
        url_link = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
        print('gz model file does not exist, downloading from internet')
        urllib.request.urlretrieve(url=url_link, filename=zip_file_path,
                                   reporthook=reporthook)

    tar = tarfile.open(zip_file_path, "r:gz")
    tar.extractall(data_dir_path)
    tar.close()


class InceptionResNetImageEnoder(object):

    def __init__(self):
        self.sess = None
        input_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input_image')
        self.input_tensor = input_tensor
        scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
        scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
        scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)
        self.scaled_input_tensor = scaled_input_tensor
        self.sess = tf.Session()
        arg_scope = inception_resnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            self.logits, self.end_points = inception_resnet_v2(self.scaled_input_tensor, is_training=False)
        self.saver = tf.train.Saver()

    def load_model(self, data_dir_path):
        download_pretrained_model(data_dir_path)
        checkpoint_file = os.path.join(data_dir_path, INCEPTION_RESNET_MODEL_NAME)
        self.saver.restore(self.sess, checkpoint_file)

    def predict_image(self, im):
        im = im.reshape(-1, 299, 299, 3)
        predict_values, logit_values = self.sess.run([self.end_points['Predictions'], self.logits],
                                                     feed_dict={self.input_tensor: im})
        return np.argmax(predict_values), np.max(predict_values)  # , np.argmax(logit_values)

    def predict_image_file(self, image_path):
        im = Image.open(image_path).resize((299, 299))
        im = np.array(im)
        return self.predict_image(im)

    def encode_image(self, im):
        im = im.reshape(-1, 299, 299, 3)
        predict_values, logit_values = self.sess.run([self.end_points['Predictions'], self.logits],
                                                     feed_dict={self.input_tensor: im})
        return predict_values

    def encode_image_file(self, image_path):
        im = Image.open(image_path).resize((299, 299))
        im = np.array(im)
        return self.encode_image(im)


def main():
    data_dir_path = '../demo/very_large_data'
    img_dir_path = '../demo/data/images'
    sample_images = [img_dir_path + '/dog.jpg', img_dir_path + '/cat.jpg']
    encoder = InceptionResNetImageEnoder()
    encoder.load_model(data_dir_path)
    for image_path in sample_images:
        class_id, predict_score = encoder.predict_image_file(image_path)
        print(class_id, predict_score)


if __name__ == '__main__':
    main()
