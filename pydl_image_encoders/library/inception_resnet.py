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


download_pretrained_model('../demo/very_large_data')

checkpoint_file = '../demo/vary_large_data/' + INCEPTION_RESNET_MODEL_NAME

sample_images = ['../demo/data/images/dog.jpg', '../demo/data/images/cat.jpg']

input_tensor = tf.placeholder(tf.float32, shape=(None,299,299,3), name='input_image')
scaled_input_tensor = tf.scalar_mul((1.0/255), input_tensor)
scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

#Load the model
sess = tf.Session()
arg_scope = inception_resnet_v2_arg_scope()
with slim.arg_scope(arg_scope):
  logits, end_points = inception_resnet_v2(scaled_input_tensor, is_training=False)
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)
for image in sample_images:
  im = Image.open(image).resize((299,299))
  im = np.array(im)
  im = im.reshape(-1,299,299,3)
  predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})
  print (np.max(predict_values), np.max(logit_values))
  print (np.argmax(predict_values), np.argmax(logit_values))