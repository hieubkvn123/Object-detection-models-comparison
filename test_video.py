import cv2
import time
import mxnet as mx
from argparse import ArgumentParser
from gluoncv import model_zoo, data, utils

def numpy_to_mx(img, model='center'):
    preprocess_func = data.transforms.presets.center_net.transform_test

    if(model == 'yolo'):
        preprocess_func = data.transforms.presets.yolo.transform_test

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = mx.nd.array(img.reshape(H, W, C))
    x, img = preprocess_func(x, short=512)

    return x, img

