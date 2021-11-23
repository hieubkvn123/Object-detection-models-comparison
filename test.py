import cv2
import time
import mxnet as mx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from argparse import ArgumentParser
from gluoncv import model_zoo, data, utils

parser = ArgumentParser()
parser.add_argument('--input', required=True, help='Path to input test image')
parser.add_argument('--model', required=False, default='center', help='Detection model used')
args = vars(parser.parse_args())

models = {
    'center' : 'center_net_resnet101_v1b_dcnv2_coco',
    'yolo' : 'yolo3_darknet53_coco'
}

if(args['model'] not in models.keys()):
    raise Exception('Invalid model name')

net = model_zoo.get_model(models[args['model']], pretrained=True)

img = cv2.imread(args['input'])
H, W, C = img.shape

start = time.time()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x = mx.nd.array(img.reshape(H, W, C))
x, img = data.transforms.presets.center_net.transform_test(x, short=512)
class_IDS, scores, bounding_boxes = net(x)
end = time.time()

print(f'[INFO] FPS = {1/(end-start)}') 

ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_IDS[0], class_names=net.classes)
plt.show()
