import cv2
import time
import mxnet as mx
from argparse import ArgumentParser
from gluoncv import model_zoo, data, utils

from imutils.video import WebcamVideoStream

parser = ArgumentParser()
parser.add_argument('--input', required=True, help='Path to input test video')
parser.add_argument('--model', required=False, default='center', help='Detection model used')
args = vars(parser.parse_args())

models = {
    'center' : 'center_net_resnet101_v1b_dcnv2_coco',
    'yolo' : 'yolo3_darknet53_coco'
}

if(args['model'] not in models.keys()):
    raise Exception('Invalid model name')

# Initialize the preprocessing function
preprocess_func = data.transforms.presets.center_net.transform_test
if(args['model'] == 'yolo'):
    preprocess_func = data.transforms.presets.yolo.transform_test

# Initialize the model
net = model_zoo.get_model(models[args['model']], pretrained=True)

def numpy_to_mx(img, model='center'):
    global preprocess_func

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, C = img.shape

    x = mx.nd.array(img.reshape(H, W, C))
    x, img = preprocess_func(x, short=512)

    return x

vs = WebcamVideoStream(src=args['input']).start()
time.sleep(2.0)

while(True):
    try:
        frame = vs.read()
        x = numpy_to_mx(frame, model = args['model'])
        class_IDS, scores, bounding_boxes = net(x)

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if(key == ord('q')):
            break
    except:
        break

vs.stop()
cv2.destroyAllWindows()
