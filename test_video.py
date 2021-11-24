import cv2
import tqdm
import sys, traceback
import time
import mxnet as mx
from argparse import ArgumentParser
from gluoncv import model_zoo, data, utils

from imutils.video import WebcamVideoStream

parser = ArgumentParser()
parser.add_argument('--input', required=True, help='Path to input test video')
parser.add_argument('--model', required=False, default='center', help='Detection model used')
parser.add_argument('--output', required=False, default='videos/output.avi', help='Path to output video')
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
frame_info = []

print('[INFO] Running frame-by-frame prediction')
while(True):
    try:
        frame = vs.read()
        if(frame is None): break

        x = numpy_to_mx(frame, model = args['model'])
        class_IDS, scores, bounding_boxes = net(x)
        frame_info.append((frame, class_IDS, scores, bounding_boxes))
    except:
        traceback.print_exc(file=sys.stdout)
        break

vs.stop()
cv2.destroyAllWindows()

print('[INFO] Writing results ...')

# Input stream
vs = cv2.VideoCapture(args['input']) 

# Output stream
frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
frame_size = (frame_width,frame_height)
fps = 20
writer = cv2.VideoWriter('videos/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)

with tqdm.tqdm(total=len(frame_info)) as pbar:
    for i in range(len(frame_info)):
        try:
            # ret, frame = vs.read()

            frame, class_IDS, scores, bboxes = frame_info[i]
            class_IDS, scores, bboxes = class_IDS.asnumpy(), scores.asnumpy(), bboxes.asnumpy()

            class_IDS = class_IDS[scores >= 0.6]
            bboxes = bboxes[scores >= 0.6].astype('int')

            for (x1, y1, x2, y2) in bboxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)

            writer.write(frame)

            pbar.update(1)
        except:
            traceback.print_exc(file=sys.stdout)
            break

writer.release()
vs.release()
cv2.destroyAllWindows()
