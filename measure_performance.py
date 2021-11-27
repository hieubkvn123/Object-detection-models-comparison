import cv2
import tqdm
import time
import numpy as np
import mxnet as mx
import sys, traceback
from argparse import ArgumentParser
from gluoncv import model_zoo, data, utils

parser = ArgumentParser()
parser.add_argument('--input', required=True, help='Path to input test video')
parser.add_argument('--model', required=False, default='center', help='Detection model used')
parser.add_argument('--frames', required=False, default=150, help='Max number of frames to read')
parser.add_argument('--metric', required=False, default='fps', help='Evaluation metric, fps or detection rate')
parser.add_argument('--thresh', required=False, default=0.6, help='Detection threshold. Minimum probability for detecting objects')
args = vars(parser.parse_args())

# List of models for detection
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

# Initialize video reader
in_stream = cv2.VideoCapture(args['input'])
frame_width = int(in_stream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
frame_height = int(in_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'[INFO] Frame width = {frame_width}, Frame height = {frame_height}')
time.sleep(2.0)

# Measure FPS - pure inference performance and detection rate
print(f'[INFO] Calculating FPS and detection rate for model {args["model"]}')
frame_count = 0
num_objects = []
start = time.time()
with tqdm.tqdm(total=args['frames']) as pbar:
    while(True):
        try:
            frame_count += 1
            if(frame_count > args['frames']):
                break

            # 1. Load frames from the camera
            ret, frame = in_stream.read()
            if(not ret): 
                break

            # 2. Image preprocessing
            frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            rgb_nd, frame = preprocess_func(frame, short=512, max_size=700)
            
            # 3. Run inference on current frame
            class_IDS, scores, bounding_boxes = net(rgb_nd)

            if(args['metric'] == 'dr'):
                # 4. Check how many predictions has score above the threshold
                scores = scores[0].asnumpy().reshape(-1)
                indices = np.where(scores >= args['thresh'])
                num_obj = scores[indices].shape[0]
                num_objects.append(num_obj)
            
            pbar.update(1)
        except KeyboardInterrupt:
            print('[INFO] Key board interrupted')
            break
        except:
            traceback.print_exc(file=sys.stdout)
            break
end = time.time()

# Release video reader
in_stream.release()
cv2.destroyAllWindows()

# Calculate the FPS and detection rate
elapsed = end - start

print('\n----------------------------------------------')
print(f'[INFO] Time elapsed : {elapsed:.2f} seconds')
if(args['metric'] == 'dr'):
    det_rate = np.array(num_objects).sum() / frame_count
    print(f'[INFO] Detection rate = {det_rate:.2f} objects per frame')
else:
    fps = frame_count / elapsed
    print(f'[INFO] FPS = {fps:.2f}')
