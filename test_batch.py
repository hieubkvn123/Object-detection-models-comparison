import os
import cv2
import glob
import json
import tqdm
import time
import pprint
import numpy as np
import mxnet as mx
import sys, traceback

from argparse import ArgumentParser
from gluoncv import model_zoo, data, utils

parser = ArgumentParser()
parser.add_argument('--input', required=True, help='Path to testing data folder')
parser.add_argument('--model', required=False, default='center', help='Detection model used')
parser.add_argument('--thresh', required=False, default=0.6, help='Probability threshold')
parser.add_argument('--output-format', required=False, default='predictions/pr_{}.json', help='Output file format')
args = vars(parser.parse_args())

# Initializations 
models = {
    'center' : 'center_net_resnet101_v1b_dcnv2_coco',
    'yolo' : 'yolo3_darknet53_coco'
}
img_extensions = ['png', 'jpg', 'jpeg']
class_names = ['car', 'motorcycle', 'truck', 'bus', 'bicycle']
annotations = {x:{} for x in class_names}

preprocess_func = data.transforms.presets.center_net.transform_test
if(args['model'] == 'yolo'):
    preprocess_func = data.transforms.presets.yolo.transform_test


net = model_zoo.get_model(models[args['model']], pretrained=True)
output_file = args['output_format'].format(args['model'])
img_files = []
for ext in img_extensions:
    img_files += glob.glob(os.path.join(args['input'], f'*.{ext}'))

# Make predictions for each test image
predictions = []
with tqdm.tqdm(total=len(img_files)) as pbar:
    for img_file in img_files:
        try:
            # 1. Read image
            img = cv2.imread(img_file)
            H, W, C = img.shape

            # 2. Preprocess the image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = mx.nd.array(img.reshape(H, W, C))
            x, img = preprocess_func(x, short=512)
            H_, W_ = x.shape[-2:] 
            
            # 3. Network inference
            class_IDS, scores, bounding_boxes = net(x)
            class_IDS, scores, bounding_boxes = class_IDS[0].asnumpy(), scores[0].asnumpy(), bounding_boxes[0].asnumpy()
            scores = scores.reshape(-1)
            indices = np.where(scores >= args['thresh'])
            
            # 4. Filter out high-score bounding boxes
            class_IDS = class_IDS[indices]
            bounding_boxes = bounding_boxes[indices]
            bounding_boxes[:, (0, 2)] *= W/W_
            bounding_boxes[:, (1, 3)] *= H/H_
            predictions.append((img_file, class_IDS, bounding_boxes, scores[indices]))

            pbar.update(1)
        except KeyboardInterrupt:
            print('[INFO] Keyboard interrupted')
            break
        except:
            traceback.print_exc(file=sys.stdout)
            break


# Store prediction results in JSON
print('[INFO] Writing result to JSON ...')
with tqdm.tqdm(total=len(predictions)) as pbar:
    for img_file, class_IDS, bboxes, scores in predictions:
        for class_id, bbox, score in zip(class_IDS, bboxes, scores):
            class_name = net.classes[int(class_id)].strip().lower()

            if(class_name not in class_names):
                continue

            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            score = np.float64(score)

            if(img_file not in annotations[class_name]):
                annotations[class_name][img_file] = {"boxes" : [], "scores" : []}

            annotations[class_name][img_file]["boxes"].append([x1, y1, x2, y2])
            annotations[class_name][img_file]["scores"].append(score)

        pbar.update(1)

# Save the annotations
with open(output_file, 'w') as f:
    json.dump(annotations, f, indent=4)

print(f'[INFO] Annotations written to {output_file}')
