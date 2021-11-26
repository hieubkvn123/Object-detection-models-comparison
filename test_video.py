import cv2
import imageio
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
parser.add_argument('--output', required=False, default='videos/output.gif', help='Path to output video')
parser.add_argument('--display', required=False, type=bool, default=False, help='Wether to display predicted video or not')
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

# Start the loop
print('[INFO] Running frame-by-frame prediction and writing result ... ')

frames = []
with tqdm.tqdm(total=None) as pbar:
    while(True):
        try:
            # 1. Load frames from the camera
            ret, frame = in_stream.read()
            if(not ret): 
                break

            # 2. Image preprocessing
            frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            rgb_nd, frame = preprocess_func(frame, short=512, max_size=700)
            
            # 3. Run inference on current frame
            class_IDS, scores, bounding_boxes = net(rgb_nd)

            # 4. Plot the bounding boxes
            frame = utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDS[0], class_names=net.classes)

            if(args['display']):
                # 5. Plot the frame with the bounding boxes
                utils.viz.cv_plot_image(frame)

            # 6. Store the result for writing
            frames.append(frame)

            key = cv2.waitKey(1)
            if(key == ord('q')): 
                break

            pbar.update(1)
        except KeyboardInterrupt:
            print('[INFO] Key board interrupted')
            break
        except:
            traceback.print_exc(file=sys.stdout)
            break

# Write the result into a gif
print(f'[INFO] Writing results to gif file {args["output"]}')
images = []
with tqdm.tqdm(total=len(frames)) as pbar:
    for frame in frames:
        images.append(imageio.core.util.Array(frame))
        pbar.update(1)

imageio.mimsave(args['output'], images, fps=10)

in_stream.release()
cv2.destroyAllWindows()

