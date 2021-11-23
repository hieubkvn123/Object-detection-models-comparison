import cv2
from gluoncv import model_zoo, data, utils
from matplotlib.pyplot as plt

net = model_zoo.get_model('center_net_resnet101_v1b_dcnv2_coco', pretrained=True)

img = cv2.imread('test.jpg')
x, img = data.transforms.presets.center_net.load_test(img, short=512)

class_IDS, scores, bounding_boxes = net(x)
ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], classIDS[0], class_names=net.classes)
plt.show()
