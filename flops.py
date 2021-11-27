from ptflops import get_model_complexity_info
from models.hourglass import get_large_hourglass_net
from models.dla import get_pose_net as get_dla_net
from models.resnet import get_pose_net as get_resnet

# As per the coco dataset
num_classes = 80
heads = {
    'hm' : num_classes,
    'wh' : num_classes * 2 
}

# Initialize the architectures of CenterNet
# Head Conv :
# -1 for hourglass
# 64 for resnets
# 256 for DLA
hg104 = get_large_hourglass_net(104, heads, head_conv=-1)
dla34 = get_dla_net(34, heads, head_conv=256)
rs18 = get_resnet(18, heads, head_conv=64)
rs101 = get_resnet(101, heads, head_conv=64)

models = {
    'HourGlass-104' : hg104,
    'DLA-34' : dla34,
    'ResNet-18' : rs18,
    'ResNet-101' : rs101
}

for model_name in models:
    model = models[model_name]
    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True,  print_per_layer_stat=False, verbose=False)
    print(f'[INFO] Model name : {model_name}, Number of params : {params}, Computational complexity : {macs}')
