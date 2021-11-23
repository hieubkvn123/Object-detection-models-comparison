# Object-detection-models-comparison
A comparison of CenterNet and other State-of-the-art models in object detection

### 1. Download videos.
To download testing videos, go to videos folder and run the download.sh script.
Videos will be downloaded one by one.
```
cd videos
sh download.sh
```

# TODO:
- [ ] Test and visualize the detection results on the testing youtube videos.
- [ ] Calculate the model complexity using FLOPS on ResNet (18 - 101), HourGlass-104 and DLA-34 architectures.
- [ ] Comparison with YoloV3:
	- [ ] Compare the FPS rate of CenterNet vs YoloV3 on given videos.
	- [ ] Visualize the Precision-Recall curve on a vehicle dataset for both models.
	- [ ] Plot the ROC curves for both models.
- [ ] Read paper reviews and discuss limitations and possible improvements for CenterNet.

# REFERENCES:
- GluonCV: Testing pre-trained CenterNet models : [Link](https://cv.gluon.ai/build/examples_detection/demo_center_net.html)
- GluonCV: Testing pre-trained YoloV3 models : [Link](https://cv.gluon.ai/build/examples_detection/train_yolo_v3.html)
- GluonCV: Model zoo : [Link](https://cv.gluon.ai/model_zoo/detection.html#centernet)
- CenterNet Paper - Objects as Points : [Link](https://arxiv.org/abs/1904.07850)
- Vehicle dataset for validation : [Link](https://drive.google.com/drive/folders/1a-v4os2Ekr-IezLE-pGNJ7R0plZyf6bE)
