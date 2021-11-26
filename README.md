# Object-detection-models-comparison
A comparison of CenterNet and other State-of-the-art models in object detection

### 1. Download videos.
To download testing videos, go to videos folder and run the download.sh script.
Videos will be downloaded one by one.
```
cd videos
sh download.sh
```

### 2. Test model on individual image.
To test the detection models (yolo and centernet) on an individual image, execute the command:
```
python3 test.py --model <model_name> --input <path_to_image>
```

Where :
- model_name : Is the name of the model you want to test ("yolo" or "center").
- input : Path to the image file.

### 3. Test model on video.
To test the detection models (yolo and centernet) on a video, execute the command:
```
python3 test_video.py --model <model_name> --input <path_to_video>
```
Where :
- model_name : Is the name of the model you want to test ("yolo" or "center").
- input : Path to the video file.

The script will output the prediction in the form of gif into the default path "videos/output.gif".
However, you can change this by providing the script with another option "--output" which specifies
the output GIF path. For example :
```
python3 test_video.py --model yolo --input videos/video_1.mp4 --output videos/output_1.gif
```

#### Sample predictions:
<table>
<tr>
<td><img src="./media/output_1.gif"/></td>		
</tr>
</table>

# TODO:
- [x] Test and visualize the detection results on the testing youtube videos.
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
- Precision-recall curve for object detection : [Link](https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734)
- How to interpret a precision-recall curve : [Link](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
