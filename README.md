# Packaged Yolov5 Object Detector

## Overview

This is the packaged version of [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

<img src="https://user-images.githubusercontent.com/26833433/103594689-455e0e00-4eae-11eb-9cdf-7d753e2ceeeb.png" width="1000">

## Installation

- Install yolov5 using pip:

```console
pip install yolov5
```

## Basic Usage

```python
from PIL import Image
from yolov5 import YOLOv5

# set model params
model_path = "yolov5/weights/yolov5s.pt"
device = "cuda"

# init yolov5 model
yolov5 = YOLOv5(model_path, device)

# load images
image1 = Image.open("yolov5/data/images/bus.jpg")
image2 = Image.open("yolov5/data/images/zidane.jpg")

# perform inference
results = yolov5.predict([image1, image2])
```

## Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)