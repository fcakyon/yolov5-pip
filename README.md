# Packaged YOLOv5 Object Detector

![CI CPU Testing](https://github.com/fcakyon/yolov5-python/workflows/CI%20CPU%20Testing/badge.svg)

You can finally install [YOLOv5 object detector](https://github.com/ultralytics/yolov5) using [pip](https://pypi.org/project/yolov5/) and integrate into your project easily.

## Overview

This package is up-to-date with the latest release of [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

<img src="https://user-images.githubusercontent.com/26833433/103594689-455e0e00-4eae-11eb-9cdf-7d753e2ceeeb.png" width="1000">

## Installation

- Install yolov5 using pip `(for Python >=3.7)`:

```console
pip install yolov5
```

- Install yolov5 using pip `(for Python 3.6)`:

```console
pip install "numpy>=1.18.5,<1.20"
pip install yolov5
```

## Basic Usage

```python
from PIL import Image
from yolov5 import YOLOv5

# set model params
model_path = "yolov5/weights/yolov5s.pt" # it automatically downloads yolov5s model to given path
device = "cuda" # or "cpu"

# init yolov5 model
yolov5 = YOLOv5(model_path, device)

# load images
image1 = Image.open("yolov5/data/images/bus.jpg")
image2 = Image.open("yolov5/data/images/zidane.jpg")

# perform inference
results = yolov5.predict(image1)

# perform inference with higher input size
results = yolov5.predict(image1, size=1280)

# perform inference with test time augmentation
results = yolov5.predict(image1, augment=True)

# perform inference on multiple images
results = yolov5.predict([image1, image2], size=1280, augment=True)
```

## Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)

## Tests

Builds for the latest commit for `Windows/Linux/MacOS` with `Python3.6/3.7/3.8`: ![CI CPU Testing](https://github.com/fcakyon/yolov5-python/workflows/CI%20CPU%20Testing/badge.svg)