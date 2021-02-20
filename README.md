# Packaged YOLOv5 Object Detector

[![Downloads](https://pepy.tech/badge/yolov5/month)](https://pepy.tech/project/yolov5)
<a href="https://github.com/fcakyon/yolov5-pip/actions/workflows/ci.yml"><img src="https://github.com/fcakyon/yolov5-python/workflows/CI%20CPU%20Testing/badge.svg" alt="CI CPU testing"></a>
<a href="https://github.com/fcakyon/yolov5-pip/actions/workflows/package_testing.yml"><img src="https://github.com/fcakyon/yolov5-python/workflows/Package%20CPU%20Testing/badge.svg" alt="Package CPU testing"></a>

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
pip install "numpy>=1.18.5,<1.20" "matplotlib>=3.2.2,<4"
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

## Scripts

You can download and use [train.py](scripts/train.py), [detect.py](scripts/detect.py) and [test.py](scripts/test.py) scripts after installing the package via `pip`:

### Training

Run commands below to reproduce results on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```

### Inference

[detect.py](scripts/detect.py) runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            rtmp://192.168.1.105/live/test  # rtmp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

To run inference on example images in `data/images`:
```bash
$ python detect.py --source data/images --weights yolov5s.pt --conf 0.25
```


## Status

Builds for the latest commit for `Windows/Linux/MacOS` with `Python3.6/3.7/3.8`: <a href="https://github.com/fcakyon/yolov5-pip/actions/workflows/ci.yml"><img src="https://github.com/fcakyon/yolov5-python/workflows/CI%20CPU%20Testing/badge.svg" alt="CI CPU testing"></a>

Status for the train/detect/test scripts: <a href="https://github.com/fcakyon/yolov5-pip/actions/workflows/package_testing.yml"><img src="https://github.com/fcakyon/yolov5-python/workflows/Package%20CPU%20Testing/badge.svg" alt="Package CPU testing"></a>