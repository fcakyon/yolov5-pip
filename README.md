<h1 align="center">
  packaged ultralytics/yolov5
</h1>

<h4 align="center">
  pip install yolov5
</h4>

<div align="center">
  <a href="https://pepy.tech/project/yolov5"><img src="https://pepy.tech/badge/yolov5" alt="total downloads"></a>
  <a href="https://pepy.tech/project/yolov5"><img src="https://pepy.tech/badge/yolov5/month" alt="monthly downloads"></a>
  <a href="https://badge.fury.io/py/yolov5"><img src="https://badge.fury.io/py/yolov5.svg" alt="pypi version"></a>
  <br>
  <a href="https://github.com/fcakyon/yolov5-pip/actions/workflows/ci.yml"><img src="https://github.com/fcakyon/yolov5-pip/actions/workflows/ci.yml/badge.svg" alt="ci testing"></a>
  <a href="https://github.com/fcakyon/yolov5-pip/actions/workflows/package_testing.yml"><img src="https://github.com/fcakyon/yolov5-pip/actions/workflows/package_testing.yml/badge.svg" alt="package testing"></a>
</div>

## <div align="center">Overview</div>

<div align="center">
You can finally install <a href="https://github.com/ultralytics/yolov5">YOLOv5 object detector</a> using <a href="https://pypi.org/project/yolov5/">pip</a> and integrate into your project easily.
</p>
<img src="https://user-images.githubusercontent.com/26833433/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png" width="1000">
</div>

## <div align="center">Install</div>

<details open>
<summary>Install yolov5 using pip (for Python >=3.7)</summary>

```console
pip install yolov5
```

</details>

<details closed>
<summary>Install yolov5 using pip `(for Python 3.6)`</summary>

```console
pip install "numpy>=1.18.5,<1.20" "matplotlib>=3.2.2,<4"
pip install yolov5
```

</details>

## <div align="center">Use from Python</div>


<details open>
<summary>Basic</summary>

```python
import yolov5

# load model
model = yolov5.load('yolov5s')

# set image
img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# perform inference
results = model(img)

# inference with larger input size
results = model(img, size=1280)

# inference with test time augmentation
results = model(img, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, x2, y1, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# show detection bounding boxes on image
results.show()

# save results into "results/" folder
results.save(save_dir='results/')

```

</details>

<details closed>
<summary>Alternative</summary>

```python
from yolov5 import YOLOv5

# set model params
model_path = "yolov5/weights/yolov5s.pt" # it automatically downloads yolov5s model to given path
device = "cuda" # or "cpu"

# init yolov5 model
yolov5 = YOLOv5(model_path, device)

# load images
image1 = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'
image2 = 'https://github.com/ultralytics/yolov5/blob/master/data/images/bus.jpg'

# perform inference
results = yolov5.predict(image1)

# perform inference with larger input size
results = yolov5.predict(image1, size=1280)

# perform inference with test time augmentation
results = yolov5.predict(image1, augment=True)

# perform inference on multiple images
results = yolov5.predict([image1, image2], size=1280, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, x2, y1, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# show detection bounding boxes on image
results.show()

# save results into "results/" folder
results.save(save_dir='results/')
```

</details>

<details open>
<summary>Train/Detect/Test/Export</summary>

- You can directly use these functions by importing them:

```python
from yolov5 import train, val, detect, export

train.run(imgsz=640, data='coco128.yaml')
val.run(imgsz=640, data='coco128.yaml', weights='yolov5s.pt')
detect.run(imgsz=640)
export.run(imgsz=640, weights='yolov5s.pt')
```

- You can pass any argument as input:

```python
from yolov5 import detect

img_url = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

detect.run(source=img_url, weights="yolov5s6.pt", conf_thres=0.25, imgsz=640)

```

</details>

## <div align="center">Use from CLI</div>

You can call `yolov5 train`, `yolov5 detect`, `yolov5 val` and `yolov5 export` commands after installing the package via `pip`:

<details closed>
<summary>Training</summary>

Run commands below to reproduce results on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).

```bash
$ yolov5 train --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                    yolov5m                                40
                                    yolov5l                                24
                                    yolov5x                                16
```

</details>

<details closed>
<summary>Inference</summary>

yolov5 detect command runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
$ yolov5 detect --source 0  # webcam
                       file.jpg  # image
                       file.mp4  # video
                       path/  # directory
                       path/*.jpg  # glob
                       rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                       rtmp://192.168.1.105/live/test  # rtmp stream
                       http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

To run inference on example images in `yolov5/data/images`:

</details>
