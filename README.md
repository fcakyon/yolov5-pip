<h1 align="center">
  packaged ultralytics/yolov5
</h1>

<h4 align="center">
  pip install yolov5
</h4>

<div align="center">
  <a href="https://pepy.tech/project/yolov5"><img src="https://pepy.tech/badge/yolov5" alt="total downloads"></a>
  <a href="https://pepy.tech/project/yolov5"><img src="https://pepy.tech/badge/yolov5/month" alt="monthly downloads"></a>
  <a href="https://twitter.com/fcakyon"><img src="https://img.shields.io/twitter/follow/fcakyon?color=blue&logo=twitter&style=flat" alt="fcakyon twitter"></a>
  <br>
  <a href="https://badge.fury.io/py/yolov5"><img src="https://badge.fury.io/py/yolov5.svg?kill_cache=1" alt="pypi version"></a>
  <a href="https://github.com/fcakyon/yolov5-pip/actions/workflows/ci.yml"><img src="https://github.com/fcakyon/yolov5-pip/actions/workflows/ci.yml/badge.svg" alt="ci testing"></a>
  <a href="https://github.com/fcakyon/yolov5-pip/actions/workflows/package_testing.yml"><img src="https://github.com/fcakyon/yolov5-pip/actions/workflows/package_testing.yml/badge.svg" alt="package testing"></a>
</div>

## <div align="center">Overview</div>

<div align="center">
You can finally install <a href="https://github.com/ultralytics/yolov5">YOLOv5 object detector</a> using <a href="https://pypi.org/project/yolov5/">pip</a> and integrate into your project easily.

<img src="https://user-images.githubusercontent.com/26833433/136901921-abcfcd9d-f978-4942-9b97-0e3f202907df.png" width="1000">
</div>

<br>
This yolov5 package contains everything from ultralytics/yolov5 <a href="https://github.com/ultralytics/yolov5/tree/357cde9ee7da13ba3095995488c5a23631467f1a">at this commit</a> plus:
<br>
1. Easy installation via pip: <b>pip install yolov5</b>
<br>
2. Full CLI integration with <a href="https://github.com/google/python-fire">fire</a> package
<br>
3. COCO dataset format support (for training)
<br>
4. Full <a href="https://huggingface.co/models?other=yolov5">🤗 Hub</a> integration
<br>
5. <a href="https://aws.amazon.com/s3/">S3</a> support (model and dataset upload)
<br>
6. <a href="https://neptune.ai/">NeptuneAI</a> logger support (metric, model and dataset logging)
<br>
7. Classwise AP logging during experiments



## <div align="center">Install</div>

Install yolov5 using pip (for Python >=3.7)

```console
pip install yolov5
```

## <div align="center">Model Zoo</div>



<div align="center">

Effortlessly explore and use finetuned YOLOv5 models with one line of code: <a href="https://github.com/keremberke/awesome-yolov5-models">awesome-yolov5-models</a>

<a href="https://github.com/keremberke/awesome-yolov5-models"><img src="https://user-images.githubusercontent.com/34196005/210134158-108b24f4-2b8e-43ea-95c8-44731625cde2.gif" width="640"></a>
</div>

## <div align="center">Use from Python</div>

```python
import yolov5

# load pretrained model
model = yolov5.load('yolov5s.pt')

# or load custom model
model = yolov5.load('train/best.pt')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

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
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# show detection bounding boxes on image
results.show()

# save results into "results/" folder
results.save(save_dir='results/')

```

<details closed>
<summary>Train/Detect/Test/Export</summary>

- You can directly use these functions by importing them:

```python
from yolov5 import train, val, detect, export
# from yolov5.classify import train, val, predict
# from yolov5.segment import train, val, predict

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

<details open>
<summary>Training</summary>

- Finetune one of the pretrained YOLOv5 models using your custom `data.yaml`:

```bash
$ yolov5 train --data data.yaml --weights yolov5s.pt --batch-size 16 --img 640
                                          yolov5m.pt              8
                                          yolov5l.pt              4
                                          yolov5x.pt              2
```

- Start a training using a COCO formatted dataset:

```yaml
# data.yml
train_json_path: "train.json"
train_image_dir: "train_image_dir/"
val_json_path: "val.json"
val_image_dir: "val_image_dir/"
```

```bash
$ yolov5 train --data data.yaml --weights yolov5s.pt
```

- Visualize your experiments via [Neptune.AI](https://neptune.ai/) (neptune-client>=0.10.10 required):

```bash
$ yolov5 train --data data.yaml --weights yolov5s.pt --neptune_project NAMESPACE/PROJECT_NAME --neptune_token YOUR_NEPTUNE_TOKEN
```

- Automatically upload weights to [Huggingface Hub](https://huggingface.co/models?other=yolov5):

```bash
$ yolov5 train --data data.yaml --weights yolov5s.pt --hf_model_id username/modelname --hf_token YOUR-HF-WRITE-TOKEN
```

- Automatically upload weights and datasets to AWS S3 (with Neptune.AI artifact tracking integration):

```bash
export AWS_ACCESS_KEY_ID=YOUR_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_KEY
```

```bash
$ yolov5 train --data data.yaml --weights yolov5s.pt --s3_upload_dir YOUR_S3_FOLDER_DIRECTORY --upload_dataset
```

- Add `yolo_s3_data_dir` into `data.yaml` to match Neptune dataset with a present dataset in S3.

```yaml
# data.yml
train_json_path: "train.json"
train_image_dir: "train_image_dir/"
val_json_path: "val.json"
val_image_dir: "val_image_dir/"
yolo_s3_data_dir: s3://bucket_name/data_dir/
```

</details>

<details open>
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

</details>

<details open>
<summary>Export</summary>

You can export your fine-tuned YOLOv5 weights to any format such as `torchscript`, `onnx`, `coreml`, `pb`, `tflite`, `tfjs`:

```bash
$ yolov5 export --weights yolov5s.pt --include torchscript,onnx,coreml,pb,tfjs
```

</details>

<details open>
<summary>Classify</summary>

Train/Val/Predict with YOLOv5 image classifier:

```bash
$ yolov5 classify train --img 640 --data mnist2560 --weights yolov5s-cls.pt --epochs 1
```

```bash
$ yolov5 classify predict --img 640 --weights yolov5s-cls.pt --source images/
```

</details>

<details open>
<summary>Segment</summary>

Train/Val/Predict with YOLOv5 instance segmentation model:

```bash
$ yolov5 segment train --img 640 --weights yolov5s-seg.pt --epochs 1
```

```bash
$ yolov5 segment predict --img 640 --weights yolov5s-seg.pt --source images/
```

</details>
