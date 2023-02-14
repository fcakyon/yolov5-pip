from pathlib import Path
import warnings

from yolov5.models.common import AutoShape, DetectMultiBackend
from yolov5.models.experimental import attempt_load
from yolov5.models.yolo import ClassificationModel, SegmentationModel
from yolov5.utils.general import LOGGER, logging
from yolov5.utils.torch_utils import select_device


def load_model(
    model_path, device=None, autoshape=True, verbose=False, hf_token: str = None
):
    """
    Creates a specified YOLOv5 model

    Arguments:
        model_path (str): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov5 logs will be silent
        hf_token (str): huggingface read token for private models

    Returns:
        pytorch model

    (Adapted from yolov5.hubconf.create)
    """
    # set logging
    if not verbose:
        LOGGER.setLevel(logging.WARNING)

    # set device
    device = select_device(device)

    try:
        model = DetectMultiBackend(
            model_path, device=device, fuse=autoshape, hf_token=hf_token
        )  # detection model
        if autoshape:
            if model.pt and isinstance(model.model, ClassificationModel):
                LOGGER.warning(
                    "WARNING ⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. "
                    "You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224)."
                )
            elif model.pt and isinstance(model.model, SegmentationModel):
                LOGGER.warning(
                    "WARNING ⚠️ YOLOv5 SegmentationModel is not yet AutoShape compatible. "
                    "You will not be able to run inference with this model."
                )
            else:
                try:
                    model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
                except Exception as e:
                    LOGGER.warning(f"WARNING ⚠️ autoshape failed: {e}")
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ DetectMultiBackend failed: {e}")
        model = attempt_load(model_path, device=device, fuse=False)  # arbitrary model

    if not verbose:
        LOGGER.setLevel(logging.INFO)  # reset to default

    return model.to(device)


class YOLOv5:
    def __init__(self, model_path, device=None, load_on_init=True):
        warnings.warn("YOLOv5 class is deprecated and will be removed in future release. Use 'model = yolov5.load()' instead.", DeprecationWarning)
        
        self.model_path = model_path
        self.device = device
        if load_on_init:
            Path(model_path).parents[0].mkdir(parents=True, exist_ok=True)
            self.model = load_model(
                model_path=model_path, device=device, autoshape=True
            )
        else:
            self.model = None

    def load_model(self):
        """
        Load yolov5 weight.
        """
        Path(self.model_path).parents[0].mkdir(parents=True, exist_ok=True)
        self.model = load_model(
            model_path=self.model_path, device=self.device, autoshape=True
        )

    def predict(self, image_list, size=640, augment=False):
        """
        Perform yolov5 prediction using loaded model weights.

        Returns results as a yolov5.models.common.Detections object.
        """
        assert self.model is not None, "before predict, you need to call .load_model()"
        results = self.model(ims=image_list, size=size, augment=augment)
        return results


def generate_model_usage_markdown(
    repo_id, ap50, task="object-detection", input_size=640, dataset_id=None
):
    from yolov5 import __version__ as yolov5_version

    if dataset_id is not None:
        datasets_str_1 = f"""
datasets:
- {dataset_id}
"""
        datasets_str_2 = f"""
    dataset:
      type: {dataset_id}
      name: {dataset_id}
      split: validation
"""
    else:
        datasets_str_1 = datasets_str_2 = ""
    return f""" 
---
tags:
- yolov5
- yolo
- vision
- {task}
- pytorch
library_name: yolov5
library_version: {yolov5_version}
inference: false
{datasets_str_1}
model-index:
- name: {repo_id}
  results:
  - task:
      type: {task}
{datasets_str_2}
    metrics:
      - type: precision  # since mAP@0.5 is not available on hf.co/metrics
        value: {ap50}  # min: 0.0 - max: 1.0
        name: mAP@0.5
---

<div align="center">
  <img width="640" alt="{repo_id}" src="https://huggingface.co/{repo_id}/resolve/main/sample_visuals.jpg">
</div>

### How to use

- Install [yolov5](https://github.com/fcakyon/yolov5-pip):

```bash
pip install -U yolov5
```

- Load model and perform prediction:

```python
import yolov5

# load model
model = yolov5.load('{repo_id}')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# perform inference
results = model(img, size={input_size})

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

- Finetune the model on your custom dataset:

```bash
yolov5 train --data data.yaml --img {input_size} --batch 16 --weights {repo_id} --epochs 10
```

**More models available at: [awesome-yolov5-models](https://github.com/keremberke/awesome-yolov5-models)**
"""


def push_model_card_to_hfhub(
    repo_id,
    exp_folder,
    ap50,
    hf_token=None,
    input_size=640,
    task="object-detection",
    private=False,
    dataset_id=None,
):
    from huggingface_hub import upload_file, create_repo

    create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        exist_ok=True,
    )

    # upload sample visual to the repo
    sample_visual_path = Path(exp_folder) / "val_batch0_labels.jpg"
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(sample_visual_path),
        path_in_repo="sample_visuals.jpg",
        commit_message="upload sample visuals",
        token=hf_token,
        repo_type="model",
    )

    # Create model card
    modelcard_markdown = generate_model_usage_markdown(
        repo_id,
        task=task,
        input_size=input_size,
        dataset_id=dataset_id,
        ap50=ap50,
    )
    modelcard_path = Path(exp_folder) / "README.md"
    with open(modelcard_path, "w") as file_object:
        file_object.write(modelcard_markdown)
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(modelcard_path),
        path_in_repo=Path(modelcard_path).name,
        commit_message="Add yolov5 model card",
        token=hf_token,
        repo_type="model",
    )


def push_config_to_hfhub(
    repo_id,
    exp_folder,
    best_ap50=None,
    input_size=640,
    task="object-detection",
    hf_token=None,
    private=False,
):
    """
    Pushes a yolov5 config to huggingface hub

    Arguments:
        repo_id (str): The name of the repository to create on huggingface.co
        exp_folder (str): The path to the experiment folder
        best_ap50 (float): The best ap50 score of the model
        input_size (int): The input size of the model (default: 640)
        task (str): The task of the model (default: object-detection)
        hf_token (str): The huggingface token to use to push the model
        private (bool): Whether the model should be private or not
    """
    from huggingface_hub import upload_file, create_repo
    import json

    config = {"input_size": input_size, "task": task, "best_ap50": best_ap50}
    config_path = Path(exp_folder) / "config.json"
    with open(config_path, "w") as file_object:
        json.dump(config, file_object)

    create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        exist_ok=True,
    )
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(config_path),
        path_in_repo=Path(config_path).name,
        commit_message="Add yolov5 config",
        token=hf_token,
        repo_type="model",
    )


def push_model_to_hfhub(repo_id, exp_folder, hf_token=None, private=False):
    """
    Pushes a yolov5 model to huggingface hub

    Arguments:
        repo_id (str): huggingface repo id to be uploaded to
        exp_folder (str): yolov5 experiment folder path
        hf_token (str): huggingface write token
        private (bool): whether to make the repo private or not
    """
    from huggingface_hub import upload_file, create_repo
    from glob import glob

    best_model_path = Path(exp_folder) / "weights/best.pt"
    tensorboard_log_path = glob(f"{exp_folder}/events.out.tfevents*")[-1]

    create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        exist_ok=True,
    )
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(best_model_path),
        path_in_repo=Path(best_model_path).name,
        commit_message="Upload yolov5 best model",
        token=hf_token,
        repo_type="model",
    )
    upload_file(
        repo_id=repo_id,
        path_or_fileobj=str(tensorboard_log_path),
        path_in_repo=Path(tensorboard_log_path).name,
        commit_message="Upload yolov5 tensorboard logs",
        token=hf_token,
        repo_type="model",
    )


def push_to_hfhub(
    hf_model_id,
    hf_token,
    hf_private,
    save_dir,
    hf_dataset_id=None,
    input_size=640,
    best_ap50=None,
    task="object-detection",
):
    from yolov5.utils.general import colorstr
    from yolov5.helpers import (
        push_config_to_hfhub,
        push_model_card_to_hfhub,
        push_model_to_hfhub,
    )

    LOGGER.info(f"{colorstr('hub:')} Pushing to hf.co/{hf_model_id}")

    push_config_to_hfhub(
        repo_id=hf_model_id,
        exp_folder=save_dir,
        best_ap50=best_ap50,
        input_size=input_size,
        task=task,
        hf_token=hf_token,
        private=hf_private,
    )
    push_model_card_to_hfhub(
        repo_id=hf_model_id,
        exp_folder=save_dir,
        input_size=input_size,
        task=task,
        hf_token=hf_token,
        private=hf_private,
        dataset_id=hf_dataset_id,
        ap50=best_ap50,
    )
    push_model_to_hfhub(
        repo_id=hf_model_id, exp_folder=save_dir, hf_token=hf_token, private=hf_private
    )


def convert_coco_dataset_to_yolo(opt, save_dir):
    import yaml
    from shutil import copyfile

    is_coco_data = False
    has_yolo_s3_data_dir = False
    with open(opt.data, errors="ignore") as f:
        data_info = yaml.safe_load(f)  # load data dict
        if data_info.get("train_json_path") is not None:
            is_coco_data = True
        if data_info.get("yolo_s3_data_dir") is not None:
            has_yolo_s3_data_dir = True

    if has_yolo_s3_data_dir and opt.upload_dataset:
        raise ValueError(
            "'--upload_dataset' argument cannot be passed when 'yolo_s3_data_dir' field is not empty in 'data.yaml'."
        )

    if is_coco_data:
        from sahi.utils.coco import export_coco_as_yolov5_via_yml
        from yolov5.utils.general import is_colab
        
        disable_symlink = False
        if is_colab():
             disable_symlink = True

        data = export_coco_as_yolov5_via_yml(
            yml_path=opt.data, output_dir=save_dir / "data", disable_symlink=disable_symlink
        )
        opt.data = data

        # add coco fields to data.yaml
        with open(data, errors="ignore") as f:
            updated_data_info = yaml.safe_load(f)  # load data dict
            updated_data_info["train_json_path"] = data_info["train_json_path"]
            updated_data_info["val_json_path"] = data_info["val_json_path"]
            updated_data_info["train_image_dir"] = data_info["train_image_dir"]
            updated_data_info["val_image_dir"] = data_info["val_image_dir"]
            if data_info.get("yolo_s3_data_dir") is not None:
                updated_data_info["yolo_s3_data_dir"] = data_info["yolo_s3_data_dir"]
            if data_info.get("coco_s3_data_dir") is not None:
                updated_data_info["coco_s3_data_dir"] = data_info["coco_s3_data_dir"]
        with open(data, "w") as f:
            yaml.dump(updated_data_info, f)

        w = save_dir / "data" / "coco"  # coco dir
        w.mkdir(parents=True, exist_ok=True)  # make dir

        # copy train.json/val.json and coco_data.yml into data/coco/ folder
        if (
            "train_json_path" in data_info
            and Path(data_info["train_json_path"]).is_file()
        ):
            copyfile(data_info["train_json_path"], str(w / "train.json"))
        if "val_json_path" in data_info and Path(data_info["val_json_path"]).is_file():
            copyfile(data_info["val_json_path"], str(w / "val.json"))


def upload_to_s3(opt, data, save_dir):
    import yaml
    import os
    from yolov5.utils.general import colorstr
    from yolov5.utils.aws import upload_file_to_s3, upload_folder_to_s3

    with open(data, errors="ignore") as f:
        data_info = yaml.safe_load(f)  # load data dict
    # upload yolo formatted data to s3
    s3_folder = "s3://" + str(
        Path(opt.s3_upload_dir.replace("s3://", "")) / save_dir.name / "data"
    ).replace(os.sep, "/")
    LOGGER.info(f"{colorstr('aws:')} Uploading yolo formatted dataset to {s3_folder}")
    s3_file = s3_folder + "/data.yaml"
    result = upload_file_to_s3(local_file=opt.data, s3_file=s3_file)
    s3_folder_train = s3_folder + "/train/"
    result = upload_folder_to_s3(
        local_folder=data_info["train"], s3_folder=s3_folder_train
    )
    s3_folder_val = s3_folder + "/val/"
    result = upload_folder_to_s3(local_folder=data_info["val"], s3_folder=s3_folder_val)
    if result:
        LOGGER.info(
            f"{colorstr('aws:')} Dataset has been successfully uploaded to {s3_folder}"
        )
