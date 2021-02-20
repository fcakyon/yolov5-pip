import sys
from pathlib import Path

from yolov5.models.yolo import Model
from yolov5.utils.google_utils import attempt_download
from yolov5.utils.torch_utils import torch


class OptFactory:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

"""
def load_model(model_path, device):
    model = attempt_load(weights=model_path, map_location=device)

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    hub_model = hub_model.autoshape()
    return hub_model
"""

def load_model(model_path, device=None, autoshape=True):
    """
    Creates a specified YOLOv5 model

    Arguments:
        model_path (str): path of the model
        config_path (str): path of the config file
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference

    Returns:
        pytorch model

    (Adapted from yolov5.hubconf.create)
    """
    # set device if not given
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # add yolov5 folder to system path
    here = Path(__file__).parent.absolute()
    yolov5_folder_dir = str(here)
    sys.path.insert(0, yolov5_folder_dir)

    attempt_download(model_path)  # download if not found locally
    model = torch.load(model_path, map_location=torch.device(device))
    if isinstance(model, dict):
        model = model["model"]  # load model
    hub_model = Model(model.yaml, verbose=0).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    model = hub_model

    # remove yolov5 folder from system path
    sys.path.remove(yolov5_folder_dir)

    if autoshape:
        model = model.autoshape()

    return model
    """
    # get config path automatically if given model name is one of the defaults
    default_yolov5_model_names = ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]

    model_name = ntpath.basename(model_path).split(".")[0]
    if model_name in default_yolov5_model_names:
        config_path = Path(__file__).parent / "models" / f"{model_name}.yaml"  # model.yaml path
        
    try:
        model = Model(config_path, verbose=0)
        if pretrained:
            fname = f"{model_name}.pt"  # checkpoint filename
            attempt_download(model_path)  # download if not found locally
            ckpt = torch.load(model_path, map_location=torch.device(device))  # load
            state_dict = ckpt["model"].float().state_dict()  # to FP32
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if model.state_dict()[k].shape == v.shape
            }  # filter
            model.load_state_dict(state_dict, strict=False)  # load
            model.names = ckpt["model"].names  # set class names attribute
            if autoshape:
                model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        return model

    except Exception as e:
        help_url = "https://github.com/ultralytics/yolov5/issues/36"
        s = (
            "Cache maybe be out of date, try force_reload=True. See %s for help."
            % help_url
        )
        raise Exception(s) from e
    """

class YOLOv5:
    def __init__(self, model_path, device=None, load_on_init=True):
        self.model_path = model_path
        self.device = device
        if load_on_init:
            self.model = load_model(model_path=model_path, device=device, autoshape=True)
        else:
            self.model = None

    def load_model(self):
        self.model = load_model(model_path=self.model_path, device=self.device, autoshape=True)

    def predict(self, image_list, size=640, augment=False):
        assert self.model is not None, "before predict, you need to call .load_model()"
        results = self.model(imgs=image_list, size=size, augment=augment)
        return results

if __name__ == "__main__":
    model_path = "yolov5/weights/yolov5s.pt"
    device = "cuda"
    model = load_model(model_path=model_path, config_path=None, device=device)

    from PIL import Image
    imgs = [Image.open(x) for x in Path("yolov5/data/images").glob("*.jpg")]
    results = model(imgs)
