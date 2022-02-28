from pathlib import Path

from yolov5.models.experimental import attempt_load
from yolov5.models.common import AutoShape, DetectMultiBackend
from yolov5.models.yolo import Model
from yolov5.utils.general import LOGGER, check_requirements, intersect_dicts, logging, yolov5_in_syspath
from yolov5.utils.downloads import attempt_download
from yolov5.utils.torch_utils import select_device, torch

from pathlib import Path


#def load_model(model_path, device=None, autoshape=True, verbose=False):
def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=False, device=None):
    """
    Creates a specified YOLOv5 model

    Arguments:
        model_path (str): path of the model
        config_path (str): path of the config file
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov5 logs will be silent

    Returns:
        pytorch model

    (Adapted from yolov5.hubconf.create)
    """

    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(exclude=('tensorboard', 'thop', 'opencv-python'))
    name = Path(name)
    path = name.with_suffix('.pt') if name.suffix == '' else name  # checkpoint path
    try:
        device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)

        if pretrained and channels == 3 and classes == 80:
            model = DetectMultiBackend(path, device=device)  # download/load FP32 model
            # model = models.experimental.attempt_load(path, map_location=device)  # download/load FP32 model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{path.stem}.yaml'))[0]  # model.yaml path
            model = Model(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
        if autoshape:
            model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
        raise Exception(s) from e


class YOLOv5:
    def __init__(self, model_path, device=None, load_on_init=True):
        self.model_path = model_path
        self.device = device
        if load_on_init:
            Path(model_path).parents[0].mkdir(parents=True, exist_ok=True)
            self.model = _create(name=model_path)
        else:
            self.model = None

    def load_model(self):
        """
        Load yolov5 weight.
        """
        Path(self.model_path).parents[0].mkdir(parents=True, exist_ok=True)
        self.model = _create(name=self.model_path)

    def predict(self, image_list, size=640, augment=False):
        """
        Perform yolov5 prediction using loaded model weights.

        Returns results as a yolov5.models.common.Detections object.
        """
        assert self.model is not None, "before predict, you need to call .load_model()"
        results = self.model(imgs=image_list, size=size, augment=augment)
        return results

if __name__ == "__main__":
    model_path = "yolov5/weights/yolov5s.pt"
    model = _create(name=model_path)

    from PIL import Image
    imgs = [Image.open(x) for x in Path(".").resolve().joinpath("data/images").glob("*.jpg")]
    results = model(imgs)
