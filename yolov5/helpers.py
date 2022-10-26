from pathlib import Path

from yolov5.models.common import AutoShape, DetectMultiBackend
from yolov5.models.experimental import attempt_load
from yolov5.models.yolo import ClassificationModel, SegmentationModel
from yolov5.utils.general import LOGGER, logging
from yolov5.utils.torch_utils import select_device


def load_model(model_path, device=None, autoshape=True, verbose=False):
    """
    Creates a specified YOLOv5 model

    Arguments:
        model_path (str): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov5 logs will be silent

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
        model = DetectMultiBackend(model_path, device=device, fuse=autoshape)  # detection model
        if autoshape:
            if model.pt and isinstance(model.model, ClassificationModel):
                LOGGER.warning('WARNING ⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. '
                                'You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224).')
            elif model.pt and isinstance(model.model, SegmentationModel):
                LOGGER.warning('WARNING ⚠️ YOLOv5 SegmentationModel is not yet AutoShape compatible. '
                                'You will not be able to run inference with this model.')
            else:
                model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
    except Exception:
        model = attempt_load(model_path, device=device, fuse=False)  # arbitrary model

    if not verbose:
        LOGGER.setLevel(logging.INFO)  # reset to default

    return model.to(device)


class YOLOv5:
    def __init__(self, model_path, device=None, load_on_init=True):
        self.model_path = model_path
        self.device = device
        if load_on_init:
            Path(model_path).parents[0].mkdir(parents=True, exist_ok=True)
            self.model = load_model(model_path=model_path, device=device, autoshape=True)
        else:
            self.model = None

    def load_model(self):
        """
        Load yolov5 weight.
        """
        Path(self.model_path).parents[0].mkdir(parents=True, exist_ok=True)
        self.model = load_model(model_path=self.model_path, device=self.device, autoshape=True)

    def predict(self, image_list, size=640, augment=False):
        """
        Perform yolov5 prediction using loaded model weights.

        Returns results as a yolov5.models.common.Detections object.
        """
        assert self.model is not None, "before predict, you need to call .load_model()"
        results = self.model(ims=image_list, size=size, augment=augment)
        return results

if __name__ == "__main__":
    model_path = "yolov5/weights/yolov5s.pt"
    device = "cuda:0"
    model = load_model(model_path=model_path, device=device)

    from PIL import Image
    imgs = [Image.open(x) for x in Path("yolov5/data/images").glob("*.jpg")]
    results = model(imgs)
