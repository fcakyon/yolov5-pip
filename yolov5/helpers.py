from pathlib import Path

from yolov5.models.experimental import attempt_load
from yolov5.models.yolo import Model
from yolov5.utils.torch_utils import select_device


class OptFactory:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def load_model(model_path, device):
    model = attempt_load(weights=model_path, map_location=device)

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    hub_model = hub_model.autoshape()
    return hub_model

class YOLOv5:
    def __init__(self, model_path, device, load_on_init=True):
        self.model_path = model_path
        self.device = device
        if load_on_init:
            self.model = load_model(model_path, device)
        else:
            self.model = None

    def load_model(self):
        self.model = load_model(model_path, device)

    def predict(image_list, size=640, augment=False):
        assert len(self.model) == 1, "before predict, you need to call .load_model()"
        results = self.model(imgs=image_list, size=size, augment=augment)
        return results

if __name__ == "__main__":
    model_path = "yolov5/weights/yolov5s.pt"
    device = "cuda"
    model = load_model(model_path, device)

    from PIL import Image
    imgs = [Image.open(x) for x in Path("yolov5/data/images").glob("*.jpg")]
    results = model(imgs)
